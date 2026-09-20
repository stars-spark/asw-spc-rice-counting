"""Compute cost of the proposed method against SAM 3, measured on the same machine.

Each configuration runs in its own interpreter, so its peak resident memory is its own and
not inherited from another model loaded earlier. Images are taken evenly from each dataset.
Times exclude loading the image from disk; for SAM 3 the one-off model load is reported
separately, and GPU timings are synchronised so they cover the whole forward pass.
"""
import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import RESULTS_ROOT, ensure_dir

METRICS_ROOT = RESULTS_ROOT / "metrics"
PER_DATASET = 10
PROMPT, THRESHOLD = "white seed", 0.40
# Hard limit per configuration. A child that hangs is killed together with every thread
# and process it started, so it cannot keep holding GPU memory after the benchmark ends.
RUN_TIMEOUT_S = 1500


def _samples(per_dataset):
    from src import io_utils, synth
    loaders = {"d1": io_utils.load_d1, "d2": synth.load_d2, "d3": io_utils.load_d3}
    picked = []
    for name, loader in loaders.items():
        samples = loader()
        step = max(len(samples) // per_dataset, 1)
        picked += [(name, samples[i * step]) for i in range(per_dataset)]
    return picked


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_ours(per_dataset):
    from src import batch, counter, io_utils
    batch.pin_threads()  # one thread per process: measured faster than letting BLAS spread
    items = [(name, io_utils.imread(s["path"])) for name, s in _samples(per_dataset)]
    counter.count_rice(items[0][1])  # warm-up: first call pays for imports and allocations

    times = {}
    for name, image in items:
        start = time.perf_counter()
        counter.count_rice(image)
        times.setdefault(name, []).append(time.perf_counter() - start)
    return {"times": times, "rss_mb": _peak_rss_mb()}


def _run_ours_gpu(per_dataset):
    import cupy as cp
    from src import batch, gpu_pipeline, io_utils
    batch.pin_threads()  # the GPU path still runs hulls and the KDE on the host
    items = [(name, io_utils.imread(s["path"])) for name, s in _samples(per_dataset)]
    gpu_pipeline.count_rice_gpu(items[0][1])  # warm-up: CUDA context and kernel compilation
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    start_free, total = cp.cuda.runtime.memGetInfo()

    times = {}
    for name, image in items:
        start = time.perf_counter()
        gpu_pipeline.count_rice_gpu(image)
        cp.cuda.Stream.null.synchronize()
        times.setdefault(name, []).append(time.perf_counter() - start)

    end_free, _ = cp.cuda.runtime.memGetInfo()
    return {"times": times, "rss_mb": _peak_rss_mb(),
            "vram_mb": max(start_free - end_free, 0) / 2**20}


def _run_student(per_dataset, device):
    """The distilled student: one forward pass per image."""
    import torch
    from src import batch, io_utils, student
    batch.pin_threads()
    items = [(name, io_utils.imread(s["path"])) for name, s in _samples(per_dataset)]

    start = time.perf_counter()
    checkpoint = torch.load(student.MODEL_ROOT / "student_sam3.pt", map_location=device)
    model = student.UNet(width=checkpoint["width"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    load_s = time.perf_counter() - start
    n_params = sum(p.numel() for p in model.parameters())

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    student.predict_count(model, items[0][1], device)   # warm-up
    sync()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = {}
    for name, image in items:
        start = time.perf_counter()
        student.predict_count(model, image, device)
        sync()
        times.setdefault(name, []).append(time.perf_counter() - start)

    vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    weights = (student.MODEL_ROOT / "student_sam3.pt").stat().st_size / 2**20
    return {"times": times, "rss_mb": _peak_rss_mb(), "load_s": load_s,
            "params": n_params, "vram_mb": vram, "weights_mb": weights}


def _run_sam3(per_dataset, device, limit=None):
    import torch
    from src import io_utils
    from src.teacher_sam import Sam3Teacher

    items = [(name, io_utils.imread(s["path"])) for name, s in _samples(per_dataset)]
    if limit:
        items = items[:: max(len(items) // limit, 1)][:limit]

    # bfloat16 has no fast path on this CPU and runs orders of magnitude slower there, so a
    # machine without a GPU would load the model in float32; that is what is measured.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    start = time.perf_counter()
    teacher = Sam3Teacher(device=device, dtype=dtype)
    load_s = time.perf_counter() - start
    n_params = sum(p.numel() for p in teacher.model.parameters())

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    teacher.segment(items[0][1], prompt=PROMPT, threshold=THRESHOLD)  # warm-up
    sync()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = {}
    for name, image in items:
        start = time.perf_counter()
        teacher.segment(image, prompt=PROMPT, threshold=THRESHOLD)
        sync()
        times.setdefault(name, []).append(time.perf_counter() - start)

    vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return {"times": times, "rss_mb": _peak_rss_mb(), "load_s": load_s,
            "params": n_params, "vram_mb": vram}


def _in_process(key, per_dataset, cpu_limit):
    """Run one configuration in a fresh interpreter and read back its JSON result line."""
    cmd = [sys.executable, "-m", "src.cost", "--only", key,
           "--per-dataset", str(per_dataset), "--cpu-limit", str(cpu_limit)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise RuntimeError(f"{key} exceeded {RUN_TIMEOUT_S}s and was killed")
    if proc.returncode != 0:
        raise RuntimeError(f"{key} failed:\n{err[-2000:]}")
    line = next(l for l in reversed(out.splitlines()) if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def _weights_mb():
    """Size of the safetensors weights in the local Hugging Face cache.

    Read from the cache directory rather than through `snapshot_download`, which refuses a
    snapshot that lacks the original `sam3.pt` checkpoint - a file transformers never loads.
    """
    from huggingface_hub.constants import HF_HUB_CACHE
    root = Path(HF_HUB_CACHE) / "models--facebook--sam3" / "snapshots"
    return sum(f.stat().st_size for f in root.glob("*/*.safetensors")) / 2**20


def main():
    parser = argparse.ArgumentParser(description="Compute cost: ours vs SAM 3")
    parser.add_argument("--per-dataset", type=int, default=PER_DATASET)
    parser.add_argument("--cpu-limit", type=int, default=3,
                        help="images for SAM 3 on CPU, which is slow")
    parser.add_argument("--only", choices=["ours_cpu", "ours_gpu", "student_gpu",
                                           "sam3_gpu", "sam3_cpu"],
                        help="run one configuration and print its result (used internally)")
    parser.add_argument("--force", action="store_true", help="re-measure saved configurations")
    args = parser.parse_args()

    if args.only:
        if args.only == "ours_cpu":
            result = _run_ours(args.per_dataset)
        elif args.only == "ours_gpu":
            result = _run_ours_gpu(args.per_dataset)
        elif args.only == "student_gpu":
            result = _run_student(args.per_dataset, "cuda")
        elif args.only == "sam3_gpu":
            result = _run_sam3(args.per_dataset, "cuda")
        else:
            result = _run_sam3(args.per_dataset, "cpu", args.cpu_limit)
        print("RESULT " + json.dumps(result), flush=True)
        # Leave without the normal interpreter shutdown: a library thread that never joins
        # would otherwise keep the process, and its GPU memory, alive after the result is in.
        os._exit(0)

    parser_force = args.force
    runs = {}
    for key in ("ours_cpu", "ours_gpu", "student_gpu", "sam3_gpu", "sam3_cpu"):
        # Each configuration is saved as soon as it finishes, so a slow or failed later one
        # does not throw away the earlier measurements.
        saved = METRICS_ROOT / f"cost_{key}.json"
        if saved.exists() and not parser_force:
            runs[key] = json.loads(saved.read_text())
            print(f"{key}: reused {saved.name}")
            continue
        runs[key] = _in_process(key, args.per_dataset, args.cpu_limit)
        ensure_dir(METRICS_ROOT)
        saved.write_text(json.dumps(runs[key]))
        print(f"{key}: done", flush=True)
    runs["sam3_weights_mb"] = _weights_mb()

    rows = []
    for key in ("ours_cpu", "ours_gpu", "student_gpu", "sam3_gpu", "sam3_cpu"):
        run = runs[key]
        all_times = [t for ts in run["times"].values() for t in ts]
        row = {"config": key, "n_images": len(all_times),
               "mean_s": float(np.mean(all_times)), "rss_mb": run["rss_mb"],
               "vram_mb": run.get("vram_mb", 0.0), "load_s": run.get("load_s", 0.0),
               "params": run.get("params", 0), "weights_mb": run.get("weights_mb", 0.0)}
        for name, ts in run["times"].items():
            row[f"{name}_s"] = float(np.mean(ts))
        rows.append(row)
        print(key, json.dumps({k: v for k, v in row.items()}, ensure_ascii=False))

    table = pd.DataFrame(rows)
    sam3_weights = runs["sam3_weights_mb"]
    table.loc[table.config.isin(["sam3_gpu", "sam3_cpu"]), "weights_mb"] = sam3_weights
    ensure_dir(METRICS_ROOT)
    table.to_csv(METRICS_ROOT / "cost.csv", index=False)
    print(f"wrote {METRICS_ROOT / 'cost.csv'}")


if __name__ == "__main__":
    main()

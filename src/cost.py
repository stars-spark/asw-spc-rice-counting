"""在同一台机器上测方法一与 SAM 3 的计算开销。

每种配置在单独的解释器里跑，峰值内存不会继承前面加载过的模型。图片从各数据集均匀抽取。
计时不含读盘；SAM 3 加载模型的一次性时间单独报告，GPU 计时做了同步，覆盖整个前向过程。
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
# 每种配置的硬性时限。子进程卡住时连同它启动的线程和进程一起杀掉，免得测完后还占着显存。
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
    batch.pin_threads()  # 每个进程单线程，实测比让 BLAS 多线程快
    items = [(name, io_utils.imread(s["path"])) for name, s in _samples(per_dataset)]
    counter.count_rice(items[0][1])  # 预热，第一次调用要付导入和分配内存的开销

    times = {}
    for name, image in items:
        start = time.perf_counter()
        counter.count_rice(image)
        times.setdefault(name, []).append(time.perf_counter() - start)
    return {"times": times, "rss_mb": _peak_rss_mb()}


def _run_ours_gpu(per_dataset):
    import cupy as cp
    from src import batch, gpu_pipeline, io_utils
    batch.pin_threads()  # GPU 版的凸包和核密度估计仍在主机上算
    items = [(name, io_utils.imread(s["path"])) for name, s in _samples(per_dataset)]
    gpu_pipeline.count_rice_gpu(items[0][1])  # 预热，CUDA 上下文和内核编译
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
    """蒸馏得到的学生模型，每张图一次前向。"""
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

    student.predict_count(model, items[0][1], device)  # 预热
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

    # 这台 CPU 上 bfloat16 没有快速路径，慢几个数量级，没有 GPU 的机器会用 float32 加载，这里测的就是它。
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    start = time.perf_counter()
    teacher = Sam3Teacher(device=device, dtype=dtype)
    load_s = time.perf_counter() - start
    n_params = sum(p.numel() for p in teacher.model.parameters())

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    teacher.segment(items[0][1], prompt=PROMPT, threshold=THRESHOLD)  # 预热
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
    """在新的解释器里跑一种配置，读回它输出的那行 JSON。"""
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
    """本地 Hugging Face 缓存里 safetensors 权重的大小。

    直接读缓存目录。snapshot_download 会因为缺少原始的 sam3.pt 而报错，但 transformers 根本不加载那个文件。
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
        # 不走正常的解释器退出流程，否则某个不 join 的库线程会让进程和显存一直留着。
        os._exit(0)

    parser_force = args.force
    runs = {}
    for key in ("ours_cpu", "ours_gpu", "student_gpu", "sam3_gpu", "sam3_cpu"):
        # 每种配置跑完立刻保存，后面的慢了或失败了也不会丢掉前面的结果。
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

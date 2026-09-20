"""Controlled-degradation study on the synthetic set.

Counting accuracy is reported against four degradations applied to the same scenes: blur,
sensor noise, contrast compression and loss of resolution. The synthetic set is used because
its counts are exact by construction, so the curves measure the methods rather than the
annotation - and because the same scene can be degraded to any level, which no photographed
set allows.

Every degradation is expressed in units of the grain itself (the nominal minor axis of a
synthesised grain), so a level means the same thing regardless of how the scenes were
rendered.
"""
import argparse
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from src import baselines, counter, io_utils, preprocess, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

METRICS_ROOT = RESULTS_ROOT / "metrics"

# Synthesised grains are drawn at a fixed major axis, so their width is known in advance and
# can serve as the unit the degradation levels are quoted in.
GRAIN_WIDTH_PX = synth.TARGET_MAJOR_PX / 2.8

LEVELS = {
    # Gaussian blur, standard deviation as a fraction of one grain width.
    "blur": (0.0, 0.1, 0.2, 0.35, 0.5, 0.7),
    # Additive Gaussian noise, standard deviation in grey levels.
    "noise": (0.0, 5.0, 10.0, 20.0, 30.0, 45.0),
    # Contrast retained after compressing every channel towards the frame mean.
    "contrast": (1.0, 0.7, 0.5, 0.3, 0.15, 0.08),
    # Downsampling factor; the method then sees genuinely fewer pixels per grain.
    "resolution": (1.0, 1.5, 2.0, 3.0, 4.5, 6.0),
}

METHODS = {
    "B1_components": lambda pre: baselines.b1_connected_components(pre["calib"]),
    "B4_erosion_watershed": lambda pre: baselines.b4_erosion_watershed(pre["mask"], pre["calib"]),
    "Ours_ASW_SPC": lambda pre: counter.count_rice(None, pre=pre),
}


def degrade(image, axis, level, rng):
    """Apply one degradation at one level. Level zero of every axis is the original."""
    if axis == "blur":
        sigma = level * GRAIN_WIDTH_PX
        if sigma <= 0:
            return image
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if axis == "noise":
        if level <= 0:
            return image
        noisy = image.astype(np.float32) + rng.normal(0.0, level, image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    if axis == "contrast":
        if level >= 1.0:
            return image
        mean = float(image.mean())
        return np.clip(mean + (image.astype(np.float32) - mean) * level, 0, 255).astype(np.uint8)

    if axis == "resolution":
        if level <= 1.0:
            return image
        height, width = image.shape[:2]
        size = (max(int(round(width / level)), 16), max(int(round(height / level)), 16))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    raise ValueError(f"unknown degradation axis: {axis}")


def _evaluate_one(job):
    """Counts of every method on one scene under one degradation level."""
    sample, axis, level = job
    rng = np.random.default_rng(abs(hash((sample["file_name"], axis, level))) % (2 ** 32))
    image = degrade(io_utils.imread(sample["path"]), axis, level, rng)

    rows = []
    try:
        pre = preprocess.preprocess(image)
    except ValueError:
        # No candidate binarisation described a field of grains. The scene is not skipped:
        # a degradation severe enough to defeat the stage is a result, and recording it as
        # a count of zero is what the error curve should show.
        for name in METHODS:
            rows.append({"axis": axis, "level": level, "method": name,
                         "file_name": sample["file_name"], "gt": sample["gt_count"],
                         "pred": 0, "failed": True})
        return rows

    for name, fn in METHODS.items():
        try:
            pred = int(fn(pre))
        except Exception:
            pred = 0
        rows.append({"axis": axis, "level": level, "method": name,
                     "file_name": sample["file_name"], "gt": sample["gt_count"],
                     "pred": pred, "failed": False})
    return rows


def run(workers=10, limit=None):
    samples = synth.load_d2()[:limit]

    jobs = []
    for axis, levels in LEVELS.items():
        for level in levels:
            for sample in samples:
                jobs.append((sample, axis, level))

    print(f"{len(samples)} scenes x {sum(len(v) for v in LEVELS.values())} levels "
          f"= {len(jobs)} runs on {workers} workers")

    from src import batch
    with Pool(workers, initializer=batch.pin_threads) as pool:
        collected = pool.map(_evaluate_one, jobs, chunksize=4)

    per_run = pd.DataFrame([row for rows in collected for row in rows])
    per_run["error"] = per_run["pred"] - per_run["gt"]

    # The mean and the median are both reported because they disagree where it matters: a
    # degradation that defeats the binarisation on a couple of scenes sends their counts
    # into the thousands, which moves the mean by two orders of magnitude while leaving the
    # typical scene untouched. Quoting only the mean would describe a method that collapses;
    # quoting only the median would hide that it can.
    summary = (per_run.groupby(["axis", "level", "method"])
               .agg(n=("error", "size"),
                    MAE=("error", lambda e: float(np.mean(np.abs(e)))),
                    median_AE=("error", lambda e: float(np.median(np.abs(e)))),
                    bias=("error", "mean"),
                    blowups=("error", lambda e: int((np.abs(e) > 100).sum())),
                    failures=("failed", "sum"))
               .reset_index())
    return per_run, summary


def main():
    parser = argparse.ArgumentParser(description="Controlled-degradation robustness study")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    per_run, summary = run(workers=args.workers, limit=args.limit)

    ensure_dir(METRICS_ROOT)
    per_run.to_csv(METRICS_ROOT / "robustness_per_run.csv", index=False)
    summary.to_csv(METRICS_ROOT / "robustness.csv", index=False)

    pd.set_option("display.width", 200)
    for axis in LEVELS:
        block = summary[summary.axis == axis]
        print(f"\n=== {axis} (MAE) ===")
        print(block.pivot_table(index="level", columns="method", values="MAE")
              .to_string(float_format=lambda v: f"{v:.2f}"))

    print(f"\nwrote {METRICS_ROOT / 'robustness.csv'} and robustness_per_run.csv")


if __name__ == "__main__":
    main()

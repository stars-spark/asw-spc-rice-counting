"""在 D2 上做可控退化实验。

对同一批场景分别加模糊、噪声、降对比度、降分辨率四种退化。用合成图是因为它的真值按构造是精确的，
曲线反映的是方法而不是标注，同一场景还可以退化到任意程度，照片做不到。
退化程度都以米粒本身为单位，即合成米粒的名义短轴，与场景怎么渲染无关。
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

# 合成米粒的长轴固定，粒宽事先已知，退化程度用它做单位。
GRAIN_WIDTH_PX = synth.TARGET_MAJOR_PX / 2.8

LEVELS = {
    # 高斯模糊，标准差以粒宽的比例计。
    "blur": (0.0, 0.1, 0.2, 0.35, 0.5, 0.7),
    # 加性高斯噪声，标准差以灰度级计。
    "noise": (0.0, 5.0, 10.0, 20.0, 30.0, 45.0),
    # 把各通道向全图均值压缩后保留的对比度比例。
    "contrast": (1.0, 0.7, 0.5, 0.3, 0.15, 0.08),
    # 下采样倍数，方法看到的每粒米像素确实变少了。
    "resolution": (1.0, 1.5, 2.0, 3.0, 4.5, 6.0),
}

METHODS = {
    "B1_components": lambda pre: baselines.b1_connected_components(pre["calib"]),
    "B4_erosion_watershed": lambda pre: baselines.b4_erosion_watershed(pre["mask"], pre["calib"]),
    "Ours_ASW_SPC": lambda pre: counter.count_rice(None, pre=pre),
}


def degrade(image, axis, level, rng):
    """按某一程度施加一种退化，每种退化的第 0 档都是原图。"""
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
    """一个场景在某一退化程度下各方法的计数。"""
    sample, axis, level = job
    rng = np.random.default_rng(abs(hash((sample["file_name"], axis, level))) % (2 ** 32))
    image = degrade(io_utils.imread(sample["path"]), axis, level, rng)

    rows = []
    try:
        pre = preprocess.preprocess(image)
    except ValueError:
        # 没有候选二值图像一片米粒。这个场景不跳过，退化严重到二值化失败本身就是结果，记为 0 粒。
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

    # 均值和中位数都报。二值化在一两个场景上失效时计数会到几千，均值被拉高两个数量级，
    # 典型场景却没变。只报均值会显得方法整体崩溃，只报中位数又会把崩溃藏起来。
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

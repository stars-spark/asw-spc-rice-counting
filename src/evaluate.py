import argparse
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines, counter, io_utils, preprocess, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

CACHE_ROOT = RESULTS_ROOT / "cache"
METRICS_ROOT = RESULTS_ROOT / "metrics"

METHODS = {
    "B1_components": lambda pre: baselines.b1_connected_components(pre["calib"]),
    "B2_area": lambda pre: baselines.b2_area_estimate(pre["calib"]),
    "B3_dist_watershed": lambda pre: baselines.b3_distance_watershed(pre["mask"], pre["calib"]),
    "B4_erosion_watershed": lambda pre: baselines.b4_erosion_watershed(pre["mask"], pre["calib"]),
    "B5_concave_ellipse": lambda pre: baselines.b5_concave_ellipse(pre["mask"], pre["calib"]),
    "Ours_ASW_SPC": lambda pre: counter.count_rice(None, pre=pre),
}


def _pipeline_fingerprint():
    """对决定缓存掩膜的模块取哈希，代码改了缓存就重建。"""
    digest = hashlib.sha256()
    for name in ("preprocess.py", "calibrate.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def load_dataset(name):
    if name == "d1":
        return io_utils.load_d1()
    if name == "d2":
        return synth.load_d2()
    if name == "d3":
        return io_utils.load_d3()
    if name == "d4":
        return io_utils.load_d4()
    raise ValueError(f"unknown dataset: {name}")


def preprocessed(name, use_cache=True):
    """做预处理或读缓存，预处理占了大部分运行时间。"""
    ensure_dir(CACHE_ROOT)
    cache_path = CACHE_ROOT / f"{name}_{_pipeline_fingerprint()}.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    items = []
    failures = []
    t0 = time.time()
    for sample in load_dataset(name):
        image = io_utils.imread(sample["path"])
        try:
            pre = preprocess.preprocess(image)
        except ValueError as exc:
            # 没有一个候选二值图像一片米粒。记下来而不是抛异常，免得一张图中断整轮评测。
            failures.append((sample["file_name"], str(exc)))
            continue
        items.append(
            {
                "file_name": sample["file_name"],
                "gt": sample["gt_count"],
                "touch_prob": sample.get("touch_prob"),
                "pre": pre,
            }
        )
    print(f"  preprocessed {name}: {len(items)} images in {time.time() - t0:.0f}s")
    if failures:
        print(f"  {len(failures)} image(s) had no usable binarisation: "
              f"{', '.join(n for n, _ in failures[:5])}")

    if use_cache:
        for old in CACHE_ROOT.glob(f"{name}_*.pkl"):
            old.unlink()
        with open(cache_path, "wb") as f:
            pickle.dump(items, f)
    return items


def b5_configuration_matrix(datasets=("d1", "d2", "d3")):
    """B5 在不同圆盘半径、椭圆修正开和关下的结果。

    角点响应要求圆盘比物体小得多，半径决定了它在某个米粒大小上能不能用。
    只报一种设置对它不公平，这里给出每个数据集上的最好情况。
    """
    rows = []
    for radius_ratio in (0.5, 1.0, 2.0):
        for use_ellipse in (True, False):
            row = {"radius_ratio": radius_ratio, "ellipse_correction": use_ellipse}
            for name in datasets:
                items = preprocessed(name)
                errors = [
                    baselines.b5_concave_ellipse(
                        it["pre"]["mask"], it["pre"]["calib"],
                        use_ellipse=use_ellipse, radius_ratio=radius_ratio) - it["gt"]
                    for it in items
                ]
                row[f"{name}_MAE"] = float(np.mean(np.abs(errors)))
            rows.append(row)
            print(f"  B5 radius={radius_ratio} ellipse={use_ellipse}: "
                  + "  ".join(f"{n.upper()}={rows[-1][f'{n}_MAE']:.2f}" for n in datasets))

    table = pd.DataFrame(rows)
    ensure_dir(METRICS_ROOT)
    table.to_csv(METRICS_ROOT / "b5_matrix.csv", index=False)
    print(f"wrote {METRICS_ROOT / 'b5_matrix.csv'}")
    return table


def metrics(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = np.abs(pred - gt)
    mape = float(np.mean(err / gt))
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    return {
        "n": int(gt.size),
        "MAE": float(err.mean()),
        "MAPE_%": mape * 100,
        "accuracy_%": (1 - mape) * 100,
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "hit_pm1_%": float(np.mean(err <= 1) * 100),
        "hit_pm2_%": float(np.mean(err <= 2) * 100),
        "worst": float(err.max()),
        "bias": float(np.mean(pred - gt)),
    }


def run(datasets=("d1", "d2", "d3"), use_cache=True):
    summary, per_image = [], []

    for name in datasets:
        items = preprocessed(name, use_cache=use_cache)
        for method, fn in METHODS.items():
            preds = [fn(it["pre"]) for it in items]
            gts = [it["gt"] for it in items]

            row = {"dataset": name, "method": method, "stratum": "all"}
            row.update(metrics(gts, preds))
            summary.append(row)

            for it, pred in zip(items, preds):
                per_image.append(
                    {
                        "dataset": name,
                        "method": method,
                        "file_name": it["file_name"],
                        "touch_prob": it["touch_prob"],
                        "gt": it["gt"],
                        "pred": pred,
                        "error": pred - it["gt"],
                    }
                )

            # D2 有生成时的粘连率。照片没有这个标签，按它自己二值化后的粘连程度分层，
            # 即标注米粒落在共享连通域里的比例。
            if name == "d4":
                for it in items:
                    calib = it["pre"]["calib"]
                    kept = sum(1 for c in calib["components"]
                               if c["area"] >= 0.3 * calib["a0"])
                    merged = 1.0 - kept / max(it["gt"], 1)
                    it["touch_prob"] = round(min(max(merged, 0.0), 0.99) * 2) / 2.0

            strata = sorted({it["touch_prob"] for it in items if it["touch_prob"] is not None})
            for level in strata:
                sel = [(it["gt"], p) for it, p in zip(items, preds) if it["touch_prob"] == level]
                row = {"dataset": name, "method": method, "stratum": f"touch={level:.1f}"}
                row.update(metrics([g for g, _ in sel], [p for _, p in sel]))
                summary.append(row)

    return pd.DataFrame(summary), pd.DataFrame(per_image)


def main():
    from src import batch
    batch.pin_threads()  # 整个数据集跑一遍属于批处理，见 src/batch.py

    parser = argparse.ArgumentParser(description="Evaluate rice counting methods")
    parser.add_argument("--datasets", nargs="+", default=["d1", "d2", "d3"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--b5-matrix", action="store_true",
                        help="also sweep B5 over disc radius and ellipse correction")
    args = parser.parse_args()

    summary, per_image = run(datasets=args.datasets, use_cache=not args.no_cache)

    ensure_dir(METRICS_ROOT)
    summary.to_csv(METRICS_ROOT / "summary.csv", index=False)
    per_image.to_csv(METRICS_ROOT / "per_image.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    for name in args.datasets:
        block = summary[(summary.dataset == name) & (summary.stratum == "all")]
        print(f"\n=== {name.upper()} ===")
        print(block[["method", "n", "MAE", "MAPE_%", "accuracy_%", "R2", "hit_pm2_%", "worst", "bias"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    strat = summary[summary.stratum != "all"]
    if not strat.empty:
        print("\n=== D2 by touching level (MAE) ===")
        print(strat.pivot_table(index="stratum", columns="method", values="MAE")
              .to_string(float_format=lambda v: f"{v:.2f}"))

    print(f"\nwrote {METRICS_ROOT / 'summary.csv'} and per_image.csv")

    if args.b5_matrix:
        print("\n=== B5 configuration matrix (MAE) ===")
        b5_configuration_matrix(datasets=tuple(args.datasets))


if __name__ == "__main__":
    main()

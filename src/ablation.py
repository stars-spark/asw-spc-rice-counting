"""方法一的消融实验，每项去掉一个设计，看误差变化。

预处理最费时，只改计数阶段的消融复用缓存的掩膜，改二值化的几项重新预处理。
"""
import argparse
from contextlib import contextmanager

import numpy as np
from skimage.measure import label
from skimage.morphology import h_maxima
import pandas as pd

from src import calibrate, correct, counter, evaluate, io_utils, preprocess, segment, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

METRICS_ROOT = RESULTS_ROOT / "metrics"
COUNTING_SETS = ("d1", "d2", "d3")
BINARISATION_SETS = ("d1", "d2")


@contextmanager
def patched(obj, **attrs):
    saved = {k: getattr(obj, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


def mae(items, **count_kwargs):
    return float(np.mean([abs(counter.count_rice(None, pre=it["pre"], **count_kwargs) - it["gt"])
                          for it in items]))


def _fixed_scale_mae(items):
    """把逐图标定的 A0 换成整个数据集的平均值，其余不变。"""
    mean_a0 = float(np.mean([it["pre"]["calib"]["a0"] for it in items]))
    errs = []
    for it in items:
        calib = it["pre"]["calib"]
        saved = calib["a0"]
        try:
            calib["a0"] = mean_a0
            errs.append(abs(counter.count_rice(None, pre=it["pre"]) - it["gt"]))
        finally:
            calib["a0"] = saved
    return float(np.mean(errs))


def _no_correction_mae(items):
    """直接数分水岭的区域，不合并碎片，也不按面积补数。"""
    with patched(correct, FRAGMENT_RATIO=0.0, RESIDUAL_RATIO=float("inf")):
        return mae(items, fragment_ratio=0.0, residual_ratio=float("inf"))


def counting_ablations(data):
    rows = []

    def record(name, fn):
        row = {"ablation": name}
        for key in COUNTING_SETS:
            row[key.upper()] = fn(data[key])
        rows.append(row)

    record("full method", lambda items: mae(items))
    record("no self-calibration (A0 = dataset mean)", _fixed_scale_mae)
    record("no shape-prior correction (M5)", _no_correction_mae)

    with patched(segment, FOREIGN_AREA_RATIO=float("inf")):
        record("no foreign-object rejection", lambda items: mae(items))

    with patched(segment, FOREIGN_WIDTH_RATIO=float("inf")):
        record("foreign rejection without the width test", lambda items: mae(items))

    def coin_only_foreign(component, calib):
        """最初的异物判据，要求既凸又比米粒圆。"""
        return (component["area"] > segment.FOREIGN_AREA_RATIO * calib["a0"]
                and component["solidity"] >= calib["solidity0"]
                and component["axis_ratio"] < 0.8 * calib["axis_ratio0"])

    with patched(segment, is_foreign_object=coin_only_foreign):
        record("foreign test also requiring roundness", lambda items: mae(items))

    def width_excess_below_4_5px(component, calib):
        """以前的粘连判据，粒宽不足 4.5 像素时用 width_excess > 1.5 代替凸实度。"""
        if component["area"] <= segment.TOUCH_AREA_RATIO * calib["a0"]:
            return False
        if calib["minor0"] < 4.5:
            return segment.width_excess(component, calib) > 1.5
        return component["solidity"] < calib["solidity0"] - segment.TOUCH_SOLIDITY_MARGIN

    with patched(segment, is_touching=width_excess_below_4_5px):
        record("touching: width excess below 4.5 px (previous rule)", lambda items: mae(items))

    with patched(segment, borders_bright_background=lambda *args: False):
        record("no bright-background test", lambda items: mae(items))

    with patched(counter, SPECK_RATIO=0.3):
        record("speck threshold 0.3 (previous)", lambda items: mae(items))

    with patched(segment, is_dark=lambda *args: False):
        record("no dark-region test", lambda items: mae(items))

    with patched(counter, recover_erased=lambda pre, occupied, reference: (None, [])):
        record("no recovery of erased grains", lambda items: mae(items))

    def or_rule(component, calib):
        return (component["area"] > segment.TOUCH_AREA_RATIO * calib["a0"]
                or component["solidity"] < calib["solidity0"] - segment.TOUCH_SOLIDITY_MARGIN)

    with patched(segment, is_touching=or_rule):
        record("touching gate: OR instead of AND", lambda items: mae(items))

    # 不合并同一峰顶碎出来的重复种子，即修正前的做法
    def unmerged(dist, minor0, beta=segment.BETA):
        h = max(beta * minor0 / 2.0, segment.MIN_DEPTH_PX)
        return label(h_maxima(dist, h) > 0)

    with patched(segment, adaptive_markers=unmerged):
        record("seed pieces not merged", lambda items: mae(items))
    with patched(segment, adaptive_markers=unmerged):
        record("seed pieces not merged, beta = 0.45", lambda items: mae(items, beta=0.45))

    for beta in (0.05, 0.1, 0.15, 0.2, 0.3, 0.45):
        record(f"beta = {beta}", lambda items, b=beta: mae(items, beta=b))

    return rows


def binarisation_ablations():
    """缩小候选范围后重新预处理。"""
    rows = []
    loaders = {"d1": io_utils.load_d1, "d2": synth.load_d2}

    # 只设下限的变体是加上限之前的状态，在 D3 上跑，上限就是为 D3 的失败加的。
    loaders["d3"] = io_utils.load_d3
    variants = {
        "full candidate grid": (dict(channel_names=("gray", "hsv_s")), BINARISATION_SETS),
        # 也跑 D3，保留饱和度通道就是为了它。
        "gray channel only": (dict(channel_names=("gray",)), BINARISATION_SETS + ("d3",)),
        "with 3x3 median filter":
            (dict(channel_names=("gray", "hsv_s"), median_ksize=3), BINARISATION_SETS),
        "one-sided binarisation priors": (dict(channel_names=("gray", "hsv_s")), ("d3",)),
    }

    variants["no post-removal re-validation"] = (dict(channel_names=("gray", "hsv_s")), ("d3",))

    def top_candidate_only(image, **kwargs):
        """直接取分数最高的候选，最初就是这样做的。"""
        best = None
        for cand in preprocess.candidate_masks(image, **kwargs):
            calib = preprocess.score_candidate(cand, image.shape)
            if calib is not None and (best is None or calib["score"] > best[1]["score"]):
                best = (cand, calib)
        if best is None:
            raise ValueError("no usable binarisation candidate")
        cand, calib = best
        mask = preprocess.remove_specks(
            preprocess.drop_substrate(cand["mask"], calib["a0"]), calib["a0"])
        return {"mask": mask, "calib": calibrate.calibrate(mask)}

    for name, (kwargs, datasets) in variants.items():
        row = {"ablation": name}
        for key in datasets:
            errs = []
            for sample in loaders[key]():
                image = io_utils.imread(sample["path"])
                try:
                    if name == "one-sided binarisation priors":
                        with patched(preprocess, MAX_AXIS_RATIO=float("inf"),
                                     MAX_GRAIN_AREA_FRACTION=1.0,
                                     MAX_GRAIN_EXTENT=float("inf")):
                            pre = preprocess.preprocess(image, **kwargs)
                    elif name == "no post-removal re-validation":
                        pre = top_candidate_only(image, **kwargs)
                    else:
                        pre = preprocess.preprocess(image, **kwargs)
                    errs.append(abs(counter.count_rice(None, pre=pre) - sample["gt_count"]))
                except ValueError:
                    errs.append(float(sample["gt_count"]))
            row[key.upper()] = float(np.mean(errs))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run ASW-SPC ablations")
    parser.add_argument("--skip-binarisation", action="store_true",
                        help="skip the ablations that re-run preprocessing")
    args = parser.parse_args()

    data = {key: evaluate.preprocessed(key) for key in COUNTING_SETS}
    rows = counting_ablations(data)
    table = pd.DataFrame(rows)

    print("\n=== counting-stage ablations (MAE) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    if not args.skip_binarisation:
        binary_table = pd.DataFrame(binarisation_ablations())
        print("\n=== binarisation ablations (MAE) ===")
        print(binary_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        table = pd.concat([table, binary_table], ignore_index=True)

    ensure_dir(METRICS_ROOT)
    table.to_csv(METRICS_ROOT / "ablation.csv", index=False)
    print(f"\nwrote {METRICS_ROOT / 'ablation.csv'}")


if __name__ == "__main__":
    main()

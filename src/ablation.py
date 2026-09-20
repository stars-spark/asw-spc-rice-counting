"""Ablations for ASW-SPC. Each entry disables one design decision and reports the cost.

The preprocessing stage dominates runtime, so ablations that only change the counting
stage reuse the cached masks; the two that change binarisation re-run it on D1 and D2 only.
"""
import argparse
from contextlib import contextmanager

import numpy as np
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
    """Replace the per-image A0 with the dataset mean, keeping everything else."""
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
    """Count raw watershed regions: no fragment merging, no concavity-gated area accounting."""
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
        """The foreign test as first written: convex AND rounder than a grain."""
        return (component["area"] > segment.FOREIGN_AREA_RATIO * calib["a0"]
                and component["solidity"] >= calib["solidity0"]
                and component["axis_ratio"] < 0.8 * calib["axis_ratio0"])

    with patched(segment, is_foreign_object=coin_only_foreign):
        record("foreign test also requiring roundness", lambda items: mae(items))

    with patched(segment, MIN_TRUSTED_MINOR_PX=0.0):
        record("no measurement-reliability gate", lambda items: mae(items))

    with patched(segment, COARSE_TOUCH_EXCESS=float("inf")):
        record("reliability gate abstains instead of using width excess",
               lambda items: mae(items))

    def coarse_area_rule(component, calib):
        """Coarse-scale splitting on size alone, with no evidence of touching."""
        if component["area"] <= segment.TOUCH_AREA_RATIO * calib["a0"]:
            return False
        if calib["minor0"] < segment.MIN_TRUSTED_MINOR_PX:
            return component["area"] > 1.9 * calib["a0"]
        return component["solidity"] < calib["solidity0"] - segment.TOUCH_SOLIDITY_MARGIN

    with patched(segment, is_touching=coarse_area_rule):
        record("coarse-scale rule on area instead of width excess", lambda items: mae(items))

    original = segment.is_touching

    def or_rule(component, calib):
        if calib["minor0"] < segment.MIN_TRUSTED_MINOR_PX:
            return False
        return (component["area"] > segment.TOUCH_AREA_RATIO * calib["a0"]
                or component["solidity"] < calib["solidity0"] - segment.TOUCH_SOLIDITY_MARGIN)

    with patched(segment, is_touching=or_rule):
        record("touching gate: OR instead of AND", lambda items: mae(items))

    segment.is_touching = original

    for beta in (0.2, 0.3, 0.45, 0.6, 0.8, 1.0):
        record(f"beta = {beta}", lambda items, b=beta: mae(items, beta=b))

    return rows


def binarisation_ablations():
    """Re-run preprocessing with a reduced candidate grid."""
    rows = []
    loaders = {"d1": io_utils.load_d1, "d2": synth.load_d2}

    # The one-sided-priors variant is the state before the upper bounds were added; it is
    # run on D3, where it is the failure the bounds were introduced for.
    loaders["d3"] = io_utils.load_d3
    variants = {
        "full candidate grid": (dict(channel_names=("gray", "hsv_s")), BINARISATION_SETS),
        "gray channel only": (dict(channel_names=("gray",)), BINARISATION_SETS),
        "with 3x3 median filter":
            (dict(channel_names=("gray", "hsv_s"), median_ksize=3), BINARISATION_SETS),
        "one-sided binarisation priors": (dict(channel_names=("gray", "hsv_s")), ("d3",)),
    }

    variants["no post-removal re-validation"] = (dict(channel_names=("gray", "hsv_s")), ("d3",))

    def top_candidate_only(image, **kwargs):
        """Accept the highest-scoring candidate outright, as the stage first did."""
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

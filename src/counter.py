from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

from src import correct, preprocess, segment

# Regions below this fraction of one grain are debris, not grains. Once concavity also
# decides touching on fine-grained photographs, more clusters are cut, and a cut leaves
# slivers between 0.3 and 0.45 grains that were being counted. Chosen on the real
# photographs alone; the low-resolution set, not consulted, improves with it as well.
SPECK_RATIO = 0.45
RESPLIT = True

# Recovering grains that mask cleaning erased (see `recover_erased`). The size window is
# on the raw-mask scale; the width and elongation bounds are relative to the calibrated
# single grain. The size floor and width floor were chosen on the real photographs alone.
RECOVER_AREA = (0.3, 1.3)
RECOVER_MIN_WIDTH = 0.5
RECOVER_MAX_ELONGATION = 3.0


def _rejected(component, calib, pre):
    """Regions that are not grains whatever their shape: foreign objects and bright edges."""
    return (segment.is_foreign_object(component, calib)
            or segment.borders_bright_background(component, calib, pre.get("bright")))


def recover_erased(pre, occupied, reference):
    """Grains the 3x3 opening erased or shrank below the speck floor, taken from the raw mask.

    Grains three to five pixels wide lose most of their area to one opening, so on the
    fine-grained sets most missed grains were thresholded correctly and then cleaned away.
    Keeping the raw mask instead lets through the noise the opening is there to remove, so
    the raw regions are recovered selectively: only those that nothing was counted or
    rejected on, that are as wide, elongated and bright as a grain, and that do not lie
    against bright background. Sizes are measured against single grains on the raw mask,
    which the opening has not shrunk.
    """
    raw, gray = pre.get("raw_mask"), pre.get("gray")
    calib = pre["calib"]
    if raw is None or gray is None or reference is None:
        return None, []
    raw_labels = label(raw > 0)
    if raw_labels.max() == 0:
        return raw_labels, []
    areas = np.bincount(raw_labels.ravel())

    # One grain on the raw scale: raw regions holding exactly one counted component.
    counted = np.where(occupied["counted"], calib["labels"], 0)
    inside = raw_labels > 0
    partners = defaultdict(set)
    for raw_label, component_label in np.unique(
            np.stack([raw_labels[inside], counted[inside]], axis=1), axis=0):
        partners[int(raw_label)].add(int(component_label))
    singles = [areas[r] for r, found in partners.items()
               if len(found) == 2 and 0 in found
               and 0.5 * calib["a0"] < areas[r] < 3 * calib["a0"]]
    a0_raw = float(np.median(singles)) if len(singles) >= 3 else calib["a0"]

    taken = set(np.unique(raw_labels[occupied["counted"] | occupied["rejected"]]).tolist())
    boxes = ndimage.find_objects(raw_labels)
    shape = {"labels": raw_labels, "minor0": calib["minor0"]}
    recovered = []
    for raw_label in range(1, len(areas)):
        if raw_label in taken:
            continue
        if not RECOVER_AREA[0] * a0_raw <= areas[raw_label] <= RECOVER_AREA[1] * a0_raw:
            continue
        rows, cols = boxes[raw_label - 1]
        region = raw_labels[rows, cols] == raw_label
        props = regionprops(region.astype(np.uint8))[0]
        minor = props.axis_minor_length
        if minor < RECOVER_MIN_WIDTH * calib["minor0"]:
            continue
        if props.axis_major_length / max(minor, 1e-6) > RECOVER_MAX_ELONGATION * calib["axis_ratio0"]:
            continue
        if gray[rows, cols][region].mean() < segment.DARK_RATIO * reference:
            continue
        bbox = (rows.start, cols.start, rows.stop, cols.stop)
        if segment.borders_bright_background({"label": raw_label, "bbox": bbox}, shape,
                                             pre.get("bright")):
            continue
        recovered.append(raw_label)
    return raw_labels, recovered


def count_rice(img_bgr, beta=None, residual_ratio=correct.RESIDUAL_RATIO,
               fragment_ratio=correct.FRAGMENT_RATIO, pre=None, return_debug=False):
    """ASW-SPC: adaptive-scale marker watershed with shape-prior correction.

    Isolated components are counted directly; only components flagged as touching are
    segmented, so a lone grain can never be split by the watershed.
    """
    pre = pre if pre is not None else preprocess.preprocess(img_bgr)
    calib = pre["calib"]
    labels, a0 = calib["labels"], calib["a0"]
    gray = pre.get("gray")

    total = 0
    debug = {"clusters": [], "n_isolated": 0, "n_clusters": 0, "n_foreign": 0,
             "singles": [], "foreign": [], "recovered": [], "raw_labels": None}

    kept = [c for c in calib["components"] if c["area"] >= SPECK_RATIO * a0]
    reference = segment.grain_brightness(
        labels, gray, [c["label"] for c in kept if not _rejected(c, calib, pre)])

    counted_labels, rejected_labels = [], []
    for component in kept:
        if (_rejected(component, calib, pre)
                or segment.is_dark(component, labels, gray, reference)):
            debug["n_foreign"] += 1
            debug["foreign"].append(component["label"])
            rejected_labels.append(component["label"])
            continue

        counted_labels.append(component["label"])
        if not segment.is_touching(component, calib):
            total += 1
            debug["n_isolated"] += 1
            debug["singles"].append(component["label"])
            continue

        # Work inside the component's bounding box: a full-frame distance transform and
        # watershed per cluster is orders of magnitude more pixels for the same result.
        r0, c0, r1, c1 = component["bbox"]
        pad = 2
        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
        r1, c1 = min(labels.shape[0], r1 + pad), min(labels.shape[1], c1 + pad)
        mask = (labels[r0:r1, c0:c1] == component["label"]).astype(np.uint8) * 255

        ws, dist, markers = segment.split_component(mask, calib, beta=beta)
        count, merged, details = correct.correct_cluster(
            ws, a0, solidity0=calib["solidity0"],
            residual_ratio=residual_ratio, fragment_ratio=fragment_ratio
        )
        if RESPLIT:
            count, merged, details = correct.resplit_residuals(
                merged, details, a0, calib["minor0"], solidity0=calib["solidity0"],
                residual_ratio=residual_ratio, fragment_ratio=fragment_ratio
            )
        total += count
        debug["n_clusters"] += 1
        if return_debug:
            debug["clusters"].append(
                {
                    "component": component,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "dist": dist,
                    "markers": markers,
                    "watershed": ws,
                    "merged": merged,
                    "count": count,
                    "details": details,
                }
            )

    occupied = {"counted": np.isin(labels, counted_labels),
                "rejected": np.isin(labels, rejected_labels)}
    raw_labels, recovered = recover_erased(pre, occupied, reference)
    total += len(recovered)
    debug["recovered"], debug["raw_labels"] = recovered, raw_labels

    return (total, pre, debug) if return_debug else total


def label_image(img_bgr=None, pre=None, **kwargs):
    """Full-frame instance map plus what each instance was counted as.

    Regions resolved by area accounting rather than by an actual cut are reported with the
    number of grains attributed to them, so a viewer can tell a genuine split from an
    estimate. The verdicts are the ones `count_rice` reached, not re-derived here.
    """
    total, pre, debug = count_rice(img_bgr, pre=pre, return_debug=True, **kwargs)
    calib = pre["calib"]
    source = calib["labels"]

    instances = np.zeros(source.shape, dtype=np.int32)
    info = {}
    next_label = 1

    for kind, count, found in (("foreign", 0, debug["foreign"]), ("grain", 1, debug["singles"])):
        for component_label in found:
            instances[source == component_label] = next_label
            info[next_label] = {"kind": kind, "count": count}
            next_label += 1

    for cluster in debug["clusters"]:
        r0, c0, r1, c1 = cluster["bbox"]
        merged = cluster["merged"]
        verdicts = {d["label"]: d for d in cluster["details"]}
        window = instances[r0:r1, c0:c1]
        for local_label in np.unique(merged):
            if local_label == 0:
                continue
            detail = verdicts.get(local_label, {"verdict": "grain", "n": 1})
            if detail["n"] == 0:
                continue
            window[merged == local_label] = next_label
            info[next_label] = {"kind": detail["verdict"], "count": detail["n"]}
            next_label += 1

    for raw_label in debug["recovered"]:
        instances[(debug["raw_labels"] == raw_label) & (instances == 0)] = next_label
        info[next_label] = {"kind": "grain", "count": 1}
        next_label += 1

    return instances, info, total, pre

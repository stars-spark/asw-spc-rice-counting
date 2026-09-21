import numpy as np

from src import correct, preprocess, segment

SPECK_RATIO = 0.3
RESPLIT = True


def count_rice(img_bgr, beta=None, residual_ratio=correct.RESIDUAL_RATIO,
               fragment_ratio=correct.FRAGMENT_RATIO, pre=None, return_debug=False):
    """ASW-SPC: adaptive-scale marker watershed with shape-prior correction.

    Isolated components are counted directly; only components flagged as touching are
    segmented, so a lone grain can never be split by the watershed.
    """
    pre = pre if pre is not None else preprocess.preprocess(img_bgr)
    calib = pre["calib"]
    labels, a0 = calib["labels"], calib["a0"]

    total = 0
    debug = {"clusters": [], "n_isolated": 0, "n_clusters": 0, "n_foreign": 0}

    for component in calib["components"]:
        if component["area"] < SPECK_RATIO * a0:
            continue

        if segment.is_foreign_object(component, calib):
            debug["n_foreign"] += 1
            continue

        if not segment.is_touching(component, calib):
            total += 1
            debug["n_isolated"] += 1
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

    return (total, pre, debug) if return_debug else total


def label_image(img_bgr=None, pre=None, **kwargs):
    """Full-frame instance map plus what each instance was counted as.

    Regions resolved by area accounting rather than by an actual cut are reported with the
    number of grains attributed to them, so a viewer can tell a genuine split from an
    estimate.
    """
    total, pre, debug = count_rice(img_bgr, pre=pre, return_debug=True, **kwargs)
    calib = pre["calib"]
    source = calib["labels"]

    instances = np.zeros(source.shape, dtype=np.int32)
    info = {}
    next_label = 1

    for component in calib["components"]:
        if component["area"] < SPECK_RATIO * calib["a0"]:
            continue
        if segment.is_foreign_object(component, calib):
            instances[source == component["label"]] = next_label
            info[next_label] = {"kind": "foreign", "count": 0}
            next_label += 1
        elif not segment.is_touching(component, calib):
            instances[source == component["label"]] = next_label
            info[next_label] = {"kind": "grain", "count": 1}
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

    return instances, info, total, pre

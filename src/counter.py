from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

from src import correct, preprocess, segment

# 小于单粒这个比例的区域当碎屑。凸实度判据在细粒照片上也启用后，切开的块变多，
# 切口会留下 0.3 到 0.45 粒大小的碎片。只在 D1 上选的值，D3 也跟着变好。
SPECK_RATIO = 0.45
RESPLIT = True

# 找回被开运算抹掉的米粒时用的门限，见 recover_erased。面积范围按原始掩膜上的单粒算，
# 宽度和长宽比相对标定的单粒。面积下限和宽度下限只在 D1 上选。
RECOVER_AREA = (0.3, 1.3)
RECOVER_MIN_WIDTH = 0.5
RECOVER_MAX_ELONGATION = 3.0


def _rejected(component, calib, pre):
    """不论形状都不算米粒的区域：异物和亮边。"""
    return (segment.is_foreign_object(component, calib)
            or segment.borders_bright_background(component, calib, pre.get("bright")))


def recover_erased(pre, occupied, reference):
    """从开运算之前的原始掩膜里找回被 3x3 开运算抹掉或削小的米粒。

    三到五像素宽的米粒开一次运算就去掉大半面积，D1、D3 上漏掉的米粒多数其实阈值分对了，
    是被去噪去掉的。直接改用原始掩膜会把噪点也放进来，所以只挑这样的区域找回：
    上面没有已计数或已剔除的区域，宽度、长宽比和亮度都像米粒，也不挨着亮背景。
    大小按原始掩膜上的单粒来量，因为开运算没有削小它们。
    """
    raw, gray = pre.get("raw_mask"), pre.get("gray")
    calib = pre["calib"]
    if raw is None or gray is None or reference is None:
        return None, []
    raw_labels = label(raw > 0)
    if raw_labels.max() == 0:
        return raw_labels, []
    areas = np.bincount(raw_labels.ravel())

    # 原始掩膜上的单粒面积，取只含一个已计数连通域的原始区域。
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
    """方法一的主流程：自适应尺度的标记分水岭，加形状先验修正。

    孤立的连通域直接计数，只有判为粘连的才去分割，所以单粒米不会被分水岭切开。
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

        # 在连通域的外接框里做距离变换和分水岭，比整幅图做快得多，结果一样。
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
    """整幅的实例标号图，以及每个实例被算作几粒。

    没有真正切开、按面积补数的区域会记下补了几粒，画图时能和真正切开的区分。
    判定结果直接取 count_rice 的，这里不重新判断。
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

"""二值化与尺度标定的 GPU 版本。

对 CPU 版做过性能剖析，Otsu、连通域、距离变换这些像素运算不到 1%，
时间几乎都花在逐个连通域算形状上，饱和度通道的候选要给约一万八千个噪点各算一次凸包。
所以这里像素运算用 CuPy 和 cuCIM 放到 GPU 上，连通域属性一次批量算，
凸包只给面积够大、后面可能用到凸实度的连通域算。
分水岭仍在 CPU 上做，cuCIM 没有 GPU 分水岭，而且它处理的是裁出来的小块，不值得上 GPU。
打分和合理性检查直接复用 CPU 版。
与 CPU 版的已知差别有两处，都实测过。OpenCV 的距离变换是近似的，
distance_transform_edt 是精确的；HSV 饱和度是这里用 CuPy 重算的，不是 cv2.cvtColor。
"""
import cv2
import numpy as np

from src import calibrate, preprocess

_MODULES = {}


def _gpu():
    """第一次用到时才导入 GPU 库，导入本模块不需要显卡。"""
    if not _MODULES:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
        from cucim.skimage.filters import threshold_multiotsu
        from cucim.skimage.measure import label, regionprops_table
        from cucim.skimage.morphology import binary_closing, binary_opening
        from cucim.skimage.segmentation import clear_border
        _MODULES.update(cp=cp, cndi=cndi, threshold_multiotsu=threshold_multiotsu,
                        label=label, regionprops_table=regionprops_table,
                        binary_opening=binary_opening, binary_closing=binary_closing,
                        clear_border=clear_border)
    return _MODULES


# 即 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ELLIPSE3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def channels_gpu(img_bgr):
    """在 GPU 上算灰度和 HSV 饱和度通道。

    按 OpenCV 的整数定义算（BT.601 灰度，S = (V - min) / V），阈值与 CPU 版落在同样的值上。
    """
    g = _gpu()
    cp = g["cp"]
    img = cp.asarray(img_bgr).astype(cp.float32)
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    grey = cp.rint(0.114 * blue + 0.587 * green + 0.299 * red).astype(cp.uint8)
    v = img.max(axis=2)
    m = img.min(axis=2)
    sat = cp.where(v > 0, cp.rint((v - m) * 255.0 / cp.maximum(v, 1e-6)), 0).astype(cp.uint8)
    return {"gray": grey, "hsv_s": sat}


def otsu_separability_gpu(channel):
    """用 GPU 直方图算 Otsu 阈值和可分性。"""
    g = _gpu()
    cp = g["cp"]
    hist = cp.bincount(channel.ravel(), minlength=256).astype(cp.float64)
    p = hist / (hist.sum() + preprocess.EPS)
    levels = cp.arange(256, dtype=cp.float64)

    omega = cp.cumsum(p)
    mu = cp.cumsum(p * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b2 = cp.where(denom > preprocess.EPS,
                        (mu_total * omega - mu) ** 2 / (denom + preprocess.EPS), 0.0)
    sigma_total2 = float(cp.sum(p * (levels - mu_total) ** 2))
    threshold = int(cp.argmax(sigma_b2))
    return threshold, float(sigma_b2[threshold] / (sigma_total2 + preprocess.EPS))


def clean_mask_gpu(mask_bool):
    g = _gpu()
    footprint = g["cp"].asarray(ELLIPSE3)
    return g["binary_closing"](g["binary_opening"](mask_bool, footprint), footprint)


def candidate_masks_gpu(img_bgr, channel_names=preprocess.DEFAULT_CHANNELS):
    """与 CPU 版相同的候选，通道 x 二类/三类 Otsu x 前景取亮或取暗。"""
    g = _gpu()
    out = []
    for name, ch in channels_gpu(img_bgr).items():
        if channel_names is not None and name not in channel_names:
            continue
        t_otsu, eta = otsu_separability_gpu(ch)
        splits = {"otsu": [t_otsu]}
        try:
            splits["multiotsu"] = [int(t) for t in g["threshold_multiotsu"](ch, classes=3)]
        except ValueError:
            pass

        for method, thresholds in splits.items():
            for polarity, raw in (("bright", ch > thresholds[-1]),
                                  ("dark", ch <= thresholds[0])):
                out.append({"channel_name": name, "channel": ch, "method": method,
                            "polarity": polarity, "thresholds": thresholds, "eta": eta,
                            "raw": raw, "mask": clean_mask_gpu(raw)})
    return out


def component_table_gpu(mask_bool, solidity_from=None, solidity_limit=None):
    """一次批量算出各连通域的属性。

    solidity_from 是面积门限，只给不小于它的连通域算凸包，None 表示都不算，没算的凸实度为 NaN。
    凸包最费时，cuCIM 也还是在主机上逐个算。
    solidity_limit 限制最多算多少个凸包，用于候选打分，那时凸实度只取单粒带内的中位数；
    噪点候选的 A0 会被标到噪点大小，"不小于 0.65 A0"仍会包括上万个噪点。最终选中的候选会完整地算。
    """
    g = _gpu()
    cp = g["cp"]
    labels = g["label"](mask_bool)
    n = int(labels.max())
    if n == 0:
        return labels, []

    props = g["regionprops_table"](
        labels, properties=("label", "area", "bbox",
                            "axis_major_length", "axis_minor_length"))
    areas = cp.asnumpy(props["area"]).astype(np.float64)
    label_ids = cp.asnumpy(props["label"]).astype(np.int64)
    major = cp.asnumpy(props["axis_major_length"]).astype(np.float64)
    minor = cp.asnumpy(props["axis_minor_length"]).astype(np.float64)
    bbox = np.stack([cp.asnumpy(props[f"bbox-{i}"]) for i in range(4)], axis=1)

    # 宽度 = 最大内切圆半径的两倍。区域之间隔着背景，整张做一次距离变换即可。
    dist = g["cndi"].distance_transform_edt(mask_bool)
    width = 2.0 * cp.asnumpy(
        g["cndi"].maximum(dist, labels=labels, index=cp.asarray(label_ids))
    ).astype(np.float64).ravel()

    solidity = np.full(areas.shape, np.nan)
    wanted = areas >= solidity_from if solidity_from is not None else np.zeros(areas.shape, bool)
    if wanted.any():
        # 先把保留的连通域紧凑地重新编号。cuCIM 会遍历 1..max 的所有标号，
        # 只把小块置零会留下几千个空号，每个都要算一次凸包。
        chosen_idx = np.flatnonzero(wanted)
        if solidity_limit is not None and chosen_idx.size > solidity_limit:
            # 按面积排序后等间隔抽样，结果是确定的。
            order = chosen_idx[np.argsort(areas[chosen_idx], kind="stable")]
            chosen_idx = np.sort(order[np.linspace(0, order.size - 1, solidity_limit).astype(int)])
        keep = cp.asarray(label_ids[chosen_idx])
        compact = g["label"](cp.isin(labels, keep))
        n_compact = int(compact.max())
        if n_compact:
            index = cp.arange(1, n_compact + 1)
            origin = cp.asnumpy(g["cndi"].maximum(labels, labels=compact, index=index))
            sub = g["regionprops_table"](compact, properties=("label", "solidity"))
            sub_sol = cp.asnumpy(sub["solidity"]).astype(np.float64)
            sub_lab = cp.asnumpy(sub["label"]).astype(np.int64)
            got = {int(origin[l - 1]): float(v) for l, v in zip(sub_lab, sub_sol)}
            solidity = np.array([got.get(int(l), np.nan) for l in label_ids])

    rows = [
        {"label": int(label_ids[i]), "area": float(areas[i]), "solidity": float(solidity[i]),
         "major": float(major[i]), "minor": float(minor[i]), "width": float(width[i]),
         "axis_ratio": float(major[i] / minor[i]) if minor[i] > 0 else np.inf,
         "bbox": tuple(int(v) for v in bbox[i])}
        for i in range(len(label_ids))
    ]
    return labels, rows


def calibrate_gpu(mask_bool, need_solidity=True):
    """尺度与形状先验，A0 只用面积估计。

    need_solidity=False 时不算凸包。候选按不含凸实度的分数排序，
    需要凸实度的检查按分数顺序后做，只有真正轮到的候选才算凸包。
    """
    # 第一遍只要面积，A0 由面积分布得到，不需要凸包。
    labels, rows = component_table_gpu(mask_bool)
    if not rows:
        raise ValueError("no foreground components")
    areas = np.array([r["area"] for r in rows])
    a0, _ = calibrate.estimate_single_area(areas)

    # A0 有了，只在后面可能用到凸实度的地方算。
    labels, rows = component_table_gpu(
        mask_bool,
        solidity_from=calibrate.SHAPE_FLOOR_RATIO * a0 if need_solidity else None)
    areas = np.array([r["area"] for r in rows])
    a0, kde_debug = calibrate.estimate_single_area(areas)

    lo, hi = calibrate.SINGLE_BAND
    singles = [r for r in rows if lo * a0 <= r["area"] <= hi * a0] or rows
    return {
        "labels": labels, "components": rows, "a0": a0,
        "r0": float(np.sqrt(a0 / np.pi)),
        "major0": float(np.median([r["major"] for r in singles])),
        "minor0": float(np.median([r["minor"] for r in singles])),
        "width0": float(np.median([r["width"] for r in singles])),
        "solidity0": float(np.nanmedian([r["solidity"] for r in singles])) if need_solidity
        else float("nan"),
        "axis_ratio0": float(np.median([r["axis_ratio"] for r in singles])),
        "n_components": len(rows), "n_singles": len(singles), "kde_debug": kde_debug,
    }


def local_contrast_gpu(channel, mask_bool):
    """中心与周边的对比度，两次膨胀都在 GPU 上做。"""
    g = _gpu()
    cp = g["cp"]
    if not bool(mask_bool.any()):
        return 0.0
    ring9 = cp.asarray(np.ones((9, 9), dtype=bool))
    inner = g["cndi"].binary_dilation(mask_bool, structure=cp.asarray(ELLIPSE3))
    ring = g["cndi"].binary_dilation(mask_bool, structure=ring9) & ~inner
    if not bool(ring.any()):
        return 0.0
    ch = channel.astype(cp.float64)
    return float(cp.abs(ch[mask_bool].mean() - ch[ring].mean()) / (ch.std() + preprocess.EPS))


def _drop_by_label(mask_bool, labels, drop_ids):
    """按标号整块置零，不对连通域写 Python 循环。"""
    g = _gpu()
    cp = g["cp"]
    if len(drop_ids) == 0:
        return mask_bool
    return mask_bool & ~cp.isin(labels, cp.asarray(np.asarray(drop_ids, dtype=np.int64)))


def _clean_selected(mask_bool, calib, a0):
    """去衬底和碎点，按连通域向量化处理。"""
    g = _gpu()
    cp = g["cp"]
    labels = calib["labels"]
    kept_border = g["clear_border"](labels)
    border_ids = set(np.asarray(cp.asnumpy(cp.unique(labels))).tolist()) - \
        set(np.asarray(cp.asnumpy(cp.unique(kept_border))).tolist())

    drop = [r["label"] for r in calib["components"]
            if (r["label"] in border_ids and r["area"] > preprocess.SUBSTRATE_AREA_RATIO * a0)
            or r["area"] < preprocess.SPECK_AREA_RATIO * a0]
    return _drop_by_label(mask_bool, labels, drop)


def preprocess_gpu(img_bgr, channel_names=preprocess.DEFAULT_CHANNELS):
    """preprocess.preprocess 的 GPU 版，返回相同结构的字典。"""
    g = _gpu()
    cp = g["cp"]

    scored = []
    for cand in candidate_masks_gpu(img_bgr, channel_names=channel_names):
        mask = cand["mask"]
        coverage = float(g["clear_border"](mask).mean())
        if coverage > preprocess.MAX_COVERAGE or coverage <= 0:
            continue
        try:
            calib = calibrate_gpu(mask, need_solidity=False)
        except ValueError:
            continue
        if not preprocess.plausible_grain_population(calib, img_bgr.shape,
                                                     check_solidity=False):
            continue
        calib["coverage"] = coverage
        calib["contrast"] = local_contrast_gpu(cand["channel"], mask)
        calib["score"] = calib["n_singles"] * calib["a0"] * calib["contrast"]
        scored.append((cand, calib))

    if not scored:
        raise ValueError("no usable binarisation candidate")

    for cand, ranked in sorted(scored, key=lambda pair: -pair[1]["score"]):
        calib = calibrate_gpu(cand["mask"])
        if not preprocess.plausible_grain_population(calib, img_bgr.shape):
            continue
        mask = _clean_selected(cand["mask"], calib, calib["a0"])
        try:
            final = calibrate_gpu(mask)
        except ValueError:
            continue
        if preprocess.plausible_grain_population(final, img_bgr.shape):
            # 计数阶段处理裁出的小块，选中的掩膜和标号在这里一次拷回主机。
            final["labels"] = cp.asnumpy(final["labels"]).astype(np.int32)
            host_mask = (cp.asnumpy(mask) > 0).astype(np.uint8) * 255
            return {
                "bright": preprocess.bright_background(img_bgr, host_mask),
                "channel_name": cand["channel_name"],
                "channel": cp.asnumpy(cand["channel"]),
                "method": cand["method"], "polarity": cand["polarity"],
                "thresholds": cand["thresholds"], "eta": cand["eta"],
                "raw_mask": cp.asnumpy(cand["raw"]).astype(np.uint8) * 255,
                "gray": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                "mask": host_mask,
                "calib": final,
            }
    raise ValueError("no binarisation candidate survived recalibration")


def count_rice_gpu(img_bgr, **kwargs):
    from src import counter
    return counter.count_rice(None, pre=preprocess_gpu(img_bgr), **kwargs)

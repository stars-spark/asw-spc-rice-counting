"""GPU implementation of the binarisation and calibration stages.

Profiling the CPU pipeline showed the pixel kernels (Otsu, connected components, distance
transform) take under 1% of the runtime; almost all of it goes into per-component shape
measurement, which for a saturation-channel candidate means a convex hull for each of some
18000 noise blobs. This module therefore does two things at once:

* every per-pixel stage runs on the GPU through CuPy and cuCIM;
* per-component properties are measured in one batched call, and the expensive ones
  (convex hull, hence solidity) only for components large enough for any later decision to
  consult them - nothing below 0.65*A0 is ever asked for its solidity, since the single
  band starts there and the touching, foreign and correction rules all start higher.

The watershed itself stays on the CPU: cuCIM has no GPU watershed, and it runs on small
cropped clusters where a GPU launch would cost more than the work. The selection logic
(scoring, plausibility) is reused unchanged from the CPU path - it is arithmetic on a
handful of scalars.

Numbers differ from the CPU path in two known ways, both measured rather than assumed:
OpenCV's distance transform is an approximation while `distance_transform_edt` is exact,
and the HSV saturation channel is rebuilt here in CuPy rather than by `cv2.cvtColor`.
"""
import cv2
import numpy as np

from src import calibrate, preprocess

_MODULES = {}


def _gpu():
    """Import the GPU stack on first use, so importing this module never needs a GPU."""
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


# cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ELLIPSE3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def channels_gpu(img_bgr):
    """Grey and HSV-saturation channels, built on the GPU.

    Both follow OpenCV's integer definitions (BT.601 grey, S = (V - min) / V) so the
    thresholds land on the same values as the CPU path.
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
    """Otsu threshold and separability, computed from a GPU histogram."""
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
    """Same candidate grid as the CPU path: channel x {2,3}-class Otsu x polarity."""
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
    """Per-component properties in one batched call.

    `solidity_from` is an area threshold: convex hulls, which dominate the cost and which
    cuCIM still computes on the host one object at a time, are built only for components at
    or above it. `None` skips them entirely. Components without one are returned with
    solidity NaN, which is safe because every rule that reads solidity first requires a
    larger area than the threshold used here.

    `solidity_limit` caps how many hulls are built. It is used while scoring candidates,
    where solidity is only consumed as a median over the single band: a noise candidate
    calibrates A0 to a speck, so "at least 0.65*A0" would still cover every one of its
    tens of thousands of blobs. The candidate that is finally chosen is measured exactly.
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

    # Width = twice the largest inscribed-disc radius. One transform over the whole mask
    # gives the same per-region maximum as transforming each region alone, because regions
    # are separated by background.
    dist = g["cndi"].distance_transform_edt(mask_bool)
    width = 2.0 * cp.asnumpy(
        g["cndi"].maximum(dist, labels=labels, index=cp.asarray(label_ids))
    ).astype(np.float64).ravel()

    solidity = np.full(areas.shape, np.nan)
    wanted = areas >= solidity_from if solidity_from is not None else np.zeros(areas.shape, bool)
    if wanted.any():
        # Relabel the kept components compactly first. cuCIM walks labels 1..max, so simply
        # zeroing the small ones would leave thousands of empty labels behind and it would
        # build a convex hull for every one of them.
        chosen_idx = np.flatnonzero(wanted)
        if solidity_limit is not None and chosen_idx.size > solidity_limit:
            # Evenly spaced in area order: a deterministic sample of the same population.
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
    """Scale and shape priors, with A0 estimated from areas alone (no hulls needed).

    `need_solidity=False` skips the convex hulls altogether. Candidates are ranked by a
    score that does not use solidity, and the gate that does is applied afterwards in score
    order, so hulls are built only for candidates that actually reach it.
    """
    # First pass: areas only. A0 comes from the area distribution, so no hulls are needed.
    labels, rows = component_table_gpu(mask_bool)
    if not rows:
        raise ValueError("no foreground components")
    areas = np.array([r["area"] for r in rows])
    a0, _ = calibrate.estimate_single_area(areas)

    # Now A0 is known, measure solidity only where a later rule could ask for it.
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
    """Centre-surround contrast, with both dilations done on the GPU."""
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
    """Zero whole components by label, without a Python loop over components."""
    g = _gpu()
    cp = g["cp"]
    if len(drop_ids) == 0:
        return mask_bool
    return mask_bool & ~cp.isin(labels, cp.asarray(np.asarray(drop_ids, dtype=np.int64)))


def _clean_selected(mask_bool, calib, a0):
    """Substrate and speck removal, vectorised over components."""
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
    """GPU version of `preprocess.preprocess`, returning the same dictionary shape."""
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
            # The counting stage works on small cropped clusters, so the chosen mask and
            # its labels come back to the host once, here.
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

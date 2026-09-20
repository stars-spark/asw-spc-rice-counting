import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from skimage.measure import label, regionprops

NOISE_FLOOR_PX = 5
# Hulls are measured from this fraction of A0 upwards. It matches the speck threshold the
# counter skips below, so every component any later rule can examine carries a real
# solidity - including the ablation variants that test it with an "or" and therefore reach
# components smaller than the single band.
SHAPE_FLOOR_RATIO = 0.3
SINGLE_BAND = (0.65, 1.45)
AREA_MODE = "dominant"


def component_table(mask, shape_from=None, shape_sample=None, defer_solidity=False):
    """Per-component measurements for one binary mask.

    Everything except solidity is obtained for all components in a single scan: connected
    components and the distance transform are OpenCV calls, the per-region distance maxima
    are one labelled reduction, and the axis lengths come from second-order moments
    accumulated with `bincount` (identical to `regionprops` to within floating point).

    Solidity is the only measurement needing a convex hull, and hulls are what made the old
    per-component loop expensive: a saturation-channel candidate holds tens of thousands of
    speck blobs. They are therefore computed only for components of at least `shape_from`,
    which is safe because nothing consults solidity below it - the single band starts at
    0.65*A0 and the touching, foreign-object and area-accounting rules all test a larger
    area first. Components without a hull come back with solidity NaN.

    That threshold depends on A0, which is estimated from the areas this function returns.
    With `defer_solidity` the hulls are left unmeasured and a third value is returned: a
    function that fills them in for a given threshold, so the caller can estimate A0 first
    without scanning the mask twice.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return (labels, [], lambda *a, **k: None) if defer_solidity else (labels, [])

    index = np.arange(1, count)
    areas = stats[index, cv2.CC_STAT_AREA].astype(np.float64)
    # (min_row, min_col, max_row, max_col), matching regionprops' bbox convention
    boxes = np.stack([stats[index, cv2.CC_STAT_TOP], stats[index, cv2.CC_STAT_LEFT],
                      stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT],
                      stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]], axis=1)

    # Twice the largest inscribed-disc radius: the width of the region at its thickest
    # point. Components are separated by background, so one distance transform over the
    # whole mask gives the same per-region maxima as transforming each region alone.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = 2.0 * np.asarray(ndi.maximum(dist, labels, index=index), dtype=np.float64).ravel()

    # Axis lengths from the second-order central moments of each label.
    rows_idx, cols_idx = np.nonzero(labels)
    flat = labels[rows_idx, cols_idx]
    n_px = np.bincount(flat, minlength=count).astype(np.float64)[1:]
    sum_x = np.bincount(flat, weights=cols_idx, minlength=count)[1:]
    sum_y = np.bincount(flat, weights=rows_idx, minlength=count)[1:]
    sum_xx = np.bincount(flat, weights=cols_idx * cols_idx, minlength=count)[1:]
    sum_yy = np.bincount(flat, weights=rows_idx * rows_idx, minlength=count)[1:]
    sum_xy = np.bincount(flat, weights=cols_idx * rows_idx, minlength=count)[1:]
    mean_x, mean_y = sum_x / n_px, sum_y / n_px
    mu_xx = sum_xx / n_px - mean_x * mean_x
    mu_yy = sum_yy / n_px - mean_y * mean_y
    mu_xy = sum_xy / n_px - mean_x * mean_y
    spread = np.sqrt(np.maximum((mu_xx - mu_yy) ** 2 + 4.0 * mu_xy ** 2, 0.0))
    major = 4.0 * np.sqrt(np.maximum((mu_xx + mu_yy + spread) / 2.0, 0.0))
    minor = 4.0 * np.sqrt(np.maximum((mu_xx + mu_yy - spread) / 2.0, 0.0))

    rows = [
        {
            "label": int(index[i]),
            "area": float(areas[i]),
            "solidity": float("nan"),
            "major": float(major[i]),
            "minor": float(minor[i]),
            "width": float(widths[i]),
            "axis_ratio": float(major[i] / minor[i]) if minor[i] > 0 else np.inf,
            "bbox": tuple(int(v) for v in boxes[i]),
        }
        for i in range(index.size)
    ]

    def fill_solidity(threshold=None, sample=None):
        """Measure convex hulls for the components at or above `threshold`."""
        wanted = np.ones(areas.shape, bool) if threshold is None else areas >= threshold
        chosen = index[wanted]
        if sample is not None and chosen.size > sample:
            # Evenly spaced in area order: a deterministic sample of the same population.
            order = chosen[np.argsort(areas[wanted], kind="stable")]
            chosen = np.sort(order[np.linspace(0, order.size - 1, sample).astype(int)])
        if not chosen.size:
            return
        # Relabel the chosen components 1..k before measuring: regionprops walks the whole
        # label range, so leaving gaps would make it visit every empty label in between.
        lut = np.zeros(count, dtype=np.int32)
        lut[chosen] = np.arange(1, chosen.size + 1)
        for region in regionprops(lut[labels]):
            rows[chosen[region.label - 1] - 1]["solidity"] = float(region.solidity)

    if defer_solidity:
        return labels, rows, fill_solidity
    fill_solidity(shape_from, shape_sample)
    return labels, rows


def estimate_single_area(areas, grid_size=512, mode=None):
    """Single-grain area A0 from the mode of the area distribution in log space.

    Log space is used because touching clusters pile up near 2*A0, 3*A0 ..., which are
    evenly spaced there, and because the kernel width then scales with the grain size
    instead of being a fixed pixel count.

    "dominant" takes the tallest mode: single grains are the most repeated object in the
    scene. "lowest" takes the lowest prominent mode, which is only safe when the image is
    free of debris, since fragments would be mistaken for grains.
    """
    mode = AREA_MODE if mode is None else mode
    areas = np.asarray([a for a in areas if a >= NOISE_FLOOR_PX], dtype=np.float64)
    if areas.size == 0:
        raise ValueError("no foreground components above the noise floor")
    if areas.size < 5:
        return float(np.median(areas)), None

    log_areas = np.log(areas)
    kde = gaussian_kde(log_areas)
    grid = np.linspace(log_areas.min(), log_areas.max(), grid_size)
    density = kde(grid)

    peak_idx, _ = find_peaks(density)
    if peak_idx.size == 0 or mode == "dominant":
        chosen = int(np.argmax(density))
    else:
        prominent = peak_idx[density[peak_idx] >= 0.3 * density.max()]
        chosen = prominent[0] if prominent.size else int(np.argmax(density))
    a0 = float(np.exp(grid[chosen]))

    band = areas[(areas >= SINGLE_BAND[0] * a0) & (areas <= SINGLE_BAND[1] * a0)]
    if band.size:
        a0 = float(band.mean())
    return a0, {"grid": grid, "density": density, "log_areas": log_areas}


def calibrate(mask, mode=None, need_solidity=True):
    """Estimate the single-grain scale and shape priors from the image itself.

    With `need_solidity=False` the convex hulls are skipped entirely: solidity comes back
    NaN for every component and so does `solidity0`. Everything the candidate score is made
    of - the number of single-sized components, A0 and the axis statistics - is measured
    without them, so ranking candidates costs no hulls at all (see `preprocess.preprocess`).
    """
    # One scan of the mask. A0 is estimated from the areas it returns, which needs no hulls,
    # and the hulls are then measured only where solidity can still be consulted.
    labels, rows, fill_solidity = component_table(mask, defer_solidity=True)
    if not rows:
        raise ValueError("no foreground components above the noise floor")
    a0, kde_debug = estimate_single_area([r["area"] for r in rows], mode=mode)

    if need_solidity:
        fill_solidity(SHAPE_FLOOR_RATIO * a0)
    singles = [r for r in rows if SINGLE_BAND[0] * a0 <= r["area"] <= SINGLE_BAND[1] * a0]
    if not singles:
        # The fallback averages over every component, so they all need a hull.
        if need_solidity:
            fill_solidity()
        singles = rows

    solidity0 = float(np.nanmedian([r["solidity"] for r in singles])) if need_solidity \
        else float("nan")

    return {
        "labels": labels,
        "components": rows,
        "a0": a0,
        "r0": float(np.sqrt(a0 / np.pi)),
        # The minor axis, not the equal-area radius, sets how close two grain centres can
        # be: grains pack side by side, and for an elongated grain the distance-transform
        # ridge inside one grain is longer than the gap between two of them.
        "major0": float(np.median([r["major"] for r in singles])),
        "minor0": float(np.median([r["minor"] for r in singles])),
        # The width of one grain. Unlike solidity it is an inscribed-disc measurement, so
        # it survives at the four-pixel scale where boundary statistics stop being usable.
        "width0": float(np.median([r["width"] for r in singles])),
        "solidity0": solidity0,
        "axis_ratio0": float(np.median([r["axis_ratio"] for r in singles])),
        "n_components": len(rows),
        "n_singles": len(singles),
        "kde_debug": kde_debug,
    }

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from src.segment import distance_transform

MIN_AREA_RATIO = 0.3
CRF_THRESHOLD = 0.6
ELLIPSE_MERGE_TOLERANCE = 0.25


def _count_labels(labels, a0, min_area_ratio=MIN_AREA_RATIO):
    return sum(1 for r in regionprops(labels) if r.area >= min_area_ratio * a0)


def b1_connected_components(calib):
    """Count every connected component as one grain (ignores touching)."""
    cutoff = MIN_AREA_RATIO * calib["a0"]
    return sum(1 for r in calib["components"] if r["area"] >= cutoff)


def b2_area_estimate(calib):
    """Total foreground area divided by the single-grain area."""
    cutoff = MIN_AREA_RATIO * calib["a0"]
    total = sum(r["area"] for r in calib["components"] if r["area"] >= cutoff)
    return int(np.round(total / calib["a0"]))


def b3_distance_watershed(mask, calib, fg_ratio=0.5):
    """Textbook distance-transform watershed: markers are the pixels above a fixed
    fraction of the GLOBAL distance maximum (the OpenCV tutorial recipe).

    The fixed global threshold is exactly what the adaptive seeding replaces: one large
    cluster raises the maximum for the whole image and wipes out the markers of every
    smaller grain.
    """
    dist = distance_transform(mask)
    if dist.max() <= 0:
        return 0
    sure_fg = (dist > fg_ratio * dist.max()).astype(np.uint8)
    markers = label(sure_fg)
    if markers.max() == 0:
        return 0
    labels = watershed(-dist, markers, mask=mask > 0)
    return _count_labels(labels, calib["a0"])


def b4_erosion_watershed(mask, calib, ksize=3, iterations=1):
    """Marker-controlled watershed with markers from morphological erosion.

    This is the pipeline of Kurade et al., Foods 2023 (3x3 structuring element, connected
    component analysis on the eroded image), the paper that reports removing touching
    grains from its dataset because this scheme cannot separate them.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=iterations)
    markers = label(eroded)
    if markers.max() == 0:
        return 0
    dist = distance_transform(mask)
    labels = watershed(-dist, markers, mask=mask > 0)
    return _count_labels(labels, calib["a0"])


def _corner_response(mask, contour, radius):
    """Corner response of Tan et al.: the foreground fraction of a disc centred on each
    contour pixel.

    On a straight stretch of boundary the disc is half inside the region, so the response
    sits near 0.5; where the boundary turns into the region, as it does at the neck between
    two touching grains, more of the disc is foreground and the response rises.
    """
    radius = max(int(round(radius)), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    area = float(kernel.sum())
    counts = cv2.filter2D((mask > 0).astype(np.float32), -1, kernel.astype(np.float32),
                          borderType=cv2.BORDER_CONSTANT)
    xs = np.clip(contour[:, 0], 0, mask.shape[1] - 1)
    ys = np.clip(contour[:, 1], 0, mask.shape[0] - 1)
    return counts[ys, xs] / area


def _corner_points(response, radius, threshold=CRF_THRESHOLD):
    """Indices of response peaks above the threshold, one per neighbourhood.

    The response stays high across the whole neck rather than at a single pixel, so peaks
    closer together than the disc radius are the same corner and only the strongest is kept.
    """
    n = len(response)
    if n < 3:
        return []
    above = np.flatnonzero(response > threshold)
    if above.size == 0:
        return []

    picked = []
    for index in above[np.argsort(-response[above])]:
        # The contour is a closed curve, so distance along it wraps around.
        if all(min(abs(index - other), n - abs(index - other)) > radius for other in picked):
            picked.append(int(index))
    return sorted(picked)


def _ellipse_residual(points):
    """RMS distance from the points to their best-fitting ellipse, in pixels.

    Measured radially: each point is taken along its own ray from the ellipse centre and
    compared with where the ellipse crosses that ray. A geometric residual is used rather
    than the algebraic one of the quadratic form, whose value depends on the size of the
    ellipse and so cannot be compared against a fixed tolerance.
    """
    if len(points) < 5:
        return None
    try:
        (cx, cy), (width, height), angle = cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return None
    a, b = max(width, 1e-6) / 2.0, max(height, 1e-6) / 2.0

    theta = np.deg2rad(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)

    radius = np.hypot(u, v)
    scale = np.hypot(u / max(a, 1e-6), v / max(b, 1e-6))
    on_ellipse = radius / np.maximum(scale, 1e-6)
    return float(np.sqrt(np.mean((radius - on_ellipse) ** 2)))


def _grains_by_ellipse(contour, corners, grain_width, tolerance=ELLIPSE_MERGE_TOLERANCE):
    """Group the segments between corners into grains by elliptical approximation.

    Grains are close to elliptical, so the segments of one grain's boundary are the ones a
    single ellipse explains. The published method searches every partition of the segments
    for the one of least total fit error, which is combinatorial in their number - the
    authors report the cost growing rapidly with the grain count - so this baseline merges
    greedily instead, repeatedly joining the pair whose single fitted ellipse has the lowest
    residual, and accepting a merge only while that residual stays within a fraction of a
    grain width. Judging a merge by the absolute quality of its fit rather than by how much
    the fit worsens is what makes the tolerance mean anything: a pair of short segments can
    always be fitted perfectly, so a change in residual is near zero either way.

    The centre-distance penalty of the published method, which discourages a partition whose
    ellipses share a centre, is not reproduced here: it serves to rank competing whole
    partitions, and this greedy version never holds two of them to compare.
    """
    segments = [contour[corners[i]:corners[i + 1] + 1] for i in range(len(corners) - 1)]
    segments.append(np.vstack([contour[corners[-1]:], contour[: corners[0] + 1]]))
    groups = [s for s in segments if len(s) >= 2]
    if len(groups) < 2:
        return max(len(groups), 1)

    limit = tolerance * grain_width
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                merged = np.vstack([groups[i], groups[j]])
                residual = _ellipse_residual(merged)
                if residual is None:
                    # Too few points to fit: such a pair is merged before any fitted one,
                    # since neither piece can be a grain outline on its own.
                    residual = 0.0
                if best is None or residual < best[0]:
                    best = (residual, i, j, merged)

        residual, i, j, merged = best
        if residual > limit:
            break
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
    return len(groups)


def b5_concave_ellipse(mask, calib, use_ellipse=True, radius_ratio=0.5):
    """Concave-point counting with elliptical correction (Avzalov et al., Vavilov J. 2025).

    Corner points are found on each component's contour with the response above, and the
    grain count of a contour follows their pairing rule: every two corners mark one contact,
    so a contour holding k grains carries 2(k-1) of them. Ellipse fitting then regroups the
    segments between corners, which is how the published method discards corners raised by
    chipped or dented grains instead of contacts.

    Everything here is measured on the boundary, which is the point of including it: it is
    the mainstream route for this problem and it is the route whose descriptors are the ones
    reported to need a minimum number of pixels across the object.

    The disc radius is tied to the calibrated grain width rather than fixed in pixels, so
    the baseline rescales with the image the same way the method under test does; half a
    grain width is the geometry the response assumes. It is exposed as a parameter because
    no one value works at every scale, which is the finding this baseline is here to show.
    """
    radius = max(calib["minor0"] * radius_ratio, 2.0)
    cutoff = MIN_AREA_RATIO * calib["a0"]
    labels = label(mask > 0)

    total = 0
    for region in regionprops(labels):
        if region.area < cutoff:
            continue
        patch = np.pad((labels[region.slice] == region.label).astype(np.uint8), int(radius) + 2)
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=len).reshape(-1, 2)
        if len(contour) < 8:
            total += 1
            continue

        corners = _corner_points(_corner_response(patch, contour, radius), radius)
        if len(corners) < 2:
            total += 1
        elif use_ellipse:
            total += _grains_by_ellipse(contour, corners, calib["minor0"])
        else:
            total += max(1, len(corners) // 2 + 1)
    return total

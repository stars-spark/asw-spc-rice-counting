import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

BETA = 0.10
MIN_DEPTH_PX = 1.0
SEED_MERGE_RATIO = 0.6
TOUCH_AREA_RATIO = 1.2
TOUCH_SOLIDITY_MARGIN = 0.06
FOREIGN_AREA_RATIO = 3.0
FOREIGN_WIDTH_RATIO = 2.5
BRIGHT_RING_FRACTION = 0.2
DARK_RATIO = 0.5


def width_excess(component, calib):
    """Area of a region against the area its own thickness would account for.

    A region that is one grain thick and holds one grain's area returns 1. Two grains side
    by side are no thicker than one, so their area doubles while the reference does not,
    and the ratio rises with the number of grains in the region. A single grain that simply
    happens to be large is thicker in proportion, so it stays near 1 - which is what makes
    this usable as evidence of touching rather than merely of size. Both inputs are an area
    integral and an inscribed-disc radius, neither of which degrades at the scale where
    boundary statistics such as solidity stop being meaningful.
    """
    if calib["width0"] <= 0 or calib["a0"] <= 0:
        return 0.0
    reference = (component["width"] / calib["width0"]) ** 2
    return (component["area"] / calib["a0"]) / max(reference, 1e-6)


def distance_transform(mask):
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)


def adaptive_markers(dist, minor0, beta=BETA):
    """Seeds = peaks of the distance map at least h deep, with fragments of one peak merged.

    Two things decide whether a watershed cut is real, and each gets its own scale.

    Depth: between two touching grains the map dips at the neck, inside one grain it does
    not, so only peaks that stand at least h above their surroundings become seeds. h
    follows the calibrated grain half-width, but never goes below one pixel, the resolution
    the distance transform is measured in; smaller undulations are pixelation of the
    boundary, not necks. Plump grains that touch along a broad front leave a shallow neck,
    which is why beta is small.

    Spacing: h_maxima on a floating-point map breaks one flat peak into several pieces that
    differ only by rounding, and each piece would otherwise become a seed of its own and cut
    one grain in two. Measured on the synthetic set, seed pieces that belong to one grain lie
    at most 0.55 grain widths apart and seeds of different grains at least 0.67, so pieces
    closer than SEED_MERGE_RATIO grain widths are joined into one seed.

    The two errors are coupled. With the pieces left unmerged, lowering beta multiplied the
    duplicate seeds, so tuning pushed beta up to 0.45 - high enough to flatten the real necks
    between plump grains, which then went uncut and were counted by area instead.
    """
    h = max(beta * minor0 / 2.0, MIN_DEPTH_PX)
    peaks = (h_maxima(dist, h) > 0).astype(np.uint8)
    radius = max(1, int(round(SEED_MERGE_RATIO * minor0 / 2.0)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    joined = label(cv2.dilate(peaks, disc) > 0)
    return np.where(peaks > 0, joined, 0).astype(np.int32)


def is_foreign_object(component, calib):
    """Reject a large region whose shape no pile of grains could have.

    Two tests, because foreign objects fail the grain model in two opposite directions.

    Width: grains meet side by side rather than merging, so however many of them a pile
    holds it is still one grain thick at its thickest point. Twice the largest value the
    distance transform takes inside a region is the diameter of the largest disc that fits
    in it, which is that region's size in the sense of morphological granulometry - the
    size at which an opening by a disc would finally erase it. That size is invariant to the
    number of grains in the region, which makes it the sharpest available signal: measured
    here as 1.0 grain widths for single grains and at most 2.0 for genuine clusters, against
    8.6 for the reference coin and 3.8 to 4.2 for the wooden frame bars in the
    low-resolution set. Being an inscribed-disc measurement rather than a boundary one, it
    also stays usable at four pixels per grain, where solidity does not.

    Convexity: grains meeting always leave concave necks, so a region several grains in
    area that is still as convex as a single grain cannot be a pile of them. This test
    carries no accompanying elongation condition. Requiring the region to also be rounder
    than a grain only ever described the reference coin; it let every elongated foreign
    object through, and dropping it costs nothing on the other sets.

    Without this stage such regions are area-counted into dozens of phantom grains.
    """
    if component["area"] <= FOREIGN_AREA_RATIO * calib["a0"]:
        return False
    if calib["width0"] > 0 and component["width"] > FOREIGN_WIDTH_RATIO * calib["width0"]:
        return True
    return component["solidity"] >= calib["solidity0"]


def is_touching(component, calib):
    """A component holds more than one grain only if it is both too large and too concave.

    Requiring both signals matters: grain areas scatter around the modal value, so size
    alone flags plenty of ordinary single grains, and boundary noise alone makes small
    grains look concave. Two grains that actually meet are always larger than one and
    always leave a neck.

    Concavity is used at every scale. An earlier version handed it, below 4.5 px of grain
    width, to the width excess on the grounds that solidity stops meaning anything there.
    Measured on the photographs whose grains are 3.7 to 4.5 px wide, it had not: touching
    pairs sit at a median of 0.20 below the single-grain solidity and single grains at 0,
    while the width excess put pairs at 1.22 and singles at 0.98 against a threshold of
    1.5, so two thirds of the pairs went uncut. On the low-resolution set the width excess
    also passed convex glints along the frame as clusters, which are then area-counted
    into several grains; concavity rejects them, since no pile of grains is that convex.
    """
    if component["area"] <= TOUCH_AREA_RATIO * calib["a0"]:
        return False
    return component["solidity"] < calib["solidity0"] - TOUCH_SOLIDITY_MARGIN


def grain_brightness(labels, gray, grain_labels):
    """Median grey level over the regions taken for grains: what a grain looks like here."""
    if gray is None or not grain_labels:
        return None
    return float(np.median(gray[np.isin(labels, grain_labels)]))


def is_dark(component, labels, gray, reference):
    """True if the region is far darker than the grains of this image.

    When the selected mask is "low saturation", as on most of the low-resolution set, black is
    as unsaturated as white rice: the shadow strips along the frame and the shadows beside
    grains come through as regions of grain size and shape. Their grey level gives them
    away at once - a few per cent of the grains' against 0.9 to 1.1 for the grains
    themselves. DARK_RATIO was chosen on the real photographs alone, where no annotated
    grain fell below it.
    """
    if gray is None or reference is None:
        return False
    r0, c0, r1, c1 = component["bbox"]
    region = labels[r0:r1, c0:c1] == component["label"]
    return float(gray[r0:r1, c0:c1][region].mean()) < DARK_RATIO * reference


def borders_bright_background(component, calib, bright):
    """True if the region lies against bright background rather than on the grain surface.

    `bright` is the map from `preprocess.bright_background`. A band around the region, from
    two pixels out to about two grain widths out, is looked at; skipping the first two
    pixels keeps a grain's own blurred rim out of it. A grain on its surface has almost
    none of that band bright, a glint on the frame edge has one whole side of it bright.
    More than BRIGHT_RING_FRACTION bright marks the second case. That value was chosen on
    the real photographs alone and left no real grain rejected on any data set.
    """
    if bright is None:
        return False
    labels = calib["labels"]
    height, width = labels.shape
    outer = int(round(2 + 2 * max(calib["minor0"], 1.0))) * 2 + 1
    r0, c0, r1, c1 = component["bbox"]
    top, left = max(r0 - outer, 0), max(c0 - outer, 0)
    bottom, right = min(r1 + outer, height), min(c1 + outer, width)
    window = labels[top:bottom, left:right]
    region = (window == component["label"]).astype(np.uint8)
    near = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    # The band lies inside the window and outside `near`. If no bright pixel is there, the
    # answer is already no, and the large dilation - nearly all of this test's cost - is
    # skipped. On a uniform background the only bright pixels are grain rims, all of them
    # inside `near`, so that is every region.
    candidates = bright[top:bottom, left:right] & ~near & (window == 0)
    if not candidates.any():
        return False
    far = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer, outer))) > 0
    band = far & ~near & (window == 0)
    if not band.any():
        return False
    return bright[top:bottom, left:right][band].mean() > BRIGHT_RING_FRACTION


def split_component(mask, calib, beta=None):
    """Marker-controlled watershed of one touching cluster."""
    beta = BETA if beta is None else beta
    dist = distance_transform(mask)
    markers = adaptive_markers(dist, calib["minor0"], beta=beta)
    if markers.max() <= 1:
        return (mask > 0).astype(np.int32), dist, markers
    return watershed(-dist, markers, mask=mask > 0), dist, markers


def split_by_count(mask, k, minor0):
    """Cut a region known, from its area, to hold k grains into k pieces.

    Used on the regions the depth test left whole, where the neck between two plump grains
    is too shallow for h-maxima to see. The k highest points of the distance map become the
    seeds, taken greedily and never closer than SEED_MERGE_RATIO grain widths - the spacing
    measured to separate seeds of different grains from pieces of one peak. Returns None when
    the region cannot hold k such seeds, so the caller keeps counting it by area instead.
    """
    from scipy import ndimage

    if k < 2:
        return None
    dist = distance_transform(mask)
    peaks = (dist == ndimage.maximum_filter(dist, size=3)) & (mask > 0)
    rows, cols = np.nonzero(peaks)
    order = np.argsort(-dist[rows, cols])
    spacing = SEED_MERGE_RATIO * minor0
    chosen = []
    for i in order:
        point = np.array([rows[i], cols[i]], dtype=np.float32)
        if all(np.linalg.norm(point - q) >= spacing for q in chosen):
            chosen.append(point)
            if len(chosen) == k:
                break
    if len(chosen) < k:
        return None
    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, (r, c) in enumerate(chosen, start=1):
        markers[int(r), int(c)] = index
    return watershed(-dist, markers, mask=mask > 0)

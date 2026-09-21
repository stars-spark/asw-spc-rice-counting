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
MIN_TRUSTED_MINOR_PX = 4.5
COARSE_TOUCH_EXCESS = 1.5
FOREIGN_AREA_RATIO = 3.0
FOREIGN_WIDTH_RATIO = 2.5


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

    Below a few pixels of grain width the boundary is too coarsely sampled for solidity to
    mean anything - measured dispersion nearly doubles - so the concavity half of the test
    is handed to the width excess, which asks the same question of the region's area and
    its thickness instead of its outline. Abstaining there instead, as this stage first
    did, is wrong in both directions at once: on the fine-grained photographs every genuine
    cluster is then counted as one grain, and every other large region is counted as one
    grain too.
    """
    if component["area"] <= TOUCH_AREA_RATIO * calib["a0"]:
        return False

    if calib["minor0"] < MIN_TRUSTED_MINOR_PX:
        return width_excess(component, calib) > COARSE_TOUCH_EXCESS

    return component["solidity"] < calib["solidity0"] - TOUCH_SOLIDITY_MARGIN


def split_component(mask, calib, beta=None):
    """Marker-controlled watershed of one touching cluster."""
    beta = BETA if beta is None else beta
    dist = distance_transform(mask)
    markers = adaptive_markers(dist, calib["minor0"], beta=beta)
    if markers.max() <= 1:
        return (mask > 0).astype(np.int32), dist, markers
    return watershed(-dist, markers, mask=mask > 0), dist, markers

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

from src import calibrate

EPS = 1e-12
MAX_COVERAGE = 0.5
MIN_SINGLES = 3
MIN_AXIS_RATIO = 1.25
MAX_AXIS_RATIO = 8.0
MIN_SOLIDITY = 0.80
MAX_GRAIN_AREA_FRACTION = 0.01
MAX_GRAIN_EXTENT = 0.5
DEFAULT_CHANNELS = ("gray", "hsv_s")
SUBSTRATE_AREA_RATIO = 20.0
SPECK_AREA_RATIO = 0.3
MEDIAN_KSIZE = 1
MEDIAN_OPTIONS = (MEDIAN_KSIZE,)


def otsu_separability(channel):
    """Otsu threshold plus its separability measure eta = sigma_b^2 / sigma_total^2."""
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    levels = np.arange(256, dtype=np.float64)

    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b2 = np.where(denom > EPS, (mu_total * omega - mu) ** 2 / (denom + EPS), 0.0)
    sigma_total2 = float(np.sum(p * (levels - mu_total) ** 2))

    threshold = int(np.argmax(sigma_b2))
    eta = float(sigma_b2[threshold] / (sigma_total2 + EPS))
    return threshold, eta


def candidate_channels(img_bgr, median_ksize=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    channels = {
        "gray": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        "hsv_s": hsv[:, :, 1],
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
    }
    ksize = MEDIAN_KSIZE if median_ksize is None else median_ksize
    if ksize <= 1:
        return channels
    return {k: cv2.medianBlur(v, ksize) for k, v in channels.items()}


def clean_mask(mask, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def candidate_masks(img_bgr, channel_names=None, ksize=3, median_ksize=None):
    """Candidates: channel x denoising x {2-class Otsu, 3-class Otsu} x foreground polarity.

    The scene may hold more than two grey levels (table / substrate / grains), so a plain
    two-class split is not always the right model; the polarity is likewise scene dependent.
    Median denoising is available on this axis but disabled by default: it was added to
    suppress the speck fields that saturation channels produce, a job the contrast term of
    the selection score now does, and it erodes grains only a few pixels wide. Letting the
    selector choose per image was measured too and came out worse than simply leaving it off.
    """
    median_options = MEDIAN_OPTIONS if median_ksize is None else (median_ksize,)

    channels = {}
    for ksize_median in median_options:
        for name, channel in candidate_channels(img_bgr, median_ksize=ksize_median).items():
            if channel_names is not None and name not in channel_names:
                continue
            channels[f"{name}+med{ksize_median}" if ksize_median > 1 else name] = channel

    out = []
    for name, ch in channels.items():
        t_otsu, eta = otsu_separability(ch)
        splits = {"otsu": [t_otsu]}
        try:
            splits["multiotsu"] = list(threshold_multiotsu(ch, classes=3))
        except ValueError:
            pass

        for method, thresholds in splits.items():
            bright = ch > thresholds[-1]
            dark = ch <= thresholds[0]
            for polarity, raw in (("bright", bright), ("dark", dark)):
                raw_mask = raw.astype(np.uint8) * 255
                out.append(
                    {
                        "channel_name": name,
                        "channel": ch,
                        "method": method,
                        "polarity": polarity,
                        "thresholds": thresholds,
                        "eta": eta,
                        "raw_mask": raw_mask,
                        "mask": clean_mask(raw_mask, ksize=ksize),
                    }
                )
    return out


def local_contrast(channel, mask):
    """Centre-surround saliency: the normalised intensity gap between the foreground and
    the ring just outside it.

    This is the centre-surround contrast that visual attention models use to decide what in
    a scene is an object at all, applied here to a whole candidate segmentation rather than
    to a point. A real grain is an object: it stands out from the background it sits on.
    Noise that survives thresholding as speck fields does not, which is what separates the
    two when the speck count alone would otherwise win the vote.
    """
    fg = mask > 0
    if not fg.any():
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ring = cv2.dilate(fg.astype(np.uint8), kernel).astype(bool) & ~inner.astype(bool)
    if not ring.any():
        return 0.0
    ch = channel.astype(np.float64)
    return float(abs(ch[fg].mean() - ch[ring].mean()) / (ch.std() + EPS))


def plausible_grain_population(calib, frame_shape, check_solidity=True):
    """Do the calibrated priors describe rice at all?

    Every test is two-sided. A one-sided prior can only reject one way of being wrong, and
    each of these was reached by a candidate that satisfied the lower bound and was absurd
    above it: a mask of the wooden frame bars passed "rice is elongated" with an axis ratio
    of 25, and passed "grains are many" with three blobs each covering three per cent of a
    224-pixel frame and running longer than the frame is wide.
    """
    if calib["n_singles"] < MIN_SINGLES:
        # A repeated unit cannot be established from one or two blobs: without this a
        # candidate that segments the substrate as one huge region scores highest.
        return False
    if not MIN_AXIS_RATIO <= calib["axis_ratio0"] <= MAX_AXIS_RATIO:
        return False
    if check_solidity and calib["solidity0"] < MIN_SOLIDITY:
        return False

    height, width = frame_shape[:2]
    if calib["a0"] > MAX_GRAIN_AREA_FRACTION * height * width:
        return False
    if calib["major0"] > MAX_GRAIN_EXTENT * max(height, width):
        return False
    return True


def score_candidate(cand, frame_shape):
    """Foreground area explained by mutually consistent single-grain blobs.

    A rice scene is many similar-sized blobs covering a minority of the frame, so a
    candidate is rewarded for the area it explains as single grains: it penalises both
    substrate blobs (few huge regions) and texture noise (many tiny regions).
    """
    mask = cand["mask"]
    # The substrate (table, cloth) runs off the frame while grains sit inside it, so the
    # coverage guard ignores border-connected regions; the mask itself keeps them.
    interior = clear_border(mask > 0)
    coverage = float(interior.mean())
    if coverage > MAX_COVERAGE or coverage <= 0:
        return None

    # No convex hulls here: the score below does not use solidity, and the gate that does is
    # applied later, to candidates taken in score order, so hulls are built only until one
    # of them passes. A speck field scores badly on contrast and never gets that far.
    try:
        calib = calibrate.calibrate(mask, need_solidity=False)
    except ValueError:
        return None

    if not plausible_grain_population(calib, frame_shape, check_solidity=False):
        return None

    calib["coverage"] = coverage
    calib["contrast"] = local_contrast(cand["channel"], mask)
    calib["score"] = calib["n_singles"] * calib["a0"] * calib["contrast"]
    return calib


def drop_substrate(mask, a0):
    """Remove regions that are both frame-connected and far too large to be a grain cluster.

    Requiring both signals keeps a legitimate dense cluster in the middle of the frame and
    keeps grains that merely graze the border, while discarding table/cloth regions.
    """
    labels = label(mask > 0)
    out = mask.copy()
    border_labels = set(np.unique(labels)) - set(np.unique(clear_border(labels)))
    for region in regionprops(labels):
        if region.label in border_labels and region.area > SUBSTRATE_AREA_RATIO * a0:
            out[labels == region.label] = 0
    return out


def remove_specks(mask, a0):
    """Drop components too small to be a grain, measured against the calibrated scale."""
    labels = label(mask > 0)
    out = mask.copy()
    for region in regionprops(labels):
        if region.area < SPECK_AREA_RATIO * a0:
            out[labels == region.label] = 0
    return out


def preprocess(img_bgr, channel_names=DEFAULT_CHANNELS, ksize=3, median_ksize=None):
    """Pick the binarisation whose components look most like a field of single grains."""
    scored = []
    for cand in candidate_masks(img_bgr, channel_names=channel_names, ksize=ksize, median_ksize=median_ksize):
        calib = score_candidate(cand, img_bgr.shape)
        if calib is not None:
            scored.append((cand, calib))

    if not scored:
        raise ValueError("no usable binarisation candidate")

    # Substrate removal can change the picture enough to invalidate the priors the
    # candidate was selected on, so the survivor is re-checked and the ranking falls
    # through to the next candidate rather than accepting whatever came out.
    chosen = None
    for cand, ranked in sorted(scored, key=lambda pair: -pair[1]["score"]):
        # Now measure the hulls this candidate needs and apply the gate that uses them.
        calib = calibrate.calibrate(cand["mask"])
        if not plausible_grain_population(calib, img_bgr.shape):
            continue
        # Substrate removal leaves a ragged fringe where the mask straddled the substrate
        # edge. Those specks are cleared against the scale calibrated before removal;
        # recalibrating on the fringe instead would collapse A0 onto the debris.
        mask = drop_substrate(cand["mask"], calib["a0"])
        mask = remove_specks(mask, calib["a0"])
        try:
            final = calibrate.calibrate(mask)
        except ValueError:
            continue
        if plausible_grain_population(final, img_bgr.shape):
            chosen = (cand, mask, final)
            break

    if chosen is None:
        raise ValueError("no binarisation candidate survived recalibration")
    cand, mask, calib = chosen

    return {
        "channel_name": cand["channel_name"],
        "channel": cand["channel"],
        "method": cand["method"],
        "polarity": cand["polarity"],
        "thresholds": cand["thresholds"],
        "eta": cand["eta"],
        "raw_mask": cand["raw_mask"],
        "mask": mask,
        "calib": calib,
    }

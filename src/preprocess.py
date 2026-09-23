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
    """Otsu 阈值及可分性 eta = 类间方差 / 总方差。"""
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
    """生成候选二值图，通道 x 中值滤波 x 二类/三类 Otsu x 前景取亮或取暗。

    画面里可能有桌面、衬底、米粒三种灰度，二类分割不一定合适，前景取亮取暗也看场景。
    中值滤波默认关闭。它原本用来压饱和度通道的噪点，现在选择分数里的对比度项已经能做到，
    而且它会腐蚀只有几像素宽的米粒。试过让程序逐图决定开不开，结果不如直接关掉。
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
    """前景与紧贴它外面一圈的灰度差，按全图标准差归一化。

    真米粒比周围背景明显亮或暗，阈值后残留的噪点一片则不会，
    只比个数时噪点候选可能胜出，加上这一项就能分开。
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
    """标定出的单粒参数像不像米粒。

    每项都有上下限。只设下限时，木框条的掩膜以长宽比 25 通过了"米粒是细长的"，
    三块各占 224 像素画面百分之三、比画面还长的区域也通过了"米粒很多"。
    """
    if calib["n_singles"] < MIN_SINGLES:
        # 一两个块定不出单粒尺度。没有这条的话，把衬底分成一整块的候选分数最高。
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
    """候选的分数，即能解释为单粒米的前景面积，再乘对比度。

    米粒图是很多大小相近的小块，只占画面一部分。按单粒能解释的面积打分，
    衬底那种少数大块和纹理噪点那种大量碎块都拿不到高分。
    """
    mask = cand["mask"]
    # 衬底会延伸出画面，米粒在画面内，所以覆盖率只统计不连边框的区域，掩膜本身不删。
    interior = clear_border(mask > 0)
    coverage = float(interior.mean())
    if coverage > MAX_COVERAGE or coverage <= 0:
        return None

    # 这里不算凸包。分数不用凸实度，用到它的检查放在后面，按分数从高到低逐个做，
    # 通过一个就停，噪点候选对比度低，排不到前面。
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
    """去掉既连着图像边框、又大得不像米堆的区域，即桌面或布。

    两个条件同时满足才去掉，画面中间密集的米堆和擦到边框的米粒都会保留。
    """
    labels = label(mask > 0)
    out = mask.copy()
    border_labels = set(np.unique(labels)) - set(np.unique(clear_border(labels)))
    for region in regionprops(labels):
        if region.label in border_labels and region.area > SUBSTRATE_AREA_RATIO * a0:
            out[labels == region.label] = 0
    return out


def remove_specks(mask, a0):
    """按标定的单粒面积去掉太小的碎点。"""
    labels = label(mask > 0)
    out = mask.copy()
    for region in regionprops(labels):
        if region.area < SPECK_AREA_RATIO * a0:
            out[labels == region.label] = 0
    return out


def bright_background(img_bgr, mask):
    """背景中比放米平面亮得多的像素。

    照片里除了放米的平面，常还拍到它外面的东西，D3 是木框，D1 是纸后面的墙。
    这些边缘上的反光大小形状和米粒差不多，形状判据去不掉，但它们总有一侧是亮的。
    先用 Otsu 把背景分成暗的一类和其余；背景整片都暗时 Otsu 仍会从噪声中间切一刀，
    所以还要求比"暗背景与米粒灰度的中点"更亮。两个条件单用各在一个数据集上出错，合起来只标出木框和墙。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg = mask > 0
    background = gray[~fg]
    if background.size == 0 or not fg.any():
        return np.zeros(gray.shape, bool)
    otsu, _ = cv2.threshold(background.reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = background[background <= otsu]
    surface = float(np.median(dark if dark.size else background))
    halfway = (surface + float(np.median(gray[fg]))) / 2.0
    return (gray > max(otsu, halfway)) & ~fg


def preprocess(img_bgr, channel_names=DEFAULT_CHANNELS, ksize=3, median_ksize=None):
    """在各候选二值图中选出最像一片单粒米的那个。"""
    scored = []
    for cand in candidate_masks(img_bgr, channel_names=channel_names, ksize=ksize, median_ksize=median_ksize):
        calib = score_candidate(cand, img_bgr.shape)
        if calib is not None:
            scored.append((cand, calib))

    if not scored:
        raise ValueError("no usable binarisation candidate")

    # 去掉衬底后画面可能变了，所以要重新检查，不合格就换下一个候选。
    chosen = None
    for cand, ranked in sorted(scored, key=lambda pair: -pair[1]["score"]):
        # 这时才给这个候选算凸包，做需要凸实度的检查。
        calib = calibrate.calibrate(cand["mask"])
        if not plausible_grain_population(calib, img_bgr.shape):
            continue
        # 去衬底会在边缘留下一圈毛刺，用去衬底前标定的尺度清掉。
        # 如果在毛刺上重新标定，A0 会落到碎屑的大小。
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
        "bright": bright_background(img_bgr, mask),
        "gray": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
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

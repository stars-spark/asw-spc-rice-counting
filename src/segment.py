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
    """区域面积与"按它的厚度应有的面积"之比，只在消融实验里用。

    单粒米约为 1。几粒米并排时厚度不变、面积成倍，比值随粒数上升；
    个头大的单粒厚度也大，比值仍接近 1。
    """
    if calib["width0"] <= 0 or calib["a0"] <= 0:
        return 0.0
    reference = (component["width"] / calib["width0"]) ** 2
    return (component["area"] / calib["a0"]) / max(reference, 1e-6)


def distance_transform(mask):
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)


def adaptive_markers(dist, minor0, beta=BETA):
    """在距离图上取种子，只保留比周围高出至少 h 的峰，同一个峰碎成的几块合成一个。

    h 按单粒短轴取，但不低于 1 像素，更浅的起伏只是边界的锯齿。
    h_maxima 在浮点图上会把一个平顶峰切成几小块，每块都当种子的话一粒米会被切成两半。
    在 D2 上量过，同一粒米的碎块相距不超过 0.55 个粒宽，不同米粒的种子至少相距 0.67，
    所以相距小于 SEED_MERGE_RATIO 个粒宽的合并。
    不合并时调参会把 beta 推到 0.45，丰满米粒之间的浅颈就切不开了。
    """
    h = max(beta * minor0 / 2.0, MIN_DEPTH_PX)
    peaks = (h_maxima(dist, h) > 0).astype(np.uint8)
    radius = max(1, int(round(SEED_MERGE_RATIO * minor0 / 2.0)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    joined = label(cv2.dilate(peaks, disc) > 0)
    return np.where(peaks > 0, joined, 0).astype(np.int32)


def is_foreign_object(component, calib):
    """大区域的形状不可能由米粒堆成时，判为异物。

    宽度判据：米粒只会并排挨着，再多粒也只有一粒厚。最大内切圆直径量下来，
    单粒约 1.0 个粒宽，粘连块最多 2.0，硬币 8.6，D3 的木框边 3.8 到 4.2。
    这个量在粒宽只有 4 像素时也还可靠，凸实度就不行了。
    凸性判据：米粒相接总会留下凹口，面积有好几粒、却和单粒一样凸的区域不是米堆。
    早先还要求"比米粒更圆"，那只能抓住硬币，放过了所有细长的异物，已去掉。
    """
    if component["area"] <= FOREIGN_AREA_RATIO * calib["a0"]:
        return False
    if calib["width0"] > 0 and component["width"] > FOREIGN_WIDTH_RATIO * calib["width0"]:
        return True
    return component["solidity"] >= calib["solidity0"]


def is_touching(component, calib):
    """面积大于 1.2 倍 A0 且凸实度比单粒低 0.06 以上，才算粘连。

    只看面积，偏大的单粒会被误判；只看凹凸，小米粒的边界噪声也像凹口。
    以前粒宽不足 4.5 像素时改用 width_excess 判断，但在 D1 粒宽 3.7 到 4.5 像素的照片上，
    粘连对的凸实度中位数仍比单粒低 0.20，width_excess 却只有 1.22，门限 1.5，
    三分之二的粘连没切开。D3 上它还把木框边的亮条当成粘连块，所以现在各尺度都用凸实度。
    """
    if component["area"] <= TOUCH_AREA_RATIO * calib["a0"]:
        return False
    return component["solidity"] < calib["solidity0"] - TOUCH_SOLIDITY_MARGIN


def grain_brightness(labels, gray, grain_labels):
    """被当作米粒的区域的灰度中位数。"""
    if gray is None or not grain_labels:
        return None
    return float(np.median(gray[np.isin(labels, grain_labels)]))


def is_dark(component, labels, gray, reference):
    """区域比本图米粒暗得多时返回 True，用来去掉阴影。

    D3 上多数图选的是"低饱和度"前景，黑色和白米一样不饱和，
    木框边的阴影条和米粒旁的影子会以米粒大小混进来。它们的灰度只有米粒的百分之几，
    米粒自己在 0.9 到 1.1 之间。DARK_RATIO 只在 D1 上定，D1 的标注米粒没有低于它的。
    """
    if gray is None or reference is None:
        return False
    r0, c0, r1, c1 = component["bbox"]
    region = labels[r0:r1, c0:c1] == component["label"]
    return float(gray[r0:r1, c0:c1][region].mean()) < DARK_RATIO * reference


def borders_bright_background(component, calib, bright):
    """区域是否挨着亮背景，而不是躺在放米的平面上。

    bright 来自 preprocess.bright_background。看区域外 2 像素到约两个粒宽的一圈，
    跳过最里面 2 像素是为了避开米粒自己模糊的亮边。平面上的米粒这一圈几乎没有亮像素，
    木框边的反光则整整一侧都是亮的。亮像素超过 BRIGHT_RING_FRACTION 就判为后者，
    这个值只在 D1 上定，各数据集都没有因此去掉真米粒。
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
    # 带状区在窗口内、near 之外。那里没有亮像素就直接返回，省掉最费时的大核膨胀。
    # 背景均匀时亮像素只有米粒边缘，都在 near 里，所以大多数区域走这条路。
    candidates = bright[top:bottom, left:right] & ~near & (window == 0)
    if not candidates.any():
        return False
    far = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer, outer))) > 0
    band = far & ~near & (window == 0)
    if not band.any():
        return False
    return bright[top:bottom, left:right][band].mean() > BRIGHT_RING_FRACTION


def split_component(mask, calib, beta=None):
    """对一个粘连块做标记控制的分水岭。"""
    beta = BETA if beta is None else beta
    dist = distance_transform(mask)
    markers = adaptive_markers(dist, calib["minor0"], beta=beta)
    if markers.max() <= 1:
        return (mask > 0).astype(np.int32), dist, markers
    return watershed(-dist, markers, mask=mask > 0), dist, markers


def split_by_count(mask, k, minor0):
    """已按面积知道区域里有 k 粒时，把它切成 k 块。

    用于深度判据没切开的区域，丰满米粒之间的颈太浅，h-maxima 看不出来。
    取距离图上最高的 k 个点作种子，彼此相距不小于 SEED_MERGE_RATIO 个粒宽。
    放不下 k 个种子时返回 None，调用方继续按面积计数。
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

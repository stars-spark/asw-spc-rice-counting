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
    """B1，每个连通域算一粒，不处理粘连。"""
    cutoff = MIN_AREA_RATIO * calib["a0"]
    return sum(1 for r in calib["components"] if r["area"] >= cutoff)


def b2_area_estimate(calib):
    """B2，前景总面积除以单粒面积。"""
    cutoff = MIN_AREA_RATIO * calib["a0"]
    total = sum(r["area"] for r in calib["components"] if r["area"] >= cutoff)
    return int(np.round(total / calib["a0"]))


def b3_distance_watershed(mask, calib, fg_ratio=0.5):
    """B3，教科书式的距离变换分水岭，即 OpenCV 教程的做法。

    种子取距离值超过全图最大值固定比例的像素。一个大粘连块会抬高全图最大值，
    较小米粒的种子就被整片抹掉，这正是自适应种子要解决的问题。
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
    """B4，用形态学腐蚀得到种子的标记分水岭。

    按 Kurade 等人 2023 年 Foods 论文的流程，3x3 结构元，腐蚀后做连通域分析。
    该文因为这种做法分不开粘连米粒，把粘连样本从数据集中去掉了。
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
    """Tan 等人的角点响应，即以轮廓点为圆心的圆盘内前景所占比例。

    直边上圆盘一半在区域内，响应约 0.5；两粒米之间的颈部边界向内拐，圆盘内前景更多，响应升高。
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
    """响应超过门限的峰，每个邻域只取一个。

    颈部附近一整段响应都高，相距小于圆盘半径的峰看作同一个角点，只留最强的。
    """
    n = len(response)
    if n < 3:
        return []
    above = np.flatnonzero(response > threshold)
    if above.size == 0:
        return []

    picked = []
    for index in above[np.argsort(-response[above])]:
        # 轮廓是闭合曲线，沿轮廓的距离要绕回去算。
        if all(min(abs(index - other), n - abs(index - other)) > radius for other in picked):
            picked.append(int(index))
    return sorted(picked)


def _ellipse_residual(points):
    """点到其拟合椭圆的均方根距离，单位像素。

    沿椭圆中心到各点的射线量径向距离。不用二次型的代数残差，它随椭圆大小变化，没法和固定容差比较。
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
    """按椭圆拟合把角点之间的轮廓段归并成米粒。

    原文遍历所有分组找总拟合误差最小的一种，组合数随段数增长很快，作者也说耗时随粒数迅速上升。
    这里改成贪心合并，每次合并单个椭圆拟合残差最小的一对，残差不超过粒宽的一定比例才合并。
    判断用拟合残差本身而不是残差的增量，两段短弧总能拟合得很好，增量几乎都是 0。
    原文防止椭圆共圆心的距离惩罚没有实现，它用于比较整套分组，贪心做法不需要。
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
                    # 点太少拟合不了，这种对先合并，两段单独都不可能是一粒米的轮廓。
                    residual = 0.0
                if best is None or residual < best[0]:
                    best = (residual, i, j, merged)

        residual, i, j, merged = best
        if residual > limit:
            break
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
    return len(groups)


def b5_concave_ellipse(mask, calib, use_ellipse=True, radius_ratio=0.5):
    """B5，凹点计数加椭圆修正，按 Avzalov 等人 2025 年 Vavilov 期刊论文。

    用上面的角点响应在每个连通域轮廓上找角点，按原文的配对规则，每两个角点对应一处接触，
    含 k 粒的轮廓应有 2(k-1) 个角点。再用椭圆拟合把角点之间的段重新分组，
    去掉米粒缺口、凹痕产生的假角点。这一类方法都在边界上取特征，是这个问题的主流做法。
    圆盘半径取标定粒宽的一半，随图像缩放。它做成参数，是因为没有一个值在各尺度都合适。
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

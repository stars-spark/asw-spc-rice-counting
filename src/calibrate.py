import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from skimage.measure import label, regionprops

NOISE_FLOOR_PX = 5
# 从 A0 的这个比例起才算凸包，与计数时的碎屑门限一致，
# 后面任何规则会看的连通域都有凸实度，包括消融里用"或"条件的变体。
SHAPE_FLOOR_RATIO = 0.3
SINGLE_BAND = (0.65, 1.45)
AREA_MODE = "dominant"


def component_table(mask, shape_from=None, shape_sample=None, defer_solidity=False):
    """一张二值图里每个连通域的测量值。

    除凸实度外都在一遍扫描里算完。连通域和距离变换用 OpenCV，
    长短轴用二阶矩加 bincount 累加，与 regionprops 的结果只差浮点误差。
    凸实度要算凸包，饱和度通道的候选常有上万个噪点，逐个算很慢，
    所以只给面积不小于 shape_from 的连通域算，其余记为 NaN。
    后面用到凸实度的规则都先要求更大的面积，不会读到 NaN。
    defer_solidity=True 时先不算凸包，多返回一个 fill_solidity 函数，
    调用方可以先估出 A0 再补算，不用把掩膜扫两遍。
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return (labels, [], lambda *a, **k: None) if defer_solidity else (labels, [])

    index = np.arange(1, count)
    areas = stats[index, cv2.CC_STAT_AREA].astype(np.float64)
    # (min_row, min_col, max_row, max_col)，与 regionprops 的 bbox 一致
    boxes = np.stack([stats[index, cv2.CC_STAT_TOP], stats[index, cv2.CC_STAT_LEFT],
                      stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT],
                      stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]], axis=1)

    # 宽度 = 最大内切圆半径的两倍。各连通域之间隔着背景，
    # 对整张掩膜做一次距离变换，与逐个区域单独做结果相同。
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = 2.0 * np.asarray(ndi.maximum(dist, labels, index=index), dtype=np.float64).ravel()

    # 由各标号的二阶中心矩求长短轴。
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
        """给面积不小于 threshold 的连通域算凸包和凸实度。"""
        wanted = np.ones(areas.shape, bool) if threshold is None else areas >= threshold
        chosen = index[wanted]
        if sample is not None and chosen.size > sample:
            # 按面积排序后等间隔抽样，结果是确定的。
            order = chosen[np.argsort(areas[wanted], kind="stable")]
            chosen = np.sort(order[np.linspace(0, order.size - 1, sample).astype(int)])
        if not chosen.size:
            return
        # 先把选中的连通域重新编号为 1..k，regionprops 会遍历整个标号范围，有空号会白跑很多次。
        lut = np.zeros(count, dtype=np.int32)
        lut[chosen] = np.arange(1, chosen.size + 1)
        for region in regionprops(lut[labels]):
            rows[chosen[region.label - 1] - 1]["solidity"] = float(region.solidity)

    if defer_solidity:
        return labels, rows, fill_solidity
    fill_solidity(shape_from, shape_sample)
    return labels, rows


def estimate_single_area(areas, grid_size=512, mode=None):
    """由对数面积分布的众数估计单粒面积 A0。

    用对数是因为粘连块堆在 2A0、3A0 附近，取对数后间距相等，核宽也随米粒大小缩放。
    "dominant" 取最高的峰，单粒是画面里重复最多的东西。
    "lowest" 取最低的显著峰，只适合没有碎屑的图，否则碎屑会被当成单粒。
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
    """从图像本身估计单粒尺度和形状先验。

    need_solidity=False 时不算凸包，凸实度和 solidity0 都是 NaN。
    候选打分用到的单粒个数、A0、长短轴都不需要凸包，所以给候选排序时省掉这一步。
    """
    # 扫一遍掩膜。A0 只用面积估计，不需要凸包，之后只在可能用到凸实度的地方算凸包。
    labels, rows, fill_solidity = component_table(mask, defer_solidity=True)
    if not rows:
        raise ValueError("no foreground components above the noise floor")
    a0, kde_debug = estimate_single_area([r["area"] for r in rows], mode=mode)

    if need_solidity:
        fill_solidity(SHAPE_FLOOR_RATIO * a0)
    singles = [r for r in rows if SINGLE_BAND[0] * a0 <= r["area"] <= SINGLE_BAND[1] * a0]
    if not singles:
        # 退回用全部连通域，所以都要有凸包。
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
        # 两粒米中心能靠多近由短轴决定，不是等面积半径。米粒是并排挨着的。
        "major0": float(np.median([r["major"] for r in singles])),
        "minor0": float(np.median([r["minor"] for r in singles])),
        # 单粒宽度。它是内切圆量出来的，粒宽只有 4 像素时仍可用，凸实度那时就不可靠了。
        "width0": float(np.median([r["width"] for r in singles])),
        "solidity0": solidity0,
        "axis_ratio0": float(np.median([r["axis_ratio"] for r in singles])),
        "n_components": len(rows),
        "n_singles": len(singles),
        "kde_debug": kde_debug,
    }

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

FRAGMENT_RATIO = 0.4
RESIDUAL_RATIO = 1.5
CONCAVITY_MARGIN = 0.03


def merge_fragments(labels, a0, fragment_ratio=FRAGMENT_RATIO):
    """太小、不像一粒米的子区域，并入与它共享边界最长的邻居。

    分水岭只管"哪里可以切"，不管"切得对不对"。从米粒边上削下来的小片还回原区域，面积不丢。
    """
    out = labels.copy()
    regions = {r.label: r for r in regionprops(out)}
    order = sorted(regions, key=lambda k: regions[k].area)

    for lab in order:
        area = int((out == lab).sum())
        if area == 0 or area >= fragment_ratio * a0:
            continue

        piece = out == lab
        ring = ndimage.binary_dilation(piece) & ~piece
        neighbours = out[ring]
        neighbours = neighbours[neighbours > 0]
        if neighbours.size == 0:
            continue

        values, counts = np.unique(neighbours, return_counts=True)
        out[piece] = values[int(np.argmax(counts))]
    return out


def count_regions(labels, a0, solidity0=None, residual_ratio=RESIDUAL_RATIO,
                  fragment_ratio=FRAGMENT_RATIO, solidity_margin=CONCAVITY_MARGIN):
    """数分割后各区域的粒数。

    过大的区域只有同时比单粒明显更凹时才按面积折算粒数。分水岭没找到颈部的粘连块仍然是凹的，
    个头大的单粒却和别的单粒一样凸，这样不会把偏大的单粒算成两粒。
    """
    total = 0
    details = []
    for region in regionprops(labels):
        ratio = region.area / a0
        if ratio < fragment_ratio:
            verdict, n = "fragment", 0
        elif ratio > residual_ratio and (
            solidity0 is None or region.solidity < solidity0 - solidity_margin
        ):
            verdict, n = "residual", max(1, int(round(ratio)))
        else:
            verdict, n = "grain", 1
        total += n
        details.append({"label": region.label, "ratio": ratio, "verdict": verdict, "n": n})
    return total, details


def correct_cluster(labels, a0, solidity0=None, residual_ratio=RESIDUAL_RATIO,
                    fragment_ratio=FRAGMENT_RATIO):
    """a0 要和 labels 在同一尺度下，见 segment 里的缩放。"""
    merged = merge_fragments(labels, a0, fragment_ratio=fragment_ratio)
    count, details = count_regions(
        merged, a0, solidity0=solidity0, residual_ratio=residual_ratio,
        fragment_ratio=fragment_ratio
    )
    return count, merged, details


def resplit_residuals(merged, details, a0, minor0, solidity0=None,
                      residual_ratio=RESIDUAL_RATIO, fragment_ratio=FRAGMENT_RATIO):
    """把"按面积算作 n 粒"的区域真正切成 n 块。

    分水岭没切开的区域按 round(面积 / A0) 计数，数目通常是对的，但画出来几粒米是一整块。
    这里按粒数再切一次，每块都不小于碎屑门限、也不超过 RESIDUAL_RATIO 倍 A0 才保留，
    否则仍按面积计数。
    """
    from src import segment

    out = merged.copy()
    next_label = int(out.max()) + 1
    for detail in details:
        if detail["verdict"] != "residual":
            continue
        region = out == detail["label"]
        pieces = segment.split_by_count(region.astype(np.uint8), detail["n"], minor0)
        # 一块区域可能由不相连的几片组成，注水够不着的那片会留成 0，这时不强行切
        if pieces is None or (region & (pieces == 0)).any():
            continue
        areas = [int((pieces == i).sum()) for i in range(1, detail["n"] + 1)]
        if min(areas) < fragment_ratio * a0 or max(areas) > residual_ratio * a0:
            continue
        for i in range(1, detail["n"] + 1):
            out[pieces == i] = next_label
            next_label += 1
    count, details = count_regions(out, a0, solidity0=solidity0,
                                   residual_ratio=residual_ratio, fragment_ratio=fragment_ratio)
    return count, out, details

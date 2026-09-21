import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

FRAGMENT_RATIO = 0.4
RESIDUAL_RATIO = 1.5
CONCAVITY_MARGIN = 0.03


def merge_fragments(labels, a0, fragment_ratio=FRAGMENT_RATIO):
    """Absorb sub-regions too small to be a grain into the neighbour they share most border with.

    The watershed answers "where might a cut go", not "was the cut real". A sliver carved off
    a grain boundary is returned to the region it came from, which conserves area instead of
    discarding it.
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
    """Count grains in a segmented region set.

    An oversized region is only re-counted by area when it is also concave relative to the
    single-grain prior. Grains that merged without the watershed finding the neck stay
    concave, whereas a merely large single grain is as convex as any other, so this keeps
    area accounting from doubling grains that are simply above the modal size.
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
    """`a0` must already be expressed in the same domain as `labels` (see segment scaling)."""
    merged = merge_fragments(labels, a0, fragment_ratio=fragment_ratio)
    count, details = count_regions(
        merged, a0, solidity0=solidity0, residual_ratio=residual_ratio,
        fragment_ratio=fragment_ratio
    )
    return count, merged, details


def resplit_residuals(merged, details, a0, minor0, solidity0=None,
                      residual_ratio=RESIDUAL_RATIO, fragment_ratio=FRAGMENT_RATIO):
    """Turn "counted by area" into an actual cut wherever the cut looks like grains.

    A region the watershed left whole is counted as round(area / A0) grains. That count is
    usually right, but the drawing then shows several grains as one. Here each such region is
    cut into that many pieces; the cut is kept only if every piece is at least a fragment's
    size and none is itself still an oversized concave region, otherwise the area count stands.
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

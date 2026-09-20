"""Render per-image counting results for visual checking.

Each counted instance is drawn as an outline only, never filled, so the grain underneath
stays visible. Nearby instances are given different colours by greedy graph colouring so
that a boundary between two touching grains is unambiguous. Where the method attributed
several grains to one region by area rather than by an actual cut, the region is marked
with a dashed box and a multiplier, so the region count and the reported count can be
reconciled by eye.
"""
import argparse

import cv2
import numpy as np
from skimage.measure import regionprops

from src import counter, io_utils, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

RENDER_ROOT = RESULTS_ROOT / "renders"

# Bright, mutually distinct, readable on both dark and light backgrounds.
PALETTE = [
    (66, 220, 66), (255, 96, 96), (80, 160, 255), (255, 220, 60),
    (255, 120, 255), (60, 235, 235), (255, 160, 40), (170, 130, 255),
]
FOREIGN_COLOUR = (190, 190, 190)


def neighbour_graph(regions, reach=2.2):
    """Link instances whose centroids lie within `reach` times their mean size."""
    centroids = np.array([r["centroid"] for r in regions], dtype=np.float64)
    sizes = np.array([max(r["major"], 4.0) for r in regions], dtype=np.float64)
    graph = {i: set() for i in range(len(regions))}
    if len(regions) < 2:
        return graph

    deltas = centroids[:, None, :] - centroids[None, :, :]
    distances = np.hypot(deltas[:, :, 0], deltas[:, :, 1])
    limits = reach * 0.5 * (sizes[:, None] + sizes[None, :])
    close = (distances < limits) & ~np.eye(len(regions), dtype=bool)
    for i, j in zip(*np.nonzero(close)):
        graph[int(i)].add(int(j))
    return graph


def greedy_colours(graph, n_colours=len(PALETTE)):
    """Assign each instance a colour different from its neighbours (Welsh-Powell order)."""
    order = sorted(graph, key=lambda k: -len(graph[k]))
    assigned = {}
    for node in order:
        taken = {assigned[n] for n in graph[node] if n in assigned}
        for colour in range(n_colours):
            if colour not in taken:
                assigned[node] = colour
                break
        else:
            assigned[node] = 0
    return assigned


def _banner(canvas, lines):
    """Caption strip sized to the image, so it stays legible on a 224 px thumbnail."""
    longest = max(len(text) for text in lines)
    scale = min(0.55, canvas.shape[1] / (longest * 20.0))
    pad = max(3, int(6 * scale / 0.55))
    line_h = max(9, int(round(26 * scale)))
    height = pad * 2 + line_h * len(lines)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (pad, pad + line_h * (index + 1) - int(line_h * 0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _dashed_box(canvas, bbox, colour, dash=6):
    r0, c0, r1, c1 = bbox
    for x in range(c0, c1, dash * 2):
        cv2.line(canvas, (x, r0), (min(x + dash, c1), r0), colour, 1, cv2.LINE_AA)
        cv2.line(canvas, (x, r1), (min(x + dash, c1), r1), colour, 1, cv2.LINE_AA)
    for y in range(r0, r1, dash * 2):
        cv2.line(canvas, (c0, y), (c0, min(y + dash, r1)), colour, 1, cv2.LINE_AA)
        cv2.line(canvas, (c1, y), (c1, min(y + dash, r1)), colour, 1, cv2.LINE_AA)


def draw(image_bgr, labels, info, banner_lines, thickness=1):
    canvas = image_bgr.copy()
    regions = []
    for region in regionprops(labels):
        meta = info.get(region.label, {"kind": "grain", "count": 1})
        regions.append(
            {
                "label": region.label,
                "centroid": region.centroid,
                "major": region.axis_major_length,
                "bbox": region.bbox,
                "kind": meta["kind"],
                "count": meta["count"],
            }
        )

    colours = greedy_colours(neighbour_graph(regions))

    for index, region in enumerate(regions):
        mask = (labels == region["label"]).astype(np.uint8)
        colour = (FOREIGN_COLOUR if region["kind"] == "foreign"
                  else PALETTE[colours.get(index, 0)])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, thickness, cv2.LINE_AA)

        cy, cx = region["centroid"]
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), 1, colour, -1, cv2.LINE_AA)

        if region["kind"] == "foreign":
            r0, c0, r1, c1 = region["bbox"]
            cv2.putText(canvas, "x", (c0, max(10, r0 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colour, 1, cv2.LINE_AA)
        elif region["count"] > 1:
            _dashed_box(canvas, region["bbox"], colour)
            r0, c0, _, _ = region["bbox"]
            cv2.putText(canvas, f"x{region['count']}", (c0, max(10, r0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # No banner when the caller supplies none: figures in the report carry the counts in
    # their own captions, in Chinese, which OpenCV cannot draw.
    if banner_lines:
        _banner(canvas, banner_lines)
    return canvas


def masks_to_labels(masks, shape):
    """Stack instance masks into one label image; earlier (higher-scoring) masks win."""
    labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        binary = np.asarray(mask).astype(bool)
        if binary.shape != shape:
            binary = cv2.resize(binary.astype(np.uint8), (shape[1], shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        labels[binary & (labels == 0)] = index
    return labels


def render_ours(samples, out_dir, pre_by_name=None):
    ensure_dir(out_dir)
    written = 0
    for sample in samples:
        image = io_utils.imread(sample["path"])
        pre = (pre_by_name or {}).get(sample["file_name"])
        labels, info, total, _ = counter.label_image(image, pre=pre)
        regions = int(labels.max())
        lines = [
            f"ASW-SPC count = {total}   ground truth = {sample['gt_count']}",
            f"regions drawn = {regions}   (x-n = counted by area)",
        ]
        cv2.imwrite(str(out_dir / sample["file_name"].replace(".jpg", ".png")),
                    draw(image, labels, info, lines))
        written += 1
    return written


def render_sam3(samples, out_dir, teacher, prompt, threshold):
    ensure_dir(out_dir)
    written = 0
    for sample in samples:
        image = io_utils.imread(sample["path"])
        result = teacher.segment(image, prompt=prompt, threshold=threshold)
        scores = np.asarray(result["scores"].float().cpu())
        keep = scores >= threshold
        masks = np.asarray(result["masks"].float().cpu())[keep]

        labels = masks_to_labels(masks, image.shape[:2])
        info = {i: {"kind": "grain", "count": 1} for i in range(1, int(labels.max()) + 1)}
        lines = [
            f"SAM 3 count = {int(keep.sum())}   ground truth = {sample['gt_count']}",
            f"prompt = '{prompt}'   threshold = {threshold}",
        ]
        cv2.imwrite(str(out_dir / sample["file_name"].replace(".jpg", ".png")),
                    draw(image, labels, info, lines))
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Render counting results for inspection")
    parser.add_argument("--methods", nargs="+", default=["ours", "sam3"])
    parser.add_argument("--datasets", nargs="+", default=["d1", "d2", "d3"])
    parser.add_argument("--prompt", default="white seed")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    loaders = {"d1": io_utils.load_d1, "d2": synth.load_d2, "d3": io_utils.load_d3,
               "d4": io_utils.load_d4}

    if "ours" in args.methods:
        from src import evaluate
        for name in args.datasets:
            cached = {it["file_name"]: it["pre"] for it in evaluate.preprocessed(name)}
            samples = loaders[name]()[: args.limit]
            n = render_ours(samples, RENDER_ROOT / "ours" / name, pre_by_name=cached)
            print(f"ours/{name}: {n} images -> {RENDER_ROOT / 'ours' / name}")

    if "sam3" in args.methods:
        from src.teacher_sam import Sam3Teacher
        teacher = Sam3Teacher()
        for name in args.datasets:
            samples = loaders[name]()[: args.limit]
            n = render_sam3(samples, RENDER_ROOT / "sam3" / name, teacher,
                            args.prompt, args.threshold)
            print(f"sam3/{name}: {n} images -> {RENDER_ROOT / 'sam3' / name}")


if __name__ == "__main__":
    main()

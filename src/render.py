"""把每张图的计数结果画出来，便于肉眼检查。

每个计数实例用半透明色填充并描同色边，一粒米是一块颜色，米粒纹理仍能看到。
相邻实例用贪心图着色分配不同颜色，两粒粘连米粒之间的分界一看就清楚。
没有真正切开、按面积算作几粒的区域画虚线框并标倍数，区域数与报告的粒数可以对上。
"""
import argparse

import cv2
import numpy as np
from skimage.measure import regionprops

from src import counter, io_utils, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

RENDER_ROOT = RESULTS_ROOT / "renders"

# 颜色鲜明、彼此好区分，深色浅色背景上都看得清，BGR 顺序。用十种而不是八种，密集的块不容易重色。
PALETTE = [
    (66, 220, 66), (255, 96, 96), (80, 160, 255), (255, 220, 60),
    (255, 120, 255), (60, 235, 235), (255, 160, 40), (170, 130, 255),
    (40, 90, 255), (150, 255, 150),
]
FILL_ALPHA = 0.6
FOREIGN_COLOUR = (190, 190, 190)


def neighbour_graph(regions, reach=2.2):
    """质心距离在 reach 倍平均尺寸以内的实例之间连边。"""
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
    """给每个实例分配与邻居不同的颜色，按 Welsh-Powell 顺序。

    每个节点从不同的颜色开始找。都从第一种开始的话，孤立的米粒全是同一种颜色，
    米粒大多分开的图会几乎一片绿。轮换起点能让颜色铺开，相邻的仍不会同色。
    """
    order = sorted(graph, key=lambda k: -len(graph[k]))
    assigned = {}
    for node in order:
        taken = {assigned[n] for n in graph[node] if n in assigned}
        start = (node * 7) % n_colours
        for step in range(n_colours):
            colour = (start + step) % n_colours
            if colour not in taken:
                assigned[node] = colour
                break
        else:
            assigned[node] = start
    return assigned


def _banner(canvas, lines):
    """按图像大小画标题条，224 像素的小图上也看得清。"""
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

    # 先把每粒米整块涂色，再统一与原图混合，颜色深浅在整张图上一致。
    # 异物不涂色，只勾灰色轮廓，一眼能看出它没有被计数。
    fill = canvas.copy()
    for index, region in enumerate(regions):
        if region["kind"] != "foreign":
            fill[labels == region["label"]] = PALETTE[colours.get(index, 0)]
    canvas = cv2.addWeighted(fill, FILL_ALPHA, canvas, 1 - FILL_ALPHA, 0)

    for index, region in enumerate(regions):
        mask = (labels == region["label"]).astype(np.uint8)
        colour = (FOREIGN_COLOUR if region["kind"] == "foreign"
                  else PALETTE[colours.get(index, 0)])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, thickness, cv2.LINE_AA)

        if region["kind"] == "foreign":
            r0, c0, r1, c1 = region["bbox"]
            cv2.putText(canvas, "x", (c0, max(10, r0 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colour, 1, cv2.LINE_AA)
        elif region["count"] > 1:
            _dashed_box(canvas, region["bbox"], colour)
            r0, c0, _, _ = region["bbox"]
            cv2.putText(canvas, f"x{region['count']}", (c0, max(10, r0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # 调用方没给标题就不画。报告里的图把粒数写在中文图题里，OpenCV 画不了中文。
    if banner_lines:
        _banner(canvas, banner_lines)
    return canvas


def masks_to_labels(masks, shape):
    """把实例掩膜叠成一张标号图，排在前面、得分高的掩膜优先。"""
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

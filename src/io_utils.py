import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "extracted"
RESULTS_ROOT = PROJECT_ROOT / "results"


def imread(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def load_coco_split(root, split, keep_categories=None, with_points=False):
    """返回 [{path, file_name, gt_count}]，gt_count 为保留下来的框数。

    with_points=True 时每个样本另带 points，即各框中心的 (row, col)。
    计数只要框的个数，训练密度图模型要位置，框中心是标注能给的最好估计。
    """
    split_dir = Path(root) / split
    with open(split_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    keep_ids = set(cat_name)
    if keep_categories is not None:
        keep_ids = {cid for cid, name in cat_name.items() if name in keep_categories}

    per_image = defaultdict(int)
    points = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in keep_ids:
            per_image[ann["image_id"]] += 1
            if with_points:
                x, y, w, h = ann["bbox"]
                points[ann["image_id"]].append((y + h / 2.0, x + w / 2.0))

    samples = []
    for img in coco["images"]:
        sample = {
            "path": split_dir / img["file_name"],
            "file_name": img["file_name"],
            "gt_count": per_image[img["id"]],
        }
        if with_points:
            sample["points"] = np.asarray(points[img["id"]], dtype=np.float32).reshape(-1, 2)
        samples.append(sample)
    return samples


def load_d1(splits=("train", "valid"), with_points=False):
    """D1，rice.v1i.coco 真实照片，真值为 'RICE' 类框的个数。"""
    root = DATA_ROOT / "rice.v1i.coco"
    samples = []
    for split in splits:
        samples.extend(load_coco_split(root, split, keep_categories={"RICE", "rice"},
                                       with_points=with_points))
    return samples


def load_d3(splits=("train", "valid", "test"), with_points=False):
    """D3，RICE.v3i.coco 的 224x224 低分辨率图，真值为框的总数。

    导出包里每张原图还有翻转增广的副本，不是独立样本，每张原图只保留第一个文件。
    类别名 '216'、'222'、'438' 没有说明，不使用。
    """
    root = DATA_ROOT / "RICE.v3i.coco"
    seen = set()
    samples = []
    for split in splits:
        for sample in load_coco_split(root, split, with_points=with_points):
            source = sample["file_name"].split("_jpg.rf.")[0]
            if source in seen:
                continue
            seen.add(source)
            samples.append(sample)
    return samples

def dataset_root(root_name):
    """数据集所在目录，默认为项目 data 目录，设置了 COUTRICE_ALT_DATA 时用它。

    项目原先放在 NTFS 分区上，那里写出的目录有时一遍历就卡死在不可中断的 I/O 里，
    所以留了这个变量，可以把数据指到本地文件系统。
    """
    primary = DATA_ROOT / root_name
    alternate = os.environ.get("COUTRICE_ALT_DATA")
    if alternate:
        candidate = Path(alternate) / root_name
        if candidate.exists():
            return candidate
    return primary


def load_coco_all(root_name, splits=("train", "valid", "test"), keep_categories=None):
    """按 data/ 下的目录名读取 Roboflow COCO 导出包的所有划分。"""
    root = dataset_root(root_name)
    samples = []
    for split in splits:
        if (root / split / "_annotations.coco.json").exists():
            samples.extend(load_coco_split(root, split, keep_categories))
    return samples


def load_d4(splits=("train", "valid", "test")):
    """D4，一组真实照片，只用于分析失败原因，不参与评测。

    找它是为了补上 D1、D2 都缺的真实粘连照片。它确实粘连较多，
    标注米粒落在共享连通域里的比例中位数为 22%，D1 只有 6%。不参与评测有两个原因，都量过。
    一是 47% 的图里米粒面积相差三倍以上，很多图把几个品种并排放在一起比较。
    自标定把面积分布的主峰当作单粒大小，一张图里有几种大小的米就违背了方法的前提。
    二是纸面背景有密集的暗色斑点，米粒与背景对比度很低，二值化有时会选中斑点而不是米粒。
    另一个候选导出包在这之前也量过，只有 5% 的米粒粘连，与 D1 相当，
    而且 242 张里有 211 张是 31 张照片的增广副本，所以没用。
    按外接框重叠估计粘连会高估，细长的米粒斜着靠近时框就重叠，其实并没挨着。
    """
    from src.dedupe import fingerprint, self_duplicates

    pooled = load_coco_all("ricecount.v2i.coco", splits,
                           keep_categories={"rice", "Rice", "RICE"})
    rows = fingerprint(pooled)
    by_name = {s["file_name"]: s for s in pooled}
    drop = {d["file_name"] for group in self_duplicates(rows) for d in group[1:]}
    return [by_name[r["file_name"]] for r in rows if r["file_name"] not in drop]

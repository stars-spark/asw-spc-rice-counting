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
    """Return [{path, file_name, gt_count}] with gt_count = number of kept boxes.

    With `with_points`, each sample also carries `points`: the (row, col) centre of every
    kept box. Counting needs only the number of boxes, but a model trained to predict where
    the grains are needs their positions, and a box centre is the best the annotation gives.
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
    """rice.v1i.coco: sparse real photos, GT = number of 'RICE' boxes."""
    root = DATA_ROOT / "rice.v1i.coco"
    samples = []
    for split in splits:
        samples.extend(load_coco_split(root, split, keep_categories={"RICE", "rice"},
                                       with_points=with_points))
    return samples


def load_d3(splits=("train", "valid", "test"), with_points=False):
    """RICE.v3i.coco: low-resolution 224x224 scenes, GT = total box count.

    The export contains flip-augmented copies of each source photo, which are not
    independent samples, so only the first file per source image is kept. The category
    labels ('216', '222', '438') are undocumented and deliberately ignored.
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
    """Where a dataset lives: the project data directory, or the fallback if set.

    COUTRICE_ALT_DATA exists because the project partition is NTFS and a directory written
    there can end up with metadata that makes enumerating it hang in uninterruptible I/O.
    Pointing this at a local filesystem keeps the pipeline usable without moving the project.
    """
    primary = DATA_ROOT / root_name
    alternate = os.environ.get("COUTRICE_ALT_DATA")
    if alternate:
        candidate = Path(alternate) / root_name
        if candidate.exists():
            return candidate
    return primary


def load_coco_all(root_name, splits=("train", "valid", "test"), keep_categories=None):
    """Every split of a Roboflow COCO export, by directory name under data/."""
    root = dataset_root(root_name)
    samples = []
    for split in splits:
        if (root / split / "_annotations.coco.json").exists():
            samples.extend(load_coco_split(root, split, keep_categories))
    return samples


def load_d4(splits=("train", "valid", "test")):
    """A set of real photographs used for failure analysis, not as a benchmark.

    It was collected to supply what D1 and D2 between them do not - real photographs of
    grains that genuinely touch - and it does contain touching, merging a median of 22 per
    cent of its annotated grains into shared components against 6 per cent for D1. It is not
    scored alongside the others because two things about it put it outside what the method
    claims to do, both established by measurement rather than by inspection alone:

    Forty-seven per cent of its images hold grain areas spanning more than a factor of three,
    because a large part of the set photographs several cultivars side by side for
    comparison - dark short grains, pale slender ones and rounded grey-green ones in one
    frame. Self-calibration takes the dominant mode of the area distribution as the size of
    one grain, so an image holding several populations of different sizes violates the
    assumption the whole method rests on. Quoting an error on such images would be scoring
    the method on a task it does not claim.

    Its paper background also carries dense dark speckle at very low grain-to-background
    contrast, which the binarisation selector sometimes prefers to the grains themselves.
    That is the same speck-field failure the controlled-degradation study found under blur,
    and this set is where it was confirmed to happen on real photographs rather than only on
    synthesised ones.

    A second candidate export was measured and dropped before this one: it merged 5 per cent
    of its grains, statistically the same as D1, and 211 of its 242 images were augmented
    copies of 31 photographs. Bounding-box overlap had suggested both sets were heavily
    touching, which is why that statistic is not used here - a grain is elongated, so two
    lying diagonally near each other have overlapping boxes without touching at all, and the
    measure reports their shape rather than their contact.
    """
    from src.dedupe import fingerprint, self_duplicates

    pooled = load_coco_all("ricecount.v2i.coco", splits,
                           keep_categories={"rice", "Rice", "RICE"})
    rows = fingerprint(pooled)
    by_name = {s["file_name"]: s for s in pooled}
    drop = {d["file_name"] for group in self_duplicates(rows) for d in group[1:]}
    return [by_name[r["file_name"]] for r in rows if r["file_name"] not in drop]

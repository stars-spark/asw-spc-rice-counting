"""检查候选的粘连数据集与 D1 之间、以及集内部有没有重复图片。

同一来源导出的两个数据集互相比较会自证，Roboflow 导出包也常含同一张照片翻转、裁剪后的副本。
这里用感知哈希找重复，把图缩成小灰度图，按各像素是否高于中位数编码，
缩放和 JPEG 重压缩后不变，不同场景仍能分开。
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np

from src import io_utils

HASH_SIZE = 8
NEAR_DUPLICATE_BITS = 5


def phash(path, size=HASH_SIZE):
    image = io_utils.imread(path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    return (small > np.median(small)).ravel()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def fingerprint(samples):
    """一个数据集的哈希，正反两个方向都存，翻转后也能匹配。"""
    rows = []
    for sample in samples:
        bits = phash(sample["path"])
        flipped = bits.reshape(HASH_SIZE, HASH_SIZE)[:, ::-1].ravel()
        rows.append({"file_name": sample["file_name"], "gt": sample["gt_count"],
                     "bits": bits, "flipped": flipped})
    return rows


def cross_duplicates(left, right, limit=NEAR_DUPLICATE_BITS):
    """两个数据集之间是同一张照片的图片对。"""
    hits = []
    for a in left:
        for b in right:
            distance = min(hamming(a["bits"], b["bits"]), hamming(a["flipped"], b["bits"]))
            if distance <= limit:
                hits.append((a["file_name"], b["file_name"], distance, a["gt"], b["gt"]))
                break
    return hits


def self_duplicates(rows, limit=NEAR_DUPLICATE_BITS):
    """一个数据集内部是同一张照片的图片组。"""
    groups = defaultdict(list)
    assigned = {}
    for index, a in enumerate(rows):
        if a["file_name"] in assigned:
            continue
        assigned[a["file_name"]] = index
        groups[index].append(a)
        for b in rows[index + 1:]:
            if b["file_name"] in assigned:
                continue
            distance = min(hamming(a["bits"], b["bits"]), hamming(a["flipped"], b["bits"]))
            if distance <= limit:
                assigned[b["file_name"]] = index
                groups[index].append(b)
    return [g for g in groups.values() if len(g) > 1]


def main():
    parser = argparse.ArgumentParser(description="Duplicate check for the candidate sets")
    parser.add_argument("--bits", type=int, default=NEAR_DUPLICATE_BITS)
    args = parser.parse_args()

    d1 = fingerprint(io_utils.load_d1())
    print(f"D1: {len(d1)} images")

    for root in ("ricecount.v2i.coco", "Rice.v1i.coco"):
        samples = io_utils.load_coco_all(root)
        rows = fingerprint(samples)
        groups = self_duplicates(rows, args.bits)
        redundant = sum(len(g) - 1 for g in groups)
        across = cross_duplicates(rows, d1, args.bits)

        print(f"\n{root}: {len(rows)} images")
        print(f"  internal duplicate groups: {len(groups)} covering {redundant} redundant copies")
        print(f"  images also present in D1: {len(across)}")
        for a, b, distance, ga, gb in across[:5]:
            print(f"    {a[:34]} ~ {b[:34]}  (distance {distance}, counts {ga} vs {gb})")


if __name__ == "__main__":
    main()

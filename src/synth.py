import argparse
import csv
import glob

import cv2
import numpy as np
from skimage.measure import label, regionprops

from src.io_utils import DATA_ROOT, RESULTS_ROOT, ensure_dir

GRAIN_SOURCE = DATA_ROOT / "RICE.v2i.folder" / "train" / "Karacadag"
D2_ROOT = RESULTS_ROOT / "synthetic"
TOUCH_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
TARGET_MAJOR_PX = 40.0
OVERLAP_BAND = (0.01, 0.15)
PLACEMENT_GAP_PX = 2


def load_grain_bank(limit=200, target_major=TARGET_MAJOR_PX):
    """从品种数据集里抠出单粒米，缩放到统一大小。"""
    bank = []
    for path in sorted(glob.glob(str(GRAIN_SOURCE / "*")))[:limit]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        regions = regionprops(label(binary > 0))
        if not regions:
            continue
        region = max(regions, key=lambda r: r.area)
        if region.axis_major_length <= 0:
            continue

        r0, c0, r1, c1 = region.bbox
        patch = img[r0:r1, c0:c1]
        mask = (label(binary > 0)[r0:r1, c0:c1] == region.label).astype(np.uint8)

        scale = target_major / region.axis_major_length
        size = (max(2, int(round(patch.shape[1] * scale))), max(2, int(round(patch.shape[0] * scale))))
        bank.append(
            {
                "patch": cv2.resize(patch, size, interpolation=cv2.INTER_AREA),
                "mask": cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
            }
        )
    if not bank:
        raise RuntimeError(f"no grains found under {GRAIN_SOURCE}")
    return bank


def _transform(grain, rng, scale_jitter=(0.85, 1.15)):
    scale = rng.uniform(*scale_jitter)
    angle = rng.uniform(0, 360)

    patch, mask = grain["patch"], grain["mask"]
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w) * scale)) + 2
    pad_y, pad_x = (diag - h) // 2 + 1, (diag - w) // 2 + 1

    patch = cv2.copyMakeBorder(patch, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = cv2.copyMakeBorder(mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)

    center = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle, scale)
    patch = cv2.warpAffine(patch, rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return patch[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1], mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _paste(canvas, occupancy, patch, mask, top, left):
    h, w = mask.shape
    region = occupancy[top: top + h, left: left + w]
    sel = mask > 0
    canvas[top: top + h, left: left + w][sel] = patch[sel]
    region[sel] = True


def _overlap(occupancy, mask, top, left):
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > occupancy.shape[0] or left + w > occupancy.shape[1]:
        return None
    return int(np.count_nonzero(occupancy[top: top + h, left: left + w] & (mask > 0)))


def build_scene(bank, n_grains, touch_prob, canvas_size=900, rng=None, bg_value=20, noise_std=4.0):
    """合成一个场景，touch_prob 控制一粒米贴着另一粒放的概率。

    真值就是实际放下的粒数，按构造是精确的。
    """
    rng = rng or np.random.default_rng()
    canvas = np.full((canvas_size, canvas_size, 3), bg_value, dtype=np.uint8)
    occupancy = np.zeros((canvas_size, canvas_size), dtype=bool)
    centers = []

    for _ in range(n_grains):
        transformed = _transform(bank[rng.integers(len(bank))], rng)
        if transformed is None:
            continue
        patch, mask = transformed
        h, w = mask.shape
        area = int(np.count_nonzero(mask))
        placed = False

        if centers and rng.random() < touch_prob:
            anchor = centers[rng.integers(len(centers))]
            theta = rng.uniform(0, 2 * np.pi)
            for dist in range(int(TARGET_MAJOR_PX * 1.4), 2, -2):
                top = int(round(anchor[0] + dist * np.sin(theta) - h / 2))
                left = int(round(anchor[1] + dist * np.cos(theta) - w / 2))
                inter = _overlap(occupancy, mask, top, left)
                if inter is None:
                    continue
                if inter > OVERLAP_BAND[1] * area:
                    break
                if inter >= max(1, OVERLAP_BAND[0] * area):
                    _paste(canvas, occupancy, patch, mask, top, left)
                    centers.append((top + h / 2, left + w / 2))
                    placed = True
                    break

        if not placed:
            pad = cv2.dilate(mask, np.ones((PLACEMENT_GAP_PX * 2 + 1,) * 2, np.uint8))
            for _ in range(60):
                top = int(rng.integers(0, canvas_size - h))
                left = int(rng.integers(0, canvas_size - w))
                if _overlap(occupancy, pad, top, left) == 0:
                    _paste(canvas, occupancy, patch, mask, top, left)
                    centers.append((top + h / 2, left + w / 2))
                    placed = True
                    break

    noise = rng.normal(0, noise_std, canvas.shape)
    canvas = np.clip(canvas.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return canvas, len(centers), centers


def build_d2(per_level=8, count_range=(40, 100), seed=2026, root=D2_ROOT):
    """生成 D2，五档粘连率，真值精确。

    每个场景的粒数不同，这样输出常数的预测器拿不到好成绩，预测值与真值的回归也有意义。
    """
    ensure_dir(root)
    bank = load_grain_bank(limit=300)
    rng = np.random.default_rng(seed)

    manifest = []
    for level in TOUCH_LEVELS:
        for idx in range(per_level):
            n_grains = int(rng.integers(count_range[0], count_range[1] + 1))
            img, count, _ = build_scene(bank, n_grains=n_grains, touch_prob=level, rng=rng)
            name = f"touch{int(level * 100):02d}_{idx:02d}.png"
            cv2.imwrite(str(root / name), img)
            manifest.append({"file_name": name, "gt_count": count, "touch_prob": level})

    with open(root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "gt_count", "touch_prob"])
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def load_d2(root=D2_ROOT):
    with open(root / "manifest.csv") as f:
        return [
            {
                "path": root / row["file_name"],
                "file_name": row["file_name"],
                "gt_count": int(row["gt_count"]),
                "touch_prob": float(row["touch_prob"]),
            }
            for row in csv.DictReader(f)
        ]


def main():
    parser = argparse.ArgumentParser(description="生成 D2 可控粘连合成数据集")
    parser.add_argument("--per-level", type=int, default=8, help="每个粘连档位的图像数")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，决定生成结果")
    args = parser.parse_args()

    manifest = build_d2(per_level=args.per_level, seed=args.seed)
    grains = sum(row["gt_count"] for row in manifest)
    print(f"生成 {len(manifest)} 张合成图、共 {grains} 粒米 -> {D2_ROOT}")


if __name__ == "__main__":
    main()

"""A small student network distilled from SAM 3, trained to predict a grain density map.

Counting by density rather than by instances is deliberate. The grains in two of the three
sets are about four pixels wide, which is too small to ask a network for clean instance
boundaries, but wide enough to place a blob at: the count is then the integral of the
predicted map, so the loss never depends on separating one grain from its neighbour - the
very thing the classical pipeline has to work hardest for.

Supervision is mixed on purpose (see `src/pseudo.py`):

* the synthetic set is supervised with its exact construction-time centres;
* the two photographed sets with SAM 3's mask centroids, which is the only supervision
  available without using their human labels.

The human labels are never trained on. They are the test set, so the student's number can
be read as "what a 2 MB network learned from a 3.2 GB teacher, measured against people".
"""
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import io_utils, pseudo, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

MODEL_ROOT = RESULTS_ROOT / "student"
METRICS_ROOT = RESULTS_ROOT / "metrics"
CROP = 192
DENSITY_SCALE = 100.0        # keeps the target away from float noise; divided out again
SIGMA_FRACTION = 0.30        # gaussian width as a fraction of the nearest-neighbour gap
SIGMA_LIMITS = (1.0, 12.0)


# --------------------------------------------------------------------------- data

def density_map(shape, points, sigma_fraction=SIGMA_FRACTION):
    """Sum of gaussians at `points`, one per grain, integrating to the grain count.

    The width follows the scene: half the typical distance to the nearest other grain, so
    the same network sees a comparable blob whether a grain is 4 or 25 pixels wide.
    """
    canvas = np.zeros(shape[:2], dtype=np.float32)
    if len(points) == 0:
        return canvas

    if len(points) > 1:
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        gap = float(np.median(d.min(axis=1)))
    else:
        gap = 20.0
    sigma = float(np.clip(sigma_fraction * gap, *SIGMA_LIMITS))

    for row, col in points:
        r, c = int(round(row)), int(round(col))
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            canvas[r, c] += 1.0
    blurred = cv2.GaussianBlur(canvas, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    # Blurring near a border loses mass; rescale so the integral is still the count.
    total = blurred.sum()
    if total > 0:
        blurred *= canvas.sum() / total
    return blurred


def build_records(datasets=("d1", "d2", "d3"), supervision="sam3"):
    """[(image_path, points, dataset, file_name)] for every training image."""
    out = []
    if "d2" in datasets:
        labels = pseudo.load("d2_exact")
        for sample in synth.load_d2():
            out.append((sample["path"], labels[sample["file_name"]], "d2", sample["file_name"]))
    for name, loader in (("d1", io_utils.load_d1), ("d3", io_utils.load_d3)):
        if name not in datasets:
            continue
        labels = pseudo.load(f"{name}_{supervision}")
        for sample in loader():
            out.append((sample["path"], labels[sample["file_name"]], name, sample["file_name"]))
    return out


def split_records(records, seed=0, fractions=(0.7, 0.1, 0.2)):
    """Split per dataset, so every split holds all three and none share a source image."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for name in sorted({r[2] for r in records}):
        block = [r for r in records if r[2] == name]
        order = rng.permutation(len(block))
        n_train = int(round(fractions[0] * len(block)))
        n_val = int(round(fractions[1] * len(block)))
        train += [block[i] for i in order[:n_train]]
        val += [block[i] for i in order[n_train:n_train + n_val]]
        test += [block[i] for i in order[n_train + n_val:]]
    return train, val, test


class Crops(torch.utils.data.Dataset):
    """Random crops with their density target; the count is the integral of the target."""

    def __init__(self, records, crop=CROP, length=2000, seed=0, balanced=True):
        self.records = records
        self.crop = crop
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        # Draw the dataset first, then an image from it. Sampling images uniformly would
        # spend 89% of training on D3, which holds 503 of the 567 training images, and the
        # two sets with the largest grains would hardly be seen.
        self.balanced = balanced
        self.by_dataset = {}
        for index, record in enumerate(records):
            self.by_dataset.setdefault(record[2], []).append(index)
        self.datasets = sorted(self.by_dataset)

    def __len__(self):
        return self.length

    def _load(self, index):
        if index not in self.cache:
            path, points, _, _ = self.records[index]
            image = io_utils.imread(path)
            self.cache[index] = (image, density_map(image.shape, points))
        return self.cache[index]

    def _pick(self):
        if not self.balanced:
            return int(self.rng.integers(len(self.records)))
        pool = self.by_dataset[self.datasets[int(self.rng.integers(len(self.datasets)))]]
        return int(pool[self.rng.integers(len(pool))])

    def __getitem__(self, _):
        index = self._pick()
        image, target = self._load(index)
        h, w = image.shape[:2]
        crop = min(self.crop, h, w)
        top = int(self.rng.integers(0, h - crop + 1))
        left = int(self.rng.integers(0, w - crop + 1))
        patch = image[top:top + crop, left:left + crop]
        label = target[top:top + crop, left:left + crop]

        if self.rng.random() < 0.5:
            patch, label = patch[:, ::-1], label[:, ::-1]
        if self.rng.random() < 0.5:
            patch, label = patch[::-1], label[::-1]

        patch = np.ascontiguousarray(patch[:, :, ::-1]).astype(np.float32) / 255.0
        label = np.ascontiguousarray(label).astype(np.float32) * DENSITY_SCALE
        return torch.from_numpy(patch).permute(2, 0, 1), torch.from_numpy(label)[None]


# -------------------------------------------------------------------------- model

def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Three-level U-Net, ~0.5 M parameters, predicting one density channel."""

    def __init__(self, width=24):
        super().__init__()
        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.enc3 = block(width * 2, width * 4)
        self.dec2 = block(width * 4 + width * 2, width * 2)
        self.dec1 = block(width * 2 + width, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        return F.softplus(self.head(d1))      # a density is never negative


# ----------------------------------------------------------------------- training

def predict_count(model, image, device, tile=512):
    """Count one whole image: the integral of the predicted density."""
    model.eval()
    with torch.no_grad():
        x = np.ascontiguousarray(image[:, :, ::-1]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
        pad_h = (-x.shape[-2]) % 4
        pad_w = (-x.shape[-1]) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        total = 0.0
        for top in range(0, x.shape[-2], tile):
            for left in range(0, x.shape[-1], tile):
                patch = x[:, :, top:top + tile, left:left + tile]
                if min(patch.shape[-2:]) < 8:
                    continue
                total += float(model(patch).sum())
    return total / DENSITY_SCALE


def evaluate(model, records, device):
    errors, rows = [], []
    for path, points, dataset, file_name in records:
        image = io_utils.imread(path)
        pred = predict_count(model, image, device)
        truth = float(len(points))
        errors.append(pred - truth)
        rows.append({"dataset": dataset, "file_name": file_name,
                     "truth": truth, "pred": pred})
    errors = np.asarray(errors)
    return float(np.abs(errors).mean()), rows


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = build_records(supervision=args.supervision)
    train_set, val_set, test_set = split_records(records, seed=args.seed)
    print(f"训练 {len(train_set)} 张 / 验证 {len(val_set)} 张 / 测试 {len(test_set)} 张")

    loader = torch.utils.data.DataLoader(
        Crops(train_set, length=args.iters * args.batch, seed=args.seed),
        batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = UNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"学生模型参数量 {n_params/1e6:.2f} M")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.iters)

    model.train()
    start = time.perf_counter()
    for step, (images, targets) in enumerate(loader, 1):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        predicted = model(images)
        # Pixel loss keeps the blobs in the right places; the count term is what is scored.
        loss = F.mse_loss(predicted, targets) + args.count_weight * F.l1_loss(
            predicted.sum(dim=(1, 2, 3)), targets.sum(dim=(1, 2, 3))) / DENSITY_SCALE
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % max(args.iters // 10, 1) == 0:
            print(f"  step {step}/{args.iters}  loss {float(loss.detach()):.4f}", flush=True)
        model.train()

    print(f"训练用时 {time.perf_counter()-start:.0f}s")
    ensure_dir(MODEL_ROOT)
    path = MODEL_ROOT / f"student_{args.supervision}.pt"
    torch.save({"state_dict": model.state_dict(), "width": args.width,
                "supervision": args.supervision, "seed": args.seed}, path)
    print(f"wrote {path}  ({path.stat().st_size/2**20:.1f} MB)")

    val_mae, _ = evaluate(model, val_set, device)
    print(f"验证集（伪标签）MAE {val_mae:.2f}")
    return model, test_set, device


def compare(args):
    """Student vs the classical pipeline vs the teacher, on the held-out images.

    Truth here is the human annotation for the photographed sets and the construction-time
    count for the synthetic one - never the teacher, which is itself under test.
    """
    import pandas as pd
    from src import baselines, counter, preprocess

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_ROOT / f"student_{args.supervision}.pt", map_location=device)
    model = UNet(width=checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # Same split as training (same seed, same record order), but labelled by people.
    truth_records = build_records(supervision="human")
    _, _, test_set = split_records(truth_records, seed=checkpoint["seed"])
    print(f"测试集 {len(test_set)} 张（训练时未见过），真值为人工标注 / 合成真值")

    teacher = None
    if not args.no_teacher:
        from src.teacher_sam import Sam3Teacher
        teacher = Sam3Teacher()

    rows = []
    for path, points, dataset, file_name in test_set:
        image = io_utils.imread(path)
        # B1 shares this binarisation with the proposed method, so the two differ only in
        # how they handle touching grains - which is what the comparison is about.
        pre = preprocess.preprocess(image)
        row = {"dataset": dataset, "file_name": file_name, "truth": float(len(points)),
               "student": predict_count(model, image, device),
               "ours": float(counter.count_rice(None, pre=pre)),
               "b1": float(baselines.b1_connected_components(pre["calib"]))}
        if teacher is not None:
            result = teacher.segment(image, prompt=pseudo.PROMPT, threshold=pseudo.THRESHOLD)
            scores = np.asarray(result["scores"].float().cpu())
            row["sam3"] = float((scores >= pseudo.THRESHOLD).sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    ensure_dir(METRICS_ROOT)
    table.to_csv(METRICS_ROOT / "student_test.csv", index=False)

    methods = [m for m in ("student", "ours", "b1", "sam3") if m in table.columns]
    print(f"\n{'数据集':<8}{'张数':>5}" + "".join(f"{m:>12}" for m in methods))
    for dataset in ["d1", "d2", "d3"]:
        block = table[table.dataset == dataset]
        if block.empty:
            continue
        cells = "".join(f"{np.abs(block[m] - block.truth).mean():>12.2f}" for m in methods)
        print(f"{dataset.upper():<8}{len(block):>5}{cells}")
    cells = "".join(f"{np.abs(table[m] - table.truth).mean():>12.2f}" for m in methods)
    print(f"{'合计':<8}{len(table):>5}{cells}")
    print(f"\nwrote {METRICS_ROOT / 'student_test.csv'}")
    return table


def main():
    parser = argparse.ArgumentParser(description="Train the SAM 3 student counter")
    parser.add_argument("--eval", action="store_true", help="evaluate a trained student")
    parser.add_argument("--no-teacher", action="store_true",
                        help="skip SAM 3 in the comparison")
    parser.add_argument("--supervision", default="sam3", choices=["sam3", "human"])
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--count-weight", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from src import batch
    batch.pin_threads()
    if args.eval:
        compare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

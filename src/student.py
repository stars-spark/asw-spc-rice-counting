"""从 SAM 3 蒸馏出的小型学生网络，预测米粒密度图。

用密度图而不是实例计数是有意的。三组数据中有两组米粒只有约 4 像素宽，
让网络给出干净的实例边界太难，但放一个斑点是够的。粒数就是预测图的积分，
损失不依赖把一粒米和邻居分开，而这恰恰是几何方法最费力的地方。

监督来源是混合的，见 src/pseudo.py。合成图用生成时的精确中心，
两组照片用 SAM 3 掩膜的质心，不用人工标注时照片只有这一种监督。
人工标注从不参与训练，只作测试集，学生的成绩就是它从教师那里学到了多少，由人工标注来衡量。
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
DENSITY_SCALE = 100.0  # 让目标值远离浮点噪声，最后再除回去
SIGMA_FRACTION = 0.30  # 高斯宽度占最近邻间距的比例
SIGMA_LIMITS = (1.0, 12.0)


# ---------------------------------------------------------------------------- 数据

def density_map(shape, points, sigma_fraction=SIGMA_FRACTION):
    """在 points 处各放一个高斯，一粒米一个，积分等于粒数。

    高斯宽度随场景取最近邻间距典型值的一半，米粒 4 像素宽还是 25 像素宽，网络看到的斑点都差不多。
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
    # 靠近边界的模糊会损失质量，重新缩放使积分仍等于粒数。
    total = blurred.sum()
    if total > 0:
        blurred *= canvas.sum() / total
    return blurred


def build_records(datasets=("d1", "d2", "d3"), supervision="sam3"):
    """每张训练图的 [(image_path, points, dataset, file_name)]。"""
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
    """按数据集分别划分，每个划分都含三组数据，且没有来自同一张原图的样本。"""
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
    """随机裁块及其密度目标，粒数是目标的积分。"""

    def __init__(self, records, crop=CROP, length=2000, seed=0, balanced=True):
        self.records = records
        self.crop = crop
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        # 先抽数据集再抽图。直接按图均匀抽样的话，567 张训练图里 D3 占 503 张，
        # 89% 的训练花在 D3 上，米粒最大的两组几乎见不到。
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


# ---------------------------------------------------------------------------- 模型

def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """三层 U-Net，默认宽度下约 27 万参数，输出一个密度通道。"""

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
        return F.softplus(self.head(d1))  # 密度不会是负的


# ---------------------------------------------------------------------------- 训练

def predict_count(model, image, device, tile=512):
    """对整张图计数，即预测密度的积分。"""
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
        # 像素损失让斑点落在正确位置，计数项才是最终考核的。
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
    """在留出图上比较学生、几何方法和教师。

    照片的真值是人工标注，合成图是生成时的粒数，从不用教师，教师本身也在被检验。
    """
    import pandas as pd
    from src import baselines, counter, preprocess

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_ROOT / f"student_{args.supervision}.pt", map_location=device)
    model = UNet(width=checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # 与训练时同样的划分，同一种子、同样的顺序，但用人工标注。
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
        # B1 与方法一用同一个二值图，两者只差在怎么处理粘连，这正是要比较的。
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

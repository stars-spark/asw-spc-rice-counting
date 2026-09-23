"""SAM 3，作为深度学习的参照。

用文本概念提示模型，它对每个检测到的物体返回一个实例掩膜，粒数就是得分过门限的实例数。
它有两个用途，一是和几何方法对比，二是给没有人工标注的图生成伪标签。
第二种用法要先在人工标注上量出教师自己的误差才站得住，validate_on_d1 做的就是这个。
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import io_utils
from src.io_utils import RESULTS_ROOT, ensure_dir

MODEL_ID = "facebook/sam3"
PROMPT = "rice grain"
THRESHOLD = 0.3
METRICS_ROOT = RESULTS_ROOT / "metrics"


class Sam3Teacher:
    def __init__(self, model_id=MODEL_ID, device=None, dtype=torch.bfloat16, local_files_only=True):
        from transformers import Sam3Model, Sam3Processor

        # 直接从本地缓存加载。权重已经在本地，走 Hub 会经过本机的 SOCKS 代理，
        # transformers 底层的 HTTP 客户端解析不了。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam3Processor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = (
            Sam3Model.from_pretrained(model_id, dtype=dtype, local_files_only=local_files_only)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def segment(self, image_bgr, prompt=PROMPT, threshold=THRESHOLD):
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
        )[0]
        return results

    def count(self, image_bgr, prompt=PROMPT, threshold=THRESHOLD):
        return int(len(self.segment(image_bgr, prompt=prompt, threshold=threshold)["scores"]))


def validate_on_d1(teacher, thresholds=(0.2, 0.3, 0.4, 0.5), prompt=PROMPT, limit=None):
    """先在 D1 的人工标注上衡量教师，再在别处使用。"""
    samples = io_utils.load_d1()[:limit]

    scores_per_image = []
    for sample in samples:
        image = io_utils.imread(sample["path"])
        result = teacher.segment(image, prompt=prompt, threshold=min(thresholds))
        scores_per_image.append((sample["gt_count"], np.asarray(result["scores"].float().cpu())))

    rows = []
    for threshold in thresholds:
        preds = [int((s >= threshold).sum()) for _, s in scores_per_image]
        gts = [g for g, _ in scores_per_image]
        err = np.abs(np.array(preds, float) - np.array(gts, float))
        rows.append(
            {
                "threshold": threshold,
                "n": len(gts),
                "MAE": float(err.mean()),
                "MAPE_%": float(np.mean(err / np.array(gts, float)) * 100),
                "bias": float(np.mean(np.array(preds, float) - np.array(gts, float))),
                "worst": float(err.max()),
            }
        )
    return pd.DataFrame(rows)


def score_dataset(teacher, samples, prompt, floor=0.05):
    """每张图只跑一次教师，保存原始实例得分。

    之后任何高于 floor 的门限都能直接算出粒数，不用重跑模型。
    """
    out = []
    for sample in samples:
        image = io_utils.imread(sample["path"])
        result = teacher.segment(image, prompt=prompt, threshold=floor)
        out.append(
            {
                "file_name": sample["file_name"],
                "gt": sample["gt_count"],
                "touch_prob": sample.get("touch_prob"),
                "scores": np.asarray(result["scores"].float().cpu()),
            }
        )
    return out


def sweep(scored, thresholds):
    rows = []
    gts = np.array([r["gt"] for r in scored], dtype=np.float64)
    for threshold in thresholds:
        preds = np.array([(r["scores"] >= threshold).sum() for r in scored], dtype=np.float64)
        err = np.abs(preds - gts)
        rows.append(
            {
                "threshold": threshold,
                "MAE": float(err.mean()),
                "MAPE_%": float(np.mean(err / gts) * 100),
                "bias": float(np.mean(preds - gts)),
                "worst": float(err.max()),
            }
        )
    return pd.DataFrame(rows)


def prompt_matrix(teacher, loaders, prompts, thresholds, limits=None):
    """每个提示词在每个数据集上能达到的最好 MAE。

    最好的提示词在数据集之间不通用，在照片上最好的短语在合成图上什么也检测不到。
    挑提示词需要标注，而零样本模型本应不依赖标注。
    """
    limits = limits or {}
    rows = []
    for prompt in prompts:
        row = {"prompt": prompt}
        for name, loader in loaders.items():
            samples = loader()[: limits.get(name)]
            scored = score_dataset(teacher, samples, prompt, floor=min(thresholds))
            table = sweep(scored, thresholds)
            best = table.loc[table.MAE.idxmin()]
            row[f"{name}_MAE"] = float(best.MAE)
            row[f"{name}_thr"] = float(best.threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    from src import synth

    parser = argparse.ArgumentParser(description="SAM 3 teacher model")
    parser.add_argument("--matrix", action="store_true",
                        help="report the prompt x dataset transfer matrix and exit")
    parser.add_argument("--d3-limit", type=int, default=200)
    parser.add_argument("--prompts", nargs="+", default=["rice grain", "grain", "white rice grain"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    parser.add_argument("--datasets", nargs="+", default=["d1", "d2", "d3"])
    args = parser.parse_args()

    teacher = Sam3Teacher()
    ensure_dir(METRICS_ROOT)
    loaders = {"d1": io_utils.load_d1, "d2": synth.load_d2, "d3": io_utils.load_d3}

    if args.matrix:
        table = prompt_matrix(
            teacher, loaders,
            prompts=args.prompts + ["white seed", "seed"],
            thresholds=[0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4],
            limits={"d3": args.d3_limit},
        )
        table.to_csv(METRICS_ROOT / "sam3_prompt_matrix.csv", index=False)
        print("\n=== SAM 3: best-case MAE per prompt, tuned separately on each dataset ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"\nwrote {METRICS_ROOT / 'sam3_prompt_matrix.csv'}")
        return

    # 只用 D1 的人工标注选提示词和门限。
    calibration = []
    for prompt in args.prompts:
        table = sweep(score_dataset(teacher, io_utils.load_d1(), prompt), args.thresholds)
        table.insert(0, "prompt", prompt)
        calibration.append(table)
    calibration = pd.concat(calibration, ignore_index=True)
    calibration.to_csv(METRICS_ROOT / "sam3_calibration_d1.csv", index=False)

    print("\n=== SAM 3 prompt/threshold calibration on D1 (human COCO labels) ===")
    print(calibration.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    best = calibration.loc[calibration.MAE.idxmin()]
    print(f"\nbest on D1: prompt={best.prompt!r} threshold={best.threshold} MAE={best.MAE:.2f}")
    print("note: these settings are fitted on D1's own labels, so the D1 row is optimistic")

    rows, per_image = [], []
    for name in args.datasets:
        scored = score_dataset(teacher, loaders[name](), best.prompt)
        gts = np.array([r["gt"] for r in scored], dtype=np.float64)
        preds = np.array([(r["scores"] >= best.threshold).sum() for r in scored], dtype=np.float64)
        err = np.abs(preds - gts)
        rows.append(
            {
                "dataset": name,
                "method": "SAM3_teacher",
                "prompt": best.prompt,
                "threshold": best.threshold,
                "n": len(gts),
                "MAE": float(err.mean()),
                "MAPE_%": float(np.mean(err / gts) * 100),
                "accuracy_%": float((1 - np.mean(err / gts)) * 100),
                "hit_pm2_%": float(np.mean(err <= 2) * 100),
                "worst": float(err.max()),
                "bias": float(np.mean(preds - gts)),
            }
        )
        for record, pred in zip(scored, preds):
            per_image.append(
                {
                    "dataset": name,
                    "method": "SAM3_teacher",
                    "file_name": record["file_name"],
                    "touch_prob": record["touch_prob"],
                    "gt": record["gt"],
                    "pred": int(pred),
                    "error": int(pred - record["gt"]),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS_ROOT / "sam3_summary.csv", index=False)
    pd.DataFrame(per_image).to_csv(METRICS_ROOT / "sam3_per_image.csv", index=False)

    print("\n=== SAM 3 with the D1-calibrated setting, applied to every dataset ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nwrote {METRICS_ROOT / 'sam3_summary.csv'}")


if __name__ == "__main__":
    main()

"""训练学生模型用的点标注，给出米粒在哪里，而不只是有几粒。

三种来源分开存放，报告里能说清每个结果用的是哪种监督。
exact，合成图，米粒中心在生成时已知。场景用同一随机种子重新生成而不存盘，粒数与清单一致，核对过。
sam3，教师模型的实例掩膜取质心。不用人工标注的前提下，照片只有这一种监督。
human，人工标注框的中心。只用于评测和衡量教师标签的误差，不用来训练学生。
compare_to_human 会报告教师的计数误差和点位与人工标注的吻合程度，
学生的结果可以对照教师标签的质量来看。
"""
import argparse

import numpy as np

from src import io_utils, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

LABEL_ROOT = RESULTS_ROOT / "labels"
PROMPT, THRESHOLD = "white seed", 0.40


def exact_points_d2():
    """重新生成合成场景，取回每粒米的中心。"""
    bank = synth.load_grain_bank(limit=300)
    rng = np.random.default_rng(2026)
    out = {}
    for level in synth.TOUCH_LEVELS:
        for idx in range(8):
            n_grains = int(rng.integers(40, 101))
            _, _, centers = synth.build_scene(bank, n_grains=n_grains, touch_prob=level,
                                              rng=rng)
            name = f"touch{int(level * 100):02d}_{idx:02d}.png"
            out[name] = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    return out


def sam3_points(samples, teacher=None, prompt=PROMPT, threshold=THRESHOLD):
    """教师返回的得分高于 threshold 的每个实例掩膜的质心。"""
    if teacher is None:
        from src.teacher_sam import Sam3Teacher
        teacher = Sam3Teacher()

    out = {}
    for sample in samples:
        image = io_utils.imread(sample["path"])
        result = teacher.segment(image, prompt=prompt, threshold=threshold)
        scores = np.asarray(result["scores"].float().cpu())
        masks = np.asarray(result["masks"].float().cpu())[scores >= threshold] > 0.5

        points = []
        for mask in masks:
            rows, cols = np.nonzero(mask)
            if rows.size:
                points.append((rows.mean(), cols.mean()))
        out[sample["file_name"]] = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return out


def human_points(samples):
    return {s["file_name"]: np.asarray(s["points"], dtype=np.float32).reshape(-1, 2)
            for s in samples}


def save(name, points):
    ensure_dir(LABEL_ROOT)
    path = LABEL_ROOT / f"{name}.npz"
    np.savez_compressed(path, **points)
    return path


def load(name):
    with np.load(LABEL_ROOT / f"{name}.npz") as data:
        return {k: data[k] for k in data.files}


def compare_to_human(teacher, human, tolerance=None):
    """教师标签有多好，看计数误差和点位是否对得上。

    教师的一个点是某个标注点的最近点、且距离在 tolerance 像素内，就算配上，按贪心配对。
    这样查准率和查全率反映的是位置，不只是个数。
    """
    rows = []
    for name, truth in human.items():
        pred = teacher.get(name, np.empty((0, 2), np.float32))
        limit = tolerance
        if limit is None:  # 随场景缩放，取典型最近邻间距的一半
            if len(truth) > 1:
                d = np.linalg.norm(truth[:, None, :] - truth[None, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                limit = float(np.median(d.min(axis=1)) / 2.0)
            else:
                limit = 10.0

        matched = 0
        if len(pred) and len(truth):
            distance = np.linalg.norm(truth[:, None, :] - pred[None, :, :], axis=2)
            used = set()
            for i in np.argsort(distance.min(axis=1)):
                j = int(np.argmin(np.where([c in used for c in range(distance.shape[1])],
                                           np.inf, distance[i])))
                if j not in used and distance[i, j] <= limit:
                    used.add(j)
                    matched += 1
        rows.append({
            "file_name": name,
            "n_human": len(truth),
            "n_teacher": len(pred),
            "count_error": len(pred) - len(truth),
            "matched": matched,
            "precision": matched / len(pred) if len(pred) else 0.0,
            "recall": matched / len(truth) if len(truth) else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build point labels for the student")
    parser.add_argument("--datasets", nargs="+", default=["d1", "d2", "d3"])
    args = parser.parse_args()

    from src import batch
    batch.pin_threads()

    if "d2" in args.datasets:
        points = exact_points_d2()
        print(f"d2 exact: {len(points)} scenes, {sum(len(v) for v in points.values())} points")
        print(f"  wrote {save('d2_exact', points)}")

    teacher = None
    for name, loader in (("d1", io_utils.load_d1), ("d3", io_utils.load_d3)):
        if name not in args.datasets:
            continue
        samples = loader(with_points=True)
        human = human_points(samples)
        print(f"  wrote {save(f'{name}_human', human)}")

        if teacher is None:
            from src.teacher_sam import Sam3Teacher
            teacher = Sam3Teacher()
        pseudo = sam3_points(samples, teacher=teacher)
        print(f"  wrote {save(f'{name}_sam3', pseudo)}")

        rows = compare_to_human(pseudo, human)
        errors = np.array([r["count_error"] for r in rows], dtype=np.float64)
        print(f"{name} teacher labels vs human: MAE {np.abs(errors).mean():.2f}, "
              f"bias {errors.mean():+.2f}, "
              f"precision {np.mean([r['precision'] for r in rows]):.3f}, "
              f"recall {np.mean([r['recall'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

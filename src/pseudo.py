"""Point labels for training the student: where the grains are, not just how many.

Three sources, deliberately kept apart so the report can say which supervision a result
came from:

* ``exact``   - the synthetic set, whose grain centres are known by construction. The
                scenes are regenerated from the same seed rather than stored, which
                reproduces the manifest counts exactly (checked).
* ``sam3``    - the teacher's instance masks, reduced to their centroids. This is the only
                supervision available for the photographs without using their labels.
* ``human``   - the centre of each annotated box. Used to *evaluate*, and to measure how
                wrong the teacher's labels are, never to train the student.

Teacher labels are not assumed to be good: `compare_to_human` reports the teacher's count
error and how well its points line up with the annotated ones, so the student's result can
be read against the quality of what it was taught from.
"""
import argparse

import numpy as np

from src import io_utils, synth
from src.io_utils import RESULTS_ROOT, ensure_dir

LABEL_ROOT = RESULTS_ROOT / "labels"
PROMPT, THRESHOLD = "white seed", 0.40


def exact_points_d2():
    """Regenerate the synthetic scenes to recover the centre of every placed grain."""
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
    """Centroid of every instance mask the teacher returns above `threshold`."""
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
    """How good the teacher's labels are: count error, and whether the points match.

    A teacher point counts as matched when it is the nearest one to an annotated point and
    within `tolerance` pixels of it, greedily, so precision and recall describe placement
    rather than count alone.
    """
    rows = []
    for name, truth in human.items():
        pred = teacher.get(name, np.empty((0, 2), np.float32))
        limit = tolerance
        if limit is None:  # scale with the scene: half the typical nearest-neighbour gap
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

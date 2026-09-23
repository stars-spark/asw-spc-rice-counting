"""多进程批量计数，每个进程处理一张图。

各图计数互不相关，可以按核并行。每个进程先限制为单线程，这一步比并行本身更重要。
在这台机器上实测，单张图会占满 17 个核反而更慢，因为 OpenBLAS 用 20 线程的线程池
处理标定里大量很小的线性代数调用。限成单线程后单张图快 1.6 到 1.8 倍，之后再按核并行才有意义。

可以当库用 count_paths，也可以在命令行运行：

    python -m src.batch photos/ --out counts.csv
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

from src import counter, io_utils

SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


_LIMITS = None


def pin_threads():
    """把本进程里各数值库限制为单线程。

    threadpool_limits 返回的对象被回收时会恢复原来的限制，所以要存在模块全局变量里。
    """
    global _LIMITS
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # 可选依赖，没有时 BLAS 用默认线程池
        return False
    _LIMITS = threadpool_limits(limits=1)
    return True


def _init_worker():
    pin_threads()


def _count_one(path):
    try:
        return str(path), counter.count_rice(io_utils.imread(path)), ""
    except Exception as exc:  # 一张图读不了或处理不了，不能让整批停下
        return str(path), None, f"{type(exc).__name__}: {exc}"


def count_paths(paths, workers=None, chunksize=1, progress=False):
    """逐张计数，按输入顺序返回 [(path, count, error), ...]。"""
    paths = [str(p) for p in paths]
    workers = workers or min(len(paths), os.cpu_count() or 1)
    if workers <= 1:
        pin_threads()
        return [_count_one(p) for p in paths]

    out = []
    with Pool(workers, initializer=_init_worker) as pool:
        for index, row in enumerate(pool.imap(_count_one, paths, chunksize=chunksize), 1):
            out.append(row)
            if progress and index % 25 == 0:
                print(f"  {index}/{len(paths)}", file=sys.stderr, flush=True)
    return out


def find_images(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)


def main():
    parser = argparse.ArgumentParser(description="Count rice grains in a folder of images")
    parser.add_argument("path", help="image file, or a folder to search recursively")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: one per core)")
    parser.add_argument("--out", default=None, help="write results to this CSV")
    args = parser.parse_args()

    paths = find_images(args.path)
    if not paths:
        raise SystemExit(f"no images under {args.path}")

    workers = args.workers or min(len(paths), os.cpu_count() or 1)
    start = time.perf_counter()
    rows = count_paths(paths, workers=workers, progress=True)
    elapsed = time.perf_counter() - start

    failed = [r for r in rows if r[1] is None]
    total = sum(r[1] for r in rows if r[1] is not None)
    print(f"{len(rows)} images on {workers} workers in {elapsed:.1f}s "
          f"({elapsed / len(rows):.3f}s per image), {total} grains, {len(failed)} failed")
    for path, _, error in failed[:5]:
        print(f"  failed: {os.path.basename(path)}  {error}")

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "count", "error"])
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Count a batch of images in parallel, one image per worker process.

Counting one image never depends on another, so a batch scales across cores. Each worker is
pinned to a single thread first, which matters more than the parallelism itself: measured on
this machine, one image alone occupied 17 cores and ran *slower* for it, because OpenBLAS
answers the many tiny linear-algebra calls inside the scale calibration with a 20-thread
pool. Pinning the threads made a single image 1.6x to 1.8x faster on its own, and only then
does running one image per core add anything.

Used as a library (`count_paths`) and as a command line tool:

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
    """Hold every numeric library in this process to one thread.

    Kept in a module global: `threadpool_limits` restores the previous limits when the
    object it returns is collected, so letting it go out of scope would undo this.
    """
    global _LIMITS
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # optional; without it BLAS keeps its default thread pool
        return False
    _LIMITS = threadpool_limits(limits=1)
    return True


def _init_worker():
    pin_threads()


def _count_one(path):
    try:
        return str(path), counter.count_rice(io_utils.imread(path)), ""
    except Exception as exc:  # an unreadable or unusable image must not stop the batch
        return str(path), None, f"{type(exc).__name__}: {exc}"


def count_paths(paths, workers=None, chunksize=1, progress=False):
    """Count every image, returning [(path, count, error), ...] in the order given."""
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

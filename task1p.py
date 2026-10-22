"""Task 1: Row-Major vs Column-Major Parallel Cache Traversal.

"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = 8000
STUDENT_ID = 230103112  


def physical_cores() -> int:
    """Physical core count (psutil -> lscpu -> logical fallback)."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    try:
        import subprocess
        out = subprocess.run(["lscpu", "-p=SOCKET,CORE"], capture_output=True,
                             text=True, check=True).stdout
        pairs = {l for l in out.splitlines() if l and not l.startswith("#")}
        if pairs:
            return len(pairs)
    except Exception:
        pass
    return os.cpu_count() or 1


def split(n: int, parts: int):
    """Contiguous, near-equal [start, end) slices."""
    return [(n * i // parts, n * (i + 1) // parts) for i in range(parts)]


def row_worker(a, r0, r1):
    """Mode A: own a block of rows; each row is a contiguous (stride-1) sum."""
    total = 0.0
    for i in range(r0, r1):
        total += a[i, :].sum()
    return total


def col_worker(a, c0, c1):
    """Mode B: own a block of columns; each column sum jumps N*8 bytes per element."""
    total = 0.0
    for j in range(c0, c1):
        total += a[:, j].sum()
    return total


def run(worker, a, slices):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(slices)) as ex:
        total = sum(ex.map(lambda s: worker(a, *s), slices))
    return (time.perf_counter() - t0) * 1000.0, total


def main():
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else STUDENT_ID
    logical = os.cpu_count() or 1
    phys = physical_cores()

    a = np.random.default_rng(sid).random((N, N), dtype=np.float64)  # C-order (row-major)
    slices = split(N, phys)

    run(row_worker, a, slices)  # warm-up (not timed)

    ms_a, sum_a = run(row_worker, a, slices)
    ms_b, sum_b = run(col_worker, a, slices)

    print("=== Task 1: Spatial Cache Profiling ===")
    print(f"Student ID        : {sid}")
    print(f"Logical cores     : {logical}")
    print(f"Physical cores    : {phys} (threads used)")
    print(f"Matrix            : {N} x {N} float64 ({N * N * 8 / 1e6:.0f} MB)")
    print(f"Mode A (Row-Major)   : {ms_a:10.2f} ms  (sum = {sum_a:.6f})")
    print(f"Mode B (Column-Major): {ms_b:10.2f} ms  (sum = {sum_b:.6f})")
    print(f"Cache penalty factor (B/A): {ms_b / ms_a:.2f}x")


if __name__ == "__main__":
    main()
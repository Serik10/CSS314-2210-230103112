"""Task 3: Prefix Scan - serial vs. 2-pass chunked parallel scan.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = 2_000_000
STUDENT_ID = 230103112  


def physical_cores():
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


def serial_scan(data):
    """output[i] = output[i-1] + input[i]  (plain sequential loop)."""
    out = [0] * len(data)
    running = 0
    for i, v in enumerate(data):
        running += v
        out[i] = running
    return out


def parallel_scan(data, workers):
    """Chunked 2-pass scan:
       Pass 1 (parallel): each worker scans its own chunk locally.
       Reduction (serial, tiny): exclusive scan of the chunk totals -> offsets.
       Pass 2 (parallel): each worker adds its chunk's offset.
    """
    n = len(data)
    bounds = [(n * i // workers, n * (i + 1) // workers) for i in range(workers)]
    out = np.empty(n, dtype=np.int64)

    def local_scan(b):
        lo, hi = b
        np.cumsum(data[lo:hi], out=out[lo:hi])

    def apply_offset(args):
        (lo, hi), off = args
        if off:
            out[lo:hi] += off

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(local_scan, bounds))                                   # pass 1
        totals = [int(out[hi - 1]) for lo, hi in bounds]                   # chunk sums
        offsets = [0] + list(np.cumsum(totals[:-1]))                       # offset reduction
        list(ex.map(apply_offset, zip(bounds, offsets)))                   # pass 2
    return out


def main():
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else STUDENT_ID
    workers = physical_cores()

    arr = np.random.default_rng(sid).integers(1, 11, size=N, dtype=np.int64)  # values 1..10
    as_list = arr.tolist()

    parallel_scan(arr, workers)  # warm-up (not timed)

    t0 = time.perf_counter()
    serial_out = serial_scan(as_list)
    serial_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    par_out = parallel_scan(arr, workers)
    par_ms = (time.perf_counter() - t0) * 1000.0

    # Extra reference: single-thread vectorised scan, to show the fair baseline.
    t0 = time.perf_counter()
    np.cumsum(arr)
    np_ms = (time.perf_counter() - t0) * 1000.0

    ok = int(par_out[-1]) == serial_out[-1] and np.array_equal(par_out, np.array(serial_out))
    status = "PASS" if ok else "FAIL"

    print("=== Task 3: Prefix Scan ===")
    print(f"Student ID       : {sid}")
    print(f"Elements         : {N:,} (values 1-10)   Workers: {workers}   Logical cores: {os.cpu_count()}")
    print(f"Serial Time      : {serial_ms:.2f} ms   (pure-Python loop)")
    print(f"Parallel Time    : {par_ms:.2f} ms   (2-pass chunked, {workers} thread(s))")
    print(f"Speedup          : {serial_ms / par_ms:.2f}x")
    print(f"Last element     : serial = {serial_out[-1]} | parallel = {int(par_out[-1])}")
    print(f"Verification     : {status}")
    print(f"(reference: single-thread np.cumsum = {np_ms:.2f} ms)")

    if not ok:
        print("ERROR: parallel_output[-1] != serial_output[-1]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
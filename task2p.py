"""Task 2: Pseudo-Random Generator State Bottleneck (Monte Carlo pi).

Run:  python task2.py [STUDENT_ID]
"""
import math
import os
import random
import sys
import threading
import time

TOTAL_POINTS = 40_000_000
THREADS = 4
STUDENT_ID = 230103112  


class SharedRng:
    """A single PRNG instance shared by every thread; each draw is mutex-guarded."""

    def __init__(self, seed):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def random(self):
        with self._lock:
            return self._rng.random()


def count_inside(next_random, n):
    inside = 0
    for _ in range(n):
        x = next_random()
        y = next_random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def run(make_rng):
    """make_rng(thread_index) -> zero-arg callable returning a float in [0, 1)."""
    per_thread = TOTAL_POINTS // THREADS
    hits = [0] * THREADS

    def worker(idx, fn):
        hits[idx] = count_inside(fn, per_thread)

    fns = [make_rng(i) for i in range(THREADS)]        # setup is outside the timed region
    threads = [threading.Thread(target=worker, args=(i, fns[i])) for i in range(THREADS)]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0

    pi = 4.0 * sum(hits) / (per_thread * THREADS)
    return pi, abs(pi - math.pi), ms


def main():
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else STUDENT_ID

    shared = SharedRng(sid)
    pi_a, err_a, ms_a = run(lambda i: shared.random)

    # Independent PRNG per thread: Seed = (Student_ID * 10) + thread_index
    pi_b, err_b, ms_b = run(lambda i: random.Random(sid * 10 + i).random)

    print("=== Task 2: PRNG State Isolation ===")
    print(f"Student ID : {sid}")
    print(f"Samples    : {TOTAL_POINTS:,}   Threads: {THREADS}   Logical cores: {os.cpu_count()}")
    print(f"Shared PRNG (mutex)          : pi = {pi_a:.8f} | abs error = {err_a:.8f} | time = {ms_a:.2f} ms")
    print(f"Independent PRNG per thread  : pi = {pi_b:.8f} | abs error = {err_b:.8f} | time = {ms_b:.2f} ms")
    print(f"Speedup from removing lock (Shared / Independent): {ms_a / ms_b:.2f}x")


if __name__ == "__main__":
    main()
"""
Multi-Scale QOBLIB Benchmark Suite Runner
=========================================
Runs RGSA across N=80, 90, 110, and 140 instances.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rgsa_solver import RGSASolver
from src.audit_n140 import parse_instance


def benchmark_suite():
    data_dir = os.path.join(os.path.dirname(__file__), "../data/01-marketsplit/instances")
    if not os.path.exists(data_dir):
        data_dir = "data/01-marketsplit/instances"

    files = [
        ("ms_09_100_000.dat", "N=80, J=9 (QOBLIB Benchmark)", 2**80),
        ("ms_10_200_000.dat", "N=90, J=10 (Scale 1)", 2**90),
        ("ms_12_200_000.dat", "N=110, J=12 (Scale 2)", 2**110),
        ("ms_15_200_000.dat", "N=140, J=15 (Flagship Challenge)", 2**140),
    ]

    print("=" * 95)
    print("QOBLIB MULTI-SCALE BENCHMARK SUITE")
    print("=" * 95)
    print(f"{'Instance File':<20} {'Scale / Description':<30} {'Residual':<12} {'Rel. Error':<12} {'Time':<8}")
    print("-" * 95)

    for fname, desc, _ in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        A, b, known_sol = parse_instance(fpath)
        cardinality = int(np.sum(known_sol)) if known_sol is not None else A.shape[1] // 2
        solver = RGSASolver(A, b, cardinality=cardinality, seed=42)
        res = solver.solve()
        print(f"{fname:<20} {desc:<30} {res['residual']:<12.4f} {res['relative_error_pct']:.2f}%       {res['total_time_sec']:.2f}s")

    print("=" * 95)


if __name__ == "__main__":
    benchmark_suite()

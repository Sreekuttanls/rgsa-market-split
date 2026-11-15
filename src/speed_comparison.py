"""Python versus C++ run time for identical RGSA settings (editor comment 4).

Both implementations run the original step-budget protocol (q = floor(N/2),
150,000 annealing steps, p_g = 0.65, T0 = 600, Tf = 1e-3) on the same
instances and seeds, one process at a time, so the timing difference reflects
the implementation language rather than the algorithm or the budget.

Usage:
    python -m src.speed_comparison --output results/revision/speed_comparison.json
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.audit_n140 import parse_instance
from src.run_paper_benchmarks import random_partition
from src.rgsa_solver import RGSASolver

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "cpp" / "rgsa"
INST = ROOT / "data" / "01-marketsplit" / "instances"
DEFAULT_INSTANCES = ("ms_09_050_000", "ms_12_050_000", "ms_15_050_000", "ms_15_200_000")
CONFIGS = {
    # name: (initialisation, p_guide, legacy polish)
    "rgsa": ("random", 0.65, False),
    "full_pipeline": ("lagrangian", 0.65, True),
}


def run_python(A, b, seed, init, p_guide, polish, steps):
    solver = RGSASolver(A, b, seed=seed)
    t0 = time.perf_counter()
    if init == "lagrangian":
        x, _, _ = solver.solve_stage1_lagrangian()
    else:
        x = random_partition(A.shape[1], solver.cardinality, seed)
    x, r, _ = solver.solve_stage2_guided_sa(x, n_steps=steps, p_guide=p_guide)
    if polish:
        x, r, _ = solver.solve_stage3_2opt_polish(x, max_2opt_passes=1)
    return time.perf_counter() - t0, float(r)


def run_cpp(path, seed, init, p_guide, polish, steps):
    cmd = [str(BIN), "--instance", str(path), "--seed", str(seed), "--init", init,
           "--p-guide", str(p_guide), "--steps", str(steps),
           "--polish", "legacy" if polish else "none", "--no-x"]
    t0 = time.perf_counter()
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    return wall, json.loads(out.stdout)["residual"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--instances", default=",".join(DEFAULT_INSTANCES))
    args = ap.parse_args(argv)
    subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True, capture_output=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for name in args.instances.split(","):
        path = INST / f"{name}.dat"
        A, b, _ = parse_instance(path)
        for cfg, (init, pg, polish) in CONFIGS.items():
            for seed in seeds:
                py_t, py_r = run_python(A, b, seed, init, pg, polish, args.steps)
                c_t, c_r = run_cpp(path, seed, init, pg, polish, args.steps)
                rows.append({"instance": name, "m": int(A.shape[0]), "n": int(A.shape[1]),
                             "config": cfg, "seed": seed,
                             "python_seconds": py_t, "cpp_seconds": c_t,
                             "python_residual": py_r, "cpp_residual": c_r})
                print(f"{name} {cfg} seed={seed}: python {py_t:.2f}s (R={py_r:.2f}) "
                      f"c++ {c_t:.3f}s (R={c_r:.2f}) speed-up {py_t / c_t:.0f}x",
                      flush=True)
    summary = {}
    for cfg in CONFIGS:
        sel = [r for r in rows if r["config"] == cfg]
        ratio = [r["python_seconds"] / r["cpp_seconds"] for r in sel]
        summary[cfg] = {"median_speedup": float(np.median(ratio)),
                        "min_speedup": float(np.min(ratio)),
                        "max_speedup": float(np.max(ratio))}
    meta = {"python": sys.version.split()[0], "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "processor": platform.processor(), "steps": args.steps,
            "compiler_flags": "-std=c++17 -O3 -march=native -ffp-contract=off"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": meta, "summary": summary, "runs": rows},
                                      indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

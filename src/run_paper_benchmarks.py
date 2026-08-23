"""Run Tasks 1 and 2 of the RGSA paper benchmark pipeline.

This runner compares random and Lagrangian starts with blind and guided
simulated annealing on QOBLIB Market Split instances.  Cardinality is always
inferred as half the number of variables; the embedded reference solution is
never used to configure a run.
"""

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# The script lives in src/, so adding its parent makes ``src`` importable even
# when this file is invoked directly from another working directory.
sys.path.insert(0, str(PACKAGE_ROOT))

from src.audit_n140 import parse_instance
from src.rgsa_solver import RGSASolver


DEFAULT_SEEDS = tuple(range(42, 72))
DEFAULT_N_STEPS = 150_000
T_START = 600.0
T_END = 0.001
CONFIG_NAMES = (
    "random_blind_sa",
    "random_guided_sa",
    "lagrangian_blind_sa",
    "full_esa",
)
RUN_FIELDS = (
    "instance",
    "m",
    "n",
    "D",
    "seed",
    "config_name",
    "stage1_residual",
    "stage2_residual",
    "final_residual",
    "relative_error_pct",
    "max_abs_violation",
    "feasible",
    "total_time_sec",
    "stage1_time",
    "stage2_time",
    "polish_time",
)
INSTANCE_RE = re.compile(
    r"^ms_(?P<m>0[9]|1[0-5])_(?P<D>\d+)_(?P<index>\d+)\.dat$"
)


def parse_seeds(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated list of integer seeds."""
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_configs(value: str) -> Tuple[str, ...]:
    """Parse and validate a comma-separated list of configuration names."""
    configs = tuple(item.strip() for item in value.split(",") if item.strip())
    if not configs:
        raise argparse.ArgumentTypeError("at least one configuration is required")
    unknown = [name for name in configs if name not in CONFIG_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown configuration(s): "
            + ", ".join(unknown)
            + "; choose from "
            + ", ".join(CONFIG_NAMES)
        )
    if len(set(configs)) != len(configs):
        raise argparse.ArgumentTypeError("configuration names must not be repeated")
    return configs


def discover_instances(instances_dir: Path, pattern: str) -> List[Path]:
    """Find matching top-level Market Split instances with m from 09 to 15."""
    if not instances_dir.is_dir():
        raise FileNotFoundError(f"instance directory does not exist: {instances_dir}")

    # Match against basenames so shell-style filters such as ms_09_050_*.dat
    # work consistently.  The regular expression enforces the paper's 09..15
    # scope independently of the user-supplied filter.
    paths = [
        path
        for path in instances_dir.iterdir()
        if path.is_file()
        and INSTANCE_RE.fullmatch(path.name)
        and fnmatch.fnmatch(path.name, pattern)
    ]
    return sorted(paths, key=lambda path: path.name)


def random_partition(n: int, cardinality: int, seed: int) -> np.ndarray:
    """Create the required deterministic random cardinality-k partition."""
    solution = np.zeros(n)
    solution[np.random.RandomState(seed).permutation(n)[:cardinality]] = 1.0
    return solution


def run_configuration(
    A: np.ndarray,
    b: np.ndarray,
    instance_name: str,
    m: int,
    D: int,
    seed: int,
    config_name: str,
    n_steps: int,
) -> Dict[str, object]:
    """Execute one isolated benchmark configuration and calculate metrics."""
    n = int(A.shape[1])
    cardinality = n // 2
    solver = RGSASolver(A, b, cardinality=cardinality, seed=seed)

    if config_name.startswith("random_"):
        stage1_start = time.perf_counter()
        stage1_solution = random_partition(n, cardinality, seed)
        stage1_residual = float(np.linalg.norm(A @ stage1_solution - b))
        stage1_time = time.perf_counter() - stage1_start
    else:
        stage1_solution, stage1_residual, stage1_time = (
            solver.solve_stage1_lagrangian(strategy="prefix")
        )
        stage1_residual = float(stage1_residual)
        stage1_time = float(stage1_time)

    p_guide = 0.65 if config_name in {"random_guided_sa", "full_esa"} else 0.0
    # The solver uses NumPy's process-global RNG.  Reset immediately
    # before every SA call so run order and prior stages cannot affect it.
    np.random.seed(seed + 10_000)
    stage2_solution, stage2_residual, stage2_time = solver.solve_stage2_guided_sa(
        stage1_solution,
        n_steps=n_steps,
        p_guide=p_guide,
        T_start=T_START,
        T_end=T_END,
    )
    stage2_residual = float(stage2_residual)
    stage2_time = float(stage2_time)

    if config_name == "full_esa":
        final_solution, final_residual, polish_time = (
            solver.solve_stage3_2opt_polish(stage2_solution)
        )
        final_residual = float(final_residual)
        polish_time = float(polish_time)
    else:
        final_solution = stage2_solution
        final_residual = stage2_residual
        polish_time = 0.0

    violations = np.abs(A @ final_solution - b)
    target_norm = float(np.linalg.norm(b))
    relative_error_pct = (
        100.0 * final_residual / target_norm
        if target_norm != 0.0
        else (0.0 if final_residual == 0.0 else float("inf"))
    )

    return {
        "instance": instance_name,
        "m": int(m),
        "n": n,
        "D": int(D),
        "seed": int(seed),
        "config_name": config_name,
        "stage1_residual": stage1_residual,
        "stage2_residual": stage2_residual,
        "final_residual": final_residual,
        "relative_error_pct": float(relative_error_pct),
        "max_abs_violation": float(np.max(violations)) if violations.size else 0.0,
        "feasible": bool(np.all(violations == 0.0)),
        "total_time_sec": float(stage1_time + stage2_time + polish_time),
        "stage1_time": float(stage1_time),
        "stage2_time": stage2_time,
        "polish_time": polish_time,
    }


def write_outputs(
    output_dir: Path,
    runs: Sequence[Dict[str, object]],
    instance_paths: Sequence[Path],
    seeds: Sequence[int],
    configs: Sequence[str],
    n_steps: int,
    instance_filter: str,
) -> Tuple[Path, Path]:
    """Write the complete JSON payload and flat per-run CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paper_benchmark_results.json"
    csv_path = output_dir / "paper_benchmark_runs.csv"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_tasks": [1, 2],
        "instance_directory": str(
            (PACKAGE_ROOT / "data" / "01-marketsplit" / "instances").resolve()
        ),
        "instance_filter": instance_filter,
        "instances": [path.name for path in instance_paths],
        "seeds": [int(seed) for seed in seeds],
        "configs": list(configs),
        "parameters": {
            "cardinality": "A.shape[1] // 2",
            "n_steps": int(n_steps),
            "T_start": T_START,
            "T_end": T_END,
            "guided_p_guide": 0.65,
            "blind_p_guide": 0.0,
            "lagrangian_strategy": "prefix",
            "sa_seed_offset": 10_000,
        },
        "num_runs": len(runs),
        "runs": list(runs),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(runs)

    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        default="ms_*.dat",
        help=(
            "basename glob applied within data/01-marketsplit/instances/ "
            "(default: %(default)s); only ms_09 through ms_15 are eligible"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated integer seeds (default: 42 through 71)",
    )
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=CONFIG_NAMES,
        help="comma-separated configuration names (default: all four)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="SA steps per run (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "paper_benchmarks",
        help="result directory (default: package results/paper_benchmarks/)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_steps <= 0:
        parser.error("--n-steps must be a positive integer")

    instances_dir = PACKAGE_ROOT / "data" / "01-marketsplit" / "instances"
    try:
        instance_paths = discover_instances(instances_dir, args.instances)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not instance_paths:
        parser.error(
            f"no ms_09..ms_15 instance files match {args.instances!r} in "
            f"{instances_dir}"
        )

    total_runs = len(instance_paths) * len(args.seeds) * len(args.configs)
    print(
        f"Discovered {len(instance_paths)} instance(s); starting {total_runs} run(s) "
        f"with n_steps={args.n_steps}.",
        flush=True,
    )

    runs: List[Dict[str, object]] = []
    run_number = 0
    for instance_path in instance_paths:
        match = INSTANCE_RE.fullmatch(instance_path.name)
        if match is None:  # Protected by discover_instances; keeps type checkers happy.
            continue
        A, b, _known_sol = parse_instance(instance_path)
        filename_m = int(match.group("m"))
        m = filename_m if filename_m == int(A.shape[0]) else int(A.shape[0])
        D = int(match.group("D"))
        print(
            f"Instance {instance_path.name}: m={m}, n={A.shape[1]}, D={D}, "
            f"cardinality={A.shape[1] // 2}",
            flush=True,
        )

        for seed in args.seeds:
            for config_name in args.configs:
                run_number += 1
                print(
                    f"[{run_number}/{total_runs}] {instance_path.name} "
                    f"seed={seed} config={config_name}",
                    flush=True,
                )
                run = run_configuration(
                    A=A,
                    b=b,
                    instance_name=instance_path.name,
                    m=m,
                    D=D,
                    seed=seed,
                    config_name=config_name,
                    n_steps=args.n_steps,
                )
                runs.append(run)
                print(
                    f"  final_residual={run['final_residual']:.6f} "
                    f"feasible={run['feasible']} "
                    f"time={run['total_time_sec']:.3f}s",
                    flush=True,
                )

    json_path, csv_path = write_outputs(
        output_dir=args.output_dir,
        runs=runs,
        instance_paths=instance_paths,
        seeds=args.seeds,
        configs=args.configs,
        n_steps=args.n_steps,
        instance_filter=args.instances,
    )
    print(f"Wrote {len(runs)} runs to {json_path}", flush=True)
    print(f"Wrote CSV rows to {csv_path}", flush=True)


if __name__ == "__main__":
    main()

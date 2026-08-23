"""Cardinality-aware RGSA screening and selected full-budget evaluation."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.audit_n140 import parse_instance
from src.rgsa_solver import RGSASolver

CARDINALITIES = tuple(range(60, 81))
STRATEGIES = ("prefix", "alternating")
DEFAULT_SCREEN_STEPS = 25_000
DEFAULT_FULL_STEPS = 150_000
DEFAULT_SEEDS = (42, 43, 44)
BASELINE_CARDINALITY = 71


def parse_int_list(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def run_candidate(A, b, cardinality, strategy, seed, n_steps):
    solver = RGSASolver(A, b, cardinality=cardinality, seed=seed)
    s1, s1_res, s1_time = solver.solve_stage1_lagrangian(strategy=strategy)

    # The solver currently uses NumPy's process-global RNG. Reset it before
    # every candidate so candidate order cannot affect stochastic results.
    np.random.seed(seed + 10_000)
    s2, s2_res, s2_time = solver.solve_stage2_guided_sa(
        s1,
        n_steps=n_steps,
        p_guide=0.65,
        T_start=600.0,
        T_end=0.001,
    )
    final, final_res, polish_time = solver.solve_stage3_2opt_polish(s2)
    violation = np.abs(A @ final - b)
    return {
        "cardinality": int(cardinality),
        "strategy": strategy,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "stage1_residual": float(s1_res),
        "stage2_residual": float(s2_res),
        "residual": float(final_res),
        "relative_error_pct": float(100.0 * final_res / np.linalg.norm(b)),
        "max_abs_violation": float(np.max(violation)),
        "feasible": bool(np.array_equal(A @ final, b)),
        "stage1_time_sec": float(s1_time),
        "stage2_time_sec": float(s2_time),
        "polish_time_sec": float(polish_time),
        "total_time_sec": float(s1_time + s2_time + polish_time),
    }


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["strategy"], run["cardinality"])].append(run)

    summaries = []
    for (strategy, cardinality), values in sorted(grouped.items()):
        residuals = np.array([item["residual"] for item in values])
        summaries.append(
            {
                "strategy": strategy,
                "cardinality": int(cardinality),
                "num_runs": len(values),
                "mean_residual": float(np.mean(residuals)),
                "median_residual": float(np.median(residuals)),
                "min_residual": float(np.min(residuals)),
                "std_residual": float(np.std(residuals, ddof=0)),
                "mean_relative_error_pct": float(
                    np.mean([item["relative_error_pct"] for item in values])
                ),
                "max_abs_violation": float(
                    np.max([item["max_abs_violation"] for item in values])
                ),
                "feasible_runs": int(sum(item["feasible"] for item in values)),
                "mean_time_sec": float(
                    np.mean([item["total_time_sec"] for item in values])
                ),
            }
        )
    return summaries


def select_candidates(summaries, baseline_strategy="prefix", baseline_cardinality=71, top_n=5):
    baseline = next(
        item
        for item in summaries
        if item["strategy"] == baseline_strategy
        and item["cardinality"] == baseline_cardinality
    )
    ranked = sorted(
        summaries,
        key=lambda item: (item["mean_residual"], item["strategy"], item["cardinality"]),
    )
    selected = {(item["strategy"], item["cardinality"]) for item in ranked[:top_n]}
    selected.update(
        (item["strategy"], item["cardinality"])
        for item in summaries
        if item["mean_residual"] < baseline["mean_residual"]
    )
    return baseline, [
        item for item in ranked if (item["strategy"], item["cardinality"]) in selected
    ]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(instance_path, output_dir, cardinalities, strategies, screen_seeds, full_seeds,
              screen_steps, full_steps):
    A, b, _ = parse_instance(instance_path)
    os.makedirs(output_dir, exist_ok=True)

    screening_runs = []
    for strategy in strategies:
        for cardinality in cardinalities:
            for seed in screen_seeds:
                print(
                    f"screen strategy={strategy} k={cardinality} seed={seed}",
                    flush=True,
                )
                screening_runs.append(
                    run_candidate(A, b, cardinality, strategy, seed, screen_steps)
                )

    screening_summary = aggregate_runs(screening_runs)
    baseline, selected_summary = select_candidates(screening_summary)
    selected = [
        {"strategy": item["strategy"], "cardinality": item["cardinality"]}
        for item in selected_summary
    ]
    print(
        f"screening baseline prefix/k=71 mean residual={baseline['mean_residual']:.6f}",
        flush=True,
    )
    print(f"selected candidates={len(selected)}", flush=True)

    full_runs = []
    for candidate in selected:
        for seed in full_seeds:
            print(
                f"full strategy={candidate['strategy']} k={candidate['cardinality']} seed={seed}",
                flush=True,
            )
            full_runs.append(
                run_candidate(
                    A,
                    b,
                    candidate["cardinality"],
                    candidate["strategy"],
                    seed,
                    full_steps,
                )
            )

    full_summary = aggregate_runs(full_runs)
    payload = {
        "instance": os.path.abspath(instance_path),
        "shape": list(A.shape),
        "target_norm": float(np.linalg.norm(b)),
        "cardinalities": list(cardinalities),
        "strategies": list(strategies),
        "screen_seeds": list(screen_seeds),
        "full_seeds": list(full_seeds),
        "screen_steps": int(screen_steps),
        "full_steps": int(full_steps),
        "baseline": baseline,
        "selected_candidates": selected,
        "screening_runs": screening_runs,
        "screening_summary": screening_summary,
        "full_runs": full_runs,
        "full_summary": full_summary,
    }
    with open(os.path.join(output_dir, "cardinality_sweep_results.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(os.path.join(output_dir, "screening_runs.csv"), screening_runs)
    write_csv(os.path.join(output_dir, "screening_summary.csv"), screening_summary)
    write_csv(os.path.join(output_dir, "full_runs.csv"), full_runs)
    write_csv(os.path.join(output_dir, "full_summary.csv"), full_summary)
    return payload


def main():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        default=os.path.join(
            package_dir, "data/01-marketsplit/instances/ms_15_200_000.dat"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(package_dir, "results/cardinality_sweep"),
    )
    parser.add_argument("--cardinalities", default=','.join(map(str, CARDINALITIES)))
    parser.add_argument("--strategies", default=','.join(STRATEGIES))
    parser.add_argument("--screen-seeds", default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--full-seeds", default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--screen-steps", type=int, default=DEFAULT_SCREEN_STEPS)
    parser.add_argument("--full-steps", type=int, default=DEFAULT_FULL_STEPS)
    args = parser.parse_args()

    payload = run_sweep(
        args.instance,
        args.output_dir,
        parse_int_list(args.cardinalities),
        tuple(item.strip() for item in args.strategies.split(",") if item.strip()),
        parse_int_list(args.screen_seeds),
        parse_int_list(args.full_seeds),
        args.screen_steps,
        args.full_steps,
    )
    print(json.dumps({
        "baseline": payload["baseline"],
        "selected_candidates": payload["selected_candidates"],
        "full_summary": payload["full_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify manuscript-facing numerical claims against committed benchmark data.

This checker performs no benchmark execution.  It validates the canonical
result artifact, recomputes the headline statistics, and checks the values used
in the manuscript and generated tables.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.paper_statistics import (
    RGSA_CONFIG,
    REFERENCE_CONFIG,
    compute_improvement_stats,
    compute_per_class_statistics,
    load_results,
)
from src.validate_publication import DEFAULT_RESULTS, validate_results


FLAGSHIP = (15, 200)


def _mean(runs, predicate, field="final_residual") -> float:
    values = [float(run[field]) for run in runs if predicate(run)]
    if not values:
        raise AssertionError("claim group is empty")
    return sum(values) / len(values)


def _check(condition: bool, message: str) -> bool:
    if condition:
        print(f"  PASS: {message}")
        return True
    print(f"  FAIL: {message}")
    return False


def run_checks(data: list[dict]) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    class_statistics = compute_per_class_statistics(data)

    flagship_rgsa = _mean(data, lambda r: (r["m"], r["D"]) == FLAGSHIP and r["config"] == RGSA_CONFIG)
    flagship_full = _mean(data, lambda r: (r["m"], r["D"]) == FLAGSHIP and r["config"] == "full_esa")
    results.append(("Flagship residuals", _check(
        abs(flagship_rgsa - 104.06) <= 1.0 and abs(flagship_full - 100.37) <= 1.0,
        f"flagship residuals (RGSA={flagship_rgsa:.2f}, Full={flagship_full:.2f})",
    )))

    d50_rgsa = [run for run in data if run["D"] == 50 and run["config"] == RGSA_CONFIG]
    d50_means = [
        _mean(d50_rgsa, lambda r, m=m: r["m"] == m)
        for m in range(9, 16)
    ]
    d50_times = [
        _mean(d50_rgsa, lambda r, m=m: r["m"] == m, field="total_time_sec")
        for m in range(9, 16)
    ]
    scaling_ok = (
        abs(min(d50_means) - 11.2) <= 0.5
        and abs(max(d50_means) - 24.2) <= 0.5
        and max(d50_times) < 34.0
    )
    results.append(("Scaling D=50", _check(
        scaling_ok,
        f"D=50 scaling (residuals {min(d50_means):.1f}–{max(d50_means):.1f}, "
        f"times {min(d50_times):.0f}–{max(d50_times):.0f}s)",
    )))

    r50 = _mean(data, lambda r: (r["m"], r["D"]) == (15, 50) and r["config"] == RGSA_CONFIG)
    r200 = flagship_rgsa
    ratio = r200 / r50
    results.append(("D=200/D=50 ratio", _check(
        3.5 < ratio < 4.5,
        f"D=200/D=50 ratio {ratio:.1f}x (~4x claimed)",
    )))

    paired = {}
    for run in data:
        if run["config"] in {RGSA_CONFIG, REFERENCE_CONFIG}:
            paired.setdefault((run["instance"], run["seed"]), {})[run["config"]] = run
    common_pairs = [pair for pair in paired.values() if RGSA_CONFIG in pair and REFERENCE_CONFIG in pair]
    better = sum(pair[RGSA_CONFIG]["final_residual"] < pair[REFERENCE_CONFIG]["final_residual"] for pair in common_pairs)
    results.append(("39/40 pairs", _check(
        len(common_pairs) == 40 and better == 39,
        f"{better}/{len(common_pairs)} pairs RGSA < Blind (39/40 claimed)",
    )))

    exact = sum(bool(run["feasible"]) for run in data)
    results.append(("No feasibility", _check(
        exact == 0 and len(data) == 160,
        f"{exact}/{len(data)} exact feasibility",
    )))

    violations_ok = True
    for D in (50, 200):
        values = {}
        for config in (REFERENCE_CONFIG, "lagrangian_blind_sa", RGSA_CONFIG):
            values[config] = _mean(
                data,
                lambda r, D=D, config=config: r["m"] == 15 and r["D"] == D and r["config"] == config,
                field="max_abs_violation",
            )
        violations_ok &= values[RGSA_CONFIG] <= values[REFERENCE_CONFIG]
        violations_ok &= values[RGSA_CONFIG] <= values["lagrangian_blind_sa"]
    results.append(("Violations RGSA smallest", _check(
        violations_ok,
        "RGSA has no larger mean V_max than standalone baselines on m=15",
    )))

    # Keep these calculations exercised so changes to the statistical API do not
    # silently break publication checks.
    improvement = compute_improvement_stats(data)
    stats_ok = (
        improvement["n_pairs"] == 40
        and math.isfinite(float(improvement["median_improvement"]))
        and all((15, D, RGSA_CONFIG) in class_statistics for D in (50, 200))
    )
    results.append(("Statistics API", _check(
        stats_ok,
        f"paired statistics and class groups available ({improvement['n_pairs']} pairs)",
    )))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    summary = validate_results(args.json)
    data = load_results(args.json)
    print(f"Loaded {summary['runs']} validated runs\n")
    results = run_checks(data)
    print()
    passed = sum(value for _, value in results)
    if passed != len(results):
        print(f"{passed}/{len(results)} checks passed, {len(results) - passed} FAILED")
        return 1
    print(f"ALL {len(results)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

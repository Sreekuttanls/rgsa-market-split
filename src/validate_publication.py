#!/usr/bin/env python3
"""Validate the committed RGSA publication benchmark artifact.

This module intentionally performs no solver work.  It validates the historical
JSON result artifact used by the manuscript and is suitable for CI and release
checks on any supported platform.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PACKAGE_ROOT / "results" / "paper_benchmarks" / "paper_benchmark_results.json"
EXPECTED_CONFIGS = {
    "random_blind_sa",
    "random_guided_sa",
    "lagrangian_blind_sa",
    "full_esa",
}
INSTANCE_RE = re.compile(r"^ms_(?P<m>\d+)_(?P<D>\d+)_(?P<index>\d+)$")
REQUIRED_RUN_FIELDS = {
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
}
NONNEGATIVE_NUMERIC_FIELDS = {
    "stage1_residual",
    "stage2_residual",
    "final_residual",
    "relative_error_pct",
    "max_abs_violation",
    "total_time_sec",
    "stage1_time",
    "stage2_time",
    "polish_time",
}


class PublicationValidationError(ValueError):
    """Raised when a committed publication artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationValidationError(message)


def _load_payload(path: str | Path) -> Mapping[str, Any]:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise PublicationValidationError(f"cannot read results JSON {input_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PublicationValidationError(f"invalid JSON in {input_path}: {exc}") from exc
    _require(isinstance(payload, dict), "benchmark artifact must be a JSON object")
    return payload


def _validate_run(run: Mapping[str, Any], index: int) -> tuple[str, int, str]:
    missing = REQUIRED_RUN_FIELDS - set(run)
    _require(not missing, f"run {index} is missing fields: {sorted(missing)}")

    instance = str(run["instance"])
    if instance.endswith(".dat"):
        instance = instance[:-4]
    match = INSTANCE_RE.fullmatch(instance)
    _require(match is not None, f"run {index} has invalid instance name: {run['instance']!r}")

    try:
        m = int(run["m"])
        n = int(run["n"])
        coefficient_range = int(run["D"])
        seed = int(run["seed"])
    except (TypeError, ValueError) as exc:
        raise PublicationValidationError(f"run {index} has invalid integer metadata") from exc

    filename_m = int(match.group("m"))
    filename_D = int(match.group("D"))
    _require(m == filename_m, f"run {index} m does not match instance name")
    _require(coefficient_range == filename_D, f"run {index} D does not match instance name")
    _require(m > 0 and n > 0, f"run {index} dimensions must be positive")
    _require(str(run["config_name"]) in EXPECTED_CONFIGS, f"run {index} has unknown configuration")
    _require(isinstance(run["feasible"], bool), f"run {index} feasible must be boolean")

    for field in NONNEGATIVE_NUMERIC_FIELDS:
        value = run[field]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 f"run {index} field {field} must be numeric")
        numeric = float(value)
        _require(math.isfinite(numeric) and numeric >= 0.0,
                 f"run {index} field {field} must be finite and nonnegative")

    residual = float(run["final_residual"])
    if bool(run["feasible"]):
        _require(residual <= 1e-9, f"run {index} is feasible but residual is {residual}")
    else:
        _require(residual > 1e-9, f"run {index} is infeasible but residual is {residual}")

    return instance, seed, str(run["config_name"])


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a loaded benchmark payload and return a summary."""
    _require("runs" in payload and isinstance(payload["runs"], list),
             "benchmark artifact must contain a 'runs' list")
    runs = payload["runs"]
    _require(runs, "benchmark artifact contains no runs")
    _require(payload.get("num_runs") == len(runs),
             f"num_runs={payload.get('num_runs')!r} does not match {len(runs)} records")

    parameters = payload.get("parameters")
    _require(isinstance(parameters, dict), "benchmark artifact must contain parameters metadata")
    _require(parameters.get("cardinality") == "A.shape[1] // 2",
             "primary benchmark cardinality must be recorded as A.shape[1] // 2")
    _require(int(parameters.get("n_steps", 0)) > 0, "n_steps must be positive")

    keys: list[tuple[str, int, str]] = []
    for index, raw_run in enumerate(runs):
        _require(isinstance(raw_run, dict), f"run {index} must be a JSON object")
        keys.append(_validate_run(raw_run, index))

    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    _require(not duplicates, f"duplicate instance/seed/config records: {duplicates[:3]}")

    instances = sorted({key[0] for key in keys})
    configs = sorted({key[2] for key in keys})
    seeds = sorted({key[1] for key in keys})
    _require(set(configs) == EXPECTED_CONFIGS, f"unexpected configurations: {configs}")
    _require(len(instances) == 8, f"expected 8 benchmark instances, found {len(instances)}")
    _require(len(seeds) == 5, f"expected 5 seeds, found {seeds}")
    _require(len(keys) == 160, f"expected 160 committed runs, found {len(keys)}")

    counts = Counter((instance, config) for instance, _, config in keys)
    _require(all(count == 5 for count in counts.values()),
             "every instance/configuration group must contain exactly 5 runs")
    expected_seeds = set(seeds)
    for instance in instances:
        for config in EXPECTED_CONFIGS:
            group_seeds = {seed for item, seed, item_config in keys
                           if item == instance and item_config == config}
            _require(group_seeds == expected_seeds,
                     f"{instance}/{config} does not contain the common seed set")

    feasible_count = sum(bool(run["feasible"]) for run in runs)
    return {
        "runs": len(runs),
        "instances": len(instances),
        "configs": configs,
        "seeds": seeds,
        "groups": len(counts),
        "feasible_runs": feasible_count,
    }


def validate_results(path: str | Path = DEFAULT_RESULTS) -> dict[str, Any]:
    """Load and validate a committed benchmark JSON artifact."""
    return validate_payload(_load_payload(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_RESULTS,
                        help="committed benchmark JSON to validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_results(args.json)
    print("Publication benchmark validation passed")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

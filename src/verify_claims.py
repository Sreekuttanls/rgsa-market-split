#!/usr/bin/env python3
"""Verify that prose claims in esa_paper.tex match generated table data.

Checks performed:
 1. Flagship residuals: RGSA~104.06, Full~100.37
 2. Scaling D=50: RGSA residual range 11.2–24.2, runtime <33s
 3. D=200/D=50 ratio at m=15 is ~4x
 4. 39/40 pairs RGSA < Blind
 5. 0/160 exact feasibility
 6. RGSA smallest V_max on m=15 classes
"""

import json
import sys

INSTANCE_MAP = {
    "ms_09_050_000.dat": "ms_09_050_000",
    "ms_10_050_000.dat": "ms_10_050_000",
    "ms_11_050_000.dat": "ms_11_050_000",
    "ms_12_050_000.dat": "ms_12_050_000",
    "ms_13_050_000.dat": "ms_13_050_000",
    "ms_14_050_000.dat": "ms_14_050_000",
    "ms_15_050_000.dat": "ms_15_050_000",
    "ms_15_200_000.dat": "ms_15_200_000",
}

CONFIG_SHORT = {
    "random_blind_sa": "sa",
    "lagrangian_blind_sa": "lag",
    "random_guided_sa": "rgsa",
    "full_esa": "full",
}


def load_results(json_path: str) -> list[dict]:
    with open(json_path) as f:
        raw = json.load(f)
    runs = raw.get("runs", raw) if isinstance(raw, dict) else raw
    for r in runs:
        r["_iid"] = INSTANCE_MAP.get(r["instance"], r["instance"])
        r["_cfg"] = CONFIG_SHORT.get(r["config_name"], r["config_name"])
    return runs


def flagship_check(data: list[dict], tol: float = 1.0) -> bool:
    flag = [r for r in data if r["_iid"] == "ms_15_200_000"]
    rg = [r for r in flag if r["_cfg"] == "rgsa"]
    full = [r for r in flag if r["_cfg"] == "full"]
    rg_mean = sum(r["final_residual"] for r in rg) / len(rg)
    full_mean = sum(r["final_residual"] for r in full) / len(full)
    ok = True
    if abs(rg_mean - 104.06) > tol:
        print(f"  FAIL: flagship RGSA mean {rg_mean:.2f}, expected ~104.06")
        ok = False
    if abs(full_mean - 100.37) > tol:
        print(f"  FAIL: flagship Full mean {full_mean:.2f}, expected ~100.37")
        ok = False
    if ok:
        print(f"  PASS: flagship residuals (RGSA={rg_mean:.2f}, Full={full_mean:.2f})")
    return ok


def scaling_d50_check(data: list[dict], tol: float = 0.5) -> bool:
    ok = True
    min_r, max_r = float("inf"), float("-inf")
    min_t, max_t = float("inf"), float("-inf")
    for m in range(9, 16):
        iid = f"ms_{m:02d}_050_000"
        runs = [r for r in data if r["_iid"] == iid and r["_cfg"] == "rgsa"]
        if not runs:
            print(f"  FAIL: no data for {iid}")
            ok = False
            continue
        mean_r = sum(r["final_residual"] for r in runs) / len(runs)
        mean_t = sum(r["total_time_sec"] for r in runs) / len(runs)
        min_r = min(min_r, mean_r)
        max_r = max(max_r, mean_r)
        min_t = min(min_t, mean_t)
        max_t = max(max_t, mean_t)
    if abs(min_r - 11.2) > tol:
        print(f"  FAIL: D=50 min residual {min_r:.2f}, expected ~11.2")
        ok = False
    if abs(max_r - 24.2) > tol:
        print(f"  FAIL: D=50 max residual {max_r:.2f}, expected ~24.2")
        ok = False
    if max_t > 34:
        print(f"  FAIL: D=50 max runtime {max_t:.1f}s, expected <33s")
        ok = False
    if ok:
        print(f"  PASS: D=50 scaling (residuals {min_r:.1f}–{max_r:.1f}, times {min_t:.0f}–{max_t:.0f}s)")
    return ok


def ratio_d200_check(data: list[dict]) -> bool:
    m15_d50 = [r for r in data if r["_iid"] == "ms_15_050_000" and r["_cfg"] == "rgsa"]
    m15_d200 = [r for r in data if r["_iid"] == "ms_15_200_000" and r["_cfg"] == "rgsa"]
    r50 = sum(r["final_residual"] for r in m15_d50) / len(m15_d50)
    r200 = sum(r["final_residual"] for r in m15_d200) / len(m15_d200)
    ratio = r200 / r50
    ok = 3.5 < ratio < 4.5
    if ok:
        print(f"  PASS: D=200/D=50 ratio {ratio:.1f}x (~4x claimed)")
    else:
        print(f"  FAIL: D=200/D=50 ratio {ratio:.1f}x, expected ~4x")
    return ok


def ablation_39_check(data: list[dict]) -> bool:
    pair_keys = set()
    for r in data:
        if r["_cfg"] in ("rgsa", "sa"):
            pair_keys.add((r["_iid"], r["seed"]))
    rgsa_better = 0
    total = len(pair_keys)
    for iid, seed in pair_keys:
        rg = [r for r in data if r["_iid"] == iid and r["seed"] == seed and r["_cfg"] == "rgsa"]
        bl = [r for r in data if r["_iid"] == iid and r["seed"] == seed and r["_cfg"] == "sa"]
        if rg and bl and rg[0]["final_residual"] < bl[0]["final_residual"]:
            rgsa_better += 1
    ok = rgsa_better == 39
    if ok:
        print(f"  PASS: {rgsa_better}/{total} pairs RGSA < Blind (39 claimed)")
    else:
        print(f"  FAIL: {rgsa_better}/{total} pairs, expected 39")
    return ok


def no_feasibility_check(data: list[dict]) -> bool:
    exact = [r for r in data if r["final_residual"] < 1e-6]
    ok = len(exact) == 0
    if ok:
        print(f"  PASS: 0/{len(data)} exact feasibility")
    else:
        print(f"  FAIL: {len(exact)}/{len(data)} exact feasibility")
    return ok


def violations_rgsa_smallest_on_m15(data: list[dict]) -> bool:
    ok = True
    for D in [50, 200]:
        iid = f"ms_15_{D:03d}_000"
        runs = [r for r in data if r["_iid"] == iid]
        vmeans = {}
        for cfg in ["sa", "lag", "rgsa", "full"]:
            cfg_runs = [r for r in runs if r["_cfg"] == cfg]
            if cfg_runs:
                vmeans[cfg] = sum(r["max_abs_violation"] for r in cfg_runs) / len(cfg_runs)
        if not vmeans:
            continue
        rg_v = vmeans.get("rgsa", float("inf"))
        sa_v = vmeans.get("sa", float("inf"))
        lag_v = vmeans.get("lag", float("inf"))
        if rg_v > sa_v or rg_v > lag_v:
            print(f"  FAIL: m=15 D={D} RGSA V_max {rg_v:.1f} > SA {sa_v:.1f} or Lag {lag_v:.1f}")
            ok = False
    if ok:
        print("  PASS: RGSA smallest V_max among standalone configs on m=15")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="results/paper_benchmarks/paper_benchmark_results.json")
    args = parser.parse_args()

    data = load_results(args.json)
    print(f"Loaded {len(data)} runs\n")

    results = []
    results.append(("Flagship residuals", flagship_check(data)))
    results.append(("Scaling D=50", scaling_d50_check(data)))
    results.append(("D=200/D=50 ratio", ratio_d200_check(data)))
    results.append(("39/40 pairs", ablation_39_check(data)))
    results.append(("No feasibility", no_feasibility_check(data)))
    results.append(("Violations RGSA smallest", violations_rgsa_smallest_on_m15(data)))

    print()
    passed = sum(1 for _, v in results if v)
    total = len(results)
    if passed == total:
        print(f"ALL {total} CHECKS PASSED")
    else:
        print(f"{passed}/{total} checks passed, {total - passed} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

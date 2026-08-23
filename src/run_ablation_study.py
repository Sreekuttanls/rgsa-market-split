"""
Formal 5-Configuration Ablation Study Runner
============================================
Evaluates:
1. Baseline: Random Start + Blind SA
2. Stage 1 Only: Pure Lagrangian Dual Sort
3. Ablation A: Lagrangian Warm-Start + Blind SA (p_guide = 0.0)
4. Ablation B: Random Start + Direction-Guided SA (p_guide = 0.65)
5. Full Composite RGSA Pipeline
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rgsa_solver import RGSASolver
from src.audit_n140 import parse_instance


def run_ablation():
    inst_path = os.path.join(os.path.dirname(__file__), "../data/01-marketsplit/instances/ms_15_200_000.dat")
    if not os.path.exists(inst_path):
        inst_path = "data/01-marketsplit/instances/ms_15_200_000.dat"

    A, b, known_sol = parse_instance(inst_path)
    cardinality = int(np.sum(known_sol)) if known_sol is not None else 71
    target_norm = np.linalg.norm(b)

    print("=" * 95)
    print("FORMAL ABLATION STUDY: N=140, J=15 (ms_15_200_000.dat)")
    print("=" * 95)

    solver = RGSASolver(A, b, cardinality=cardinality, seed=42)
    results = []

    # Config 1: Random + Blind SA
    np.random.seed(42)
    rand_sol = np.zeros(140)
    rand_sol[np.random.permutation(140)[:cardinality]] = 1.0
    c1_sol, c1_res, c1_time = solver.solve_stage2_guided_sa(rand_sol, p_guide=0.0)
    results.append(("1. Random Start + Blind SA", "Random Split", "Blind SA (p=0.0)", "None", c1_res, (c1_res/target_norm)*100, c1_time))

    # Config 2: Pure Lagrangian Only
    c2_sol, c2_res, c2_time = solver.solve_stage1_lagrangian()
    results.append(("2. Pure Lagrangian Dual Sort", "Lagrangian Sort", "None", "None", c2_res, (c2_res/target_norm)*100, c2_time))

    # Config 3: Lag Start + Blind SA
    c3_sol, c3_res, c3_time_sa = solver.solve_stage2_guided_sa(c2_sol, p_guide=0.0)
    results.append(("3. Lag Warm-Start + Blind SA", "Lagrangian Sort", "Blind SA (p=0.0)", "None", c3_res, (c3_res/target_norm)*100, c2_time + c3_time_sa))

    # Config 4: Random Start + Guided SA
    c4_sol, c4_res, c4_time = solver.solve_stage2_guided_sa(rand_sol, p_guide=0.65)
    results.append(("4. Random Start + Guided SA", "Random Split", "Guided SA (p=0.65)", "None", c4_res, (c4_res/target_norm)*100, c4_time))

    # Config 5: Full Pipeline
    c5_sa_sol, c5_sa_res, c5_time_sa = solver.solve_stage2_guided_sa(c2_sol, p_guide=0.65)
    c5_sol, c5_res, c5_time_polish = solver.solve_stage3_2opt_polish(c5_sa_sol)
    results.append(("5. Full RGSA Pipeline", "Lagrangian Sort", "Guided SA (p=0.65)", "2-Opt Polish", c5_res, (c5_res/target_norm)*100, c2_time + c5_time_sa + c5_time_polish))

    print("\n" + "=" * 95)
    print(f"{'Configuration':<30} {'Warm-Start':<16} {'Search Engine':<20} {'Residual':<12} {'Rel. Error':<12} {'Time':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r[0]:<30} {r[1]:<16} {r[2]:<20} {r[4]:<12.4f} {r[5]:.2f}%       {r[6]:.2f}s")
    print("=" * 95)


if __name__ == "__main__":
    run_ablation()

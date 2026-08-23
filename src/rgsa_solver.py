"""
Residual-Guided Simulated Annealing (RGSA) Core Module
=====================================================
Implements the 3-stage RGSA pipeline for Multi-Dimensional 0-1 Subset Sum
(Market Split) optimization:
- Stage 1: Lagrangian Dynamic Price Learning (warm-start construction)
- Stage 2: Residual-Guided Simulated Annealing (compensating swap proposals)
- Stage 3: Deterministic 1-/2-item exchange polishing

Note: benchmark configuration identifier ``full_esa`` is retained for
reproducibility with published results; it denotes the Full RGSA pipeline.
"""

from itertools import combinations
import time

import numpy as np


class RGSASolver:
    def __init__(self, A: np.ndarray, b: np.ndarray, cardinality: int = None, seed: int = 42):
        """
        Initialize the RGSA Solver.
        
        Parameters:
        -----------
        A : np.ndarray
            Integer constraint matrix of shape (J, N) where J is attributes, N is items.
        b : np.ndarray
            Target partition vector of shape (J,).
        cardinality : int, optional
            Number of items to assign to Pile A (default: N // 2).
        seed : int, optional
            Deterministic random seed for exact reproducibility.
        """
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = self.A.shape
        self.cardinality = cardinality if cardinality is not None else self.n // 2
        self.seed = seed
        np.random.seed(self.seed)

    def _solution_from_sorted_indices(self, sorted_idx: np.ndarray, strategy: str = "prefix"):
        """Construct a cardinality-feasible partition from sorted item indices."""
        if strategy not in {"prefix", "alternating"}:
            raise ValueError(f"unknown initialization strategy: {strategy}")
        if not 0 <= self.cardinality <= self.n:
            raise ValueError("cardinality must be between 0 and the number of items")

        solution = np.zeros(self.n)
        if strategy == "prefix":
            solution[sorted_idx[:self.cardinality]] = 1.0
            return solution

        # Start with sorted positions 0, 2, 4, ... in Pile A and the
        # interleaved positions in Pile B.  This is exactly balanced when
        # cardinality is ceil(N / 2); greedy residual-guided moves adapt it
        # to any requested cardinality without using reference-solution data.
        pile_a = set(int(i) for i in sorted_idx[::2])
        pile_b = set(range(self.n)) - pile_a
        solution[list(pile_a)] = 1.0
        residual = self.A @ solution - self.b
        target = self.cardinality

        while len(pile_a) > target:
            candidates = np.array(sorted(pile_a), dtype=int)
            candidate_residuals = residual[:, None] - self.A[:, candidates]
            candidate_magnitudes = np.linalg.norm(candidate_residuals, axis=0)
            item = int(candidates[int(np.argmin(candidate_magnitudes))])
            pile_a.remove(item)
            pile_b.add(item)
            residual -= self.A[:, item]

        while len(pile_a) < target:
            candidates = np.array(sorted(pile_b), dtype=int)
            candidate_residuals = residual[:, None] + self.A[:, candidates]
            candidate_magnitudes = np.linalg.norm(candidate_residuals, axis=0)
            item = int(candidates[int(np.argmin(candidate_magnitudes))])
            pile_b.remove(item)
            pile_a.add(item)
            residual += self.A[:, item]

        solution = np.zeros(self.n)
        solution[list(pile_a)] = 1.0
        return solution

    def solve_stage1_lagrangian(self, max_iters: int = 3000, alpha_0: float = 0.001,
                                strategy: str = "prefix"):
        """
        Stage 1: Pure Lagrangian Dynamic Price Learning.
        Projects J dimensions onto a 1D scalar line z_i = lambda . w_i and slices at cardinality.
        """
        t0 = time.time()
        lam = np.ones(self.m) / self.m
        best_res = float('inf')
        best_sol = None

        for t in range(max_iters):
            scores = lam @ self.A
            sorted_idx = np.argsort(scores)
            sol = self._solution_from_sorted_indices(sorted_idx, strategy)
            
            res_vec = self.A @ sol - self.b
            res_mag = np.linalg.norm(res_vec)
            
            if res_mag < best_res:
                best_res = res_mag
                best_sol = sol.copy()
                
            step = alpha_0 / np.sqrt(t + 1)
            lam = np.maximum(lam + step * res_vec, 1e-4)
            lam = lam / np.sum(lam)

        return best_sol, best_res, time.time() - t0

    def solve_stage2_guided_sa(self, init_sol: np.ndarray, n_steps: int = 150000, 
                               p_guide: float = 0.65, T_start: float = 600.0, T_end: float = 0.001):
        """
        Stage 2: Direction-Guided Simulated Annealing.
        Uses the ideal missing puzzle piece target formula w* = w_i - Delta with probability p_guide.
        """
        t0 = time.time()
        pile_a = set(np.where(init_sol == 1.0)[0])
        pile_b = set(range(self.n)) - pile_a

        curr_res = self.A @ init_sol - self.b
        curr_mag = np.linalg.norm(curr_res)

        best_mag = curr_mag
        best_sol = init_sol.copy()

        cooling = (T_end / T_start) ** (1.0 / n_steps)
        T = T_start

        for _ in range(n_steps):
            i = np.random.choice(list(pile_a))
            if p_guide > 0 and np.random.rand() < p_guide:
                ideal_w = self.A[:, i] - curr_res
                b_list = list(pile_b)
                dists = [np.linalg.norm(self.A[:, j] - ideal_w) for j in b_list]
                j = b_list[np.argmin(dists)]
            else:
                j = np.random.choice(list(pile_b))

            delta = self.A[:, j] - self.A[:, i]
            cand_res = curr_res + delta
            cand_mag = np.linalg.norm(cand_res)
            dE = cand_mag - curr_mag

            if dE < 0 or np.random.rand() < np.exp(-dE / max(T, 1e-4)):
                pile_a.remove(i)
                pile_a.add(j)
                pile_b.remove(j)
                pile_b.add(i)
                curr_res = cand_res
                curr_mag = cand_mag
                if curr_mag < best_mag:
                    best_mag = curr_mag
                    best_sol = np.zeros(self.n)
                    best_sol[list(pile_a)] = 1.0
            T *= cooling

        return best_sol, best_mag, time.time() - t0

    def solve_stage3_2opt_polish(self, init_sol: np.ndarray, max_2opt_passes: int = 1):
        """
        Deterministic local polishing with one-item and two-item exchanges.

        A two-item exchange removes two items from Pile A and inserts two
        items from Pile B.  It therefore preserves the cardinality exactly.
        For each pass, all C(|A|, 2) x C(|B|, 2) exchanges are evaluated;
        the best strictly improving exchange is applied.  Pair sums on the
        B side are precomputed and candidate residuals are evaluated in
        NumPy batches to keep the N=140 case practical.

        Parameters
        ----------
        init_sol : np.ndarray
            Binary starting partition.
        max_2opt_passes : int, optional
            Number of exhaustive best-improvement two-item passes.  Set to
            zero to reproduce the former one-item-only polish.
        """
        if max_2opt_passes < 0:
            raise ValueError("max_2opt_passes must be non-negative")

        t0 = time.time()
        pile_a = set(np.where(init_sol == 1.0)[0])
        pile_b = set(range(self.n)) - pile_a

        curr_res = self.A @ init_sol - self.b
        curr_mag = np.linalg.norm(curr_res)

        # 1-Opt Pass: retain the original greedy single-item polishing step.
        for i in list(pile_a):
            if i not in pile_a:
                continue
            for j in list(pile_b):
                if j not in pile_b:
                    continue
                delta = self.A[:, j] - self.A[:, i]
                cand_mag = np.linalg.norm(curr_res + delta)
                if cand_mag < curr_mag - 1e-5:
                    curr_mag = cand_mag
                    pile_a.remove(i)
                    pile_a.add(j)
                    pile_b.remove(j)
                    pile_b.add(i)
                    curr_res = self.A[:, list(pile_a)].sum(axis=1) - self.b
                    break

        # 2-Opt Passes: replace two selected items with two unselected items.
        for _ in range(max_2opt_passes):
            a_indices = np.array(sorted(pile_a), dtype=int)
            b_indices = np.array(sorted(pile_b), dtype=int)
            if len(a_indices) < 2 or len(b_indices) < 2:
                break

            a_pairs = np.array(list(combinations(a_indices, 2)), dtype=int)
            b_pairs = np.array(list(combinations(b_indices, 2)), dtype=int)
            b_pair_sums = self.A[:, b_pairs[:, 0]] + self.A[:, b_pairs[:, 1]]

            best_mag = curr_mag
            best_a_pair = None
            best_b_pair = None
            for a_pair in a_pairs:
                removed_sum = self.A[:, a_pair[0]] + self.A[:, a_pair[1]]
                candidate_residuals = curr_res[:, None] + b_pair_sums - removed_sum[:, None]
                candidate_magnitudes = np.linalg.norm(candidate_residuals, axis=0)
                b_pos = int(np.argmin(candidate_magnitudes))
                candidate_mag = float(candidate_magnitudes[b_pos])
                if candidate_mag < best_mag - 1e-5:
                    best_mag = candidate_mag
                    best_a_pair = a_pair
                    best_b_pair = b_pairs[b_pos]

            if best_a_pair is None:
                break

            a1, a2 = map(int, best_a_pair)
            b1, b2 = map(int, best_b_pair)
            pile_a.remove(a1)
            pile_a.remove(a2)
            pile_a.add(b1)
            pile_a.add(b2)
            pile_b.remove(b1)
            pile_b.remove(b2)
            pile_b.add(a1)
            pile_b.add(a2)
            curr_res = self.A[:, list(pile_a)].sum(axis=1) - self.b
            curr_mag = np.linalg.norm(curr_res)

        final_sol = np.zeros(self.n)
        final_sol[list(pile_a)] = 1.0
        return final_sol, curr_mag, time.time() - t0

    def solve(self):
        """
        Run the complete 3-stage RGSA pipeline end-to-end.
        """
        t_start = time.time()
        
        # Stage 1
        s1_sol, s1_res, s1_time = self.solve_stage1_lagrangian()
        
        # Stage 2
        s2_sol, s2_res, s2_time = self.solve_stage2_guided_sa(s1_sol)
        
        # Stage 3
        final_sol, final_res, s3_time = self.solve_stage3_2opt_polish(s2_sol)
        
        total_time = time.time() - t_start
        rel_error = (final_res / np.linalg.norm(self.b)) * 100.0
        
        return {
            "solution": final_sol,
            "residual": final_res,
            "relative_error_pct": rel_error,
            "total_time_sec": total_time,
            "pile_a_indices": sorted(list(np.where(final_sol == 1.0)[0])),
            "pile_b_indices": sorted(list(np.where(final_sol == 0.0)[0])),
            "stage_breakdown": {
                "stage1_res": s1_res, "stage1_time": s1_time,
                "stage2_res": s2_res, "stage2_time": s2_time,
                "stage3_res": final_res, "stage3_time": s3_time
            }
        }


# Backward-compatibility alias (legacy name used in earlier internal drafts)
ESASolver = RGSASolver

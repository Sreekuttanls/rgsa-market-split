import unittest

import numpy as np

from src.rgsa_solver import RGSASolver


class TestInitializationStrategies(unittest.TestCase):
    def setUp(self):
        self.A = np.array(
            [
                [4, 1, 8, 2, 7, 3, 6, 5, 9, 0, 2, 4],
                [0, 7, 2, 9, 1, 8, 3, 6, 4, 5, 9, 2],
                [5, 3, 1, 8, 6, 2, 9, 4, 0, 7, 5, 1],
            ],
            dtype=float,
        )
        self.b = np.array([25, 25, 25], dtype=float)
        self.sorted_idx = np.arange(self.A.shape[1])

    def test_prefix_and_alternating_have_requested_cardinality(self):
        for cardinality in range(self.A.shape[1] + 1):
            solver = RGSASolver(self.A, self.b, cardinality=cardinality)
            for strategy in ("prefix", "alternating"):
                solution = solver._solution_from_sorted_indices(
                    self.sorted_idx, strategy
                )
                self.assertEqual(int(solution.sum()), cardinality)
                self.assertTrue(np.all((solution == 0) | (solution == 1)))

    def test_alternating_balanced_case_uses_sorted_even_positions(self):
        solver = RGSASolver(self.A, self.b, cardinality=6)
        solution = solver._solution_from_sorted_indices(
            self.sorted_idx, "alternating"
        )
        np.testing.assert_array_equal(
            solution, np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        )

    def test_alternating_adjustment_is_deterministic(self):
        for cardinality in (2, 5, 8, 11):
            solver = RGSASolver(self.A, self.b, cardinality=cardinality)
            first = solver._solution_from_sorted_indices(
                self.sorted_idx, "alternating"
            )
            second = solver._solution_from_sorted_indices(
                self.sorted_idx, "alternating"
            )
            np.testing.assert_array_equal(first, second)

    def test_unknown_strategy_is_rejected(self):
        solver = RGSASolver(self.A, self.b, cardinality=6)
        with self.assertRaises(ValueError):
            solver._solution_from_sorted_indices(self.sorted_idx, "unknown")

    def test_stage1_strategy_outputs_requested_cardinality(self):
        for cardinality in (2, 6, 10):
            for strategy in ("prefix", "alternating"):
                solver = RGSASolver(self.A, self.b, cardinality=cardinality)
                solution, _, _ = solver.solve_stage1_lagrangian(
                    max_iters=5, strategy=strategy
                )
                self.assertEqual(int(solution.sum()), cardinality)


if __name__ == "__main__":
    unittest.main()


class TestCandidateSelection(unittest.TestCase):
    def test_selects_top_five_and_all_baseline_beaters(self):
        from src.run_cardinality_sweep import select_candidates

        summaries = []
        for cardinality in range(60, 81):
            summaries.append(
                {
                    "strategy": "prefix",
                    "cardinality": cardinality,
                    "mean_residual": float(abs(cardinality - 70)),
                }
            )
        summaries.append(
            {"strategy": "alternating", "cardinality": 65, "mean_residual": 0.5}
        )
        baseline, selected = select_candidates(summaries)
        selected_ids = {
            (item["strategy"], item["cardinality"]) for item in selected
        }
        self.assertEqual((baseline["strategy"], baseline["cardinality"]), ("prefix", 71))
        self.assertTrue(("alternating", 65) in selected_ids)
        self.assertTrue(("prefix", 70) in selected_ids)
        self.assertGreaterEqual(len(selected_ids), 5)

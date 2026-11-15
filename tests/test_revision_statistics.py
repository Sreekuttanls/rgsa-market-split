"""Known-answer tests for the statistics used in the revised paper."""

import math
import unittest

import numpy as np
from scipy import stats

from src.revision_statistics import INF, compare, holm, rank_biserial, vs


class TestStatistics(unittest.TestCase):

    def test_holm_matches_hand_computation(self):
        # sorted p: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
        self.assertEqual([round(x, 6) for x in holm([0.04, 0.01, 0.02])], [0.04, 0.03, 0.04])
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])  # 0.8*2 capped at 1

    def test_rank_biserial_extremes(self):
        self.assertEqual(rank_biserial([1, 2, 3], [2, 3, 4]), -1.0)  # a always better (lower)
        self.assertEqual(rank_biserial([5, 5], [1, 1]), 1.0)
        self.assertEqual(rank_biserial([1, 2], [1, 2]), 0.0)

    def _summ(self, table):
        return {(f"i{k}", m): {"median": v} for k, row in enumerate(table)
                for m, v in row.items()}

    def test_failures_are_ranked_last_not_dropped(self):
        table = [{"a": 1.0, "b": 2.0, "c": INF} for _ in range(6)]
        res = compare(self._summ(table), ["a", "b", "c"], "a", [f"i{k}" for k in range(6)])
        self.assertEqual(res["mean_rank"], {"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertEqual(res["n_instances"], 6)
        w, t, l, p, r = vs(res, "c", "a")
        self.assertEqual((w, t, l), (0, 0, 6))
        self.assertGreater(r, 0)

    def test_friedman_matches_scipy(self):
        rng = np.random.default_rng(0)
        table = [{"a": float(x), "b": float(y), "c": float(z)} for x, y, z in rng.random((12, 3))]
        res = compare(self._summ(table), ["a", "b", "c"], "a", [f"i{k}" for k in range(12)])
        ref = stats.friedmanchisquare(*[[row[m] for row in table] for m in "abc"])
        self.assertTrue(math.isclose(res["friedman"]["p"], ref.pvalue))

    def test_ties_get_average_rank(self):
        table = [{"a": 0.0, "b": 0.0, "c": 1.0} for _ in range(4)]
        res = compare(self._summ(table), ["a", "b", "c"], "a", [f"i{k}" for k in range(4)])
        self.assertEqual(res["mean_rank"]["a"], 1.5)
        self.assertEqual(vs(res, "b", "a")[:3], (0, 4, 0))


if __name__ == "__main__":
    unittest.main()

"""Equivalence tests between the C++ solver (cpp/rgsa) and the Python reference.

The deterministic components must agree exactly with a Python specification
that fixes the two details the pseudocode leaves open:
  * ties in the Lagrangian scores are broken by item index (stable sort) and
    scores are accumulated row by row in double precision;
  * the polishing sweeps visit items in increasing index order.
The legacy solver in src/rgsa_solver.py (used for the original 160 runs)
leaves both to numpy's unstable sort and CPython set order; it agrees with
the specification whenever those happen to coincide (checked below for the
small instances).  Every C++ result is also re-verified in Python.

The tests are skipped when the C++ binary cannot be built (e.g. no compiler).
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from src.audit_n140 import parse_instance
from src.rgsa_solver import RGSASolver

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "cpp" / "rgsa"
INST = ROOT / "data" / "01-marketsplit" / "instances"
CASES = ("ms_03_050_002.dat", "ms_09_050_000.dat", "ms_15_200_000.dat")


def build_binary():
    if shutil.which("make") is None:
        return False
    try:
        subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True,
                       capture_output=True, timeout=300)
    except (subprocess.SubprocessError, OSError):
        return False
    return BIN.exists()


def run_cpp(*args):
    out = subprocess.run([str(BIN), *map(str, args)], check=True,
                         capture_output=True, text=True, timeout=600)
    return json.loads(out.stdout)


def spec_lagrangian(A, b, q, iters, alpha0=0.001, floor=1e-4):
    """Algorithm 1 with index tie-breaking and row-order accumulation."""
    m, n = A.shape
    lam = np.ones(m) / m
    best, best_r = None, np.inf
    for t in range(iters):
        sc = np.zeros(n)
        for r in range(m):          # same accumulation order as the C++ code
            sc = sc + lam[r] * A[r]
        x = np.zeros(n)
        x[np.argsort(sc, kind="stable")[:q]] = 1.0
        res = A @ x - b
        R = np.linalg.norm(res)
        if R < best_r:
            best_r, best = R, x.copy()
        lam = np.maximum(lam + alpha0 / np.sqrt(t + 1) * res, floor)
        tot = 0.0
        for r in range(m):
            tot += lam[r]
        lam = lam / tot
    return best, best_r


def spec_legacy_polish(A, b, x0, passes=1, tol=1e-5):
    """Algorithm 3 (legacy rule) visiting items in increasing index order."""
    x = x0.copy()
    res = A @ x - b
    for i in sorted(np.where(x == 1)[0]):
        if x[i] != 1:
            continue
        R = np.linalg.norm(res)
        for j in sorted(np.where(x == 0)[0]):
            if np.linalg.norm(res - A[:, i] + A[:, j]) < R - tol:
                x[i], x[j] = 0, 1
                res = res - A[:, i] + A[:, j]
                break
    for _ in range(passes):
        S = sorted(np.where(x == 1)[0]); U = sorted(np.where(x == 0)[0])
        pu = [(j1, j2) for k, j1 in enumerate(U) for j2 in U[k + 1:]]
        add = np.stack([A[:, j1] + A[:, j2] for j1, j2 in pu], axis=1)
        best, move = np.linalg.norm(res), None
        for k, i1 in enumerate(S):
            for i2 in S[k + 1:]:
                cand = np.linalg.norm(res[:, None] - (A[:, i1] + A[:, i2])[:, None] + add, axis=0)
                p = int(np.argmin(cand))
                if cand[p] < best - tol:
                    best, move = cand[p], (i1, i2, *pu[p])
        if move is None:
            break
        i1, i2, j1, j2 = move
        x[[i1, i2]] = 0; x[[j1, j2]] = 1
        res = A @ x - b
    return x, np.linalg.norm(res)


def to_bits(x):
    return "".join("1" if v else "0" for v in np.asarray(x).astype(int))


@unittest.skipUnless(build_binary(), "C++ solver could not be built")
class TestCppMatchesPython(unittest.TestCase):

    def check_result(self, A, b, d):
        x = np.array([int(c) for c in d["x"]], dtype=float)
        res = A @ x - b
        self.assertAlmostEqual(float(np.linalg.norm(res)), d["residual"], places=6)
        self.assertEqual(int(np.abs(res).max()), d["max_abs_violation"])
        self.assertEqual(int(x.sum()), d["cardinality"])
        self.assertEqual(d["feasible"], bool(np.all(res == 0)))
        return x

    def test_lagrangian_projection_identical(self):
        for name in CASES:
            with self.subTest(instance=name):
                A, b, _ = parse_instance(INST / name)
                q = A.shape[1] // 2
                py_x, py_res = spec_lagrangian(A, b, q, 300)
                d = run_cpp("--instance", INST / name, "--init", "lagrangian",
                            "--lag-iters", 300, "--steps", 0, "--polish", "none")
                cpp_x = self.check_result(A, b, d)
                self.assertEqual(to_bits(py_x), to_bits(cpp_x))
                self.assertAlmostEqual(py_res, d["residual"], places=6)

    def test_legacy_polish_identical(self):
        for name in CASES:
            for seed in (0, 1, 2):
                with self.subTest(instance=name, seed=seed):
                    A, b, _ = parse_instance(INST / name)
                    n = A.shape[1]
                    q = n // 2
                    x0 = np.zeros(n)
                    x0[np.random.RandomState(seed).permutation(n)[:q]] = 1.0
                    py_x, py_res = spec_legacy_polish(A, b, x0, passes=1)
                    d = run_cpp("--instance", INST / name, "--init", "given",
                                "--init-x", to_bits(x0), "--steps", 0,
                                "--polish", "legacy", "--polish-passes", 1)
                    cpp_x = self.check_result(A, b, d)
                    self.assertAlmostEqual(py_res, d["residual"], places=6)
                    self.assertEqual(to_bits(py_x), to_bits(cpp_x))

    def test_legacy_python_solver_agrees_on_small_instances(self):
        """Where tie-breaking cannot differ, the legacy solver matches too."""
        for name in ("ms_03_050_002.dat", "ms_09_050_000.dat"):
            with self.subTest(instance=name):
                A, b, _ = parse_instance(INST / name)
                n = A.shape[1]
                q = n // 2
                x0 = np.zeros(n)
                x0[np.random.RandomState(7).permutation(n)[:q]] = 1.0
                legacy, _, _ = RGSASolver(A, b, cardinality=q).solve_stage3_2opt_polish(x0, 1)
                spec, _ = spec_legacy_polish(A, b, x0, passes=1)
                self.assertEqual(to_bits(legacy), to_bits(spec))

    def test_converge_polish_is_local_optimum(self):
        """No improving 1-swap or 2-swap remains after converge polishing."""
        name = "ms_09_050_000.dat"
        A, b, _ = parse_instance(INST / name)
        d = run_cpp("--instance", INST / name, "--seed", 3, "--steps", 2000,
                    "--polish", "converge")
        x = self.check_result(A, b, d)
        R = np.linalg.norm(A @ x - b)
        S = np.where(x == 1)[0]
        U = np.where(x == 0)[0]
        res = A @ x - b
        one = min(np.linalg.norm(res - A[:, i] + A[:, j]) for i in S for j in U)
        self.assertGreaterEqual(one, R - 1e-5)
        pu = [(j1, j2) for k, j1 in enumerate(U) for j2 in U[k + 1:]]
        add = np.stack([A[:, j1] + A[:, j2] for j1, j2 in pu], axis=1)
        for k, i1 in enumerate(S):
            for i2 in S[k + 1:]:
                base = res - A[:, i1] - A[:, i2]
                best = np.linalg.norm(base[:, None] + add, axis=0).min()
                self.assertGreaterEqual(best, R - 1e-5)

    def test_band_and_checkpoints(self):
        name = "ms_09_050_000.dat"
        A, b, _ = parse_instance(INST / name)
        n = A.shape[1]
        d = run_cpp("--instance", INST / name, "--seed", 5, "--band", 2,
                    "--time", 1.0, "--checkpoints", "0.25,0.5,1.0",
                    "--polish", "converge")
        self.check_result(A, b, d)
        self.assertEqual(sorted(p["q"] for p in d["per_q"]),
                         list(range(n // 2 - 2, n // 2 + 3))[:len(d["per_q"])]
                         if d["feasible"] else list(range(n // 2 - 2, n // 2 + 3)))
        vals = [c["residual"] for c in d["checkpoints"]]
        self.assertTrue(all(v is not None for v in vals))
        self.assertTrue(all(x >= y - 1e-9 for x, y in zip(vals, vals[1:])),
                        "best-so-far must not increase")
        self.assertAlmostEqual(vals[-1], d["residual"], places=6)
        self.assertLess(d["total_time"], 1.5)

    def test_every_mode_respects_time_limit(self):
        """Including the full pipeline, whose polishing must fit in the budget."""
        name = "ms_15_200_000.dat"
        limit = 2.0
        variants = (("--mode", "sa", "--init", "random", "--polish", "none"),
                    ("--mode", "sa", "--init", "lagrangian", "--polish", "converge"),
                    ("--mode", "sa", "--p-guide", "0", "--polish", "none"),
                    ("--mode", "greedy"))
        for v in variants:
            with self.subTest(variant=v):
                d = run_cpp("--instance", INST / name, "--seed", 1, "--band", 2,
                            "--time", limit, "--no-x", *v)
                self.assertLessEqual(d["total_time"], limit + 0.05)
                self.assertLessEqual(d["time_to_best"], limit)

    def test_greedy_mode(self):
        name = "ms_03_050_002.dat"
        A, b, _ = parse_instance(INST / name)
        d = run_cpp("--instance", INST / name, "--mode", "greedy", "--seed", 1,
                    "--time", 0.3, "--band", 1)
        self.check_result(A, b, d)
        self.assertGreater(d["restarts"], 0)


if __name__ == "__main__":
    unittest.main()

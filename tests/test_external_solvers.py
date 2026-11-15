"""Sanity checks for the external-method wrappers (reviewer 1, comment Q3.2).

* The exact solvers must solve a small instance to a verified zero residual.
* Every wrapper must return a vector, best-so-far checkpoints that never get
  worse, and a residual that matches an independent recomputation.
* Solver failures must be reported as status "error", not hidden.
Skipped when the optional solver packages are not installed.
"""

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from src.audit_n140 import parse_instance

ROOT = Path(__file__).resolve().parent.parent
INST = ROOT / "data" / "01-marketsplit" / "instances"
HAVE = {name: importlib.util.find_spec(mod) is not None
        for name, mod in (("cpsat", "ortools"), ("scip", "pyscipopt"),
                          ("neal", "dwave.samplers"), ("ga", "pymoo"))}


def run(solver, name, seconds, band=2, seed=1, checkpoints="0.2,0.5,1,2"):
    from src import external_solvers

    buf = io.StringIO()
    with redirect_stdout(buf):
        external_solvers.main(["--solver", solver, "--instance", str(INST / name),
                               "--time", str(seconds), "--seed", str(seed),
                               "--band", str(band), "--checkpoints", checkpoints])
    return json.loads(buf.getvalue())


class TestExternalSolvers(unittest.TestCase):

    def verify(self, d, name):
        A, b, _ = parse_instance(INST / name)
        self.assertNotEqual(d["status"], "error", d.get("error"))
        x = np.array([int(c) for c in d["x"]])
        r = A.astype(int) @ x - b.astype(int)
        self.assertAlmostEqual(float(np.linalg.norm(r)), d["residual"], places=9)
        self.assertEqual(d["feasible"], bool(np.all(r == 0)))
        n = A.shape[1]
        self.assertTrue(n // 2 - 2 <= int(x.sum()) <= n // 2 + 2)
        vals = [c["residual"] for c in d["checkpoints"] if c["residual"] is not None]
        self.assertTrue(all(u >= v - 1e-12 for u, v in zip(vals, vals[1:])))
        self.assertLessEqual(d["total_time"], d["time_limit"] + 1.0)

    def test_exact_solvers_solve_small_instance(self):
        for solver in ("cpsat", "scip"):
            if not HAVE[solver]:
                continue
            with self.subTest(solver=solver):
                d = run(solver, "ms_03_050_002.dat", 20)
                self.verify(d, "ms_03_050_002.dat")
                self.assertTrue(d["feasible"])
                self.assertEqual(d["status"], "feasible")
                self.assertEqual(d["residual"], 0.0)

    def test_all_wrappers_return_vectors_on_medium_instance(self):
        for solver, ok in HAVE.items():
            if not ok:
                continue
            with self.subTest(solver=solver):
                d = run(solver, "ms_09_050_000.dat", 2)
                self.verify(d, "ms_09_050_000.dat")
                self.assertIn(d["status"], ("time_limit", "feasible",
                                            "proven_no_exact_solution_in_band"))

    def test_errors_are_reported_not_hidden(self):
        if not HAVE["cpsat"]:
            self.skipTest("ortools missing")
        d = run("cpsat", "ms_03_050_002.dat", 0.0, checkpoints="")
        self.assertEqual(d["status"], "error")
        self.assertIsNone(d["residual"])


if __name__ == "__main__":
    unittest.main()


class TestDeadlineEnforced(unittest.TestCase):
    """Improvements found after the time limit must never count."""

    def test_recorder_ignores_late_improvements(self):
        import time
        from src.external_solvers import Recorder
        A = np.array([[1, 2, 3, 4]]); b = np.array([5])
        rec = Recorder(A, b, 0, 4, [0.5, 1.0], time.perf_counter(), deadline=0.05)
        rec.offer([1, 0, 0, 0])            # residual 4, in time
        time.sleep(0.1)
        rec.offer([1, 0, 0, 1])            # residual 0, but too late
        out = rec.result({})
        self.assertEqual(out["squared_residual"], 16)
        self.assertEqual(out["late_improvements_ignored"], 1)
        self.assertTrue(out["deadline_enforced"])
        self.assertEqual(out["checkpoints"][-1]["residual"], 4.0)

    def test_fj_driver_reports_deadline(self):
        import subprocess
        root = Path(__file__).resolve().parent.parent
        binary = root / "cpp" / "fj_driver"
        if not binary.exists():
            self.skipTest("fj_driver not built")
        inst = root / "data" / "01-marketsplit" / "instances" / "ms_09_050_000.dat"
        out = json.loads(subprocess.run([str(binary), "--instance", str(inst), "--time", "0.5",
                                         "--seed", "1"], capture_output=True, text=True,
                                        check=True).stdout)
        self.assertTrue(out["deadline_enforced"])
        self.assertLessEqual(out["time_to_best"], 0.5)

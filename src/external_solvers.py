"""External comparison methods for the revision experiments.

Every method receives the same instance, cardinality band, wall-clock budget
and single thread, and every method reports the *best vector it has seen*
(not only its terminal answer), measured by the same Euclidean residual
||Ax - b||_2 over vectors whose cardinality lies in the band.  Output is one
JSON object per run with the same keys as the C++ solvers (cpp/rgsa,
cpp/fj_driver), plus a solver ``status`` that separates proven results,
time limits with a vector, time limits without a vector, and errors.

Methods
-------
cpsat : Google OR-Tools CP-SAT on the linear model
        min sum_i (s+_i + s-_i)  s.t.  A_i x - b_i = s+_i - s-_i,
        q-k <= 1'x <= q+k, x binary, s >= 0 integer.
        Improving solutions are captured by a solution callback.
scip  : SCIP (PySCIPOpt) on the same model; improving solutions captured by
        a BESTSOLFOUND event handler.
neal  : D-Wave simulated-annealing QUBO sampler (dwave-samplers) on
        E(x) = ||Ax - b||^2 + P (1'x - q)^2, restarted until the budget ends.
ga    : pymoo genetic algorithm on binary strings with a cardinality repair
        operator, objective ||Ax - b||_2.

Usage:
    python -m src.external_solvers --solver cpsat --instance FILE --time 10 \
        --seed 1 --band 2 --checkpoints 1,2.5,5,10
"""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.audit_n140 import parse_instance


class Recorder:
    """Best-so-far tracking with wall-clock checkpoints."""

    def __init__(self, A, b, qlo, qhi, checkpoints, t0, deadline=None):
        self.A = A.astype(np.int64)
        self.b = b.astype(np.int64)
        self.qlo, self.qhi = qlo, qhi
        self.cps = sorted(checkpoints)
        self.cp_vals = [None] * len(self.cps)
        self.next_cp = 0
        self.t0 = t0
        self.best_sq = None
        self.best_x = None
        self.time_to_best = 0.0
        self.offers = 0
        # Some solvers only check the clock between batches of work (a Neal
        # read, a GA generation), so they can overrun.  Improvements found
        # after the deadline are ignored, so no method gains extra time.
        self.deadline = deadline
        self.late_improvements_ignored = 0

    def elapsed(self):
        return time.perf_counter() - self.t0

    def tick(self, now=None):
        now = self.elapsed() if now is None else now
        while self.next_cp < len(self.cps) and now >= self.cps[self.next_cp]:
            self.cp_vals[self.next_cp] = self.best_sq
            self.next_cp += 1

    def offer(self, x):
        self.offers += 1
        x = np.asarray(x, dtype=np.int64)
        card = int(x.sum())
        if card < self.qlo or card > self.qhi:
            return
        r = self.A @ x - self.b
        sq = int(r @ r)
        if self.best_sq is None or sq < self.best_sq:
            now = self.elapsed()
            if self.deadline is not None and now > self.deadline:
                self.late_improvements_ignored += 1
                return
            self.tick(now)
            self.best_sq, self.best_x, self.time_to_best = sq, x.copy(), now

    def result(self, extra):
        self.tick(float("inf"))
        out = dict(extra)
        out["time_to_best"] = self.time_to_best
        out["late_improvements_ignored"] = self.late_improvements_ignored
        out["deadline_enforced"] = self.deadline is not None
        out["checkpoints"] = [
            {"t": t, "residual": None if v is None else math.sqrt(v)}
            for t, v in zip(self.cps, self.cp_vals)
        ]
        if self.best_x is None:
            out.update(residual=None, squared_residual=None, max_abs_violation=None,
                       feasible=False, cardinality=None)
        else:
            r = self.A @ self.best_x - self.b
            out.update(residual=math.sqrt(int(r @ r)), squared_residual=int(r @ r),
                       max_abs_violation=int(np.abs(r).max()), feasible=bool(np.all(r == 0)),
                       cardinality=int(self.best_x.sum()),
                       x="".join(str(int(v)) for v in self.best_x))
        return out


# ------------------------------------------------------------------ CP-SAT
def run_cpsat(A, b, rec, qlo, qhi, time_limit, seed):
    from ortools.sat.python import cp_model

    m, n = A.shape
    Ai = A.astype(int)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{j}") for j in range(n)]
    slack = []
    for i in range(m):
        ub = int(Ai[i].sum())
        sp = model.NewIntVar(0, ub, f"sp{i}")
        sn = model.NewIntVar(0, ub, f"sn{i}")
        model.Add(sum(int(Ai[i, j]) * x[j] for j in range(n)) - int(b[i]) == sp - sn)
        slack += [sp, sn]
    model.Add(sum(x) >= qlo)
    model.Add(sum(x) <= qhi)
    model.Minimize(sum(slack))

    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.count = 0

        def on_solution_callback(self):
            self.count += 1
            rec.offer([self.Value(v) for v in x])

    remaining = time_limit - rec.elapsed()
    if remaining <= 0:
        return {"status": "error", "error": "budget exhausted before the solve started"}
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = int(seed)
    cb = CB()
    status = solver.Solve(model, cb)
    name = solver.StatusName(status)
    proven = name == "OPTIMAL"
    obj = solver.ObjectiveValue() if name in ("OPTIMAL", "FEASIBLE") else None
    if name == "INFEASIBLE":
        st = "proven_no_vector_in_band"
    elif proven and obj == 0:
        st = "feasible"
    elif proven:
        st = "proven_no_exact_solution_in_band"
    elif name == "FEASIBLE":
        st = "time_limit"
    elif name == "UNKNOWN":
        st = "time_limit_no_vector"
    else:
        st = "error"
    return {"status": st, "solver_status": name, "solutions_seen": cb.count,
            "l1_objective": obj, "l1_bound": solver.BestObjectiveBound()}


# ------------------------------------------------------------------ SCIP
def run_scip(A, b, rec, qlo, qhi, time_limit, seed):
    from pyscipopt import Eventhdlr, Model, SCIP_EVENTTYPE, quicksum

    m, n = A.shape
    Ai = A.astype(int)
    model = Model()
    model.hideOutput()
    x = [model.addVar(vtype="B", name=f"x{j}") for j in range(n)]
    slack = []
    for i in range(m):
        sp = model.addVar(vtype="I", lb=0, name=f"sp{i}")
        sn = model.addVar(vtype="I", lb=0, name=f"sn{i}")
        model.addCons(quicksum(int(Ai[i, j]) * x[j] for j in range(n)) - int(b[i]) == sp - sn)
        slack += [sp, sn]
    model.addCons(quicksum(x) >= qlo)
    model.addCons(quicksum(x) <= qhi)
    model.setObjective(quicksum(slack), "minimize")
    remaining = time_limit - rec.elapsed()
    if remaining <= 0:
        return {"status": "error", "error": "budget exhausted before the solve started"}
    model.setParam("limits/time", remaining)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("lp/threads", 1)
    model.setParam("randomization/randomseedshift", int(seed))

    counter = {"n": 0}

    class Best(Eventhdlr):
        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            counter["n"] += 1
            sol = self.model.getBestSol()
            rec.offer([int(round(self.model.getSolVal(sol, v))) for v in x])

    model.includeEventhdlr(Best(), "best", "record incumbents")
    model.optimize()
    name = model.getStatus()
    nsols = model.getNSols()
    obj = model.getObjVal() if nsols > 0 else None
    if name == "infeasible":
        st = "proven_no_vector_in_band"
    elif name == "optimal" and obj is not None and abs(obj) < 0.5:
        st = "feasible"
    elif name == "optimal":
        st = "proven_no_exact_solution_in_band"
    elif name in ("timelimit", "userinterrupt") and nsols > 0:
        st = "time_limit"
    elif name == "timelimit":
        st = "time_limit_no_vector"
    else:
        st = "error"
    return {"status": st, "solver_status": name, "solutions_seen": counter["n"],
            "l1_objective": obj, "l1_bound": model.getDualbound()}


# ------------------------------------------------------------------ Neal QUBO
def qubo_bqm(A, b, q, penalty):
    import dimod

    A = A.astype(float)
    b = b.astype(float)
    n = A.shape[1]
    G = A.T @ A
    h = A.T @ b
    lin = np.diag(G) + penalty - 2 * h - 2 * penalty * q
    quad = 2 * G + 2 * penalty
    Q = {}
    for j in range(n):
        Q[(j, j)] = float(lin[j])
        for k in range(j + 1, n):
            Q[(j, k)] = float(quad[j, k])
    offset = float(b @ b + penalty * q * q)
    return dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)


def run_neal(A, b, rec, qlo, qhi, time_limit, seed, num_sweeps=1000, penalty_scale=1.0):
    from dwave.samplers import SimulatedAnnealingSampler

    n = A.shape[1]
    q = (qlo + qhi) // 2
    # penalty: the largest single-column energy change, so that leaving the
    # band is never cheaper than the worst one-item residual change
    penalty = penalty_scale * float(max((A.astype(float) ** 2).sum(axis=0)))
    bqm = qubo_bqm(A, b, q, penalty)
    sampler = SimulatedAnnealingSampler()
    reads = 0
    rng = np.random.RandomState(seed)
    while rec.elapsed() < time_limit:
        ss = sampler.sample(bqm, num_reads=1, num_sweeps=num_sweeps,
                            seed=int(rng.randint(2**31 - 1)))
        rec.offer([ss.first.sample[j] for j in range(n)])
        rec.tick()
        reads += 1
    return {"status": "feasible" if rec.best_sq == 0 else
            ("time_limit" if rec.best_x is not None else "time_limit_no_vector"),
            "reads": reads, "num_sweeps": num_sweeps, "penalty": penalty,
            "penalty_scale": penalty_scale}


# ------------------------------------------------------------------ GA
def run_ga(A, b, rec, qlo, qhi, time_limit, seed, pop_size=100, mutation_rate=1.0):
    """mutation_rate: expected number of flipped bits per offspring."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination.max_time import TimeBasedTermination

    m, n = A.shape
    Af = A.astype(float)
    bf = b.astype(float)
    remaining = max(0.0, time_limit - rec.elapsed())

    class MS(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=1, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, *args, **kwargs):
            R = X.astype(float) @ Af.T - bf
            out["F"] = np.sqrt((R ** 2).sum(axis=1))

    class CardRepair(Repair):
        """Move cardinality into the band by flipping random bits."""

        def _do(self, problem, X, **kwargs):
            X = X.astype(bool)
            for row in X:
                c = int(row.sum())
                if c > qhi:
                    on = np.flatnonzero(row)
                    row[np.random.choice(on, c - qhi, replace=False)] = False
                elif c < qlo:
                    off = np.flatnonzero(~row)
                    row[np.random.choice(off, qlo - c, replace=False)] = True
            return X

    class Rec(Callback):
        def notify(self, algorithm):
            pop = algorithm.pop
            k = int(np.argmin(pop.get("F")[:, 0]))
            rec.offer(pop.get("X")[k].astype(int))
            rec.tick()

    np.random.seed(seed)
    algo = GA(pop_size=pop_size, sampling=BinaryRandomSampling(),
              crossover=TwoPointCrossover(),
              mutation=BitflipMutation(prob=1.0, prob_var=min(1.0, mutation_rate / n)),
              repair=CardRepair(), eliminate_duplicates=True)
    res = minimize(MS(), algo, TimeBasedTermination(max_time=remaining), seed=seed,
                   callback=Rec(), verbose=False)
    return {"status": "feasible" if rec.best_sq == 0 else
            ("time_limit" if rec.best_x is not None else "time_limit_no_vector"),
            "generations": int(res.algorithm.n_gen), "pop_size": pop_size,
            "mutation_rate": mutation_rate}


SOLVERS = {"cpsat": run_cpsat, "scip": run_scip, "neal": run_neal, "ga": run_ga}

# Libraries are imported before the clock starts: loading a native solver
# library can take seconds (tens of seconds on the first load after install
# on macOS), and charging that to the search budget would leave the solver
# with no search time at all.
PRELOAD = {
    "cpsat": ("ortools.sat.python.cp_model",),
    "scip": ("pyscipopt",),
    "neal": ("dwave.samplers", "dimod"),
    "ga": ("pymoo.algorithms.soo.nonconvex.ga", "pymoo.optimize",
           "pymoo.termination.max_time"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one external comparison method.")
    ap.add_argument("--solver", choices=sorted(SOLVERS), required=True)
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band", type=int, default=0)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--no-x", action="store_true")
    ap.add_argument("--num-sweeps", type=int, default=1000, help="neal")
    ap.add_argument("--penalty-scale", type=float, default=1.0, help="neal")
    ap.add_argument("--pop-size", type=int, default=100, help="ga")
    ap.add_argument("--mutation-rate", type=float, default=1.0, help="ga")
    args = ap.parse_args(argv)

    import importlib

    t_load = time.perf_counter()
    for mod in PRELOAD[args.solver]:
        importlib.import_module(mod)
    load_seconds = time.perf_counter() - t_load

    t0 = time.perf_counter()
    A, b, _ = parse_instance(args.instance)
    n = A.shape[1]
    q = args.q if args.q is not None else n // 2
    qlo, qhi = max(0, q - args.band), min(n, q + args.band)
    cps = [float(t) for t in args.checkpoints.split(",") if t]
    rec = Recorder(A, b, qlo, qhi, cps, t0, deadline=args.time)
    base = {"instance": str(args.instance), "m": int(A.shape[0]), "n": int(n),
            "mode": args.solver, "seed": args.seed, "band": args.band,
            "time_limit": args.time, "library_load_seconds": load_seconds}
    try:
        extra = {"neal": {"num_sweeps": args.num_sweeps, "penalty_scale": args.penalty_scale},
                 "ga": {"pop_size": args.pop_size, "mutation_rate": args.mutation_rate}}
        info = SOLVERS[args.solver](A, b, rec, qlo, qhi, args.time, args.seed,
                                    **extra.get(args.solver, {}))
    except Exception as exc:  # report, never hide, solver failures
        info = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}
    base.update(info)
    out = rec.result(base)
    out["total_time"] = rec.elapsed()
    if args.no_x:
        out.pop("x", None)
    sys.stdout.write(json.dumps(out) + "\n")
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())

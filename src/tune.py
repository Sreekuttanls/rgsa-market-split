"""Parameter tuning for the revision experiments (reviewer 2, comments 3 and 5).

Protocol
--------
* Tuning instances: the first instance (lowest identifier) of each of the 39
  (m, D) classes.  The other 117 instances are never seen during tuning.
* Budget per evaluation: the same wall-clock budget as the main experiment
  (N/10 seconds), single thread.
* Objective of a configuration: mean over tuning instances of log(1 + R),
  where R is the Euclidean residual of the best vector found.  Differences in
  this mean are average log-ratios of residuals, so instances with large
  coefficients do not dominate, and an exact solution (R = 0) scores best.
* Search: Optuna's TPE sampler (Akiba et al., 2019) with a fixed seed.  Each
  trial evaluates the tuning instances one by one in a fixed shuffled order
  and reports its running mean; trials clearly worse than the median trial
  at the same point are stopped early (median pruning), a racing scheme in
  the spirit of irace.
* Equal effort: every tuned method gets the same number of trials, and the
  method's default configuration is always evaluated first, so tuning cannot
  return a configuration that did worse than the default on the tuning set.
* Seeds used for tuning (10_000 + trial number) are disjoint from the seeds
  used in the evaluation experiments (1-30).

Methods tuned: RGSA (guidance probability, start/end temperature), blind SA
(start/end temperature), the genetic algorithm (population size, mutation
rate) and the QUBO sampler (sweeps per read, penalty scale).  The full RGSA
pipeline reuses RGSA's settings; Feasibility Jump, CP-SAT and SCIP use their
authors' defaults; the greedy baseline has no parameters.

Usage:
    python -m src.tune --method rgsa --trials 60 --workers 4
"""

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import optuna

from src.run_revision_experiments import (DEFAULT_CONFIG, OUT_DIR, PY, RGSA, all_instances,
                                          budget, run_one, tuning_set)

SPACES = {
    "rgsa": {
        "p_guide": ("float", 0.05, 1.0, False),
        "T0": ("float", 1.0, 2000.0, True),
        "Tf": ("float", 1e-4, 1.0, True),
    },
    "blind_sa": {
        "T0": ("float", 1.0, 2000.0, True),
        "Tf": ("float", 1e-4, 1.0, True),
    },
    "ga": {
        "pop_size": ("int", 20, 400, True),
        "mutation_rate": ("float", 0.2, 8.0, True),
    },
    "neal": {
        "num_sweeps": ("int", 100, 20000, True),
        # first run used [0.05, 5]; its optimum hit the lower bound (0.0505),
        # so the range was widened and the method re-tuned (records kept as
        # neal_range1_*).  The first widened run (neal_range2_contended_*)
        # shared the machine with the main experiment, so it was repeated on
        # an idle machine as study "neal_range2_idle".
        "penalty_scale": ("float", 0.001, 5.0, True),
    },
}
BAND = DEFAULT_CONFIG["rgsa"]["band"]


def command(method, params, inst, seed):
    T = budget(inst)
    base = ["--instance", str(inst["path"]), "--seed", str(seed), "--time", f"{T:g}",
            "--band", str(BAND)]
    if method in ("rgsa", "blind_sa"):
        pg = params["p_guide"] if method == "rgsa" else 0.0
        return [str(RGSA), *base, "--mode", "sa", "--init", "random", "--p-guide", str(pg),
                "--T0", str(params["T0"]), "--Tf", str(params["Tf"]), "--polish", "none",
                "--no-x"]
    if method == "ga":
        return [PY, "-m", "src.external_solvers", "--solver", "ga", *base, "--no-x",
                "--pop-size", str(params["pop_size"]),
                "--mutation-rate", str(params["mutation_rate"])]
    if method == "neal":
        return [PY, "-m", "src.external_solvers", "--solver", "neal", *base, "--no-x",
                "--num-sweeps", str(params["num_sweeps"]),
                "--penalty-scale", str(params["penalty_scale"])]
    raise ValueError(method)


def suggest(trial, space):
    out = {}
    for name, (kind, lo, hi, log) in space.items():
        if kind == "int":
            out[name] = trial.suggest_int(name, lo, hi, log=log)
        else:
            out[name] = trial.suggest_float(name, lo, hi, log=log)
    return out


def collect():
    settings, sources = {}, {}
    for method in SPACES:
        path = OUT_DIR / "tuning" / f"{method}_summary.json"
        if not path.exists():
            print(f"missing {path}; tune {method} first", file=sys.stderr)
            return 1
        summ = json.loads(path.read_text())
        settings[method] = summ["best_params"]
        sources[method] = {"best_value": summ["best_value"],
                           "default_value": summ["default_value"],
                           "n_trials": summ["n_trials"]}
    out = {"settings": settings, "sources": sources,
           "note": "Band fixed at +/-2 (QOBLIB generator documentation); not tuned."}
    (OUT_DIR / "tuned_config.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=sorted(SPACES))
    ap.add_argument("--collect", action="store_true",
                    help="write results/revision/tuned_config.json from the summaries")
    ap.add_argument("--smoke", type=Path, default=None,
                    help="smoke test: 3 instances, results written to this directory")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--study-name", default=None,
                    help="Optuna study name (default: the method name)")
    args = ap.parse_args(argv)
    if args.collect:
        return collect()
    if not args.method:
        ap.error("--method is required")

    insts = [i for i in all_instances() if i["name"] in tuning_set(all_instances())]
    order = sorted(insts, key=lambda i: i["name"])
    random.Random(args.seed).shuffle(order)
    if args.smoke:
        order = sorted(order, key=lambda i: i["n"])[:3]
    out_dir = args.smoke or (OUT_DIR / "tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.method}_evaluations.jsonl"

    storage = optuna.storages.RDBStorage(f"sqlite:///{out_dir / 'optuna.db'}")
    study = optuna.create_study(
        study_name=args.study_name or args.method, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10))
    if len(study.trials) == 0:
        defaults = {k: DEFAULT_CONFIG[args.method][k] for k in SPACES[args.method]}
        study.enqueue_trial(defaults)

    def objective(trial):
        params = suggest(trial, SPACES[args.method])
        seed = 10_000 + trial.number
        total = 0.0
        for step, inst in enumerate(order):
            rec = run_one(command(args.method, params, inst, seed), budget(inst) + 120)
            R = rec.get("residual")
            ok = rec.get("status") != "error" and R is not None
            score = math.log1p(R) if ok else math.log1p(1e6)  # failed run = very bad
            with log_path.open("a") as fh:
                fh.write(json.dumps({"trial": trial.number, "params": params, "seed": seed,
                                     "instance": inst["name"], "residual": R,
                                     "status": rec.get("status"), "score": score}) + "\n")
            total += score
            trial.report(total / (step + 1), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return total / len(order)

    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL)
    remaining = args.trials - sum(t.state in finished for t in study.trials)
    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=args.workers)
    best = study.best_trial
    summary = {"method": args.method, "best_params": best.params, "best_value": best.value,
               "n_trials": len(study.trials),
               "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE
                                 for t in study.trials),
               "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
               "default_value": study.trials[0].value,
               "tuning_instances": [i["name"] for i in order],
               "search_space": {k: list(v) for k, v in SPACES[args.method].items()},
               "optuna_version": optuna.__version__, "sampler_seed": args.seed,
               "wall_seconds_this_session": time.time() - t0}
    (out_dir / f"{args.method}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

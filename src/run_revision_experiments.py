"""Runner for the revision experiments (all 156 QOBLIB Market Split instances).

Each (experiment, instance, method, seed) is one independent single-threaded
run with a wall-clock budget.  Results are appended to a JSON-lines file as
soon as a run finishes, so an interrupted experiment resumes where it stopped
(runs already present are skipped).  Every returned vector is re-verified here
against the instance before it is stored.

Experiments
-----------
main      every heuristic, all instances, 30 seeds, budget N/10 s
ablation  start x guidance x polishing variants, all instances, 10 seeds
exact     CP-SAT and SCIP, all instances, one run, 300 s
long      hardest classes (m = 14, 15), 300 s, 5 seeds

Usage:
    python -m src.run_revision_experiments --experiment main --workers 4
    python -m src.run_revision_experiments --experiment main --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INST_DIR = ROOT / "data" / "01-marketsplit" / "instances"
OUT_DIR = ROOT / "results" / "revision"
RGSA = ROOT / "cpp" / "rgsa"
FJ = ROOT / "cpp" / "fj_driver"
PY = sys.executable
NAME_RE = re.compile(r"^ms_(\d\d)_(\d{3})_(\d{3})\.dat$")
CHECKPOINT_FRACTIONS = (0.1, 0.25, 0.5, 1.0)

# Default settings: the original paper's values.  Replaced by the tuned
# configuration (results/revision/tuned_config.json) once tuning has run.
DEFAULT_CONFIG = {
    "rgsa": {"p_guide": 0.65, "T0": 600.0, "Tf": 1e-3, "band": 2},
    "blind_sa": {"T0": 600.0, "Tf": 1e-3, "band": 2},
    "neal": {"num_sweeps": 1000, "penalty_scale": 1.0},
    "ga": {"pop_size": 100, "mutation_rate": 1.0},
}


def load_config():
    path = OUT_DIR / "tuned_config.json"
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        tuned = json.loads(path.read_text())
        for k, v in tuned.get("settings", {}).items():
            cfg.setdefault(k, {}).update(v)
    return cfg


def all_instances():
    out = []
    for p in sorted(INST_DIR.glob("ms_*.dat")):
        mt = NAME_RE.match(p.name)
        if mt:
            m, D, idx = (int(g) for g in mt.groups())
            out.append({"path": p, "name": p.stem, "m": m, "D": D, "idx": idx,
                        "n": 10 * (m - 1)})
    return out


def tuning_set(instances):
    """First instance (lowest identifier) of every (m, D) class: 39 instances."""
    first = {}
    for inst in instances:
        key = (inst["m"], inst["D"])
        if key not in first or inst["idx"] < first[key]["idx"]:
            first[key] = inst
    return {inst["name"] for inst in first.values()}


def budget(inst):
    return inst["n"] / 10.0


def rgsa_cmd(inst, seed, secs, cfg, *, init, guided, polish, mode="sa"):
    c = cfg["rgsa"] if guided else cfg["blind_sa"]
    cmd = [str(RGSA), "--instance", str(inst["path"]), "--seed", str(seed),
           "--time", f"{secs:g}", "--band", str(c["band"]), "--mode", mode,
           "--init", init, "--p-guide", str(c["p_guide"] if guided else 0.0),
           "--T0", str(c["T0"]), "--Tf", str(c["Tf"]), "--polish", polish]
    return cmd


def method_table(cfg):
    """name -> function(inst, seed, secs) -> argv."""
    band = cfg["rgsa"]["band"]
    t = {
        "rgsa": lambda i, s, T: rgsa_cmd(i, s, T, cfg, init="random", guided=True, polish="none"),
        "rgsa_full": lambda i, s, T: rgsa_cmd(i, s, T, cfg, init="lagrangian", guided=True,
                                              polish="converge"),
        "blind_sa": lambda i, s, T: rgsa_cmd(i, s, T, cfg, init="random", guided=False,
                                             polish="none"),
        "greedy": lambda i, s, T: [str(RGSA), "--instance", str(i["path"]), "--seed", str(s),
                                   "--time", f"{T:g}", "--band", str(band), "--mode", "greedy"],
        "fj": lambda i, s, T: [str(FJ), "--instance", str(i["path"]), "--seed", str(s),
                               "--time", f"{T:g}", "--band", str(band)],
    }
    solver_flags = {
        "neal": ["--num-sweeps", str(cfg["neal"]["num_sweeps"]),
                 "--penalty-scale", str(cfg["neal"]["penalty_scale"])],
        "ga": ["--pop-size", str(cfg["ga"]["pop_size"]),
               "--mutation-rate", str(cfg["ga"]["mutation_rate"])],
    }
    for name in ("neal", "ga", "cpsat", "scip"):
        t[name] = (lambda nm: lambda i, s, T: [PY, "-m", "src.external_solvers", "--solver", nm,
                                               "--instance", str(i["path"]), "--seed", str(s),
                                               "--time", f"{T:g}", "--band", str(band)]
                   + solver_flags.get(nm, []))(name)
    # ablation: (start, guidance, polishing) -- 8 variants
    for init in ("random", "lagrangian"):
        for guided in (False, True):
            for polish in ("none", "converge"):
                key = f"abl_{init[:3]}_{'guided' if guided else 'blind'}_{polish}"
                t[key] = (lambda a, g, p: lambda i, s, T: rgsa_cmd(
                    i, s, T, cfg, init=a, guided=g, polish=p))(init, guided, polish)
    return t


EXPERIMENTS = {
    "main": {"methods": ["rgsa", "rgsa_full", "blind_sa", "greedy", "fj", "neal", "ga"],
             "seeds": list(range(1, 31)), "budget": budget, "filter": None},
    "ablation": {"methods": [f"abl_{a}_{g}_{p}" for a in ("ran", "lag")
                             for g in ("blind", "guided") for p in ("none", "converge")],
                 "seeds": list(range(1, 11)), "budget": budget, "filter": None},
    "exact": {"methods": ["cpsat", "scip"], "seeds": [1],
              "budget": lambda inst: 300.0, "filter": None},
    "long": {"methods": ["rgsa", "rgsa_full", "greedy", "fj"], "seeds": list(range(1, 6)),
             "budget": lambda inst: 300.0, "filter": lambda inst: inst["m"] >= 14},
}


def verify(inst, rec):
    """Recompute residual from the stored vector; raise on any mismatch."""
    if rec.get("x") is None:
        return
    lines = [l for l in inst["path"].read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    m, n = (int(v) for v in lines[0].split()[:2])
    rows = np.array([[int(v) for v in l.split()] for l in lines[1:1 + m]], dtype=np.int64)
    A, b = rows[:, :-1], rows[:, -1]
    x = np.array([int(c) for c in rec["x"]], dtype=np.int64)
    r = A @ x - b
    if int(r @ r) != rec["squared_residual"]:
        raise ValueError(f"residual mismatch for {inst['name']}")
    if bool(np.all(r == 0)) != rec["feasible"]:
        raise ValueError(f"feasibility flag mismatch for {inst['name']}")


def run_one(argv, timeout):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "hard timeout", "wall": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    if not p.stdout.strip():
        return {"status": "error", "error": f"exit {p.returncode}: {p.stderr[-2000:]}",
                "wall": wall}
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    rec.setdefault("status", "feasible" if rec.get("feasible") else "time_limit")
    rec["wall"] = wall
    if p.returncode not in (0, 1):
        rec["status"] = "error"
        rec["error"] = f"exit {p.returncode}: {p.stderr[-2000:]}"
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many jobs")
    ap.add_argument("--instances", default=None, help="comma-separated instance names")
    ap.add_argument("--methods", default=None, help="comma-separated subset of methods")
    args = ap.parse_args(argv)

    exp = EXPERIMENTS[args.experiment]
    cfg = load_config()
    table = method_table(cfg)
    insts = all_instances()
    tune = tuning_set(insts)
    if exp["filter"]:
        insts = [i for i in insts if exp["filter"](i)]
    if args.instances:
        keep = set(args.instances.split(","))
        insts = [i for i in insts if i["name"] in keep]
    methods = args.methods.split(",") if args.methods else exp["methods"]

    out_dir = OUT_DIR / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "runs.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["instance_name"], r["method"], r["seed"]))

    jobs = []
    for inst in insts:
        T = exp["budget"](inst)
        cps = ",".join(f"{f * T:g}" for f in CHECKPOINT_FRACTIONS)
        for meth in methods:
            for seed in exp["seeds"]:
                if (inst["name"], meth, seed) in done:
                    continue
                argv_ = table[meth](inst, seed, T) + ["--checkpoints", cps]
                jobs.append((inst, meth, seed, T, argv_))
    if args.limit:
        jobs = jobs[:args.limit]
    est = sum(j[3] for j in jobs) / max(1, args.workers) / 3600
    print(f"{args.experiment}: {len(jobs)} runs to do ({len(done)} already done), "
          f"~{est:.1f} h with {args.workers} workers", flush=True)
    if args.dry_run:
        for j in jobs[:5]:
            print(" ", " ".join(j[4]))
        return 0

    subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True, capture_output=True)
    meta = {"experiment": args.experiment, "config": cfg,
            "checkpoint_fractions": CHECKPOINT_FRACTIONS,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "workers": args.workers, "uname": " ".join(os.uname())}
    (out_dir / f"meta_{int(time.time())}.json").write_text(json.dumps(meta, indent=2) + "\n")

    lock = threading.Lock()
    finished = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, j[4], j[3] + 120): j for j in jobs}
        for fut in as_completed(futs):
            inst, meth, seed, T, argv_ = futs[fut]
            rec = fut.result()
            try:
                verify(inst, rec)
            except ValueError as exc:
                rec["status"] = "error"
                rec["error"] = str(exc)
            rec.update(instance_name=inst["name"], method=meth, seed=seed, budget=T,
                       m=inst["m"], D=inst["D"], n=inst["n"],
                       tuning_instance=inst["name"] in tune, argv=argv_[1:])
            with lock:
                with out_file.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                finished += 1
                if finished % 50 == 0 or rec["status"] == "error":
                    el = time.time() - t_start
                    print(f"[{finished}/{len(jobs)}] {el / 60:.0f} min elapsed; last "
                          f"{inst['name']} {meth} s={seed} -> {rec['status']} "
                          f"R={rec.get('residual')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

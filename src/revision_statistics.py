"""Statistics and LaTeX tables for the revised manuscript (Discover Computing).

Single source of truth for every number in the Results section.  Reads the
run records in results/revision/<experiment>/runs.jsonl, drops records with
invalid timing (src.purge_invalid_runs.reason), and writes

  results/revision/statistics.json          every number used in the paper
  results/revision/per_instance_<exp>.csv   per-instance supplement
  paper/revision/tables/*.tex               LaTeX tables (algorithms as columns)

Analysis (Derrac et al. 2011; Benavoli et al. 2016):
  * unit of analysis: the instance.  Per instance and method the 30 runs are
    summarised by the median residual; a run that returned no vector counts as
    an infinitely bad residual, so failures are ranked last, never dropped;
  * Friedman test over instances, mean ranks (average ranks for ties);
  * all pairwise Wilcoxon signed-rank tests on the per-instance medians
    (zero differences split between the signs), Holm-corrected; effect size
    is the matched-pairs rank-biserial correlation;
  * paired bootstrap (resampling instances) of the mean difference in
    log(1 + median residual) between each method and the control.

Usage: python -m src.revision_statistics [--control greedy] [--experiments main ablation exact long]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from src.purge_invalid_runs import reason

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "revision"
TABLES = ROOT / "paper" / "revision" / "tables"

LABELS = {
    "greedy": "Greedy", "rgsa": "RGSA", "rgsa_full": "RGSA full", "blind_sa": "Blind SA",
    "fj": "FJ", "neal": "QUBO SA", "ga": "GA", "cpsat": "CP-SAT", "scip": "SCIP",
}
MAIN_ORDER = ["greedy", "rgsa", "rgsa_full", "blind_sa", "fj", "neal", "ga"]
ABL_ORDER = [f"abl_{a}_{g}_{p}" for a in ("ran", "lag") for g in ("blind", "guided")
             for p in ("none", "converge")]
for k in ABL_ORDER:
    _, a, g, p = k.split("_")
    LABELS[k] = f"{'Lag' if a == 'lag' else 'Rand'}/{'G' if g == 'guided' else 'B'}/{'P' if p == 'converge' else '--'}"
INF = float("inf")
BOOT = 10_000


# ----------------------------------------------------------------- loading
def load(exp):
    path = RES / exp / "runs.jsonl"
    if not path.exists():
        return [], 0
    recs, dropped = {}, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if reason(d):
            dropped += 1
            continue
        recs[(d["instance_name"], d["method"], d["seed"])] = d
    return list(recs.values()), dropped


def canonical(r):
    """Exact residual from a printed one.

    R^2 is an integer, so sqrt(round(R^2)) recovers the exact value.  This
    matters because the C++ programs print 10 significant digits and the
    Python wrappers print full precision: without it, equal residuals from
    different programs (e.g. sqrt(2)) would compare as unequal.
    """
    return math.sqrt(round(float(r) ** 2))


def residual(d):
    if d.get("status") == "error":
        return INF
    sq = d.get("squared_residual")
    if isinstance(sq, int):
        return math.sqrt(sq)
    r = d.get("residual")
    return INF if r is None else canonical(r)


def per_instance(recs, n_seeds):
    """(instance, method) -> summary; only complete cells are returned."""
    cell = collections.defaultdict(list)
    meta = {}
    for d in recs:
        cell[(d["instance_name"], d["method"])].append(d)
        meta[d["instance_name"]] = {"m": d["m"], "n": d["n"], "D": d.get("D"),
                                    "tuning": bool(d.get("tuning_instance"))}
    out = {}
    for (inst, meth), rs in cell.items():
        if len(rs) < n_seeds:
            continue
        R = [residual(d) for d in rs]
        cps = collections.defaultdict(list)
        for d in rs:
            for k, c in enumerate(d.get("checkpoints") or []):
                cps[k].append(INF if c.get("residual") is None else canonical(c["residual"]))
        out[(inst, meth)] = {
            "median": float(np.median(R)), "best": min(R), "mean_finite": (
                float(np.mean([r for r in R if r < INF])) if any(r < INF for r in R) else None),
            "exact": sum(r == 0 for r in R), "runs": len(R),
            "coverage": sum(r < INF for r in R) / len(R),
            "median_vmax": float(np.median([d["max_abs_violation"] for d in rs
                                            if d.get("max_abs_violation") is not None] or [math.nan])),
            "median_ttb": float(np.median([d.get("time_to_best") or 0.0 for d in rs])),
            "checkpoints": [float(np.median(cps[k])) for k in sorted(cps)],
            "statuses": dict(collections.Counter(d.get("status") for d in rs)),
            **meta[inst],
        }
    return out


# -------------------------------------------------------------- statistics
def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def finite(x):
    """Map inf (no vector) to a value above every real residual, for ranking."""
    return np.where(np.isinf(x), 1e300, x)


def rank_biserial(a, b):
    d = finite(np.asarray(a)) - finite(np.asarray(b))
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(summ, methods, control, instances, seed=2026):
    """Friedman + pairwise Wilcoxon/Holm + bootstrap on the given instances."""
    X = np.array([[summ[(i, m)]["median"] for m in methods] for i in instances])
    ranks = np.array([stats.rankdata(finite(row)) for row in X])
    res = {"n_instances": len(instances), "methods": methods,
           "mean_rank": dict(zip(methods, ranks.mean(axis=0).round(4).tolist()))}
    if len(instances) >= 2 and len(methods) >= 3:
        chi2, p = stats.friedmanchisquare(*[finite(X[:, k]) for k in range(len(methods))])
        k, n = len(methods), len(instances)
        ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else INF
        res["friedman"] = {"chi2": float(chi2), "p": float(p),
                           "iman_davenport_F": float(ff), "df1": k - 1, "df2": (k - 1) * (n - 1)}
    pairs, pv = [], []
    for a in range(len(methods)):
        for b in range(a + 1, len(methods)):
            xa, xb = finite(X[:, a]), finite(X[:, b])
            if np.all(xa == xb):
                p = 1.0
            else:
                p = float(stats.wilcoxon(xa, xb, zero_method="zsplit").pvalue)
            pairs.append((methods[a], methods[b]))
            pv.append(p)
    adj = holm(pv)
    res["pairwise"] = [
        {"a": a, "b": b, "p": p, "p_holm": q,
         "rank_biserial": rank_biserial(X[:, methods.index(a)], X[:, methods.index(b)]),
         "a_better": int(np.sum(finite(X[:, methods.index(a)]) < finite(X[:, methods.index(b)]))),
         "b_better": int(np.sum(finite(X[:, methods.index(a)]) > finite(X[:, methods.index(b)]))),
         "ties": int(np.sum(finite(X[:, methods.index(a)]) == finite(X[:, methods.index(b)])))}
        for (a, b), p, q in zip(pairs, pv, adj)]
    # paired bootstrap of the mean difference in log(1 + median R) vs control
    rng = np.random.default_rng(seed)
    L = np.log1p(np.where(np.isinf(X), np.nan, X))
    c = methods.index(control)
    res["bootstrap_vs_control"] = {}
    for k, m in enumerate(methods):
        if k == c:
            continue
        diff = L[:, k] - L[:, c]
        ok = ~np.isnan(diff)
        if ok.sum() < 2:
            continue
        diff = diff[ok]
        idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
        boots = diff[idx].mean(axis=1)
        res["bootstrap_vs_control"][m] = {
            "mean_log1p_diff": float(diff.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_pairs": int(ok.sum()), "unit": "instance"}
    return res


def vs(res, a, b):
    for p in res["pairwise"]:
        if {p["a"], p["b"]} == {a, b}:
            if p["a"] == a:
                return p["a_better"], p["ties"], p["b_better"], p["p_holm"], p["rank_biserial"]
            return p["b_better"], p["ties"], p["a_better"], p["p_holm"], -p["rank_biserial"]
    return None


# ------------------------------------------------------------------ tables
def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x == INF:
        return "fail"
    return f"{x:.{nd}f}"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def by_m_table(summ, methods, instances, control, comp, caption, label, value="median",
               summary=True):
    ms = sorted({summ[(i, methods[0])]["m"] for i in instances})
    cols = "l" + "r" * len(methods)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             "$m$ & " + " & ".join(LABELS[m] for m in methods) + r" \\", r"\midrule"]
    for mm in ms:
        insts = [i for i in instances if summ[(i, methods[0])]["m"] == mm]
        row = []
        for m in methods:
            if value == "median":
                vals = [summ[(i, m)]["median"] for i in insts]
                row.append(fmt(float(np.mean(vals)) if all(v < INF for v in vals) else INF))
            else:  # exact runs
                row.append(str(sum(summ[(i, m)]["exact"] for i in insts)))
        best = None
        if value == "median":
            nums = [float(r) if r not in ("fail", "--") else INF for r in row]
            best = min(nums)
            row = [rf"\textbf{{{r}}}" if r not in ("fail", "--") and float(r) == best else r
                   for r in row]
        lines.append(f"{mm} & " + " & ".join(row) + r" \\")
    if not summary:
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines) + "\n"
    lines.append(r"\midrule")
    lines.append("Mean rank & " + " & ".join(f"{comp['mean_rank'][m]:.2f}" for m in methods) + r" \\")
    wtl, pr, rb = [], [], []
    for m in methods:
        if m == control:
            wtl.append("--"); pr.append("--"); rb.append("--")
            continue
        w, t, l, p, r = vs(comp, m, control)
        wtl.append(f"{w}/{t}/{l}"); pr.append(fmt_p(p)); rb.append(f"{r:+.2f}")
    lines += ["W/T/L & " + " & ".join(wtl) + r" \\",
              "Holm $p$ & " + " & ".join(pr) + r" \\",
              r"$r$ & " + " & ".join(rb) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def checkpoint_table(summ, methods, instances, caption, label):
    fr = ["10\\%", "25\\%", "50\\%", "100\\%"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * len(methods) + "}", r"\toprule",
             "Budget & " + " & ".join(LABELS[m] for m in methods) + r" \\", r"\midrule"]
    out = {}
    for k, f in enumerate(fr):
        X = np.array([[summ[(i, m)]["checkpoints"][k] if len(summ[(i, m)]["checkpoints"]) > k
                       else INF for m in methods] for i in instances])
        mr = np.array([stats.rankdata(finite(r)) for r in X]).mean(axis=0)
        out[f] = dict(zip(methods, mr.round(3).tolist()))
        lines.append(f"{f} & " + " & ".join(f"{v:.2f}" for v in mr) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", out


def exact_table(summ, instances):
    ms = sorted({summ[(i, 'cpsat')]["m"] for i in instances if (i, "cpsat") in summ})
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Time-limited exact solvers (300\,s, one thread, one run per instance; "
             r"4 instances per $(m,D)$ class, 12 per $m$). Exact: instances solved with residual zero; "
             r"Infeas.: band proven to contain no exact solution; Vector: instances with any vector "
             r"in the band at the time limit; $\tilde R$: median best residual over instances with a vector.}"
             r"\label{tab:exact}",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{CP-SAT} & \multicolumn{4}{c}{SCIP} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"$m$ & Exact & Infeas. & Vector & $\tilde R$ & Exact & Infeas. & Vector & $\tilde R$ \\",
             r"\midrule"]
    agg = {}
    for mm in ms:
        row = [str(mm)]
        for s in ("cpsat", "scip"):
            cells = [summ[(i, s)] for i in instances if (i, s) in summ and summ[(i, s)]["m"] == mm]
            ex = sum(c["exact"] > 0 for c in cells)
            inf_ = sum(any("proven_no" in (k or "") for k in c["statuses"]) for c in cells)
            vec = sum(c["coverage"] > 0 for c in cells)
            Rs = [c["median"] for c in cells if c["median"] < INF]
            row += [str(ex), str(inf_), str(vec), fmt(float(np.median(Rs))) if Rs else "--"]
            agg.setdefault(s, {})[mm] = {"instances": len(cells), "exact": ex,
                                        "proven_infeasible_band": inf_, "with_vector": vec}
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", agg


def ablation_stage_summary(recs):
    """Median residual after each stage and median stage times, per variant."""
    out = {}
    by = collections.defaultdict(list)
    for d in recs:
        by[d["method"]].append(d)
    for m, rs in by.items():
        ai, asa, ap = [], [], []
        for d in rs:
            for q in d.get("per_q") or []:
                if q.get("after_init") is not None:
                    ai.append(q["after_init"]); asa.append(q["after_sa"]); ap.append(q["after_polish"])
        out[m] = {"median_after_init": float(np.median(ai)) if ai else None,
                  "median_after_sa": float(np.median(asa)) if asa else None,
                  "median_after_polish": float(np.median(ap)) if ap else None,
                  "median_stage1_s": float(np.median([d.get("stage1_time") or 0 for d in rs])),
                  "median_stage2_s": float(np.median([d.get("stage2_time") or 0 for d in rs])),
                  "median_stage3_s": float(np.median([d.get("stage3_time") or 0 for d in rs])),
                  "polish_incomplete_runs": sum(bool(d.get("polish_incomplete")) for d in rs),
                  "runs": len(rs)}
    return out


def write_csv(path, summ, methods):
    insts = sorted({i for i, _ in summ})
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "m", "n", "D", "tuning_instance", "method", "runs",
                    "median_residual", "best_residual", "exact_runs", "coverage",
                    "median_max_violation", "median_time_to_best_s"])
        for i in insts:
            for m in methods:
                s = summ.get((i, m))
                if s:
                    w.writerow([i, s["m"], s["n"], s["D"], int(s["tuning"]), m, s["runs"],
                                s["median"], s["best"], s["exact"], s["coverage"],
                                s["median_vmax"], s["median_ttb"]])


# -------------------------------------------------------------------- main
def complete_instances(summ, methods):
    insts = sorted({i for i, _ in summ})
    return [i for i in insts if all((i, m) in summ for m in methods)]


def speed_table(path):
    """Python vs C++ table from results/revision/speed_comparison.json (median over seeds)."""
    data = json.loads(path.read_text())
    rows = collections.defaultdict(list)
    for r in data["runs"]:
        rows[(r["instance"], r["config"])].append(r)
    names = {"rgsa": "RGSA", "full_pipeline": "Full pipeline"}
    caption = ("Run time of the Python reference and the C++ implementation on identical settings "
               "(150{,}000 annealing steps, $q=\\lfloor N/2\\rfloor$), instances and seeds, one process "
               "at a time on an otherwise idle Apple M1. Times and speed-ups are medians and residuals "
               "$\\bar R$ are means over three seeds.")
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}\\label{{tab:speed}}",
             "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Instance & Variant & Python (s) & C++ (s) & Speed-up & $\\bar R$ Python & $\\bar R$ C++ \\\\",
             "\\midrule"]
    for (inst, cfg), rs in rows.items():
        py = float(np.median([r["python_seconds"] for r in rs]))
        cc = float(np.median([r["cpp_seconds"] for r in rs]))
        sp = float(np.median([r["python_seconds"] / r["cpp_seconds"] for r in rs]))
        tex_name = inst.replace("_", "\\_")
        lines.append(f"{tex_name} & {names.get(cfg, cfg)} & {py:.1f} & {cc:.3f} & "
                     f"{sp:.0f} & {np.mean([r['python_residual'] for r in rs]):.1f} & "
                     f"{np.mean([r['cpp_residual'] for r in rs]):.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    allr = [r["python_seconds"] / r["cpp_seconds"] for r in data["runs"]]
    summary = {"median_speedup": float(np.median(allr)), "min_speedup": float(np.min(allr)),
               "max_speedup": float(np.max(allr)), "by_config": data["summary"]}
    return "\n".join(lines) + "\n", summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control", default="greedy")
    ap.add_argument("--experiments", nargs="*", default=["main", "ablation", "exact", "long"])
    args = ap.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = {"control": args.control}

    if "main" in args.experiments:
        recs, dropped = load("main")
        summ = per_instance(recs, 30)
        methods = [m for m in MAIN_ORDER if any(k[1] == m for k in summ)]
        insts = complete_instances(summ, methods)
        held = [i for i in insts if not summ[(i, methods[0])]["tuning"]]
        comp_all = compare(summ, methods, args.control, insts)
        comp_held = compare(summ, methods, args.control, held)
        out["main"] = {"records": len(recs), "dropped_invalid": dropped,
                       "complete_instances": len(insts), "all": comp_all, "held_out": comp_held,
                       "exact_runs": {m: sum(summ[(i, m)]["exact"] for i in insts) for m in methods},
                       "exact_instances": {m: sum(summ[(i, m)]["exact"] > 0 for i in insts)
                                           for m in methods},
                       "exact_runs_m_ge_8": {m: sum(summ[(i, m)]["exact"] for i in insts
                                                    if summ[(i, m)]["m"] >= 8) for m in methods},
                       "coverage_min": {m: min(summ[(i, m)]["coverage"] for i in insts)
                                        for m in methods}}
        n = len(insts)
        (TABLES / "main_residual.tex").write_text(by_m_table(
            summ, methods, insts, args.control, comp_all,
            f"Heuristics under equal single-thread budgets ($N/10$\\,s), {'all ' if n == 156 else ''}{n} instances, 30 runs "
            "each. Rows: mean over the instances with that $m$ of the per-instance median residual "
            "$R$ (lower is better; bold: best). Greedy: greedy exchange search with restarts; RGSA full: "
            "full RGSA pipeline; Blind SA: annealing without guidance; FJ: Feasibility Jump; QUBO SA: "
            "QUBO annealing; GA: genetic algorithm. Summary rows: Friedman mean rank (1 = best); "
            f"W/T/L: instances on which the method is better/tied/worse than {LABELS[args.control]}; "
            f"Holm $p$: Holm-corrected Wilcoxon signed-rank $p$ against {LABELS[args.control]}; "
            "$r$: rank-biserial effect size (negative: "
            f"better than {LABELS[args.control]}). Runs without a vector are ranked last.",
            "tab:main"))
        (TABLES / "main_exact.tex").write_text(by_m_table(
            summ, methods, insts, args.control, comp_all,
            "Number of runs (out of 30 per instance, 12 instances per $m$) that found an exact "
            "solution ($R=0$). Column labels as in Table~\\ref{tab:main}.",
            "tab:mainexact", value="exact", summary=False))
        tex, cp = checkpoint_table(summ, methods, insts,
                                   "Anytime behaviour: mean rank of the per-instance median "
                                   "best-so-far residual at fractions of the budget (1 = best).",
                                   "tab:anytime")
        (TABLES / "main_anytime.tex").write_text(tex)
        out["main"]["anytime_mean_rank"] = cp
        write_csv(RES / "per_instance_main.csv", summ, methods)
        descents = collections.defaultdict(list)
        for d in recs:
            if d.get("method") == "greedy" and "restarts" in d:
                descents[d["m"]].append(d["restarts"])
        out["main"]["greedy_descents_median_by_m"] = {int(k): float(np.median(v))
                                                      for k, v in sorted(descents.items())}

    if "ablation" in args.experiments:
        recs, dropped = load("ablation")
        summ = per_instance(recs, 10)
        methods = [m for m in ABL_ORDER if any(k[1] == m for k in summ)]
        insts = complete_instances(summ, methods)
        if insts and len(methods) >= 3:
            ctrl = "abl_lag_guided_converge" if "abl_lag_guided_converge" in methods else methods[-1]
            comp = compare(summ, methods, ctrl, insts)
            out["ablation"] = {"records": len(recs), "dropped_invalid": dropped,
                               "complete_instances": len(insts), "comparison": comp,
                               "stages": ablation_stage_summary(recs)}
            (TABLES / "ablation.tex").write_text(by_m_table(
                summ, methods, insts, ctrl, comp,
                "Ablation, 10 runs per instance. Variant = start (Rand: random, Lag: Lagrangian) / "
                "move (B: blind, G: guided) / polishing (P: to convergence, --: none). "
                f"Rows as in Table~\\ref{{tab:main}}, with comparisons against {LABELS.get(ctrl, ctrl)} "
                "(the full pipeline).", "tab:ablation"))
            write_csv(RES / "per_instance_ablation.csv", summ, methods)

    if "exact" in args.experiments:
        recs, dropped = load("exact")
        summ = per_instance(recs, 1)
        insts = sorted({i for i, _ in summ})
        if insts:
            tex, agg = exact_table(summ, insts)
            (TABLES / "exact.tex").write_text(tex)
            out["exact"] = {"records": len(recs), "dropped_invalid": dropped, "by_m": agg,
                            "statuses": dict(collections.Counter(d.get("status") for d in recs))}
            write_csv(RES / "per_instance_exact.csv", summ, ["cpsat", "scip"])

    if "long" in args.experiments:
        recs, dropped = load("long")
        summ = per_instance(recs, 5)
        methods = [m for m in MAIN_ORDER if any(k[1] == m for k in summ)]
        insts = complete_instances(summ, methods)
        if insts and len(methods) >= 3:
            ctrl = args.control if args.control in methods else methods[0]
            comp = compare(summ, methods, ctrl, insts)
            tex, cp = checkpoint_table(summ, methods, insts,
                                       "Long budget (300\\,s, $m\\ge14$, 5 runs): mean rank of the "
                                       "median best-so-far residual at 10/25/50/100\\% of the budget.",
                                       "tab:long")
            (TABLES / "long.tex").write_text(tex)
            out["long"] = {"records": len(recs), "dropped_invalid": dropped,
                           "complete_instances": len(insts), "comparison": comp,
                           "anytime_mean_rank": cp,
                           "median_residual_300s": {m: float(np.median([summ[(i, m)]["median"]
                                                                        for i in insts]))
                                                    for m in methods}}
            write_csv(RES / "per_instance_long.csv", summ, methods)

    speed = RES / "speed_comparison.json"
    if speed.exists():
        tex, out["speed"] = speed_table(speed)
        (RES / "speed_table.tex").write_text(tex)  # for the response letter, not the paper

    (RES / "statistics.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    m = out.get("main")
    if m:
        print(f"main: {m['complete_instances']} complete instances, {m['dropped_invalid']} invalid "
              f"records ignored")
        print("mean rank:", m["all"]["mean_rank"])
        if "friedman" in m["all"]:
            print("Friedman p =", m["all"]["friedman"]["p"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

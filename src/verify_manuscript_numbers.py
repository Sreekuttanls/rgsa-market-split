"""Check that the numbers quoted in the manuscript text match the run data.

Each check computes a value from results/revision/statistics.json or the
per-instance CSV files, formats it the way the manuscript writes it, and
requires that text to appear in paper/revision/manuscript.tex.  It also checks
that no placeholder or revision-history wording is left and that the abstract
has fewer than 250 words and no mathematical symbols.

Usage: python -m src.verify_manuscript_numbers   (exit status 1 on any failure)
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "revision"
TEX = ROOT / "paper" / "revision" / "manuscript.tex"

FORBIDDEN = r"\\TBD|\bTBD\b|editor|reviewer|original (submission|study|version|manuscript)|earlier work"


def pair(rows, a, b):
    for r in rows:
        if (r["a"], r["b"]) == (a, b):
            return r["a_better"], r["ties"], r["b_better"], r["p_holm"]
        if (r["a"], r["b"]) == (b, a):
            return r["b_better"], r["ties"], r["a_better"], r["p_holm"]
    raise KeyError((a, b))


def csv_rows(name):
    return {(r["instance"], r["method"]): r for r in csv.DictReader(open(RES / name))}


def expected_snippets():
    st = json.loads((RES / "statistics.json").read_text())
    main, held = st["main"]["all"], st["main"]["held_out"]
    out = []
    add = lambda what, text: out.append((what, text))

    mr = main["mean_rank"]
    add("Friedman chi2 main", f"$\\chi^2_6={main['friedman']['chi2']:.1f}$")
    add("mean ranks main", f"({mr['greedy']:.2f}), followed by RGSA ({mr['rgsa']:.2f}) and the full RGSA "
                           f"pipeline ({mr['rgsa_full']:.2f})")
    w, t, l, _ = pair(main["pairwise"], "greedy", "rgsa")
    add("greedy vs rgsa W/T/L", f"better on {w} instances, tied on {t} and worse on {l}")
    bt = main["bootstrap_vs_control"]["rgsa"]
    add("ratio rgsa", f"is {math.exp(bt['mean_log1p_diff']):.3f} for RGSA relative to the greedy search "
                      f"(95\\% CI {math.exp(bt['ci95'][0]):.3f}--{math.exp(bt['ci95'][1]):.3f})")
    for m, name in (("blind_sa", "blind annealing"), ("fj", "Feasibility Jump"),
                    ("ga", "the genetic algorithm"), ("neal", "QUBO annealing")):
        v = math.exp(main["bootstrap_vs_control"][m]["mean_log1p_diff"])
        add(f"ratio {m}", f"{v:.2f} for {name}")
    w, t, l, p = pair(main["pairwise"], "rgsa", "rgsa_full")
    add("rgsa vs full", f"({w} versus {l} instances, {t} ties; Holm $p={p:.2f}$)")
    add("held-out chi2", f"$\\chi^2_6={held['friedman']['chi2']:.1f}$")
    w, t, l, _ = pair(held["pairwise"], "greedy", "rgsa")
    add("held-out W/T/L", f"better on {w} test instances, tied on {t} and worse on {l}")
    w, t, l, _ = pair(main["pairwise"], "rgsa", "blind_sa")
    add("rgsa vs blind", f"it is better on {w} instances, tied on {t}")
    ex = st["main"]["exact_runs"]
    add("exact runs m=3 FJ/QUBO/GA", f"({ex['fj']}, {ex['neal']} and {ex['ga']} runs)")
    any_ = st["main"]["anytime_mean_rank"]["10\\%"]
    add("anytime 10%", f"(mean ranks {any_['greedy']:.2f} and {any_['rgsa']:.2f})")

    ab = st["ablation"]["comparison"]
    add("ablation chi2", f"$\\chi^2_7={ab['friedman']['chi2']:.1f}$")
    g = [v for k, v in ab["mean_rank"].items() if "guided" in k]
    b = [v for k, v in ab["mean_rank"].items() if "blind" in k]
    add("ablation guided ranks", f"between {min(g):.2f} and {max(g):.2f}")
    add("ablation blind ranks", f"between {min(b):.2f} and {max(b):.2f}")
    w, t, l, _ = pair(ab["pairwise"], "abl_ran_guided_none", "abl_ran_blind_none")
    add("ablation guided vs blind", f"better on {w} instances, tied on {t} and never worse")
    others = [r["p_holm"] for r in ab["pairwise"]
              if ("guided" in r["a"]) == ("guided" in r["b"])]
    add("ablation min Holm p (same move)", f"Holm $p\\ge{math.floor(min(others) * 100) / 100:.2f}$")

    E, M = csv_rows("per_instance_exact.csv"), csv_rows("per_instance_main.csv")
    insts = sorted({i for i, _ in E})
    for solver, name in (("cpsat", "CP-SAT"), ("scip", "SCIP")):
        times = [float(E[(i, solver)]["median_time_to_best_s"]) for i in insts
                 if float(E[(i, solver)]["best_residual"]) == 0]
        add(f"{solver} median time", f"{np.median(times):.1f} seconds for {name}")
        a = np.array([float(M[(i, "greedy")]["median_residual"]) for i in insts])
        bb = np.array([float(E[(i, solver)]["best_residual"]) for i in insts])
        add(f"greedy vs {solver}", f"on {int((a < bb).sum())} instances (equal on {int((a == bb).sum())}, "
                                   f"higher on {int((a > bb).sum())}")
    chk = json.loads((RES / "exact_solution_check.json").read_text())
    add("checker", f"{chk['distinct_vectors_checked']} distinct vectors on {chk['instances']} instances")
    assert chk["failed"] == 0

    lg = st["long"]["comparison"]
    w, t, l, p = pair(lg["pairwise"], "greedy", "rgsa")
    add("long W/L", f"to {w} wins and {l} losses")
    add("long Holm", f"(Holm $p={p:.2f}$)")
    bt = lg["bootstrap_vs_control"]["rgsa"]
    add("long ratio", f"is {math.exp(bt['mean_log1p_diff']):.3f} (95\\% CI {math.exp(bt['ci95'][0]):.3f}--"
                      f"{math.exp(bt['ci95'][1]):.3f})")
    mr = lg["mean_rank"]
    add("long ranks", f"({mr['greedy']:.2f}, against {mr['rgsa']:.2f} for RGSA, {mr['rgsa_full']:.2f} for "
                      f"the full pipeline and {mr['fj']:.2f} for Feasibility Jump")
    dm = st["main"]["greedy_descents_median_by_m"]
    add("descents m=15", f"about {round(dm['15'], -2):,.0f}".replace(",", "{,}") + " per run for $m=15$")
    lo, hi = min(dm[k] for k in ("4", "5", "6", "7")), max(dm[k] for k in ("4", "5", "6", "7"))
    add("descents m=4-7", f"between {math.floor(lo / 1000):.0f}{{,}}000 and {math.ceil(hi / 1000):.0f}{{,}}000")
    return out


def main():
    text = TEX.read_text()
    flat = re.sub(r"\s+", " ", text)
    failures = []
    for what, snippet in expected_snippets():
        if re.sub(r"\s+", " ", snippet) not in flat:
            failures.append(f"{what}: expected text not found: {snippet}")
    for m in re.finditer(FORBIDDEN, text, flags=re.I):
        line = text[:m.start()].count("\n") + 1
        failures.append(f"forbidden wording on line {line}: {m.group(0)!r}")
    ab = re.search(r"\\abstract\{(.*?)\}\n", text, re.S).group(1)
    if len(ab.split()) >= 250:
        failures.append(f"abstract has {len(ab.split())} words")
    if "$" in ab:
        failures.append("abstract contains mathematical symbols")
    for f in failures:
        print("FAIL", f)
    n = len(expected_snippets())
    print(f"{n} number checks, {len(failures)} failures; abstract {len(ab.split())} words")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

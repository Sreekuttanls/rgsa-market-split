"""Fill page, line, table and algorithm references in the response letter.

Reads paper/revision/response_to_reviewers.src.md, replaces each <<token>> with
the reference taken from the compiled manuscript (manuscript.aux), and writes
paper/revision/response_to_reviewers.md.  Compile the manuscript first.

Tokens:  <<L:key>>   line label  -> "p. 7, line 160"
         <<B:key>>   reference   -> "ref. 17, p. 23"
         <<tab:key>> <<alg:key>> -> "Table 3 (p. 12)", "Algorithm 2 (p. 8)"
         <<sec:key>>             -> "Section 6.6"
         <<SPEED_ROWS>> <<SPEED_MEDIAN>> <<SPEED_MIN>> <<SPEED_MAX>> <<ABSTRACT_WORDS>>
Exits with status 1 if any token cannot be resolved.

Usage: python -m src.render_response
"""

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "paper" / "revision"
SPEED = ROOT / "results" / "revision" / "speed_comparison.json"


def labels(aux_text):
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", aux_text):
        out[m.group(1)] = (m.group(2), m.group(3))
    # reference numbers come from \bibcite; the page from the label placed after \bibitem
    for m in re.finditer(r"\\bibcite\{([^}]*)\}\{\{?(\d+)", aux_text):
        key = "B:" + m.group(1)
        if key in out:
            out[key] = (m.group(2), out[key][1])
    return out


def speed_tokens():
    data = json.loads(SPEED.read_text())
    groups = collections.OrderedDict()
    for r in data["runs"]:
        groups.setdefault((r["instance"], r["config"]), []).append(r)
    names = {"rgsa": "RGSA", "full_pipeline": "Full pipeline"}
    rows = []
    for (inst, cfg), rs in groups.items():
        py = np.median([r["python_seconds"] for r in rs])
        cc = np.median([r["cpp_seconds"] for r in rs])
        sp = np.median([r["python_seconds"] / r["cpp_seconds"] for r in rs])
        rows.append(f"| {inst} | {names.get(cfg, cfg)} | {py:.1f} | {cc:.3f} | {sp:.0f} |")
    ratios = [r["python_seconds"] / r["cpp_seconds"] for r in data["runs"]]
    return {"SPEED_ROWS": "\n".join(rows), "SPEED_MEDIAN": f"{np.median(ratios):.0f}",
            "SPEED_MIN": f"{min(ratios):.0f}", "SPEED_MAX": f"{max(ratios):,.0f}"}


def render(src, labs, extra):
    missing = []

    def sub(m):
        tok = m.group(1)
        if tok in extra:
            return extra[tok]
        if tok not in labs:
            missing.append(tok)
            return m.group(0)
        num, page = labs[tok]
        kind = tok.split(":", 1)[0]
        if kind == "L":
            return f"p. {page}, line {num}"
        if kind == "B":
            return f"ref. {num}, p. {page}"
        if kind == "tab":
            return f"Table {num} (p. {page})"
        if kind == "alg":
            return f"Algorithm {num} (p. {page})"
        if kind == "sec":
            return f"Section {num}"
        missing.append(tok)
        return m.group(0)

    return re.sub(r"<<([^<>]+)>>", sub, src), missing


def main():
    labs = labels((REV / "manuscript.aux").read_text())
    tex = (REV / "manuscript.tex").read_text()
    abstract = re.search(r"\\abstract\{(.*?)\}\n", tex, re.S).group(1)
    extra = speed_tokens()
    extra["ABSTRACT_WORDS"] = str(len(abstract.split()))
    out, missing = render((REV / "response_to_reviewers.src.md").read_text(), labs, extra)
    if missing:
        print("unresolved:", ", ".join(sorted(set(missing))))
        return 1
    (REV / "response_to_reviewers.md").write_text(out)
    n_refs = len(re.findall(r"p\. \d+", out))
    print(f"wrote response_to_reviewers.md ({n_refs} page references)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

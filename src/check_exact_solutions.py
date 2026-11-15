"""Confirm every claimed exact solution with QOBLIB's own checker.

Every run record with residual zero is passed, as a 0/1 string, to the
QOBLIB Rust checker (data/01-marketsplit/check).  Distinct (instance, vector)
pairs are checked once.  Exits with status 1 if any claimed solution fails.

Usage: python -m src.check_exact_solutions [--experiments main ablation exact long]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK_DIR = ROOT / "data" / "01-marketsplit" / "check"
CHECKER = CHECK_DIR / "target" / "release" / "check_marketsplit"
INST = ROOT / "data" / "01-marketsplit" / "instances"
RES = ROOT / "results" / "revision"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiments", nargs="*", default=["main", "ablation", "exact", "long"])
    args = ap.parse_args(argv)
    if not CHECKER.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=CHECK_DIR, check=True)
    claims, missing_x = {}, 0
    per_exp = {}
    for exp in args.experiments:
        n = 0
        for line in (RES / exp / "runs.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("residual") != 0:
                continue
            n += 1
            if not d.get("x"):
                missing_x += 1
                continue
            claims.setdefault((d["instance_name"], d["x"]), []).append(exp)
        per_exp[exp] = n
    failed = []
    for (inst, x), _ in sorted(claims.items()):
        out = subprocess.run([str(CHECKER), str(INST / f"{inst}.dat"), x],
                             capture_output=True, text=True)
        if out.returncode != 0:
            failed.append((inst, x, out.stdout + out.stderr))
    summary = {"claimed_exact_runs": per_exp, "distinct_vectors_checked": len(claims),
               "instances": len({i for i, _ in claims}), "runs_without_vector": missing_x,
               "failed": len(failed)}
    print(json.dumps(summary, indent=2))
    (RES / "exact_solution_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    for inst, x, msg in failed[:10]:
        print("FAILED", inst, x, msg.strip()[-300:])
    return 1 if failed or missing_x else 0


if __name__ == "__main__":
    sys.exit(main())

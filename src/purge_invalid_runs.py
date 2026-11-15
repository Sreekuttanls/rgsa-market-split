"""Remove run records whose timing is invalid, so the resumable runner redoes them.

A record is invalid when
  * the solver's own clock ran past the budget by more than 10 % + 0.1 s while
    the solver itself enforces the limit (rgsa, rgsa_full, blind_sa, greedy):
    this only happens when the computer went to sleep during the run;
  * the solver's clock reads more than 0.5 s above the runner's wall clock
    (the solver's clock counted a sleep that the runner's did not); or
  * an external solver (fj, neal, ga) without the deadline guard reported an
    answer found after the time limit.

The original file is kept as runs.jsonl.before_purge_<timestamp>.

Optional extra rules:
  --contended-lines N   drop every record of any instance that has a record
                        among the first N lines (runs that shared the machine
                        with another job); the whole instance is redone so all
                        methods on it run under the same conditions.
  --drop-method M       drop every record of method M (e.g. after re-tuning M).

Usage: python -m src.purge_invalid_runs results/revision/main/runs.jsonl [--apply]
"""

import argparse
import collections
import json
import shutil
import time
from pathlib import Path

SELF_TIMED = {"rgsa", "rgsa_full", "blind_sa", "greedy"}
GUARDED = {"fj", "neal", "ga", "cpsat", "scip"}


def reason(d):
    tl, tt = d.get("time_limit"), d.get("total_time")
    if tl is None or tt is None:
        return None
    wall = d.get("wall")
    if wall is not None and tt > wall + 0.5:
        return "clock counted sleep"
    method = d.get("method")
    ttb = d.get("time_to_best")
    if ttb is not None and ttb > tl + 1e-9:
        return "answer found after deadline"
    if (method in SELF_TIMED or str(method).startswith("abl_")) and tt > 1.1 * tl + 0.1:
        return "overran budget (sleep)"
    if method in GUARDED and not d.get("deadline_enforced"):
        ttb = d.get("time_to_best")
        if ttb is not None and ttb > tl:
            return "answer found after deadline"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--contended-lines", type=int, default=0)
    ap.add_argument("--drop-method", action="append", default=[])
    args = ap.parse_args()
    lines = [l for l in args.path.read_text().splitlines() if l.strip()]
    contended = {json.loads(l).get("instance_name") for l in lines[:args.contended_lines]}
    keep, drop = [], collections.Counter()
    for line in lines:
        d = json.loads(line)
        r = reason(d)
        if r is None and d.get("method") in args.drop_method:
            r = "method re-tuned"
        if r is None and d.get("instance_name") in contended:
            r = "shared machine with another job"
        if r:
            drop[(d.get("method"), r)] += 1
        else:
            keep.append(line)
    for k, v in sorted(drop.items()):
        print(f"{k[0]:10s} {k[1]:30s} {v}")
    print(f"total invalid: {sum(drop.values())}; kept: {len(keep)}")
    if args.apply and drop:
        backup = args.path.with_name(args.path.name + f".before_purge_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(args.path, backup)
        args.path.write_text("\n".join(keep) + "\n")
        print(f"backup: {backup}")


if __name__ == "__main__":
    main()

#!/bin/zsh
# Resumes the long experiment after the 11 Nov pause, with purge-and-rerun passes.
cd "/Users/sreekuttan/Documents/Bloq/Bloq Research/rgsa-market-split"
PY=.venv/bin/python
LOG=results/revision/after_main.log
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
say "long resumed after pause ($(wc -l < results/revision/long/runs.jsonl) records)"
for pass in 1 2 3; do
  say "long pass $pass"
  $PY -m src.run_revision_experiments --experiment long --workers 4 >> results/revision/long_run.log 2>&1
  out=$($PY -m src.purge_invalid_runs results/revision/long/runs.jsonl --apply)
  echo "$out" >> $LOG
  echo "$out" | grep -q "total invalid: 0;" && break
done
say "LONG DONE"

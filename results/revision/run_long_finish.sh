#!/bin/zsh
# Waits for mains power (battery low-power mode slows the CPU and would bias timings),
# then redoes the long runs invalidated by sleep.
cd "/Users/sreekuttan/Documents/Bloq/Bloq Research/rgsa-market-split"
PY=.venv/bin/python
LOG=results/revision/after_main.log
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
say "long finish: waiting for AC power"
until pmset -g batt | grep -q "AC Power"; do sleep 30; done
for pass in 1 2 3; do
  say "long finish pass $pass"
  caffeinate -i $PY -m src.run_revision_experiments --experiment long --workers 4 >> results/revision/long_run.log 2>&1
  out=$($PY -m src.purge_invalid_runs results/revision/long/runs.jsonl --apply)
  echo "$out" >> $LOG
  echo "$out" | grep -q "total invalid: 0;" && break
done
say "LONG FINISH DONE"

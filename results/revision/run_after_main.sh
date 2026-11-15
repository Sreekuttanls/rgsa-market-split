#!/bin/zsh
# Runs after the main experiment: clean Neal retune, purge invalid runs,
# rerun them, then the remaining experiments.  Each step is logged.
cd "/Users/sreekuttan/Documents/Bloq/Bloq Research/rgsa-market-split"
PY=.venv/bin/python
LOG=results/revision/after_main.log
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
while pgrep -f "run_revision_experiments --experiment main" >/dev/null; do sleep 60; done
say "main finished: $(wc -l < results/revision/main/runs.jsonl) records"
say "neal retune on idle machine"
$PY -m src.tune --method neal --study-name neal_range2_idle --trials 60 --workers 4 > results/revision/tuning_neal_idle.log 2>&1 || { say "neal tune FAILED"; exit 1; }
$PY -m src.tune --collect >> results/revision/tuning_neal_idle.log 2>&1
say "tuned_config: $(tr -d '\n ' < results/revision/tuned_config.json | cut -c1-400)"
$PY -m src.purge_invalid_runs results/revision/main/runs.jsonl --contended-lines 6650 --drop-method neal --apply >> $LOG 2>&1
for exp in main ablation exact long; do
  for pass in 1 2 3; do
    say "$exp pass $pass"
    $PY -m src.run_revision_experiments --experiment $exp --workers 4 >> results/revision/${exp}_run.log 2>&1
    out=$($PY -m src.purge_invalid_runs results/revision/$exp/runs.jsonl --apply)
    echo "$out" >> $LOG
    echo "$out" | grep -q "total invalid: 0;" && break
  done
done
say "ALL DONE"

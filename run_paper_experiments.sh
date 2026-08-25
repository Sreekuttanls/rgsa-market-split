#!/usr/bin/env bash
# ==============================================================================
# RGSA Paper — Master Reproduction Script
# ==============================================================================
# Reproduces all experimental results reported in the paper.
# Requirements: Python 3.9–3.12 and NumPy 1.26.4
# Estimated runtime: ~30-40 minutes on a modern laptop (single core)
# Set PYTHON to another interpreter when needed, for example PYTHON=python3.
# This script intentionally reruns the full benchmark; CI uses the lightweight
# validation commands documented in README.md instead.
# ============================================================================
set -e

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "  RGSA FOR MARKET SPLIT — PAPER REPRODUCTION"
echo "=============================================================================="
echo ""
echo "Architecture: $(uname -m 2>/dev/null || echo unknown)"
echo "Python:      $($PYTHON --version)"
echo "NumPy:       $($PYTHON -c 'import numpy; print(numpy.__version__)')"
echo ""

# Step 0: Unit tests
echo ">>> Step 0/4: Running unit tests..."
$PYTHON -m unittest discover -s tests -v
echo ""

# Step 1: Benchmark D=50 flagship instances (m=9..15)
echo ">>> Step 1/4: Running D=50 benchmark (7 instances × 4 configs × 5 seeds = 140 runs)..."
$PYTHON src/run_paper_benchmarks.py \
  --instances 'ms_*_050_000.dat' \
  --seeds '42,43,44,45,46' \
  --configs 'full_esa,random_blind_sa,random_guided_sa,lagrangian_blind_sa' \
  --n-steps 150000 \
  --output-dir results/paper_benchmarks_d50
echo ""

# Step 2: Benchmark D=200 flagship instance (m=15)
echo ">>> Step 2/4: Running D=200 benchmark (1 instance × 4 configs × 5 seeds = 20 runs)..."
$PYTHON src/run_paper_benchmarks.py \
  --instances 'ms_15_200_000.dat' \
  --seeds '42,43,44,45,46' \
  --configs 'full_esa,random_blind_sa,random_guided_sa,lagrangian_blind_sa' \
  --n-steps 150000 \
  --output-dir results/paper_benchmarks_d200
echo ""

# Step 3: Merge results while retaining publication protocol metadata
echo ">>> Step 3/4: Merging results and computing statistics..."
$PYTHON - <<'PY'
import json
import os

source_paths = [
    'results/paper_benchmarks_d50/paper_benchmark_results.json',
    'results/paper_benchmarks_d200/paper_benchmark_results.json',
]
all_runs = []
source_payload = None
for path in source_paths:
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if source_payload is None:
        source_payload = payload
    all_runs.extend(payload['runs'])

merged = dict(source_payload)
merged.update({
    'artifact_version': '1.0',
    'source_artifacts': source_paths,
    'instances': sorted({run['instance'] for run in all_runs}),
    'seeds': sorted({run['seed'] for run in all_runs}),
    'configs': sorted({run['config_name'] for run in all_runs}),
    'num_runs': len(all_runs),
    'runs': all_runs,
})
os.makedirs('results/paper_benchmarks', exist_ok=True)
with open('results/paper_benchmarks/paper_benchmark_results.json', 'w', encoding='utf-8') as handle:
    json.dump(merged, handle, indent=2)
    handle.write('\n')
print(f"  Merged {len(all_runs)} runs.")
PY

# Step 4: Generate LaTeX tables
mkdir -p paper/tables
$PYTHON src/paper_statistics.py \
  --input results/paper_benchmarks/paper_benchmark_results.json \
  --output-dir paper/tables
echo ""

echo "=============================================================================="
echo "  REPRODUCTION COMPLETE"
echo "  Results: results/paper_benchmarks/"
echo "  Tables:  paper/tables/"
echo "  Paper:   paper/esa_paper.tex (RGSA manuscript; filename kept for continuity)"
echo "=============================================================================="

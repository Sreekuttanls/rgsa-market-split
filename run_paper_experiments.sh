#!/usr/bin/env bash
# ==============================================================================
# RGSA Paper — Master Reproduction Script
# ==============================================================================
# Reproduces all experimental results reported in the paper.
# Requirements: Python 3.8+, numpy, scipy
# Estimated runtime: ~30-40 minutes on a modern laptop (single core)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "  RGSA FOR MARKET SPLIT — PAPER REPRODUCTION"
echo "=============================================================================="
echo ""
echo "Hardware: $(uname -m) $(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1 | cut -d: -f2)"
echo "Python:   $(python3 --version)"
echo "NumPy:    $(python3 -c 'import numpy; print(numpy.__version__)')"
echo ""

# Step 0: Unit tests
echo ">>> Step 0/4: Running unit tests..."
python3 -m unittest discover -s tests -v
echo ""

# Step 1: Benchmark D=50 flagship instances (m=9..15)
echo ">>> Step 1/4: Running D=50 benchmark (7 instances × 4 configs × 5 seeds = 140 runs)..."
python3 src/run_paper_benchmarks.py \
  --instances 'ms_*_050_000.dat' \
  --seeds '42,43,44,45,46' \
  --configs 'full_esa,random_blind_sa,random_guided_sa,lagrangian_blind_sa' \
  --n-steps 150000 \
  --output-dir results/paper_benchmarks_d50
echo ""

# Step 2: Benchmark D=200 flagship instance (m=15)
echo ">>> Step 2/4: Running D=200 benchmark (1 instance × 4 configs × 5 seeds = 20 runs)..."
python3 src/run_paper_benchmarks.py \
  --instances 'ms_15_200_000.dat' \
  --seeds '42,43,44,45,46' \
  --configs 'full_esa,random_blind_sa,random_guided_sa,lagrangian_blind_sa' \
  --n-steps 150000 \
  --output-dir results/paper_benchmarks_d200
echo ""

# Step 3: Merge results
echo ">>> Step 3/4: Merging results and computing statistics..."
python3 - <<'PY'
import json, os
all_runs = []
for path in ['results/paper_benchmarks_d50/paper_benchmark_results.json',
             'results/paper_benchmarks_d200/paper_benchmark_results.json']:
    d = json.load(open(path))
    all_runs.extend(d['runs'])
os.makedirs('results/paper_benchmarks', exist_ok=True)
with open('results/paper_benchmarks/paper_benchmark_results.json', 'w') as f:
    json.dump({"num_runs": len(all_runs), "runs": all_runs}, f, indent=2)
print(f"  Merged {len(all_runs)} runs.")
PY

# Step 4: Generate LaTeX tables
mkdir -p paper/tables
python3 src/paper_statistics.py \
  --input results/paper_benchmarks/paper_benchmark_results.json \
  --output-dir paper/tables
echo ""

echo "=============================================================================="
echo "  REPRODUCTION COMPLETE"
echo "  Results: results/paper_benchmarks/"
echo "  Tables:  paper/tables/"
echo "  Paper:   paper/esa_paper.tex (RGSA manuscript; filename kept for continuity)"
echo "=============================================================================="

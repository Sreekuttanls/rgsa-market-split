#!/usr/bin/env bash
# ==============================================================================
# Residual-Guided Simulated Annealing (RGSA) - Master Reproduction Script
# ==============================================================================
set -e

echo "=============================================================================="
echo "      RGSA - INDEPENDENT AUDIT & REPRODUCTION                                 "
echo "=============================================================================="

echo ""
echo ">>> Step 1/3: Running Complete Audit & Sum Verification on N=140 (ms_15_200_000.dat)..."
python3 src/audit_n140.py

echo ""
echo ">>> Step 2/3: Running 5-Configuration Component Ablation Study..."
python3 src/run_ablation_study.py

echo ""
echo ">>> Step 3/3: Running Multi-Scale QOBLIB Benchmark Suite..."
python3 src/benchmark_qoblib_suite.py

echo ""
echo "=============================================================================="
echo "  REPRODUCTION COMPLETE: All mathematical invariants and benchmarks verified. "
echo "=============================================================================="

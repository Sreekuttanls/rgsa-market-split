# Residual-Guided Exchange Search for Multi-Dimensional Subset Sum / Market Split

This repository contains the implementation, the record of every experimental
run, and the analysis and reproduction checks for the manuscript **Residual-Guided
Exchange Search for Multi-Dimensional Subset Sum: Greedy Restarts versus
Simulated Annealing** (submitted to *Discover Computing*; earlier title:
*Residual-Guided Simulated Annealing for Multi-Dimensional Subset Sum*).

## Revised study (current)

| What | Where |
|---|---|
| C++ solvers (RGSA, greedy search, blind SA; Feasibility Jump driver) | `cpp/` (`make -C cpp`) |
| Python reference implementation | `src/rgsa_solver.py` |
| Experiment runner (resumable) | `python -m src.run_revision_experiments --experiment main\|ablation\|exact\|long` |
| External solvers (FJ, QUBO annealing, GA, CP-SAT, SCIP) | `src/external_solvers.py` |
| Parameter tuning (Optuna) | `python -m src.tune` |
| Discard runs made under invalid conditions | `python -m src.purge_invalid_runs` |
| Statistics and generated tables | `python -m src.revision_statistics` → `results/revision/statistics.json`, `paper/revision/tables/` |
| Confirm exact solutions with the QOBLIB checker | `python -m src.check_exact_solutions` |
| Check the numbers quoted in the manuscript | `python -m src.verify_manuscript_numbers` |
| Run records (one JSON line per run) | `results/revision/{main,ablation,exact,long}/runs.jsonl` |
| Manuscript and response letter | `paper/revision/` |

Tests: `python -m unittest discover -s tests`.

The sections below describe the original study and its committed data.

The author is **Sreekuttan L. S.**, Bloq Quantum, India. Contact:
`sreekuttan@bloq.in`.

RGSA solves fixed-cardinality QOBLIB Market Split instances with a
residual-guided simulated-annealing neighborhood. For a selected item with
attribute vector `w_i` and current residual `Delta = A x - b`, the guided swap
searches for an available vector near `w_i - Delta`. The move preserves
cardinality while using residual geometry to direct exploration.

## Quickstart: validate the published results

The committed benchmark result is already included. No benchmark rerun is
needed to verify the paper tables and headline claims.

Requires Python 3.9–3.12 and NumPy 1.26.4:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.validate_publication
python -m src.verify_claims
python -m src.publication_smoke
```

The checks validate 160 committed runs, recompute the manuscript-facing claims,
and regenerate all six LaTeX tables in a temporary directory to detect stale
artifacts. They do not run the expensive solver benchmark.

Tables may be regenerated explicitly:

```text
python -m src.paper_statistics \
  --input results/paper_benchmarks/paper_benchmark_results.json \
  --output-dir paper/tables
```

The full benchmark runner is available for independent reproduction. It is
expected to take approximately 30–40 minutes and is intentionally not run by
CI:

```text
python src/run_paper_benchmarks.py --help
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the historical protocol,
artifact provenance, and the distinction between lightweight validation and a
full benchmark rerun.

## Main published experiment

The primary result file is:

`results/paper_benchmarks/paper_benchmark_results.json`

It contains 160 runs:

- 8 selected QOBLIB instances;
- 4 configurations;
- 5 common seeds (42–46);
- 150,000 SA steps per run;
- 3,000 Lagrangian iterations;
- cardinality `A.shape[1] // 2`.

The four raw configuration identifiers are retained for reproducibility:

- `random_blind_sa` — Blind SA;
- `lagrangian_blind_sa` — Lagrangian + Blind SA;
- `random_guided_sa` — RGSA;
- `full_esa` — Full RGSA pipeline.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE                         # Original RGSA code: MIT
├── LICENSE.data                    # Bundled QOBLIB data: CC BY 4.0
├── THIRD_PARTY_NOTICES.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── data/01-marketsplit/            # QOBLIB Market Split material
├── src/
│   ├── rgsa_solver.py              # Core three-stage solver
│   ├── run_paper_benchmarks.py     # Full benchmark runner
│   ├── paper_statistics.py         # LaTeX table generation
│   ├── validate_publication.py     # Result schema and protocol checks
│   ├── verify_claims.py            # Manuscript-facing claim checks
│   └── publication_smoke.py        # Non-mutating release smoke checks
├── tests/
├── paper/esa_paper.tex
├── paper/tables/                    # Generated publication tables
└── results/paper_benchmarks/       # Canonical committed result artifact
```

## Data and licensing

The original RGSA source code is licensed under the MIT License. The bundled
QOBLIB instances and related third-party files retain their upstream licensing
and attribution requirements. QOBLIB identifies code as Apache-2.0 and instance
/raw data as CC BY 4.0. See `LICENSE.data` and `THIRD_PARTY_NOTICES.md`.

Please cite QOBLIB as:

> T. Koch et al., “The Quantum Optimization Benchmarking Library,” *Nature
> Computational Science*, 6, 653–671 (2026),
> https://doi.org/10.1038/s43588-026-00991-1.

## Citation

See `CITATION.cff`. The final public release should replace the current
anonymous review URL with the archival repository URL and DOI once minted.

# Reproducibility notes

The committed benchmark artifact contains the 160 runs used by the manuscript:
8 selected QOBLIB instances × 4 configurations × 5 common seeds. The repository
also contains the source scripts used to regenerate the benchmark, but the
full benchmark is intentionally not part of the lightweight CI or publication
smoke checks.

## Validate the committed results

From the repository root, after installing the pinned dependency:

```text
python -m unittest discover -s tests -v
python -m src.validate_publication
python -m src.verify_claims
python -m src.publication_smoke
```

These commands validate the JSON schema and run structure, recompute the
manuscript-facing claims, and regenerate the six LaTeX tables in a temporary
directory to detect stale committed artifacts. They do not run the solver
benchmark.

## Regenerate tables

```text
python -m src.paper_statistics \
  --input results/paper_benchmarks/paper_benchmark_results.json \
  --output-dir paper/tables
```

## Full benchmark

The full benchmark requires approximately 30–40 minutes on the documented
single-core environment and is not required to audit the committed results.
The historical protocol uses five seeds (42–46), 150,000 SA steps per run,
3,000 Lagrangian iterations, and cardinality `A.shape[1] // 2`.

```text
python src/run_paper_benchmarks.py --help
```

The exact wall-clock times in the manuscript are hardware-dependent. Residuals,
configuration definitions, seeds, and protocol metadata are the publication
record; runtime values should not be interpreted as hardware-independent speed
claims.

## Provenance

The primary data file is
`results/paper_benchmarks/paper_benchmark_results.json`. It was assembled from
the D=50 and D=200 result artifacts listed in its `source_artifacts` metadata.
QOBLIB instances are third-party data; see `LICENSE.data` and
`THIRD_PARTY_NOTICES.md`.

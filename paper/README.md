# Manuscript package

`esa_paper.tex` is the manuscript source. The six files under `tables/` are
generated from the canonical benchmark artifact by `src/paper_statistics.py`.
Do not edit generated tables by hand.

## Regenerate publication tables

From the repository root:

```text
python -m src.paper_statistics \
  --input results/paper_benchmarks/paper_benchmark_results.json \
  --output-dir paper/tables
```

## Build the manuscript

The revised manuscript is `paper/revision/manuscript.tex`, built with the
Springer Nature template (`sn-jnl.cls` and `sn-mathphys-num.bst` are in the same
folder): run `pdflatex manuscript.tex` three times there. Regenerate the tables
first with `python -m src.revision_statistics`, and fill the page and line
references of the response letter with `python -m src.render_response`. The
original manuscript below is kept for reference. The source in this
repository uses standard LaTeX packages and can be compiled locally when a
LaTeX distribution is installed:

```text
pdflatex -interaction=nonstopmode esa_paper.tex
pdflatex -interaction=nonstopmode esa_paper.tex
```

The repository’s lightweight CI validates source structure, table freshness,
metadata, and numerical claims. It does not require a LaTeX installation and
does not rerun the benchmark.

## Submission variants

- The current URL is an anonymized review mirror.
- A camera-ready/public version should replace the review URL with the final
  repository URL and persistent archive DOI.
- The manuscript contains the named author and affiliation supplied for the
  publication package; use the journal’s peer-review policy to decide whether
  the review submission must temporarily restore anonymized author metadata.

#!/usr/bin/env python3
"""Run lightweight publication checks without rerunning experiments."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.paper_statistics import load_results, write_latex_tables
from src.validate_publication import DEFAULT_RESULTS, validate_results

DEFAULT_TABLE_DIR = PACKAGE_ROOT / "paper" / "tables"
DEFAULT_MANUSCRIPT = PACKAGE_ROOT / "paper" / "esa_paper.tex"
EXPECTED_TABLES = (
    "comprehensive_table.tex",
    "scaling_table.tex",
    "ablation_table.tex",
    "literature_comparison.tex",
    "best_residuals_table.tex",
    "violations_table.tex",
)


def check_table_artifacts(results_path: Path, table_dir: Path) -> None:
    runs = load_results(results_path)
    from src.paper_statistics import compute_per_class_statistics

    with tempfile.TemporaryDirectory(prefix="rgsa-publication-") as temporary:
        generated_dir = Path(temporary)
        write_latex_tables(
            runs,
            compute_per_class_statistics(runs),
            generated_dir,
        )
        for filename in EXPECTED_TABLES:
            committed = table_dir / filename
            fresh = generated_dir / filename
            if not committed.exists():
                raise AssertionError(f"missing generated table: {committed}")
            if committed.read_text(encoding="utf-8") != fresh.read_text(encoding="utf-8"):
                raise AssertionError(f"generated table is stale: {committed}")


def check_manuscript(manuscript_path: Path) -> None:
    text = manuscript_path.read_text(encoding="utf-8")
    required_inputs = {
        "tables/comprehensive_table.tex",
        "tables/scaling_table.tex",
        "tables/ablation_table.tex",
        "tables/literature_comparison.tex",
        "tables/best_residuals_table.tex",
        "tables/violations_table.tex",
    }
    for include in required_inputs:
        if f"\\input{{{include}}}" not in text:
            raise AssertionError(f"manuscript does not include {include}")
    if "\\begin{document}" not in text or "\\end{document}" not in text:
        raise AssertionError("manuscript is not a complete LaTeX document")

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
    if abstract_match is None:
        raise AssertionError("manuscript has no abstract")
    abstract_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", abstract_match.group(1))
    if not 150 <= len(abstract_words) <= 250:
        raise AssertionError(f"abstract has {len(abstract_words)} words; expected 150–250")

    keyword_match = re.search(r"\\textbf\{Keywords:\}(.*)", text)
    if keyword_match is None:
        raise AssertionError("manuscript has no keywords")
    keyword_count = len([item for item in keyword_match.group(1).split(";") if item.strip()])
    if not 4 <= keyword_count <= 6:
        raise AssertionError(f"manuscript has {keyword_count} keywords; expected 4–6")

    for stale in ("Anonymous", "XXXX", "esa_solver.py", "environment manifests"):
        if stale in text:
            raise AssertionError(f"stale manuscript placeholder: {stale}")
    for required in ("Sreekuttan L. S.", "Bloq Quantum", "sreekuttan@bloq.in"):
        if required not in text:
            raise AssertionError(f"missing named-author metadata: {required}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_results(args.json)
    check_table_artifacts(args.json, args.tables)
    check_manuscript(args.manuscript)
    print("Publication smoke checks passed")
    print(f"  validated runs: {summary['runs']}")
    print(f"  checked tables: {len(EXPECTED_TABLES)}")
    print(f"  manuscript: {args.manuscript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

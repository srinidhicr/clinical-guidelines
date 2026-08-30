"""AC-08 tests for the committed, clause-resolvable golden evaluation set."""

from __future__ import annotations

from pathlib import Path

from eval.golden_set import load_golden_set, validate_expected_clauses
from src.ingestion.loaders import load_corpus


def test_golden_set_has_min_20_entries_with_expected_context() -> None:
    """AC-08: committed questions cover grounded, multi-clause, and abstention cases."""
    root = Path(__file__).resolve().parents[1]
    entries = load_golden_set(root / "data" / "golden_eval_set.json")
    available = {section.clause_id for section in load_corpus(root / "data" / "raw")}
    validate_expected_clauses(entries, available)

    assert len(entries) >= 20
    assert {"single_clause", "multi_clause", "abstain"} <= {entry["type"] for entry in entries}
    assert all(entry["reference_answer"] for entry in entries)

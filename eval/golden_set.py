"""Golden-set validation and deterministic retrieval metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"id", "question", "type", "expected_context", "reference_answer", "ac_ref"}
VALID_TYPES = {"single_clause", "multi_clause", "abstain"}


@dataclass(frozen=True)
class RetrievalMetric:
    question_id: str
    context_precision: float
    context_recall: float


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Load and structurally validate committed evaluation examples."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) < 20:
        raise ValueError("Golden evaluation set must contain at least 20 entries.")
    ids: set[str] = set()
    for entry in entries:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Golden entry is missing required fields: {sorted(missing)}")
        if entry["id"] in ids:
            raise ValueError(f"Duplicate golden-set ID: {entry['id']}")
        if entry["type"] not in VALID_TYPES:
            raise ValueError(f"Invalid golden-set type: {entry['type']}")
        if not isinstance(entry["expected_context"], list):
            raise ValueError(f"Expected context must be a list for {entry['id']}")
        if entry["type"] == "abstain" and entry["expected_context"]:
            raise ValueError(f"Abstention case {entry['id']} cannot name expected context.")
        if entry["type"] != "abstain" and not entry["expected_context"]:
            raise ValueError(f"Grounded case {entry['id']} needs expected context.")
        ids.add(entry["id"])
    return entries


def validate_expected_clauses(entries: list[dict[str, Any]], available_clause_ids: set[str]) -> None:
    """Ensure every golden reference remains valid after corpus edits."""
    missing = {
        clause_id
        for entry in entries
        for clause_id in entry["expected_context"]
        if clause_id not in available_clause_ids
    }
    if missing:
        raise ValueError(f"Golden set references missing clauses: {sorted(missing)}")


def score_retrieval(expected_clause_ids: list[str], retrieved_clause_ids: list[str]) -> tuple[float, float]:
    """Compute deterministic clause-level precision and recall for one result list."""
    expected = set(expected_clause_ids)
    retrieved = set(retrieved_clause_ids)
    if not expected:
        return (1.0 if not retrieved else 0.0, 1.0)
    precision = len(expected & retrieved) / len(retrieved) if retrieved else 0.0
    recall = len(expected & retrieved) / len(expected)
    return precision, recall

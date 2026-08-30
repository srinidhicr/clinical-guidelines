"""NFR-08 provenance-only logging tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.generation.generator import abstain
from src.generation.schema import ClauseCitation, GuidelineAnswer
from src.utils.logging import configure_provenance_logger, log_answer, log_generation_failure


def test_query_logged_with_provenance(tmp_path: Path) -> None:
    """NFR-08: audit records include citations but never accept or store query prose."""
    logger = configure_provenance_logger(tmp_path)
    answer = GuidelineAnswer(
        answer="Synthetic evidence answer.",
        citations=[
            ClauseCitation(
                document_id="CARD-001",
                guideline_source="SYN-GUIDE-CARD-001",
                clause_id="CARD-001-S2",
                section="Section 2: First-Line Management",
            )
        ],
        guideline_source="SYN-GUIDE-CARD-001",
        section="Section 2: First-Line Management",
        strength_of_recommendation="Strong",
        grounding_confidence=0.9,
        grounded=True,
    )
    log_answer(logger, "request-123", answer)
    secret_like_error = RuntimeError("request contents=patient question should never be logged")
    log_generation_failure(logger, "request-124", secret_like_error)

    for handler in logger.handlers:
        handler.flush()
    events = [json.loads(line) for line in (tmp_path / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]

    assert events[0]["citation_clause_ids"] == ["CARD-001-S2"]
    assert events[1]["error_type"] == "RuntimeError"
    assert "Synthetic evidence answer" not in (tmp_path / "provenance.jsonl").read_text(encoding="utf-8")
    assert "patient question" not in (tmp_path / "provenance.jsonl").read_text(encoding="utf-8")


def test_reconfiguring_logger_replaces_old_file_handler(tmp_path: Path) -> None:
    """Changing a process-wide logger destination does not duplicate events or retain handles."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_logger = configure_provenance_logger(first_dir)
    log_answer(first_logger, "first-request", abstain())
    second_logger = configure_provenance_logger(second_dir)
    log_answer(second_logger, "second-request", abstain())
    for handler in second_logger.handlers:
        handler.flush()

    first_events = (first_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    second_events = (second_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(second_logger.handlers) == 1
    assert len(first_events) == 1
    assert len(second_events) == 1
    assert "second-request" in second_events[0]

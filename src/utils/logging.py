"""Privacy-aware operational logging for provenance and provider failures.

Logs deliberately exclude raw user questions, prompts, model responses, and credentials.
They retain only a request ID, result state, confidence, citations, and safe diagnostic
information needed to investigate an integration failure.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.generation.schema import GuidelineAnswer


LOGGER_NAME = "clinical_guidelines_assistant"


class JsonLineFormatter(logging.Formatter):
    """Render structured log records as single JSON objects for repeatable audit review."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "application_log"),
        }
        payload.update(getattr(record, "audit_fields", {}))
        return json.dumps(payload, sort_keys=True)


def configure_provenance_logger(log_dir: Path) -> logging.Logger:
    """Configure one local JSONL audit destination; no query text is accepted."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = (log_dir / "provenance.jsonl").resolve()
    # This named logger is process-wide. Close and replace prior file handlers so a later
    # configuration cannot duplicate audit events or retain a Windows file lock elsewhere.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    return logger


def log_generation_failure(logger: logging.Logger, request_id: str, error: Exception) -> None:
    """Log a safe failure category without ever serialising untrusted error text."""
    logger.warning(
        "generation failure",
        extra={
            "event": "generation_failure",
            "audit_fields": {
                "request_id": request_id,
                "error_type": type(error).__name__,
            },
        },
    )


def log_answer(logger: logging.Logger, request_id: str, answer: GuidelineAnswer) -> None:
    """Record response provenance without recording the question or generated prose."""
    logger.info(
        "answer completed",
        extra={
            "event": "answer_completed",
            "audit_fields": {
                "request_id": request_id,
                "grounded": answer.grounded,
                "grounding_confidence": answer.grounding_confidence,
                "citation_clause_ids": [citation.clause_id for citation in answer.citations],
            },
        },
    )

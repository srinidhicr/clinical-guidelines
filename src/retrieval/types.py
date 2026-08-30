"""Shared, serialisable retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """One candidate chunk and its retriever/reranker score."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str

    def with_score(self, score: float, source: str | None = None) -> "RetrievedChunk":
        return replace(self, score=float(score), source=source or self.source)

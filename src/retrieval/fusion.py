"""Reciprocal Rank Fusion for independently produced candidate lists."""

from __future__ import annotations

from collections.abc import Iterable

from src.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: Iterable[list[RetrievedChunk]], rrf_k: int, limit: int | None = None
) -> list[RetrievedChunk]:
    """Fuse rankings by rank, avoiding incomparable raw BM25/vector score scales."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused_scores: dict[str, float] = {}
    canonical_chunks: dict[str, RetrievedChunk] = {}
    for result_set in result_sets:
        for rank, result in enumerate(result_set, start=1):
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            canonical_chunks.setdefault(result.chunk_id, result)
    fused = [
        canonical_chunks[chunk_id].with_score(score, source="rrf")
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda result: (-result.score, result.chunk_id))
    return fused[:limit] if limit is not None else fused

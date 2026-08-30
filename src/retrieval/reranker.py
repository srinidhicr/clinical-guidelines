"""Cross-encoder reranking of fused candidates before generation."""

from __future__ import annotations

from typing import Any, Protocol

from src.retrieval.types import RetrievedChunk


class PairScorer(Protocol):
    def predict(self, sentence_pairs: list[list[str]], **kwargs: Any) -> Any: ...


class CrossEncoderReranker:
    """Lazy cross-encoder wrapper; a supplied scorer keeps tests fast and deterministic."""

    def __init__(self, model_name: str, scorer: PairScorer | None = None) -> None:
        self.model_name = model_name
        self._scorer = scorer

    @property
    def scorer(self) -> PairScorer:
        if self._scorer is None:
            from sentence_transformers import CrossEncoder

            self._scorer = CrossEncoder(self.model_name)
        return self._scorer

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0 or not candidates:
            return []
        scores = self.scorer.predict([[query, candidate.text] for candidate in candidates], show_progress_bar=False)
        rescored = [
            candidate.with_score(float(score), source="cross_encoder")
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda candidate: (-candidate.score, candidate.chunk_id))[:top_k]

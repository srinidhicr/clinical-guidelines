"""NFR-02 pipeline wiring test without an external model or network dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline import ClinicalGuidelinesPipeline


class DeterministicEmbedder:
    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        return np.array(
            [[float(len(sentence) % 19), float(sum(map(ord, sentence)) % 29), 1.0] for sentence in sentences],
            dtype=np.float32,
        )


class DeterministicReranker:
    def predict(self, sentence_pairs: list[list[str]], **_: object) -> list[float]:
        return [float(len(pair[1])) for pair in sentence_pairs]


def test_pipeline_wires_transform_retrieval_fusion_and_reranking(tmp_path: Path) -> None:
    """NFR-02: one pipeline object executes all local stages over committed sample data."""
    repository_root = Path(__file__).resolve().parents[1]
    settings = {
        "project": {
            "corpus_dir": str(repository_root / "data" / "raw"),
            "index_dir": str(tmp_path / "index"),
            "log_dir": str(tmp_path / "logs"),
        },
        "ingestion": {
            "chunk_strategy": "section_aware",
            "max_chunk_characters": 1800,
            "chunk_overlap_characters": 160,
            "deduplicate_by": "content_sha256",
        },
        "embedding": {"model_name": "test-embedder", "normalize_embeddings": True},
        "retrieval": {
            "bm25_candidate_count": 6,
            "vector_candidate_count": 6,
            "rrf_k": 60,
            "reranker_model": "test-reranker",
            "reranker_candidate_count": 8,
            "final_context_count": 3,
            "minimum_grounding_confidence": 0.42,
        },
        "generation": {"model_name": "test", "temperature": 0.0, "max_retries": 1, "retry_backoff_seconds": 0.0},
    }
    pipeline = ClinicalGuidelinesPipeline(
        settings=settings,
        embedder=DeterministicEmbedder(),
        reranker_scorer=DeterministicReranker(),
    )

    plan, fused, reranked = pipeline.retrieve(
        "Compare the second-line management of APES and Chronic Rhythm Irregularity Disorder (CRID)."
    )

    assert plan.strategy == "comparison_decomposition"
    assert fused
    assert reranked
    assert len(reranked) <= settings["retrieval"]["final_context_count"]
    assert all(candidate.source == "cross_encoder" for candidate in reranked)

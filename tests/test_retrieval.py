"""AC-03, AC-04, and AC-07 retrieval-stage tests."""

from __future__ import annotations

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.query_transform import transform_query
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.types import RetrievedChunk


def _result(chunk_id: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, f"Evidence {chunk_id}", {"clause_id": chunk_id}, score, source)


def test_hybrid_fusion_combines_both_sources() -> None:
    """AC-03: independent lexical/semantic results are combined using RRF."""
    lexical_chunks = [
        {"chunk_id": "CARD-001-S2::p1", "text": "APES Cardaprilol first line", "metadata": {}},
        {"chunk_id": "CARD-001-S3::p1", "text": "APES Rhythmostat second line", "metadata": {}},
    ]
    lexical = BM25Retriever(lexical_chunks).search("APES first line", limit=2)
    semantic = [_result("ENDO-004-S2::p1", 0.91, "vector"), _result(lexical[0].chunk_id, 0.83, "vector")]

    fused = reciprocal_rank_fusion([lexical, semantic], rrf_k=60)

    assert {result.chunk_id for result in fused} == {
        "CARD-001-S2::p1",
        "CARD-001-S3::p1",
        "ENDO-004-S2::p1",
    }
    assert fused[0].chunk_id == lexical[0].chunk_id
    assert all(result.source == "rrf" for result in fused)


class ReverseScorer:
    def predict(self, sentence_pairs: list[list[str]], **_: object) -> list[float]:
        return [float(index) for index in range(len(sentence_pairs))]


def test_reranker_reorders_candidates() -> None:
    """AC-04: cross-encoder scores, rather than RRF order, select final context."""
    candidates = [_result("first", 0.9, "rrf"), _result("second", 0.8, "rrf")]
    reranked = CrossEncoderReranker("test", scorer=ReverseScorer()).rerank("query", candidates, top_k=2)

    assert [candidate.chunk_id for candidate in reranked] == ["second", "first"]
    assert all(candidate.source == "cross_encoder" for candidate in reranked)


def test_multi_part_query_is_decomposed() -> None:
    """AC-07: comparison queries create focused subqueries before retrieval."""
    plan = transform_query("Compare the second-line management of APES and Chronic Rhythm Irregularity Disorder (CRID).")

    assert plan.strategy == "comparison_decomposition"
    assert len(plan.retrieval_queries) == 3
    assert "APES" in plan.retrieval_queries[1]
    assert "CRID" in plan.retrieval_queries[2]

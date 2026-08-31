"""Single entry point that orchestrates indexing, retrieval, grounding, and logging."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from src.generation.generator import GenerationSettings, GeminiClient, generate_grounded_answer
from src.generation.schema import GuidelineAnswer
from src.ingestion.indexer import build_index_from_settings, load_persisted_index
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.query_transform import QueryPlan, transform_query
from src.retrieval.reranker import CrossEncoderReranker, PairScorer
from src.retrieval.types import RetrievedChunk
from src.retrieval.vector_retriever import QueryEmbedder, VectorRetriever
from src.utils.config import load_settings, repository_path
from src.utils.logging import configure_provenance_logger, log_answer


class ClinicalGuidelinesPipeline:
    """Composable RAG pipeline with explicit stage boundaries for testability."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        embedder: QueryEmbedder | None = None,
        reranker_scorer: PairScorer | None = None,
        gemini_client: GeminiClient | None = None,
        generation_model_override: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._embedder = embedder
        self._reranker_scorer = reranker_scorer
        self._gemini_client = gemini_client
        self._generation_model_override = generation_model_override
        self._bm25: BM25Retriever | None = None
        self._vector: VectorRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None

    def _embedder_instance(self) -> QueryEmbedder:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings["embedding"]["model_name"])
        return self._embedder

    def prepare(self) -> None:
        """Build/reuse the index and initialise retrieval components exactly once."""
        if self._bm25 is not None:
            return
        build_index_from_settings(self.settings, embedder=self._embedder_instance())
        index_dir = repository_path(self.settings["project"]["index_dir"])
        _, chunks, _ = load_persisted_index(index_dir)
        self._bm25 = BM25Retriever(chunks)
        self._vector = VectorRetriever(
            index_dir,
            self._embedder_instance(),
            normalize_embeddings=bool(self.settings["embedding"]["normalize_embeddings"]),
        )
        self._reranker = CrossEncoderReranker(
            self.settings["retrieval"]["reranker_model"], scorer=self._reranker_scorer
        )

    def retrieve(self, query: str) -> tuple[QueryPlan, list[RetrievedChunk], list[RetrievedChunk]]:
        """Transform, independently retrieve, fuse with RRF, then cross-encoder rerank."""
        self.prepare()
        assert self._bm25 is not None and self._vector is not None and self._reranker is not None
        config = self.settings["retrieval"]
        plan = transform_query(query)
        result_sets: list[list[RetrievedChunk]] = []
        for retrieval_query in plan.retrieval_queries:
            result_sets.append(self._bm25.search(retrieval_query, int(config["bm25_candidate_count"])))
            result_sets.append(self._vector.search(retrieval_query, int(config["vector_candidate_count"])))
        fused = reciprocal_rank_fusion(
            result_sets,
            rrf_k=int(config["rrf_k"]),
            limit=int(config["reranker_candidate_count"]),
        )
        reranked = self._reranker.rerank(query, fused, int(config["final_context_count"]))
        return plan, fused, reranked

    def ask(
        self,
        query: str,
        request_id: str | None = None,
        contexts: list[RetrievedChunk] | None = None,
    ) -> GuidelineAnswer:
        """Return the only public answer object, with provenance logged after completion."""
        active_contexts = contexts if contexts is not None else self.retrieve(query)[2]
        generation = self.settings["generation"]
        retrieval = self.settings["retrieval"]
        active_request_id = request_id or str(uuid.uuid4())
        logger = configure_provenance_logger(repository_path(self.settings["project"]["log_dir"]))
        answer = generate_grounded_answer(
            query=query,
            contexts=active_contexts,
            settings=GenerationSettings(
                model_name=self._generation_model_override
                or os.getenv("GEMINI_MODEL", generation["model_name"]),
                temperature=float(generation["temperature"]),
                max_retries=int(generation["max_retries"]),
                retry_backoff_seconds=float(generation["retry_backoff_seconds"]),
                minimum_grounding_confidence=float(retrieval["minimum_grounding_confidence"]),
                minimum_evidence_term_overlap=float(retrieval["minimum_evidence_term_overlap"]),
            ),
            client=self._gemini_client,
            logger=logger,
            request_id=active_request_id,
        )
        log_answer(logger, active_request_id, answer)
        return answer


def run_end_to_end(
    settings: dict[str, Any] | None = None,
    pipeline: ClinicalGuidelinesPipeline | None = None,
    run_ragas: bool = False,
    evaluator: Any | None = None,
) -> dict[str, Any]:
    """Execute the documented NFR-02 pipeline and evaluation smoke path.

    The default evaluation is deterministic clause retrieval so one command is safe to
    rerun without silently consuming Gemini quota. ``run_ragas=True`` explicitly adds
    the live Gemini-backed evaluation stage.
    """
    from eval.golden_set import load_golden_set
    from eval.ragas_eval import evaluate

    active_settings = settings or load_settings()
    active_pipeline = pipeline or ClinicalGuidelinesPipeline(active_settings)
    golden_set = load_golden_set(repository_path(active_settings["project"]["golden_set_path"]))
    sample = next(entry for entry in golden_set if entry["type"] != "abstain")

    active_pipeline.prepare()
    answer = active_pipeline.ask(sample["question"], request_id="nfr-02-end-to-end-smoke")
    evaluate_runner = evaluator or evaluate
    evaluation_report = evaluate_runner(
        settings=active_settings,
        pipeline=active_pipeline,
        run_ragas=run_ragas,
        report_filename="pipeline_run_evaluation_report.json",
    )
    return {
        "sample_question_id": sample["id"],
        "sample_answer": answer.model_dump(),
        "evaluation_report": "eval/reports/pipeline_run_evaluation_report.json",
        "evaluation_scope": evaluation_report.get("evaluation_scope"),
        "ragas_status": evaluation_report.get("ragas", {}).get("status"),
    }


def main() -> None:
    """Minimal CLI for one locally indexed question; UI polish is intentionally deferred."""
    parser = argparse.ArgumentParser(description="Synthetic Clinical Guidelines Assistant")
    parser.add_argument("query", nargs="?", help="Question to ask against the synthetic corpus")
    parser.add_argument("--build-index", action="store_true", help="Build/reuse the persisted FAISS index")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run indexing, one committed sample answer, then full deterministic evaluation.",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="With --run-all, include the quota-consuming live Gemini RAGAS stage.",
    )
    args = parser.parse_args()

    if args.with_ragas and not args.run_all:
        parser.error("--with-ragas can only be used with --run-all")
    if args.run_all:
        print(json.dumps(run_end_to_end(run_ragas=args.with_ragas), indent=2))
        return

    pipeline = ClinicalGuidelinesPipeline()
    if args.build_index:
        pipeline.prepare()
        print("Index is ready.")
    if args.query:
        print(json.dumps(pipeline.ask(args.query).model_dump(), indent=2))
    elif not args.build_index:
        parser.print_help()


if __name__ == "__main__":
    main()

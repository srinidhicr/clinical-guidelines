"""Re-runnable RAGAS evaluation using Gemini plus deterministic retrieval diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from eval.golden_set import load_golden_set, score_retrieval, validate_expected_clauses
from src.ingestion.loaders import load_corpus
from src.pipeline import ClinicalGuidelinesPipeline
from src.utils.config import load_settings, repository_path


def run_deterministic_evaluation(pipeline: ClinicalGuidelinesPipeline, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate clause retrieval over each golden question without requiring generation."""
    item_results: list[dict[str, Any]] = []
    for entry in entries:
        _, _, contexts = pipeline.retrieve(entry["question"])
        retrieved_clause_ids = [str(context.metadata["clause_id"]) for context in contexts]
        precision, recall = score_retrieval(entry["expected_context"], retrieved_clause_ids)
        item_results.append(
            {
                "id": entry["id"],
                "type": entry["type"],
                "ac_ref": entry["ac_ref"],
                "expected_clause_ids": entry["expected_context"],
                "retrieved_clause_ids": retrieved_clause_ids,
                "context_precision": precision,
                "context_recall": recall,
            }
        )
    return {
        "items": item_results,
        "aggregate": {
            "context_precision": sum(item["context_precision"] for item in item_results) / len(item_results),
            "context_recall": sum(item["context_recall"] for item in item_results) / len(item_results),
        },
    }


REQUIRED_RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


class LegacyRagasEmbeddingAdapter:
    """Expose RAGAS legacy sync method names over a modern embedding provider."""

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_texts(texts)


def _ragas_result_payload(result: Any, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve every RAGAS item score and derive auditable aggregate values."""
    if not hasattr(result, "to_pandas"):
        raise TypeError(f"Unsupported RAGAS result type: {type(result).__name__}")
    rows = result.to_pandas().to_dict(orient="records")
    if len(rows) != len(entries):
        raise ValueError(f"RAGAS returned {len(rows)} rows for {len(entries)} golden questions.")

    item_scores: list[dict[str, Any]] = []
    for entry, row in zip(entries, rows, strict=True):
        # RAGAS has used both names for the same rubric metric across releases.
        if "answer_relevancy" not in row and "response_relevancy" in row:
            row["answer_relevancy"] = row["response_relevancy"]
        scores: dict[str, float] = {}
        for metric in REQUIRED_RAGAS_METRICS:
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"RAGAS did not compute {metric} for golden question {entry['id']}.")
            scores[metric] = float(value)
        item_scores.append(
            {
                "id": entry["id"],
                "type": entry["type"],
                "ac_ref": entry["ac_ref"],
                **scores,
            }
        )

    aggregate = {
        metric: sum(item[metric] for item in item_scores) / len(item_scores)
        for metric in REQUIRED_RAGAS_METRICS
    }
    return {"items": item_scores, "aggregate": aggregate}


def run_ragas_if_configured(
    pipeline: ClinicalGuidelinesPipeline,
    entries: list[dict[str, Any]],
    model_name: str,
    embedding_model: str,
) -> dict[str, Any]:
    """Run all required RAGAS metrics with Gemini when credentials are configured."""
    # Evaluation is normally launched as `python -m eval.ragas_eval`, so load the
    # repository-root .env before checking the key. This mirrors the generation client.
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "status": "not_run",
            "reason": "GOOGLE_API_KEY is not configured; faithfulness and answer relevancy require Gemini-backed RAGAS judging.",
            "faithfulness": None,
            "answer_relevancy": None,
        }

    from datasets import Dataset
    from google import genai
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    try:
        from ragas.metrics import AnswerRelevancy
    except ImportError:  # Newer RAGAS releases use the clearer ResponseRelevancy name.
        from ragas.metrics import ResponseRelevancy as AnswerRelevancy

    dataset_rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for entry in entries:
        # The pipeline's validated generation path supplies the answer and the exact
        # reranked context that RAGAS is meant to judge.
        _, _, contexts = pipeline.retrieve(entry["question"])
        answer = pipeline.ask(
            entry["question"],
            request_id=f"ragas-{entry['id']}",
            contexts=contexts,
        )
        dataset_rows["question"].append(entry["question"])
        dataset_rows["answer"].append(answer.answer)
        dataset_rows["contexts"].append([context.text for context in contexts])
        dataset_rows["ground_truth"].append(entry["reference_answer"])

    client = genai.Client(api_key=api_key)
    judge_llm = llm_factory(model_name, provider="google", client=client, temperature=0.0)
    judge_embeddings = LegacyRagasEmbeddingAdapter(
        GoogleEmbeddings(client=client, model=embedding_model)
    )
    evaluation_settings = pipeline.settings["evaluation"]
    run_config = RunConfig(
        timeout=int(evaluation_settings.get("request_timeout_seconds", 180)),
        max_retries=int(evaluation_settings.get("max_retries", 8)),
        max_wait=int(evaluation_settings.get("max_wait_seconds", 60)),
        max_workers=int(evaluation_settings.get("max_workers", 1)),
    )
    result = ragas_evaluate(
        Dataset.from_dict(dataset_rows),
        metrics=[
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
            Faithfulness(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=True,
    )
    scores = _ragas_result_payload(result, entries)
    return {
        "status": "completed",
        "judge_model": model_name,
        "embedding_model": embedding_model,
        "question_count": len(entries),
        **scores,
    }


def evaluate(
    settings: dict[str, Any] | None = None,
    pipeline: ClinicalGuidelinesPipeline | None = None,
    run_ragas: bool = True,
    question_ids: list[str] | None = None,
    report_filename: str = "ragas_report.json",
) -> dict[str, Any]:
    """Run the golden set and optionally invoke live Gemini-backed RAGAS scoring."""
    active_settings = settings or load_settings()
    corpus_dir = repository_path(active_settings["project"]["corpus_dir"])
    all_entries = load_golden_set(repository_path(active_settings["project"]["golden_set_path"]))
    validate_expected_clauses(all_entries, {section.clause_id for section in load_corpus(corpus_dir)})
    if question_ids:
        requested_ids = set(question_ids)
        entries = [entry for entry in all_entries if entry["id"] in requested_ids]
        missing_ids = requested_ids - {entry["id"] for entry in entries}
        if missing_ids:
            raise ValueError(f"Unknown golden question IDs: {sorted(missing_ids)}")
    else:
        entries = all_entries
    active_pipeline = pipeline or ClinicalGuidelinesPipeline(active_settings)
    report = run_deterministic_evaluation(active_pipeline, entries)
    evaluation_settings = active_settings["evaluation"]
    ragas_model = evaluation_settings.get(
        "judge_model", active_settings.get("generation", {}).get("model_name", "gemini-3.6-flash")
    )
    ragas_embedding_model = evaluation_settings.get("embedding_model", "gemini-embedding-001")
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "evaluation_type": "deterministic_retrieval_and_gemini_ragas" if run_ragas else "deterministic_clause_retrieval",
            "evaluation_scope": "subset" if question_ids else "full_golden_set",
            "question_ids": [entry["id"] for entry in entries],
            "question_count": len(entries),
            "ragas": (
                run_ragas_if_configured(
                    active_pipeline,
                    entries,
                    ragas_model,
                    ragas_embedding_model,
                )
                if run_ragas
                else {
                    "status": "not_run",
                    "reason": "Live RAGAS is disabled for this invocation.",
                    "faithfulness": None,
                    "answer_relevancy": None,
                }
            ),
        }
    )
    report_dir = repository_path(active_settings["evaluation"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / report_filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the synthetic guideline RAG pipeline")
    parser.add_argument(
        "--question-id",
        action="append",
        help="Evaluate only this golden-set ID; repeat to select more than one.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Run only deterministic clause-retrieval diagnostics.",
    )
    args = parser.parse_args()
    is_subset = bool(args.question_id)
    report = evaluate(
        run_ragas=not args.skip_ragas,
        question_ids=args.question_id,
        report_filename="ragas_smoke_report.json" if is_subset else "ragas_report.json",
    )
    print(
        json.dumps(
            {
                "report": "ragas_smoke_report.json" if is_subset else "ragas_report.json",
                "retrieval": report["aggregate"],
                "ragas": report["ragas"].get("aggregate", report["ragas"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

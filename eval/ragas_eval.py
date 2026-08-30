"""Re-runnable RAGAS evaluation using Gemini plus deterministic retrieval diagnostics."""

from __future__ import annotations

import argparse
import json
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


def _ragas_result_dict(result: Any) -> dict[str, float]:
    """Normalise RAGAS result versions into JSON-safe aggregate metric values."""
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return {str(key): float(value) for key, value in frame.mean(numeric_only=True).to_dict().items()}
    if isinstance(result, dict):
        return {str(key): float(value) for key, value in result.items()}
    raise TypeError(f"Unsupported RAGAS result type: {type(result).__name__}")


def run_ragas_if_configured(
    pipeline: ClinicalGuidelinesPipeline, entries: list[dict[str, Any]], model_name: str
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
    from ragas.llms import llm_factory
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness

    try:
        from ragas.metrics import AnswerRelevancy
    except ImportError:  # Newer RAGAS releases use the clearer ResponseRelevancy name.
        from ragas.metrics import ResponseRelevancy as AnswerRelevancy

    dataset_rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for entry in entries:
        # The pipeline's validated generation path supplies the answer and the exact
        # reranked context that RAGAS is meant to judge.
        answer = pipeline.ask(entry["question"], request_id=f"ragas-{entry['id']}")
        _, _, contexts = pipeline.retrieve(entry["question"])
        dataset_rows["question"].append(entry["question"])
        dataset_rows["answer"].append(answer.answer)
        dataset_rows["contexts"].append([context.text for context in contexts])
        dataset_rows["ground_truth"].append(entry["reference_answer"])

    client = genai.Client(api_key=api_key)
    judge_llm = llm_factory(model_name, provider="google", client=client, temperature=0.0)
    result = ragas_evaluate(
        Dataset.from_dict(dataset_rows),
        metrics=[
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
            Faithfulness(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm),
        ],
    )
    scores = _ragas_result_dict(result)
    # RAGAS's spelling has differed across versions; publish the rubric's stable name.
    answer_relevancy = scores.get("answer_relevancy", scores.get("response_relevancy"))
    return {
        "status": "completed",
        "judge_model": model_name,
        "question_count": len(entries),
        "context_precision": scores.get("context_precision"),
        "context_recall": scores.get("context_recall"),
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": answer_relevancy,
        "raw_metric_names": sorted(scores),
    }


def evaluate(
    settings: dict[str, Any] | None = None,
    pipeline: ClinicalGuidelinesPipeline | None = None,
    run_ragas: bool = True,
) -> dict[str, Any]:
    """Run the golden set and optionally invoke live Gemini-backed RAGAS scoring."""
    active_settings = settings or load_settings()
    corpus_dir = repository_path(active_settings["project"]["corpus_dir"])
    entries = load_golden_set(repository_path(active_settings["project"]["golden_set_path"]))
    validate_expected_clauses(entries, {section.clause_id for section in load_corpus(corpus_dir)})
    active_pipeline = pipeline or ClinicalGuidelinesPipeline(active_settings)
    report = run_deterministic_evaluation(active_pipeline, entries)
    ragas_model = active_settings.get("generation", {}).get("model_name", "gemini-2.0-flash")
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "evaluation_type": "deterministic_clause_retrieval",
            "question_count": len(entries),
            "ragas": (
                run_ragas_if_configured(active_pipeline, entries, ragas_model)
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
    (report_dir / "ragas_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the synthetic guideline RAG pipeline")
    parser.parse_args()
    report = evaluate()
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()

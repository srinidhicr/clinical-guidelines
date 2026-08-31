"""Instructor-authorized Groq-judged RAGAS evaluation for Gemini-generated answers.

The application generation model remains Gemini. Groq is used only as the RAGAS judge
when the instructor permits this evaluation-provider exception to the original brief.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from eval.golden_set import load_golden_set, validate_expected_clauses
from eval.ragas_eval import REQUIRED_RAGAS_METRICS, _ragas_result_payload, run_deterministic_evaluation
from src.ingestion.loaders import load_corpus
from src.pipeline import ClinicalGuidelinesPipeline
from src.utils.config import load_settings, repository_path


class LocalRagasEmbeddingAdapter:
    """Use the already-approved local Sentence-Transformer for answer relevancy."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class GenerationStageUnavailable(RuntimeError):
    """Raised when an answerable golden question cannot obtain a grounded answer."""

    def __init__(self, question_id: str, audit: dict[str, Any]) -> None:
        super().__init__(f"Gemini did not produce a grounded answer for {question_id}.")
        self.question_id = question_id
        self.audit = audit


def _select_entries(all_entries: list[dict[str, Any]], question_ids: list[str] | None) -> list[dict[str, Any]]:
    if not question_ids:
        return all_entries
    requested = set(question_ids)
    entries = [entry for entry in all_entries if entry["id"] in requested]
    missing = requested - {entry["id"] for entry in entries}
    if missing:
        raise ValueError(f"Unknown golden question IDs: {sorted(missing)}")
    return entries


def create_groq_judge_llm(
    api_key: str,
    judge_model: str,
    max_tokens: int = 4096,
    reasoning_effort: str = "low",
    client_factory: Callable[..., Any] | None = None,
    llm_builder: Callable[..., Any] | None = None,
) -> Any:
    """Adapt Groq's OpenAI-compatible endpoint to RAGAS 0.4's OpenAI adapter.

    RAGAS 0.4.3 incorrectly patches the current native ``Groq`` client as though it
    exposed an Anthropic-style ``messages`` API. Groq's supported OpenAI-compatible
    endpoint instead exposes the ``chat.completions`` API RAGAS expects here.
    """
    if client_factory is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,
            timeout=180.0,
        )
    else:
        client = client_factory(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    if llm_builder is None:
        from ragas.llms import llm_factory

        llm_builder = llm_factory
    return llm_builder(
        judge_model,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Avoid corrupting a resumable checkpoint if evaluation is interrupted."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _load_checkpoint(path: Path, generation_model: str) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "generation_model": generation_model, "items": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("generation_model") != generation_model:
        return {"version": 1, "generation_model": generation_model, "items": {}}
    if not isinstance(payload.get("items"), dict):
        return {"version": 1, "generation_model": generation_model, "items": {}}
    return payload


def _load_metrics_checkpoint(path: Path, judge_model: str) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "judge_model": judge_model, "items": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("judge_model") != judge_model:
        return {"version": 1, "judge_model": judge_model, "items": {}}
    if not isinstance(payload.get("items"), dict):
        return {"version": 1, "judge_model": judge_model, "items": {}}
    return payload


def _generation_audit(
    pipeline: ClinicalGuidelinesPipeline,
    items: list[dict[str, Any]],
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_model": pipeline.settings["generation"]["model_name"],
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "question_count": len(items),
        "grounded_answer_count": sum(item["grounded"] for item in items),
        "abstained_answer_count": sum(not item["grounded"] for item in items),
        "reused_checkpoint_answer_count": sum(item["source"] == "checkpoint" for item in items),
        "new_generation_answer_count": sum(item["source"] == "gemini" for item in items),
        "items": items,
    }


def collect_generation_dataset(
    pipeline: ClinicalGuidelinesPipeline,
    entries: list[dict[str, Any]],
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Generate answers, reusing only prior grounded answers from an atomic checkpoint."""
    from src.generation.schema import GuidelineAnswer

    dataset_rows: dict[str, list[Any]] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    items: list[dict[str, Any]] = []
    model_name = pipeline.settings["generation"]["model_name"]
    checkpoint = _load_checkpoint(checkpoint_path, model_name) if checkpoint_path is not None else None
    for entry in entries:
        entry_type = entry.get("type")
        if entry_type not in {"single_clause", "multi_clause", "abstain"}:
            raise ValueError(
                f"Golden question {entry.get('id', '<unknown>')} has an invalid or missing type."
            )
        _, _, contexts = pipeline.retrieve(entry["question"])
        cached = checkpoint["items"].get(entry["id"]) if checkpoint is not None else None
        if cached and cached.get("question") == entry["question"]:
            try:
                answer = GuidelineAnswer.model_validate(cached["answer"])
                answer_source = "checkpoint"
            except (KeyError, ValueError):
                cached = None
        if not cached:
            answer = pipeline.ask(entry["question"], request_id=f"groq-ragas-{entry['id']}", contexts=contexts)
            answer_source = "gemini"

        item = {
            "id": entry["id"],
            "grounded": answer.grounded,
            "grounding_confidence": answer.grounding_confidence,
            "citation_clause_ids": [citation.clause_id for citation in answer.citations],
            "source": answer_source,
        }
        items.append(item)
        if not answer.grounded and entry_type != "abstain":
            raise GenerationStageUnavailable(entry["id"], _generation_audit(pipeline, items, checkpoint_path))
        if checkpoint is not None and answer_source == "gemini" and (answer.grounded or entry_type == "abstain"):
            checkpoint["items"][entry["id"]] = {
                "question": entry["question"],
                "answer": answer.model_dump(),
            }
            _write_json_atomically(checkpoint_path, checkpoint)
        dataset_rows["question"].append(entry["question"])
        dataset_rows["answer"].append(answer.answer)
        dataset_rows["contexts"].append([context.text for context in contexts])
        dataset_rows["ground_truth"].append(entry["reference_answer"])
    return dataset_rows, _generation_audit(pipeline, items, checkpoint_path)


def run_groq_ragas(
    pipeline: ClinicalGuidelinesPipeline,
    entries: list[dict[str, Any]],
    judge_model: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run all four RAGAS metrics through Groq, retaining each item score."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "status": "not_run",
            "reason": "GROQ_API_KEY is not configured.",
            "required_metrics": list(REQUIRED_RAGAS_METRICS),
        }

    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    try:
        from ragas.metrics import AnswerRelevancy
    except ImportError:
        from ragas.metrics import ResponseRelevancy as AnswerRelevancy

    report_dir = repository_path(pipeline.settings["evaluation"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    generation_audit_path = report_dir / "groq_ragas_generation_audit.json"
    try:
        dataset_rows, generation_audit = collect_generation_dataset(pipeline, entries, checkpoint_path)
    except GenerationStageUnavailable as error:
        generation_audit_path.write_text(json.dumps(error.audit, indent=2), encoding="utf-8")
        return {
            "status": "generation_incomplete",
            "reason": str(error),
            "failed_question_id": error.question_id,
            "generation_audit_report": "eval/reports/groq_ragas_generation_audit.json",
            "generation_checkpoint": "eval/reports/groq_ragas_generation_checkpoint.json",
        }
    generation_audit_path.write_text(json.dumps(generation_audit, indent=2), encoding="utf-8")

    evaluation_settings = pipeline.settings["evaluation"]
    judge_llm = create_groq_judge_llm(
        api_key,
        judge_model,
        max_tokens=int(evaluation_settings.get("groq_judge_max_tokens", 4096)),
        reasoning_effort=str(evaluation_settings.get("groq_judge_reasoning_effort", "low")),
    )
    embeddings = LocalRagasEmbeddingAdapter(pipeline.settings["embedding"]["model_name"])
    run_config = RunConfig(
        timeout=int(evaluation_settings.get("request_timeout_seconds", 180)),
        max_retries=int(evaluation_settings.get("max_retries", 8)),
        max_wait=int(evaluation_settings.get("max_wait_seconds", 60)),
        max_workers=int(evaluation_settings.get("max_workers", 4)),
    )
    metrics_checkpoint_path = report_dir / "groq_ragas_metrics_checkpoint.json"
    metrics_checkpoint = _load_metrics_checkpoint(metrics_checkpoint_path, judge_model)

    pending_indices = [
        i for i, entry in enumerate(entries) if entry["id"] not in metrics_checkpoint["items"]
    ]

    if pending_indices:
        batch_size = int(evaluation_settings.get("batch_size", 5))
        for b_start in range(0, len(pending_indices), batch_size):
            batch_idx = pending_indices[b_start : b_start + batch_size]
            batch_entries = [entries[i] for i in batch_idx]
            batch_dataset = {k: [dataset_rows[k][i] for i in batch_idx] for k in dataset_rows}
            batch_result = ragas_evaluate(
                Dataset.from_dict(batch_dataset),
                metrics=[
                    ContextPrecision(llm=judge_llm),
                    ContextRecall(llm=judge_llm),
                    Faithfulness(llm=judge_llm),
                    AnswerRelevancy(llm=judge_llm, embeddings=embeddings, strictness=1),
                ],
                llm=judge_llm,
                embeddings=embeddings,
                run_config=run_config,
                raise_exceptions=True,
            )
            batch_payload = _ragas_result_payload(batch_result, batch_entries)
            for item in batch_payload["items"]:
                metrics_checkpoint["items"][item["id"]] = item
            _write_json_atomically(metrics_checkpoint_path, metrics_checkpoint)

    all_item_scores = [metrics_checkpoint["items"][entry["id"]] for entry in entries]
    aggregate = {
        metric: sum(item[metric] for item in all_item_scores) / len(all_item_scores)
        for metric in REQUIRED_RAGAS_METRICS
    }
    return {
        "status": "completed",
        "judge_provider": "groq",
        "judge_model": judge_model,
        "embedding_provider": "local_sentence_transformer",
        "embedding_model": pipeline.settings["embedding"]["model_name"],
        "generation_audit_report": "eval/reports/groq_ragas_generation_audit.json",
        "generation_checkpoint": "eval/reports/groq_ragas_generation_checkpoint.json",
        "metrics_checkpoint": "eval/reports/groq_ragas_metrics_checkpoint.json",
        "question_count": len(entries),
        "items": all_item_scores,
        "aggregate": aggregate,
    }


def evaluate_groq(
    settings: dict[str, Any] | None = None,
    question_ids: list[str] | None = None,
    judge_model: str | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Create a report combining deterministic retrieval and Groq-judged RAGAS."""
    active_settings = settings or load_settings()
    corpus_dir = repository_path(active_settings["project"]["corpus_dir"])
    all_entries = load_golden_set(repository_path(active_settings["project"]["golden_set_path"]))
    validate_expected_clauses(all_entries, {section.clause_id for section in load_corpus(corpus_dir)})
    entries = _select_entries(all_entries, question_ids)
    pipeline = ClinicalGuidelinesPipeline(active_settings)
    report = run_deterministic_evaluation(pipeline, entries)
    report_dir = repository_path(active_settings["evaluation"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    active_checkpoint = checkpoint_path or report_dir / "groq_ragas_generation_checkpoint.json"
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "evaluation_type": "deterministic_retrieval_and_groq_ragas",
            "evaluation_scope": "subset" if question_ids else "full_golden_set",
            "question_ids": [entry["id"] for entry in entries],
            "question_count": len(entries),
            "ragas": run_groq_ragas(
                pipeline,
                entries,
                judge_model or active_settings["evaluation"].get("groq_judge_model", "openai/gpt-oss-20b"),
                active_checkpoint,
            ),
        }
    )
    filename = "ragas_groq_smoke_report.json" if question_ids else "ragas_groq_report.json"
    (report_dir / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run instructor-authorized Groq-judged RAGAS evaluation")
    parser.add_argument("--question-id", action="append", help="Evaluate only this golden-set ID; repeat as needed.")
    parser.add_argument("--judge-model", help="Groq model ID; defaults to evaluation.groq_judge_model.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show checkpointed Gemini-generation progress without calling Gemini or Groq.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Discard cached grounded Gemini answers before evaluating; normally leave the checkpoint intact.",
    )
    args = parser.parse_args()
    settings = load_settings()
    checkpoint_path = repository_path(settings["evaluation"]["report_dir"]) / "groq_ragas_generation_checkpoint.json"
    if args.status:
        if args.question_id or args.judge_model or args.reset_checkpoint:
            parser.error("--status cannot be combined with evaluation options")
        entries = load_golden_set(repository_path(settings["project"]["golden_set_path"]))
        checkpoint = _load_checkpoint(checkpoint_path, settings["generation"]["model_name"])
        completed_ids = list(checkpoint["items"])
        answerable_ids = [entry["id"] for entry in entries if entry["type"] != "abstain"]
        print(
            json.dumps(
                {
                    "checkpoint": "eval/reports/groq_ragas_generation_checkpoint.json",
                    "generation_model": settings["generation"]["model_name"],
                    "completed_grounded_count": len(completed_ids),
                    "expected_grounded_count": len(answerable_ids),
                    "completed_grounded_ids": completed_ids,
                    "remaining_answerable_ids": [item_id for item_id in answerable_ids if item_id not in completed_ids],
                },
                indent=2,
            )
        )
        return
    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    report = evaluate_groq(settings=settings, question_ids=args.question_id, judge_model=args.judge_model, checkpoint_path=checkpoint_path)
    print(json.dumps({"report": "ragas_groq_smoke_report.json" if args.question_id else "ragas_groq_report.json", "ragas": report["ragas"]}, indent=2))


if __name__ == "__main__":
    main()

"""Compare configured Gemini generation models on the same committed golden set."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import UTC, datetime
from typing import Any, Callable

from eval.ragas_eval import evaluate
from src.pipeline import ClinicalGuidelinesPipeline
from src.utils.config import load_settings, repository_path


REQUIRED_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def _safe_filename_fragment(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()


def _selection_score(metrics: dict[str, float]) -> float:
    """Use a documented, grounding-first score only after all metrics exist."""
    return (
        0.40 * metrics["faithfulness"]
        + 0.25 * metrics["answer_relevancy"]
        + 0.20 * metrics["context_precision"]
        + 0.15 * metrics["context_recall"]
    )


def compare_models(
    settings: dict[str, Any] | None = None,
    model_names: list[str] | None = None,
    question_ids: list[str] | None = None,
    run_ragas: bool = True,
    evaluator: Callable[..., dict[str, Any]] = evaluate,
) -> dict[str, Any]:
    """Run each candidate with identical retrieval, golden-set, and judge settings."""
    active_settings = copy.deepcopy(settings or load_settings())
    configured_models = model_names or active_settings["evaluation"].get("candidate_models", [])
    candidates = list(dict.fromkeys(configured_models))
    if len(candidates) < 2:
        raise ValueError("AC-10 requires at least two distinct Gemini candidate models.")

    report_dir = repository_path(active_settings["evaluation"]["report_dir"])
    comparisons: list[dict[str, Any]] = []
    for model_name in candidates:
        candidate_settings = copy.deepcopy(active_settings)
        candidate_settings["generation"]["model_name"] = model_name
        pipeline = ClinicalGuidelinesPipeline(
            candidate_settings,
            generation_model_override=model_name,
        )
        model_report = evaluator(
            settings=candidate_settings,
            pipeline=pipeline,
            run_ragas=run_ragas,
            question_ids=question_ids,
            report_filename=f"ragas_{_safe_filename_fragment(model_name)}.json",
        )
        ragas = model_report["ragas"]
        aggregate = ragas.get("aggregate") if ragas.get("status") == "completed" else None
        metrics = (
            {metric: float(aggregate[metric]) for metric in REQUIRED_METRICS}
            if aggregate is not None and all(metric in aggregate for metric in REQUIRED_METRICS)
            else None
        )
        comparisons.append(
            {
                "model": model_name,
                "generation_model": model_name,
                "judge_model": candidate_settings["evaluation"].get("judge_model"),
                "embedding_model": candidate_settings["evaluation"].get("embedding_model"),
                "report": str(report_dir / f"ragas_{_safe_filename_fragment(model_name)}.json"),
                "ragas_status": ragas.get("status"),
                "metrics": metrics,
                "selection_score": _selection_score(metrics) if metrics is not None else None,
            }
        )

    eligible = [item for item in comparisons if item["metrics"] is not None]
    selected = max(eligible, key=lambda item: (item["selection_score"], item["model"])) if eligible else None
    scope = "subset" if question_ids else "full_golden_set"
    complete = len(eligible) == len(comparisons) and scope == "full_golden_set"
    selection = (
        {
            "selected_generation_model": selected["model"],
            "selection_score": selected["selection_score"],
            "rationale": (
                "Selected from completed full-golden-set runs using the documented weighting: "
                "faithfulness 40%, answer relevancy 25%, context precision 20%, and context recall 15%. "
                "Faithfulness receives the highest weight because grounded clinical reference answers are the priority."
            ),
        }
        if selected is not None and complete
        else {
            "selected_generation_model": None,
            "selection_score": None,
            "rationale": "No selection is made until two completed full-golden-set RAGAS runs are available.",
        }
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "completed" if complete else "incomplete",
        "evaluation_scope": scope,
        "question_ids": question_ids,
        "candidate_count": len(comparisons),
        "comparison_controls": {
            "corpus": active_settings["project"]["corpus_dir"],
            "golden_set": active_settings["project"]["golden_set_path"],
            "judge_model": active_settings["evaluation"].get("judge_model"),
            "embedding_model": active_settings["evaluation"].get("embedding_model"),
            "temperature": active_settings["generation"].get("temperature"),
        },
        "selection_method": "0.40 faithfulness + 0.25 answer_relevancy + 0.20 context_precision + 0.15 context_recall",
        "models": comparisons,
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two or more Gemini models with the RAGAS golden set")
    parser.add_argument("--model", action="append", help="Candidate Gemini model; repeat for each model.")
    parser.add_argument("--question-id", action="append", help="Smoke-test only this golden-set ID; repeat to select more.")
    parser.add_argument("--skip-ragas", action="store_true", help="Exercise comparison wiring without live Gemini judging.")
    args = parser.parse_args()
    settings = load_settings()
    report = compare_models(
        settings=settings,
        model_names=args.model,
        question_ids=args.question_id,
        run_ragas=not args.skip_ragas,
    )
    report_path = repository_path(settings["evaluation"]["report_dir"]) / "llm_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status": report["status"], "selection": report["selection"]}, indent=2))


if __name__ == "__main__":
    main()

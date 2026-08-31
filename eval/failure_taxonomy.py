"""Classify evaluation evidence into retrieval, grounding, and synthesis failures.

The classifier deliberately uses only values present in a committed evaluation report.
It does not infer answer quality when live RAGAS scoring was unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from src.utils.config import repository_path


THRESHOLDS = {
    "deterministic_context_precision": 0.50,
    "context_recall": 1.00,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
}

FAILURE_CATALOG = {
    "retrieval_miss": {
        "layer": "retrieval",
        "meaning": "One or more expected clause IDs were absent from the final reranked context.",
        "action": "Inspect the query plan, BM25/vector candidates, RRF ranks, and reranker top-K for this question.",
    },
    "retrieval_noise": {
        "layer": "retrieval",
        "meaning": "The expected clause was retrieved, but the final context contains substantial unrelated evidence.",
        "action": "Tune top-K, RRF depth, or reranker selection; retain the expected clause while reducing distractors.",
    },
    "abstention_retrieval_risk": {
        "layer": "retrieval",
        "meaning": "An abstention test retrieved passages despite having no expected supporting clause.",
        "action": "Review confidence/abstention gating so irrelevant passages cannot trigger an answer.",
    },
    "grounding_failure": {
        "layer": "grounding",
        "meaning": "Completed RAGAS faithfulness was below the documented target.",
        "action": "Inspect the generated answer and cited clauses; tighten evidence-only prompting or abstain when support is incomplete.",
    },
    "synthesis_failure": {
        "layer": "synthesis",
        "meaning": "Completed RAGAS answer relevancy was below the documented target.",
        "action": "Inspect question decomposition and answer focus; remove unsupported or off-topic synthesis.",
    },
    "ragas_unassessed": {
        "layer": "assessment",
        "meaning": "Live RAGAS did not complete, so grounding and synthesis quality cannot be classified.",
        "action": "Resolve provider availability/quota and rerun the same golden set before making a quality claim.",
    },
}


def analyse_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable per-question taxonomy for an evaluation report."""
    ragas = report.get("ragas", {})
    ragas_completed = ragas.get("status") == "completed"
    ragas_by_id = {item["id"]: item for item in ragas.get("items", [])}
    findings: list[dict[str, Any]] = []

    for item in report.get("items", []):
        labels: list[str] = []
        expected = item.get("expected_clause_ids", [])
        retrieved = item.get("retrieved_clause_ids", [])
        precision = float(item["context_precision"])
        recall = float(item["context_recall"])

        if expected:
            if recall < THRESHOLDS["context_recall"]:
                labels.append("retrieval_miss")
            if precision < THRESHOLDS["deterministic_context_precision"]:
                labels.append("retrieval_noise")
        elif retrieved:
            labels.append("abstention_retrieval_risk")

        ragas_item = ragas_by_id.get(item["id"])
        if ragas_completed and ragas_item is not None:
            if float(ragas_item["faithfulness"]) < THRESHOLDS["faithfulness"]:
                labels.append("grounding_failure")
            if float(ragas_item["answer_relevancy"]) < THRESHOLDS["answer_relevancy"]:
                labels.append("synthesis_failure")
        elif not ragas_completed:
            labels.append("ragas_unassessed")

        findings.append(
            {
                "id": item["id"],
                "type": item["type"],
                "ac_ref": item["ac_ref"],
                "deterministic_metrics": {
                    "context_precision": precision,
                    "context_recall": recall,
                },
                "ragas_metrics": (
                    {
                        "faithfulness": ragas_item["faithfulness"],
                        "answer_relevancy": ragas_item["answer_relevancy"],
                    }
                    if ragas_item is not None
                    else None
                ),
                "categories": labels,
            }
        )

    counts = Counter(category for finding in findings for category in finding["categories"])
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_evaluation": {
            "scope": report.get("evaluation_scope"),
            "question_count": report.get("question_count", len(findings)),
            "ragas_status": ragas.get("status", "not_run"),
        },
        "thresholds": THRESHOLDS,
        "catalog": FAILURE_CATALOG,
        "summary": {
            "questions_with_findings": sum(bool(finding["categories"]) for finding in findings),
            "category_counts": dict(sorted(counts.items())),
        },
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a RAG failure-taxonomy report")
    parser.add_argument("--input", default="eval/reports/ragas_report.json")
    parser.add_argument("--output", default="eval/reports/failure_taxonomy_report.json")
    args = parser.parse_args()
    input_path = repository_path(args.input)
    output_path = repository_path(args.output)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    taxonomy = analyse_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output_path), "summary": taxonomy["summary"]}, indent=2))


if __name__ == "__main__":
    main()

"""AC-09 and NFR-06 deterministic evaluation-harness tests."""

from __future__ import annotations

from pathlib import Path

from eval.ragas_eval import evaluate
from src.retrieval.query_transform import QueryPlan
from src.retrieval.types import RetrievedChunk


class FakePipeline:
    """Returns the expected clause for every supported case, without model downloads."""

    def __init__(self, clause_by_question: dict[str, list[str]]) -> None:
        self.clause_by_question = clause_by_question

    def retrieve(self, question: str):  # type: ignore[no-untyped-def]
        clauses = self.clause_by_question[question]
        contexts = [
            RetrievedChunk(
                chunk_id=f"{clause}::p1",
                text="test evidence",
                metadata={"clause_id": clause},
                score=1.0,
                source="cross_encoder",
            )
            for clause in clauses
        ]
        return QueryPlan(question, [question], "identity"), contexts, contexts


def test_ragas_report_is_generated(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """AC-09/NFR-06: re-runnable evaluation writes deterministic numeric retrieval metrics."""
    root = Path(__file__).resolve().parents[1]
    import json

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    entries = json.loads((root / "data" / "golden_eval_set.json").read_text(encoding="utf-8"))
    pipeline = FakePipeline({entry["question"]: entry["expected_context"] for entry in entries})
    settings = {
        "project": {"corpus_dir": str(root / "data" / "raw"), "golden_set_path": str(root / "data" / "golden_eval_set.json")},
        "evaluation": {"report_dir": str(tmp_path / "reports")},
    }

    first = evaluate(settings=settings, pipeline=pipeline, run_ragas=False)  # type: ignore[arg-type]
    second = evaluate(settings=settings, pipeline=pipeline, run_ragas=False)  # type: ignore[arg-type]
    report_path = tmp_path / "reports" / "ragas_report.json"

    assert report_path.exists()
    assert first["aggregate"] == second["aggregate"]
    assert first["aggregate"]["context_recall"] == 1.0
    assert first["ragas"]["faithfulness"] is None

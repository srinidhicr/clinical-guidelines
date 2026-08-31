"""AC-09 and NFR-06 deterministic evaluation-harness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.ragas_eval import LegacyRagasEmbeddingAdapter, _ragas_result_payload, evaluate
from eval.failure_taxonomy import analyse_report
from eval.llm_comparison import compare_models
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


class FakeFrame:
    def __init__(self, rows: list[dict[str, float]]) -> None:
        self.rows = rows

    def to_dict(self, orient: str) -> list[dict[str, float]]:
        assert orient == "records"
        return self.rows


class FakeRagasResult:
    def __init__(self, rows: list[dict[str, float]]) -> None:
        self.rows = rows

    def to_pandas(self) -> FakeFrame:
        return FakeFrame(self.rows)


class ModernEmbeddingFake:
    def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


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


def test_ragas_payload_preserves_per_question_scores() -> None:
    """AC-09: all four judge metrics are retained for every golden question."""
    entries = [
        {"id": "Q01", "type": "single_clause", "ac_ref": "AC-02"},
        {"id": "Q02", "type": "multi_clause", "ac_ref": "AC-07"},
    ]
    rows = [
        {"context_precision": 0.8, "context_recall": 0.9, "faithfulness": 1.0, "answer_relevancy": 0.7},
        {"context_precision": 0.6, "context_recall": 1.0, "faithfulness": 0.8, "response_relevancy": 0.9},
    ]

    payload = _ragas_result_payload(FakeRagasResult(rows), entries)

    assert [item["id"] for item in payload["items"]] == ["Q01", "Q02"]
    assert payload["items"][1]["answer_relevancy"] == 0.9
    assert payload["aggregate"] == {
        "context_precision": pytest.approx(0.7),
        "context_recall": pytest.approx(0.95),
        "faithfulness": pytest.approx(0.9),
        "answer_relevancy": pytest.approx(0.8),
    }


def test_ragas_payload_rejects_missing_item_metric() -> None:
    """AC-09: an incomplete live evaluation cannot be reported as completed."""
    entries = [{"id": "Q01", "type": "single_clause", "ac_ref": "AC-02"}]
    rows = [{"context_precision": 1.0, "context_recall": 1.0, "faithfulness": 1.0}]

    with pytest.raises(ValueError, match="answer_relevancy.*Q01"):
        _ragas_result_payload(FakeRagasResult(rows), entries)


def test_embedding_adapter_supports_legacy_answer_relevancy_methods() -> None:
    """The pinned legacy metric can consume the modern Gemini embedding provider."""
    adapter = LegacyRagasEmbeddingAdapter(ModernEmbeddingFake())

    assert adapter.embed_query("abc") == [3.0]
    assert adapter.embed_documents(["a", "abcd"]) == [[1.0], [4.0]]


def test_subset_evaluation_writes_separate_report(tmp_path: Path) -> None:
    """A smoke run selects one question without overwriting the full report."""
    root = Path(__file__).resolve().parents[1]
    import json

    entries = json.loads((root / "data" / "golden_eval_set.json").read_text(encoding="utf-8"))
    selected = entries[0]
    pipeline = FakePipeline({selected["question"]: selected["expected_context"]})
    settings = {
        "project": {
            "corpus_dir": str(root / "data" / "raw"),
            "golden_set_path": str(root / "data" / "golden_eval_set.json"),
        },
        "evaluation": {"report_dir": str(tmp_path / "reports")},
    }

    report = evaluate(
        settings=settings,
        pipeline=pipeline,  # type: ignore[arg-type]
        run_ragas=False,
        question_ids=[selected["id"]],
        report_filename="ragas_smoke_report.json",
    )

    assert report["question_count"] == 1
    assert report["evaluation_scope"] == "subset"
    assert (tmp_path / "reports" / "ragas_smoke_report.json").exists()
    assert not (tmp_path / "reports" / "ragas_report.json").exists()


def test_failure_taxonomy_separates_retrieval_grounding_and_synthesis() -> None:
    """RAG evaluation failures are assigned to the layer supported by evidence."""
    report = {
        "evaluation_scope": "full_golden_set",
        "question_count": 2,
        "items": [
            {"id": "Q01", "type": "single_clause", "ac_ref": "AC-02", "expected_clause_ids": ["CLAUSE-1"], "retrieved_clause_ids": ["OTHER"], "context_precision": 0.0, "context_recall": 0.0},
            {"id": "Q20", "type": "abstain", "ac_ref": "AC-05", "expected_clause_ids": [], "retrieved_clause_ids": ["OTHER"], "context_precision": 0.0, "context_recall": 1.0},
        ],
        "ragas": {"status": "completed", "items": [
            {"id": "Q01", "faithfulness": 0.7, "answer_relevancy": 0.6},
            {"id": "Q20", "faithfulness": 1.0, "answer_relevancy": 1.0},
        ]},
    }

    taxonomy = analyse_report(report)

    assert taxonomy["findings"][0]["categories"] == [
        "retrieval_miss", "retrieval_noise", "grounding_failure", "synthesis_failure"
    ]
    assert taxonomy["findings"][1]["categories"] == ["abstention_retrieval_risk"]
    assert taxonomy["summary"]["category_counts"]["grounding_failure"] == 1


def test_model_comparison_requires_two_completed_full_runs_for_selection(tmp_path: Path) -> None:
    """AC-10: model comparison keeps the judge fixed and records a defensible choice."""
    settings = {
        "project": {"corpus_dir": "data/raw", "golden_set_path": "data/golden_eval_set.json"},
        "generation": {"model_name": "gemini-a", "temperature": 0.0},
        "evaluation": {
            "report_dir": str(tmp_path / "reports"),
            "judge_model": "gemini-judge",
            "embedding_model": "gemini-embedding-001",
            "candidate_models": ["gemini-a", "gemini-b"],
        },
    }

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        model = kwargs["settings"]["generation"]["model_name"]  # type: ignore[index]
        assert kwargs["pipeline"]._generation_model_override == model  # type: ignore[attr-defined,index]
        return {
            "ragas": {
                "status": "completed",
                "aggregate": {
                    "context_precision": 0.8,
                    "context_recall": 0.9,
                    "faithfulness": 0.95 if model == "gemini-b" else 0.85,
                    "answer_relevancy": 0.9,
                },
            }
        }

    report = compare_models(settings=settings, evaluator=fake_evaluate)  # type: ignore[arg-type]

    assert report["status"] == "completed"
    assert report["selection"]["selected_generation_model"] == "gemini-b"
    assert all(item["judge_model"] == "gemini-judge" for item in report["models"])

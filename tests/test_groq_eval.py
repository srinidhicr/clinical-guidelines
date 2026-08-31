"""Offline tests for the optional, instructor-authorized Groq RAGAS runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.ragas_groq_eval import (
    GenerationStageUnavailable,
    _select_entries,
    collect_generation_dataset,
    create_groq_judge_llm,
)
from src.generation.schema import ClauseCitation, GuidelineAnswer
from src.retrieval.types import RetrievedChunk


def test_select_entries_rejects_unknown_question_id() -> None:
    entries = [{"id": "Q01"}]

    try:
        _select_entries(entries, ["Q99"])
    except ValueError as error:
        assert "Q99" in str(error)
    else:
        raise AssertionError("Unknown question ID should be rejected.")


def test_groq_judge_uses_openai_compatible_endpoint() -> None:
    """Pinned RAGAS receives the chat-completions interface it expects."""
    captured: dict[str, object] = {}

    def fake_client_factory(**kwargs: object) -> object:
        captured["client"] = kwargs
        return "groq-compatible-client"

    def fake_llm_builder(*args: object, **kwargs: object) -> str:
        captured["builder"] = {"args": args, "kwargs": kwargs}
        return "judge"

    judge = create_groq_judge_llm(
        "test-key",
        "openai/gpt-oss-20b",
        client_factory=fake_client_factory,
        llm_builder=fake_llm_builder,
    )

    assert judge == "judge"
    assert captured["client"] == {
        "api_key": "test-key", "base_url": "https://api.groq.com/openai/v1"
    }
    assert captured["builder"] == {
        "args": ("openai/gpt-oss-20b",),
        "kwargs": {
            "provider": "openai",
            "client": "groq-compatible-client",
            "temperature": 0.0,
            "max_tokens": 4096,
            "reasoning_effort": "low",
        },
    }


def test_generation_audit_counts_grounded_and_abstained_answers() -> None:
    """The audit documents every generation result without storing its answer text."""
    context = RetrievedChunk(
        chunk_id="C1::p1",
        text="Evidence.",
        metadata={"clause_id": "C1"},
        score=1.0,
        source="cross_encoder",
    )

    class FakePipeline:
        settings = {"generation": {"model_name": "gemini-test"}}

        def retrieve(self, question: str):  # type: ignore[no-untyped-def]
            return None, None, [context]

        def ask(self, question: str, request_id: str, contexts: list[RetrievedChunk]):  # type: ignore[no-untyped-def]
            return GuidelineAnswer(
                answer="Stored only in the RAGAS dataset, not the audit.",
                citations=(
                    [
                        ClauseCitation(
                            document_id="D1", guideline_source="S1", clause_id="C1", section="Section 1"
                        )
                    ]
                    if question == "supported"
                    else []
                ),
                grounding_confidence=1.0 if question == "supported" else 0.0,
                grounded=question == "supported",
            )

    _, audit = collect_generation_dataset(
        FakePipeline(),  # type: ignore[arg-type]
        [
            {"id": "Q01", "question": "supported", "reference_answer": "reference", "type": "single_clause"},
            {"id": "Q02", "question": "unsupported", "reference_answer": "reference", "type": "abstain"},
        ],
    )

    assert audit["question_count"] == 2
    assert audit["grounded_answer_count"] == 1
    assert audit["abstained_answer_count"] == 1
    assert "answer" not in audit["items"][0]


def test_generation_checkpoint_resumes_after_provider_failure(tmp_path: Path) -> None:
    """A retry reuses grounded answers instead of spending Gemini calls again."""
    context = RetrievedChunk("C1::p1", "Evidence.", {"clause_id": "C1"}, 1.0, "cross_encoder")
    entries = [
        {"id": "Q01", "question": "one", "reference_answer": "reference", "type": "single_clause"},
        {"id": "Q02", "question": "two", "reference_answer": "reference", "type": "single_clause"},
    ]
    checkpoint = tmp_path / "generation_checkpoint.json"

    def answer(grounded: bool) -> GuidelineAnswer:
        return GuidelineAnswer(
            answer="Evidence." if grounded else "Unsupported.",
            citations=(
                [ClauseCitation(document_id="D1", guideline_source="S1", clause_id="C1", section="Section 1")]
                if grounded else []
            ),
            guideline_source="S1" if grounded else None,
            section="Section 1" if grounded else None,
            grounding_confidence=1.0 if grounded else 0.0,
            grounded=grounded,
        )

    class FirstAttempt:
        settings = {"generation": {"model_name": "gemini-test"}}
        def retrieve(self, question: str):  # type: ignore[no-untyped-def]
            return None, None, [context]
        def ask(self, question: str, **_: object) -> GuidelineAnswer:
            return answer(question == "one")

    with pytest.raises(GenerationStageUnavailable, match="Q02"):
        collect_generation_dataset(FirstAttempt(), entries, checkpoint)  # type: ignore[arg-type]

    class RetryAttempt:
        settings = {"generation": {"model_name": "gemini-test"}}
        calls: list[str] = []
        def retrieve(self, question: str):  # type: ignore[no-untyped-def]
            return None, None, [context]
        def ask(self, question: str, **_: object) -> GuidelineAnswer:
            self.calls.append(question)
            return answer(True)

    retry = RetryAttempt()
    _, audit = collect_generation_dataset(retry, entries, checkpoint)  # type: ignore[arg-type]

    assert retry.calls == ["two"]
    assert audit["reused_checkpoint_answer_count"] == 1
    assert audit["new_generation_answer_count"] == 1

"""AC-02, AC-05, AC-06 and NFR-05 tests for grounded generation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.generation.generator import GenerationSettings, generate_grounded_answer, parse_model_response
from src.generation.schema import GuidelineAnswer, gemini_response_schema
from src.retrieval.types import RetrievedChunk


def _context() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="CARD-001-S2::p1",
        text="Cardaprilol is first-line.",
        metadata={
            "document_id": "CARD-001",
            "source": "SYN-GUIDE-CARD-001",
            "clause_id": "CARD-001-S2",
            "section_heading": "Section 2: First-Line Management",
        },
        score=0.9,
        source="cross_encoder",
    )


def _valid_json() -> str:
    return json.dumps(
        {
            "answer": "The guideline lists Cardaprilol as first-line.",
            "citations": [
                {
                    "document_id": "CARD-001",
                    "guideline_source": "SYN-GUIDE-CARD-001",
                    "clause_id": "CARD-001-S2",
                    "section": "Section 2: First-Line Management",
                }
            ],
            "guideline_source": "SYN-GUIDE-CARD-001",
            "section": "Section 2: First-Line Management",
            "strength_of_recommendation": "Strong",
            "grounding_confidence": 0.92,
            "grounded": True,
        }
    )


class ScriptedModel:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate_content(self, **_: object) -> object:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"text": outcome})()


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.models = ScriptedModel(outcomes)


def _settings() -> GenerationSettings:
    return GenerationSettings("test-model", 0.0, 3, 0.0, 0.42)


def test_answer_has_valid_citation(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-02: a generated citation must correspond to retrieved clause evidence."""
    monkeypatch.setattr("src.generation.generator._generation_config", lambda _: {})
    answer = generate_grounded_answer("What is first line?", [_context()], _settings(), FakeClient([_valid_json()]))

    assert answer.grounded is True
    assert answer.citations[0].clause_id == "CARD-001-S2"


def test_abstains_on_unsupported_question() -> None:
    """AC-05: no retrieved context takes the explicit abstention path without an LLM call."""
    answer = generate_grounded_answer("Unsupported question", [], _settings())

    assert answer.grounded is False
    assert answer.citations == []
    assert "does not support" in answer.answer


def test_invalid_citation_becomes_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-05: a model cannot fabricate a clause ID outside the supplied evidence."""
    payload = json.loads(_valid_json())
    payload["citations"][0]["clause_id"] = "INVENTED-S9"
    monkeypatch.setattr("src.generation.generator._generation_config", lambda _: {})

    answer = generate_grounded_answer("What is first line?", [_context()], _settings(), FakeClient([json.dumps(payload)]))

    assert answer.grounded is False
    assert answer.citations == []


def test_response_matches_pydantic_schema() -> None:
    """AC-06: schema violations are rejected instead of silently accepted."""
    with pytest.raises(ValidationError):
        GuidelineAnswer.model_validate({"answer": "Missing required fields"})

    assert parse_model_response(_valid_json(), [_context()]).grounded is True
    assert "additionalProperties" not in str(gemini_response_schema())


def test_retries_on_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR-05: transient provider failures are retried before returning a safe result."""
    monkeypatch.setattr("src.generation.generator._generation_config", lambda _: {})
    monkeypatch.setattr("src.generation.generator.time.sleep", lambda _: None)
    client = FakeClient([RuntimeError("temporary"), RuntimeError("temporary"), _valid_json()])

    answer = generate_grounded_answer("What is first line?", [_context()], _settings(), client)

    assert client.models.calls == 3
    assert answer.grounded is True

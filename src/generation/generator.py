"""Grounded Gemini generation, retries, citation validation, and abstention."""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
from pydantic import ValidationError

from src.generation.prompts import build_grounded_prompt
from src.generation.schema import GuidelineAnswer, gemini_response_schema
from src.retrieval.types import RetrievedChunk
from src.utils.logging import log_generation_failure


ABSTENTION_TEXT = "The synthetic guideline corpus does not support an answer to this question."


class GenerateContent(Protocol):
    def generate_content(self, *, model: str, contents: str, config: Any) -> Any: ...


class GeminiClient(Protocol):
    models: GenerateContent


@dataclass(frozen=True)
class GenerationSettings:
    model_name: str
    temperature: float
    max_retries: int
    retry_backoff_seconds: float
    minimum_grounding_confidence: float


def abstain(reason: str = ABSTENTION_TEXT, confidence: float = 0.0) -> GuidelineAnswer:
    """Return the safe structured response used for unsupported or failed requests."""
    return GuidelineAnswer(
        answer=reason,
        citations=[],
        guideline_source=None,
        section=None,
        strength_of_recommendation=None,
        grounding_confidence=confidence,
        grounded=False,
    )


def _create_client() -> GeminiClient:
    """Construct the Gemini Developer API client only when generation is required."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured in .env or the process environment.")
    from google import genai

    return genai.Client(api_key=api_key)


def _generation_config(temperature: float) -> Any:
    """Use Gemini JSON-schema mode, with the Pydantic contract as response schema."""
    from google.genai import types

    return types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=gemini_response_schema(),
    )


def _supported_citation_keys(contexts: list[RetrievedChunk]) -> set[tuple[str, str, str, str]]:
    return {
        (
            str(context.metadata["document_id"]),
            str(context.metadata["source"]),
            str(context.metadata["clause_id"]),
            str(context.metadata["section_heading"]),
        )
        for context in contexts
    }


def validate_citations(answer: GuidelineAnswer, contexts: list[RetrievedChunk]) -> GuidelineAnswer:
    """Reject citations absent from the actual retrieved evidence."""
    supported = _supported_citation_keys(contexts)
    for citation in answer.citations:
        key = (citation.document_id, citation.guideline_source, citation.clause_id, citation.section)
        if key not in supported:
            raise ValueError(f"Generated citation is not present in retrieved evidence: {citation.clause_id}")
    return answer


def parse_model_response(raw_json: str, contexts: list[RetrievedChunk]) -> GuidelineAnswer:
    """Validate schema and citations; callers can distinguish validation failures clearly."""
    answer = GuidelineAnswer.model_validate_json(raw_json)
    return validate_citations(answer, contexts)


def generate_grounded_answer(
    query: str,
    contexts: list[RetrievedChunk],
    settings: GenerationSettings,
    client: GeminiClient | None = None,
    logger: logging.Logger | None = None,
    request_id: str = "unassigned",
) -> GuidelineAnswer:
    """Generate a cited answer or safely abstain after unsupported/repeatedly failed calls."""
    if not contexts:
        return abstain()
    active_client = client
    last_error: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            active_client = active_client or _create_client()
            response = active_client.models.generate_content(
                model=settings.model_name,
                contents=build_grounded_prompt(query, contexts),
                config=_generation_config(settings.temperature),
            )
            answer = parse_model_response(str(response.text), contexts)
            if not answer.grounded or answer.grounding_confidence < settings.minimum_grounding_confidence:
                return abstain(confidence=answer.grounding_confidence)
            return answer
        except (ValidationError, ValueError) as error:
            # Invalid structured output/citations must never be returned as a plausible answer.
            return abstain(reason=f"{ABSTENTION_TEXT} Generated output could not be validated.")
        except Exception as error:  # Provider/network failures are retried, then made safe.
            last_error = error
            if attempt + 1 < settings.max_retries:
                time.sleep(settings.retry_backoff_seconds * (2**attempt))
    if last_error is not None and logger is not None:
        log_generation_failure(logger, request_id, last_error)
    return abstain(reason=f"{ABSTENTION_TEXT} The generation service is currently unavailable.")

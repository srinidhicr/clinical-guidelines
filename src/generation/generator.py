"""Grounded Gemini generation, retries, citation validation, and abstention."""

from __future__ import annotations

import os
import re
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
    minimum_evidence_term_overlap: float = 0.10


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


def _cited_context_text(answer: GuidelineAnswer, contexts: list[RetrievedChunk]) -> str:
    """Return only the passages explicitly claimed as support by the answer."""
    citation_keys = {
        (citation.document_id, citation.guideline_source, citation.clause_id, citation.section)
        for citation in answer.citations
    }
    return "\n".join(
        context.text
        for context in contexts
        if (
            str(context.metadata["document_id"]),
            str(context.metadata["source"]),
            str(context.metadata["clause_id"]),
            str(context.metadata["section_heading"]),
        )
        in citation_keys
    )


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "with", "what", "which", "who", "will", "would",
    "guideline", "guidelines", "list", "lists", "states", "state", "synthetic", "corpus",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _has_minimum_evidence_support(query: str, contexts: list[RetrievedChunk], minimum_overlap: float) -> bool:
    """Reject clearly unrelated retrieval before a model confidence value is considered."""
    query_terms = _content_tokens(query)
    if not query_terms:
        return True
    evidence_terms = _content_tokens(" ".join(context.text for context in contexts))
    return len(query_terms & evidence_terms) / len(query_terms) >= minimum_overlap


def _is_patient_specific_request(query: str) -> bool:
    """Catch first-person requests for an individual decision without blocking generic cases."""
    return bool(
        re.search(
            r"\b(?:should i|should we|for me|my (?:dose|dosing|treatment|medication|test|lab|symptom|condition)|"
            r"my patient|patient named|this patient should)\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def validate_claim_support(answer: GuidelineAnswer, contexts: list[RetrievedChunk], query: str = ""
) -> GuidelineAnswer:
    """Require each factual answer sentence to be materially supported by its cited text.

    This local guard is intentionally conservative: quantities must occur verbatim in the
    cited clause, and each factual sentence needs at least two meaningful terms and a
    majority of its meaningful terms in the cited evidence. It is a supplement to, not a
    replacement for, RAGAS faithfulness evaluation.
    """
    cited_text = _cited_context_text(answer, contexts)
    evidence_terms = _content_tokens(cited_text)
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", cited_text))
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
    allowed_numbers = evidence_numbers | query_numbers
    for sentence in re.split(r"(?<=[A-Za-z])(?:[.!?]+)(?:\s+|$)", answer.answer):
        terms = _content_tokens(sentence)
        if len(terms) < 2:
            continue
        sentence_numbers = set(re.findall(r"\d+(?:\.\d+)?", sentence))
        overlap = len(terms & evidence_terms) / len(terms)
        if not sentence_numbers.issubset(allowed_numbers) or overlap < 0.60:
            raise ValueError("Generated answer contains a claim not sufficiently supported by its cited evidence.")
    return answer


def validate_citations(answer: GuidelineAnswer, contexts: list[RetrievedChunk]) -> GuidelineAnswer:
    """Reject absent citations and inconsistent top-level provenance."""
    supported = _supported_citation_keys(contexts)
    for citation in answer.citations:
        key = (citation.document_id, citation.guideline_source, citation.clause_id, citation.section)
        if key not in supported:
            raise ValueError(f"Generated citation is not present in retrieved evidence: {citation.clause_id}")
    if answer.grounded:
        if answer.guideline_source is None or answer.section is None:
            raise ValueError("Grounded answers require top-level guideline source and section.")
        citation_provenance = {(citation.guideline_source, citation.section) for citation in answer.citations}
        if (answer.guideline_source, answer.section) not in citation_provenance:
            raise ValueError("Top-level guideline source and section must match one cited clause.")
    return answer


def parse_model_response(raw_json: str, contexts: list[RetrievedChunk], query: str = "") -> GuidelineAnswer:
    """Validate schema, provenance, and local claim support before returning an answer."""
    answer = GuidelineAnswer.model_validate_json(raw_json)
    answer = validate_citations(answer, contexts)
    return validate_claim_support(answer, contexts, query) if answer.grounded else answer


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
    if _is_patient_specific_request(query):
        return abstain("This assistant is a synthetic-guideline reference tool and cannot make an individual treatment decision.")
    if not _has_minimum_evidence_support(query, contexts, settings.minimum_evidence_term_overlap):
        return abstain("The retrieved synthetic guideline evidence is not sufficiently relevant to support an answer.")
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
            print(f"[DEBUG raw response for {request_id}]: {response.text}")
            answer = parse_model_response(str(response.text), contexts, query)
            if not answer.grounded or answer.grounding_confidence < settings.minimum_grounding_confidence:
                return abstain(confidence=answer.grounding_confidence)
            return answer
        except (ValidationError, ValueError) as error:
            # Invalid structured output/citations must never be returned as a plausible answer.
            print(f"[DEBUG] validation failure for request_id={request_id}: {error}")
            return abstain(reason=f"{ABSTENTION_TEXT} Generated output could not be validated.")
        except Exception as error:  # Provider/network failures are retried, then made safe.
            print(f"[DEBUG] provider/parse exception (attempt {attempt+1}) for {request_id}: {type(error).__name__}: {error}")
            last_error = error
            if attempt + 1 < settings.max_retries:
                time.sleep(settings.retry_backoff_seconds * (2**attempt))
    if last_error is not None and logger is not None:
        log_generation_failure(logger, request_id, last_error)
    return abstain(reason=f"{ABSTENTION_TEXT} The generation service is currently unavailable.")

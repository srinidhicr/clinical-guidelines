"""Validated response contract for grounded clinical-guideline answers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClauseCitation(BaseModel):
    """A human-readable provenance pointer into the synthetic corpus."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    guideline_source: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    section: str = Field(min_length=1)


class GuidelineAnswer(BaseModel):
    """Only this schema may cross the generation/pipeline boundary."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[ClauseCitation] = Field(default_factory=list)
    guideline_source: str | None = None
    section: str | None = None
    strength_of_recommendation: str | None = None
    grounding_confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool

    @model_validator(mode="after")
    def enforce_grounding_contract(self) -> "GuidelineAnswer":
        if self.grounded and not self.citations:
            raise ValueError("Grounded answers require at least one clause-level citation.")
        if not self.grounded and self.citations:
            raise ValueError("Abstained answers must not include citations.")
        return self


def gemini_response_schema() -> dict[str, object]:
    """Return Gemini's supported schema subset, then validate strictly with Pydantic.

    Pydantic's JSON schema uses `additionalProperties: false` for `extra="forbid"`.
    Gemini's Developer API currently rejects that keyword, so its response schema is kept
    deliberately minimal here; `GuidelineAnswer` still rejects unwanted fields locally.
    """
    citation_properties = {
        "document_id": {"type": "STRING"},
        "guideline_source": {"type": "STRING"},
        "clause_id": {"type": "STRING"},
        "section": {"type": "STRING"},
    }
    return {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING"},
            "citations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": citation_properties,
                    "required": list(citation_properties),
                },
            },
            "guideline_source": {"type": "STRING", "nullable": True},
            "section": {"type": "STRING", "nullable": True},
            "strength_of_recommendation": {"type": "STRING", "nullable": True},
            "grounding_confidence": {"type": "NUMBER"},
            "grounded": {"type": "BOOLEAN"},
        },
        "required": [
            "answer",
            "citations",
            "guideline_source",
            "section",
            "strength_of_recommendation",
            "grounding_confidence",
            "grounded",
        ],
    }

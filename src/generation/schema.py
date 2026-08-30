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

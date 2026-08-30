"""Clause-preserving chunking for clinical guideline retrieval.

Sections are the natural semantic and citation boundary in this corpus: every section has
one stable clause ID. We therefore preserve a full section as one chunk where possible,
and split only overly long sections at paragraph/table-row boundaries. This avoids naive
fixed-size splitting that could separate a dose from its qualification or citation.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from src.ingestion.loaders import GuidelineSection


@dataclass(frozen=True)
class Chunk:
    """Retrieval unit with provenance sufficient for clause-level citation."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int]

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {"chunk_id": self.chunk_id, "text": self.text, "metadata": self.metadata}


def _joined_length(units: list[str]) -> int:
    """Return the rendered length of content joined with paragraph boundaries."""
    return len("\n\n".join(units))


def _overlap_tail(
    units: list[str], overlap_characters: int, available_characters: int
) -> str:
    """Return bounded trailing context without allowing overlap to exceed chunk size."""
    if overlap_characters <= 0 or available_characters <= 0:
        return ""
    previous_text = "\n\n".join(units)
    retained_length = min(overlap_characters, available_characters)
    return previous_text[-retained_length:].lstrip()


def _split_at_boundaries(
    text: str, max_characters: int, overlap_characters: int
) -> list[str]:
    """Split at semantic boundaries and retain bounded trailing context when necessary."""
    if len(text) <= max_characters:
        return [text]
    units = [unit.strip() for unit in text.split("\n\n") if unit.strip()]
    if any(len(unit) > max_characters for unit in units):
        expanded: list[str] = []
        for unit in units:
            if len(unit) <= max_characters:
                expanded.append(unit)
            else:
                expanded.extend(line.strip() for line in unit.splitlines() if line.strip())
        units = expanded

    chunks: list[str] = []
    current_units: list[str] = []
    for unit in units:
        candidate_units = [*current_units, unit]
        if current_units and _joined_length(candidate_units) > max_characters:
            chunks.append("\n\n".join(current_units))
            available_for_overlap = max_characters - len(unit) - 2
            overlap = _overlap_tail(current_units, overlap_characters, available_for_overlap)
            current_units = [*([overlap] if overlap else []), unit]
        else:
            current_units = candidate_units
    if current_units:
        chunks.append("\n\n".join(current_units))
    return chunks


def chunk_sections(
    sections: list[GuidelineSection], max_characters: int, overlap_characters: int = 0
) -> list[Chunk]:
    """Create citation-safe chunks with inherited document and clause metadata."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    if overlap_characters < 0:
        raise ValueError("overlap_characters cannot be negative")
    chunks: list[Chunk] = []
    for section in sections:
        prefix = f"{section.document_title}\n{section.section_heading}\n[clause id: {section.clause_id}]"
        parts = _split_at_boundaries(section.text, max_characters, overlap_characters)
        for part_number, part in enumerate(parts, start=1):
            metadata = asdict(section)
            metadata.pop("text")
            metadata["chunk_part"] = part_number
            metadata["chunk_parts"] = len(parts)
            chunk_id = f"{section.clause_id}::p{part_number}"
            chunks.append(Chunk(chunk_id=chunk_id, text=f"{prefix}\n\n{part}", metadata=metadata))
    return chunks

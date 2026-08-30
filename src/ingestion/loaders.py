"""Explicit markdown corpus loader.

The loader deliberately performs only parsing and metadata extraction. Retrieval and
embedding decisions belong to later pipeline stages, keeping corpus provenance auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


METADATA_PATTERN = re.compile(r"^(Source|Specialty|Version|Issuing Body):\s*(.+)$")
CLAUSE_PATTERN = re.compile(r"^\[clause id:\s*([^\]]+)\]\s*$", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$")


@dataclass(frozen=True)
class GuidelineSection:
    """A citation-addressable section in one source document."""

    document_id: str
    source: str
    specialty: str
    version: str
    issuing_body: str
    document_title: str
    document_type: str
    section_heading: str
    clause_id: str
    text: str
    path: str


def infer_document_type(path: Path) -> str:
    """Derive the controlled document type from the explicit filename convention."""
    name = path.stem.lower()
    if "dosing" in name or "contraindication" in name:
        return "dosing_contraindication_reference"
    if "referral" in name:
        return "referral_criteria"
    if "comorbidity" in name:
        return "comorbidity_pathway"
    if "care_pathway" in name:
        return "care_pathway"
    return "other"


def load_markdown_document(path: Path) -> list[GuidelineSection]:
    """Parse one markdown guideline into sections, requiring stable clause IDs."""
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    metadata: dict[str, str] = {}
    title = ""
    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        # Document metadata is valid only in the pre-section header. Body prose may
        # legitimately contain labels such as "Source:" and must not override it.
        if SECTION_PATTERN.match(line):
            break
        match = METADATA_PATTERN.match(line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()

    required = ("Source", "Specialty", "Version", "Issuing Body")
    missing = [field for field in required if not metadata.get(field)]
    if not title or missing:
        raise ValueError(f"{path.name} is missing title or required metadata: {', '.join(missing)}")

    starts: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        heading = SECTION_PATTERN.match(line)
        if not heading:
            continue
        clause_id = ""
        for following in lines[index + 1 : min(index + 5, len(lines))]:
            clause = CLAUSE_PATTERN.match(following)
            if clause:
                clause_id = clause.group(1)
                break
            if following.strip() and not following.startswith("<!--"):
                break
        if not clause_id:
            raise ValueError(f"{path.name}: section '{heading.group(1)}' lacks a clause id.")
        starts.append((index, heading.group(1), clause_id))

    if not starts:
        raise ValueError(f"{path.name} contains no clause-addressable sections.")

    sections: list[GuidelineSection] = []
    for position, (start, heading, clause_id) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        body = CLAUSE_PATTERN.sub("", body, count=1).strip()
        if not body:
            raise ValueError(f"{path.name}: clause {clause_id} has no text.")
        sections.append(
            GuidelineSection(
                document_id=path.stem.split("_")[0],
                source=metadata["Source"],
                specialty=metadata["Specialty"],
                version=metadata["Version"],
                issuing_body=metadata["Issuing Body"],
                document_title=title,
                document_type=infer_document_type(path),
                section_heading=heading,
                clause_id=clause_id,
                text=body,
                path=str(path),
            )
        )
    return sections


def load_corpus(corpus_dir: Path) -> list[GuidelineSection]:
    """Load supported markdown files in deterministic filename order."""
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown guideline documents found in {corpus_dir}")
    return [section for path in paths for section in load_markdown_document(path)]

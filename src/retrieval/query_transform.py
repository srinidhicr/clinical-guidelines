"""Transparent, deterministic query transformation and decomposition before retrieval.

Avoids unpredictable LLM latency during retrieval while providing structured multi-part
query decomposition, contrastive comparison splitting, and clinical acronym expansion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYNTHETIC_ACRONYM_MAP = {
    "APES": "Stage 2 Arterial Pressure Elevation Syndrome",
    "CRID": "Chronic Rhythm Irregularity Disorder",
    "GDS": "Type S Glycemic Dysregulation Syndrome",
    "THID": "Thyroid Hormone Imbalance Disorder",
    "SMRS": "Systemic Microbial Response Syndrome",
    "RSTID": "Recurrent Soft-Tissue Infection Disorder",
    "PRFDS": "Progressive Renal Filtration Decline Syndrome",
    "ERD": "Electrolyte Regulation Disorder",
    "CARS": "Chronic Airway Restriction Syndrome",
    "RBID": "Recurrent Bronchial Inflammation Disorder",
    "CJIS": "Chronic Joint Inflammation Syndrome",
    "ACTD": "Autoimmune Connective Tissue Disorder",
}


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    retrieval_queries: list[str]
    strategy: str


def _expand_acronyms(text: str) -> str:
    """Expand synthetic clinical acronyms with their full condition names for hybrid search."""
    expanded = text
    for acronym, full_name in SYNTHETIC_ACRONYM_MAP.items():
        pattern = rf"\b{acronym}\b"
        if re.search(pattern, expanded) and full_name.lower() not in expanded.lower():
            expanded = re.sub(pattern, f"{acronym} ({full_name})", expanded)
    return expanded


def transform_query(query: str) -> QueryPlan:
    """Transform, expand, or decompose complex queries into retrievable sub-queries."""
    cleaned = " ".join(query.split())

    # Strategy 1: Explicit comparison decomposition
    comparison = re.match(
        r"^(?:compare|what is the difference between)\s+(.+?)\s+(?:of|between|in)\s+(.+?)\s+and\s+(.+?)[?.]?$",
        cleaned,
        re.IGNORECASE,
    )
    if comparison:
        subject, left, right = comparison.group(1), comparison.group(2), comparison.group(3)
        expanded_left = _expand_acronyms(f"{subject} of {left}")
        expanded_right = _expand_acronyms(f"{subject} of {right}")
        return QueryPlan(
            original_query=cleaned,
            retrieval_queries=[cleaned, f"{subject} of {left}", f"{subject} of {right}"],
            strategy="comparison_decomposition",
        )

    # Strategy 2: Multi-part conjunction decomposition
    if re.search(r"\b(and|also|as well as|alongside)\b", cleaned, re.IGNORECASE):
        parts = re.split(r"\s+(?:and|also|as well as|alongside)\s+", cleaned, flags=re.IGNORECASE)
        focused = [part.strip(" ?.!") for part in parts if len(part.strip()) >= 12]
        if len(focused) >= 2:
            return QueryPlan(cleaned, [cleaned, *focused], "conjunction_decomposition")

    # Strategy 3: Acronym expansion for single-focus queries
    expanded = _expand_acronyms(cleaned)
    if expanded != cleaned:
        return QueryPlan(cleaned, [cleaned, expanded], "acronym_expansion")

    return QueryPlan(cleaned, [cleaned], "identity")


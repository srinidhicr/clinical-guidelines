"""Transparent deterministic query decomposition before retrieval.

The initial implementation avoids an unnecessary LLM call and keeps behaviour reproducible.
It preserves the original query and adds focused subqueries for comparison/multi-part forms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    retrieval_queries: list[str]
    strategy: str


def transform_query(query: str) -> QueryPlan:
    """Decompose explicit comparisons and conjunctions into retrievable focused queries."""
    cleaned = " ".join(query.split())
    comparison = re.match(r"^(compare)\s+(.+?)\s+of\s+(.+?)\s+and\s+(.+?)[?.]?$", cleaned, re.IGNORECASE)
    if comparison:
        subject, left, right = comparison.group(2), comparison.group(3), comparison.group(4)
        return QueryPlan(
            original_query=cleaned,
            retrieval_queries=[cleaned, f"{subject} of {left}", f"{subject} of {right}"],
            strategy="comparison_decomposition",
        )
    if re.search(r"\b(and|also|as well as)\b", cleaned, re.IGNORECASE):
        parts = re.split(r"\s+(?:and|also|as well as)\s+", cleaned, flags=re.IGNORECASE)
        focused = [part.strip(" ?.!") for part in parts if len(part.strip()) >= 12]
        if len(focused) >= 2:
            return QueryPlan(cleaned, [cleaned, *focused], "conjunction_decomposition")
    return QueryPlan(cleaned, [cleaned], "identity")

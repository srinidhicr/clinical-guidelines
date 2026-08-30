"""Prompts that restrict Gemini to retrieved, clause-addressable evidence."""

from __future__ import annotations

from src.retrieval.types import RetrievedChunk


SYSTEM_INSTRUCTION = """You are a reference assistant for a fully synthetic clinical-guideline corpus.
Answer only from the supplied evidence. Never use external medical knowledge, make a
patient-specific treatment decision, or invent a dose, threshold, source, or clause ID.
Every grounded claim needs support from a supplied clause. If the evidence does not fully
support an answer, return an abstention with grounded=false and no citations."""


def build_grounded_prompt(query: str, contexts: list[RetrievedChunk]) -> str:
    """Render retrieved clauses with all fields required for verifiable citations."""
    evidence = "\n\n".join(
        "\n".join(
            [
                f"[Evidence {index}]",
                f"Document ID: {context.metadata['document_id']}",
                f"Guideline source: {context.metadata['source']}",
                f"Clause ID: {context.metadata['clause_id']}",
                f"Section: {context.metadata['section_heading']}",
                f"Text: {context.text}",
            ]
        )
        for index, context in enumerate(contexts, start=1)
    )
    return f"""{SYSTEM_INSTRUCTION}

Question: {query}

Retrieved evidence:
{evidence}
"""

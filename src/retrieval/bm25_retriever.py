"""Independent lexical retrieval using BM25 over clause-preserving chunks."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.types import RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Use a deterministic lightweight tokenizer suitable for synthetic guideline text."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Lexical retriever kept independent from the FAISS semantic retriever."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk.")
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(str(chunk["text"])) for chunk in chunks])

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        if limit <= 0:
            return []
        scores = np.asarray(self.index.get_scores(tokenize(query)))
        ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:limit]
        return [
            RetrievedChunk(
                chunk_id=str(self.chunks[index]["chunk_id"]),
                text=str(self.chunks[index]["text"]),
                metadata=dict(self.chunks[index]["metadata"]),
                score=float(scores[index]),
                source="bm25",
            )
            for index in ranked_indices
        ]

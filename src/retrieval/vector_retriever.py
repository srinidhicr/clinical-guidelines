"""Independent semantic retrieval against the persisted local FAISS index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.ingestion.indexer import load_persisted_index
from src.retrieval.types import RetrievedChunk


class QueryEmbedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class VectorRetriever:
    """Semantic retriever that does not depend on BM25 or fusion implementation."""

    def __init__(self, index_dir: Path, embedder: QueryEmbedder, normalize_embeddings: bool = True) -> None:
        self.index, self.chunks, self.manifest = load_persisted_index(index_dir)
        self.embedder = embedder
        self.normalize_embeddings = normalize_embeddings

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        if limit <= 0:
            return []
        query_vector = np.asarray(
            self.embedder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("Query embedding must have shape (1, dimensions).")
        if not self.normalize_embeddings:
            faiss.normalize_L2(query_vector)
        scores, positions = self.index.search(query_vector, min(limit, self.index.ntotal))
        results: list[RetrievedChunk] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk["chunk_id"]),
                    text=str(chunk["text"]),
                    metadata=dict(chunk["metadata"]),
                    score=float(score),
                    source="vector",
                )
            )
        return results

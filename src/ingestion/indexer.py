"""Idempotent local FAISS indexing for clause-preserving guideline chunks.

An index manifest fingerprints the corpus plus chunking and embedding settings. Re-running
with the same signature loads the existing index without calling the embedding model.
Changing the corpus or signature causes a clean rebuild, preventing stale vectors from
surviving a source-document update.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.ingestion.chunking import Chunk, chunk_sections
from src.ingestion.loaders import GuidelineSection, load_corpus
from src.utils.config import load_settings, repository_path


INDEX_FILENAME = "guidelines.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"


class Embedder(Protocol):
    """Minimal protocol that permits deterministic test doubles."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class IndexBuildResult:
    """Auditable outcome of one indexing invocation."""

    index_dir: Path
    chunk_count: int
    vector_count: int
    embedded_chunk_count: int
    reused_existing_index: bool
    signature: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _corpus_fingerprint(corpus_dir: Path) -> str:
    """Fingerprint filenames and bytes, so an edited corpus cannot reuse stale vectors."""
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _create_embedder(model_name: str) -> Embedder:
    """Instantiate locally only when a rebuild actually requires embeddings."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_chunks(
    chunks: list[Chunk], embedder: Embedder, normalize_embeddings: bool
) -> np.ndarray:
    """Encode chunks into the float32 matrix required by FAISS."""
    vectors = embedder.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Embedding model returned an invalid matrix shape.")
    if not normalize_embeddings:
        faiss.normalize_L2(matrix)
    return matrix


def _read_manifest(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _deduplicate_chunks(chunks: list[Chunk], strategy: str) -> list[Chunk]:
    """Apply the configured exact-content deduplication strategy before embedding."""
    if strategy != "content_sha256":
        raise ValueError(f"Unsupported deduplication strategy: {strategy}")
    unique: list[Chunk] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        content_hash = chunk.content_hash()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique.append(chunk)
    return unique


def load_persisted_index(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    """Load a persisted index and its citation metadata together."""
    index_path = index_dir / INDEX_FILENAME
    chunks_path = index_dir / CHUNKS_FILENAME
    manifest = _read_manifest(index_dir)
    if not (index_path.exists() and chunks_path.exists() and manifest):
        raise FileNotFoundError(f"Persisted index is incomplete in {index_dir}")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise ValueError("Persisted FAISS vector count does not match chunk metadata.")
    return index, chunks, manifest


def build_index(
    sections: list[GuidelineSection],
    index_dir: Path,
    settings: dict[str, Any],
    corpus_fingerprint: str,
    embedder: Embedder | None = None,
) -> IndexBuildResult:
    """Build or reuse a FAISS IndexFlatIP plus JSON provenance metadata."""
    ingestion = settings["ingestion"]
    chunks = _deduplicate_chunks(chunk_sections(
        sections,
        max_characters=int(ingestion["max_chunk_characters"]),
        overlap_characters=int(ingestion["chunk_overlap_characters"]),
    ), strategy=str(ingestion["deduplicate_by"]))
    if not chunks:
        raise ValueError("No chunks were produced from the corpus.")

    signature_material = {
        "corpus_sha256": corpus_fingerprint,
        "embedding_model": settings["embedding"]["model_name"],
        "normalize_embeddings": settings["embedding"]["normalize_embeddings"],
        "chunk_strategy": ingestion["chunk_strategy"],
        "max_chunk_characters": ingestion["max_chunk_characters"],
        "chunk_overlap_characters": ingestion["chunk_overlap_characters"],
        "deduplicate_by": ingestion["deduplicate_by"],
    }
    signature = _sha256_bytes(json.dumps(signature_material, sort_keys=True).encode("utf-8"))
    manifest = _read_manifest(index_dir)
    if manifest and manifest.get("signature") == signature:
        try:
            index, persisted_chunks, _ = load_persisted_index(index_dir)
            return IndexBuildResult(
                index_dir=index_dir,
                chunk_count=len(persisted_chunks),
                vector_count=index.ntotal,
                embedded_chunk_count=0,
                reused_existing_index=True,
                signature=signature,
            )
        except (FileNotFoundError, ValueError):
            # Recover by rebuilding rather than trusting a partial or mismatched index.
            pass

    active_embedder = embedder or _create_embedder(settings["embedding"]["model_name"])
    vectors = _encode_chunks(chunks, active_embedder, bool(settings["embedding"]["normalize_embeddings"]))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # A complete directory replacement avoids mixed old/new index artefacts.
    temporary_dir = index_dir.with_name(f"{index_dir.name}.building")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    faiss.write_index(index, str(temporary_dir / INDEX_FILENAME))
    (temporary_dir / CHUNKS_FILENAME).write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2), encoding="utf-8"
    )
    (temporary_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "signature": signature,
                "corpus_sha256": corpus_fingerprint,
                "embedding_model": settings["embedding"]["model_name"],
                "chunk_count": len(chunks),
                "vector_count": index.ntotal,
                "vector_dimension": index.d,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if index_dir.exists():
        shutil.rmtree(index_dir)
    temporary_dir.replace(index_dir)
    return IndexBuildResult(
        index_dir=index_dir,
        chunk_count=len(chunks),
        vector_count=index.ntotal,
        embedded_chunk_count=len(chunks),
        reused_existing_index=False,
        signature=signature,
    )


def build_index_from_settings(
    settings: dict[str, Any] | None = None, embedder: Embedder | None = None
) -> IndexBuildResult:
    """Load the configured corpus and build/reuse its persisted index."""
    active_settings = settings or load_settings()
    corpus_dir = repository_path(active_settings["project"]["corpus_dir"])
    index_dir = repository_path(active_settings["project"]["index_dir"])
    return build_index(
        sections=load_corpus(corpus_dir),
        index_dir=index_dir,
        settings=active_settings,
        corpus_fingerprint=_corpus_fingerprint(corpus_dir),
        embedder=embedder,
    )

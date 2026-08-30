"""AC-01 tests: section-aware, persisted, idempotent indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ingestion.chunking import chunk_sections
from src.ingestion.indexer import build_index, load_persisted_index
from src.ingestion.loaders import GuidelineSection, load_corpus


class CountingEmbedder:
    """Small deterministic embedding stub: no network/model download in unit tests."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        self.calls += 1
        return np.array(
            [[float(len(text) % 17), float(sum(map(ord, text)) % 23), 1.0] for text in sentences],
            dtype=np.float32,
        )


def _settings() -> dict[str, object]:
    return {
        "ingestion": {
            "chunk_strategy": "section_aware",
            "max_chunk_characters": 1800,
            "chunk_overlap_characters": 160,
            "deduplicate_by": "content_sha256",
        },
        "embedding": {"model_name": "test-embedder", "normalize_embeddings": True},
    }


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    """AC-01: a second unchanged run embeds nothing and retains one vector per chunk."""
    repository_root = Path(__file__).resolve().parents[1]
    sections = load_corpus(repository_root / "data" / "raw")
    embedder = CountingEmbedder()
    index_dir = tmp_path / "index"

    first = build_index(sections, index_dir, _settings(), "stable-corpus", embedder)
    second = build_index(sections, index_dir, _settings(), "stable-corpus", embedder)
    index, chunks, manifest = load_persisted_index(index_dir)

    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert first.embedded_chunk_count == len(chunks)
    assert second.reused_existing_index is True
    assert second.embedded_chunk_count == 0
    assert embedder.calls == 1
    assert index.ntotal == len(chunks)
    assert manifest["chunk_count"] == len(chunks)


def test_oversized_clause_keeps_configured_overlap() -> None:
    """Section-aware chunking retains context when a clause exceeds the configured limit."""
    section = GuidelineSection(
        document_id="TEST-001",
        source="SYN-GUIDE-TEST-001",
        specialty="Test Specialty",
        version="2026.1",
        issuing_body="Synthetic Clinical Standards Board",
        document_title="Test Guideline",
        document_type="care_pathway",
        section_heading="Section 2: Long Rule",
        clause_id="TEST-001-S2",
        text=("alpha " * 30).strip() + "\n\n" + ("beta " * 30).strip() + "\n\n" + ("gamma " * 30).strip(),
        path="data/raw/TEST-001.md",
    )

    chunks = chunk_sections([section], max_characters=250, overlap_characters=80)

    assert len(chunks) == 3
    assert "alpha" in chunks[0].text and "alpha" in chunks[1].text
    assert "beta" in chunks[1].text and "beta" in chunks[2].text
    assert all("[clause id: TEST-001-S2]" in chunk.text for chunk in chunks)

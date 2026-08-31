"""NFR-01 and NFR-04 configuration safety checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.config import load_settings


def test_no_secret_environment_file_is_tracked() -> None:
    """NFR-01: a local credentials file remains local while its template is committed."""
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".env", ".env.example"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert ".env" not in tracked
    template = root / ".env.example"
    assert template.exists()
    assert template.read_text(encoding="utf-8").splitlines()[2] == "GOOGLE_API_KEY="


def test_retrieval_parameters_are_externalized() -> None:
    """NFR-04: tunable retrieval values are available through committed settings."""
    load_settings.cache_clear()
    settings = load_settings()

    assert settings["ingestion"]["max_chunk_characters"] > 0
    assert settings["ingestion"]["chunk_overlap_characters"] >= 0
    assert settings["retrieval"]["final_context_count"] > 0
    assert 0.0 <= settings["retrieval"]["minimum_grounding_confidence"] <= 1.0

"""Configuration loading with paths resolved relative to the repository root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPOSITORY_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load the committed settings file, failing clearly if it is malformed."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"Settings at {settings_path} must contain a YAML mapping.")
    return settings


def repository_path(relative_path: str) -> Path:
    """Resolve a project-relative configured path."""
    return REPOSITORY_ROOT / relative_path

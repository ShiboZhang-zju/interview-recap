from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .config import project_path


def load_replacements(config: dict[str, Any]) -> dict[str, str]:
    path = project_path(config, config["knowledge"]["normalization_file"])
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return {str(key): str(value) for key, value in (payload.get("replacements") or {}).items()}


def normalize_text(text: str, replacements: dict[str, str]) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    # Longest first prevents `http` from partially rewriting `https`.
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        flags = re.IGNORECASE if source.isascii() else 0
        if source.isascii() and re.fullmatch(r"[A-Za-z0-9+ ._-]+", source):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])"
        else:
            pattern = re.escape(source)
        normalized = re.sub(pattern, lambda _: target, normalized, flags=flags)
    normalized = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", normalized)
    return normalized.strip()

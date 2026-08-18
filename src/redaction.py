from __future__ import annotations

import re
from collections import Counter
from typing import Any


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential",
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|password|passwd|secret)\b\s*[:=]\s*[^\s,;]+"
        ),
    ),
    ("email", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")),
    ("cn_id", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")),
    (
        "ipv4",
        re.compile(
            r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
        ),
    ),
)


class Redactor:
    """Local deterministic redaction with count-only audit metadata."""

    def __init__(self, custom_terms: list[str] | None = None) -> None:
        self.counts: Counter[str] = Counter()
        terms = sorted(
            {term.strip() for term in (custom_terms or []) if term and term.strip()},
            key=len,
            reverse=True,
        )
        self._custom_pattern = (
            re.compile("|".join(re.escape(term) for term in terms)) if terms else None
        )

    def _replacement(self, category: str) -> str:
        self.counts[category] += 1
        return f"[REDACTED_{category.upper()}_{self.counts[category]}]"

    def redact_text(self, text: str) -> str:
        value = str(text)
        for category, pattern in PATTERNS:
            value = pattern.sub(lambda _: self._replacement(category), value)
        if self._custom_pattern:
            value = self._custom_pattern.sub(lambda _: self._replacement("custom"), value)
        return value

    def redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.redact_value(item) for key, item in value.items()}
        return value

    def audit_counts(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))

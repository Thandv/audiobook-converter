"""Pronunciation preprocessing.

Whole-word, case-insensitive substitution before text reaches Kokoro.
We keep this purely textual — Kokoro / espeak-ng will phonemize the result.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml


class PronunciationMap:
    def __init__(self, mapping: dict[str, str]):
        # Sort by length descending so multi-word keys win over single-word ones.
        items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
        # Build a single big alternation regex for speed.
        # We escape and require word boundaries (allowing inner hyphens / apostrophes).
        parts = [re.escape(k) for k, _ in items]
        # Lookbehind / lookahead enforce a "word-ish" boundary that respects
        # apostrophes and hyphens inside our keys.
        self._re = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(parts) + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        self._lookup = {k.lower(): v for k, v in mapping.items()}

    def apply(self, text: str) -> str:
        if not text:
            return text

        def repl(m: re.Match[str]) -> str:
            original = m.group(1)
            replacement = self._lookup[original.lower()]
            # Preserve all-caps emphasis on the original.
            if original.isupper() and len(original) > 1:
                return replacement.upper()
            return replacement

        return self._re.sub(repl, text)


def load_pronunciations(path: Path) -> PronunciationMap:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return PronunciationMap({str(k): str(v) for k, v in raw.items()})

"""Fine-tuning subsystem for emotion-aware TTS.

Provides dataset ingestion (RAVDESS, ESD, custom CSV manifests), XTTS v2
fine-tuning via Coqui's community fork, and inference for use as an
alternative backend in the audiobook pipeline.

Optional dependency group: ``pip install -e ".[training]"``.
"""
from __future__ import annotations

__all__ = [
    "EMOTIONS",
    "MANIFEST_COLUMNS",
]

#: Canonical emotion vocabulary used across the subsystem.
EMOTIONS: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "whispered",
    "excited",
    "calm",
)

#: Columns required in a manifest CSV.
MANIFEST_COLUMNS: tuple[str, ...] = ("audio_path", "text", "speaker", "emotion")

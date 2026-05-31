"""Backend-agnostic synthesis façade and voice cast loader.

This module no longer owns the TTS implementation — it loads the voice
cast YAML and selects a backend (kokoro / xtts). The Backend interface
lives in `audiobook.backends`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from .backends.base import Backend, BackendError, VoiceConfig


# Re-export common constants so older imports keep working.
SAMPLE_RATE = 24_000


@dataclass
class VoiceCast:
    """Loaded voice configuration. Each entry is a VoiceConfig."""

    narrator: VoiceConfig
    cast: dict[str, VoiceConfig]

    def for_speaker(self, speaker: str) -> VoiceConfig:
        if speaker == "NARRATOR":
            return self.narrator
        return self.cast.get(speaker, self.narrator)


def _voice_from_dict(d: dict | None, default: VoiceConfig | None = None) -> VoiceConfig:
    d = d or {}
    voice = d.get("voice", default.voice if default else "bm_george")
    speed = float(d.get("speed", default.speed if default else 1.0))
    emotions = d.get("emotions") or {}
    if not isinstance(emotions, dict):
        emotions = {}
    return VoiceConfig(voice=str(voice), speed=speed, emotions=dict(emotions))


def load_voice_cast(path: Path) -> VoiceCast:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    narrator = _voice_from_dict(raw.get("narrator"))
    cast_raw = raw.get("cast") or {}
    cast: dict[str, VoiceConfig] = {}
    for name, cfg in cast_raw.items():
        cast[str(name)] = _voice_from_dict(cfg, default=narrator)
    return VoiceCast(narrator=narrator, cast=cast)


def make_backend(name: str, **kwargs) -> Backend:
    """Factory — instantiate a backend by short name.

    Backends:
      - kokoro:     local, free, no training/library required (default).
      - xtts:       fine-tuned XTTS v2 model — needs `model_dir`.
      - cloning:    zero-shot XTTS v2 voice cloning from a voice library —
                    needs `library_root`.
      - chatterbox: Chatterbox emotion-aware TTS from a voice library —
                    needs `library_root`.
    """
    name = (name or "kokoro").lower()
    if name == "kokoro":
        from .backends.kokoro import KokoroBackend
        return KokoroBackend()
    if name == "xtts":
        from .backends.xtts import XTTSBackend
        model_dir = kwargs.get("model_dir")
        if not model_dir:
            raise BackendError("XTTS backend requires `model_dir` (path to fine-tuned model).")
        return XTTSBackend(Path(model_dir))
    if name == "cloning":
        from .backends.cloning import CloningBackend
        library_root = kwargs.get("library_root")
        if not library_root:
            raise BackendError(
                "Cloning backend requires `library_root` (path to voice library)."
            )
        return CloningBackend(Path(library_root))
    if name == "chatterbox":
        from .backends.chatterbox import ChatterboxBackend
        library_root = kwargs.get("library_root")
        if not library_root:
            raise BackendError(
                "Chatterbox backend requires `library_root` (path to voice library)."
            )
        return ChatterboxBackend(Path(library_root))
    raise BackendError(
        f"Unknown backend: {name!r}. "
        "Available: kokoro, xtts, cloning, chatterbox."
    )


def iter_voices_used(cast: VoiceCast, speakers: Iterator[str]) -> list[str]:
    seen: dict[str, None] = {}
    seen[cast.narrator.voice] = None
    for sp in speakers:
        seen[cast.for_speaker(sp).voice] = None
    return list(seen.keys())

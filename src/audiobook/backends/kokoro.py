"""Kokoro TTS backend.

Wraps the official `kokoro` Python package. Produces 24 kHz mono float32.
Lazy-initializes pipelines per language code to avoid reloading.
"""
from __future__ import annotations

import re

import numpy as np

from .base import Backend, BackendError, VoiceConfig


SAMPLE_RATE = 24_000
# Soft cap on chars per Kokoro call. Quality dips above ~600 chars.
MAX_CHUNK_CHARS = 600

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'A-Z])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    bits = _SENTENCE_RE.split(text)
    return [b.strip() for b in bits if b.strip()]


def chunk_for_synth(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    out: list[str] = []
    buf = ""
    for sent in split_sentences(text):
        if len(sent) > max_chars:
            for piece in _hard_wrap(sent, max_chars):
                if buf and len(buf) + 1 + len(piece) > max_chars:
                    out.append(buf)
                    buf = piece
                else:
                    buf = f"{buf} {piece}".strip()
            continue
        if buf and len(buf) + 1 + len(sent) > max_chars:
            out.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return out


def _hard_wrap(sent: str, max_chars: int) -> list[str]:
    if len(sent) <= max_chars:
        return [sent]
    pieces: list[str] = []
    cursor = 0
    while cursor < len(sent):
        end = min(cursor + max_chars, len(sent))
        if end == len(sent):
            pieces.append(sent[cursor:].strip())
            break
        slice_ = sent[cursor:end]
        idx = max(slice_.rfind(", "), slice_.rfind("; "), slice_.rfind(" — "), slice_.rfind(" "))
        if idx <= 0:
            idx = max_chars
        pieces.append(sent[cursor : cursor + idx].strip())
        cursor += idx
    return [p for p in pieces if p]


class KokoroBackend:
    """Kokoro v1.0 (or compatible) TTS backend.

    Kokoro has no native emotion control. Emotion is conveyed via the
    per-emotion overrides in VoiceConfig (typically slowing the speed
    and/or swapping in a softer voice).
    """

    name = "kokoro"
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, voice_id: str):
        lang_code = voice_id[0] if voice_id else "a"
        if lang_code not in self._pipelines:
            try:
                from kokoro import KPipeline  # lazy import
            except ImportError as e:
                raise BackendError(
                    "Kokoro is not installed. Run: pip install kokoro>=0.9.4"
                ) from e
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: VoiceConfig, emotion: str = "neutral") -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        resolved = voice.resolve(emotion)
        pipeline = self._pipeline(resolved.voice)
        chunks: list[np.ndarray] = []
        for piece in chunk_for_synth(text):
            for _gs, _ps, audio in pipeline(piece, voice=resolved.voice, speed=resolved.speed):
                arr = np.asarray(audio, dtype=np.float32)
                if arr.size:
                    chunks.append(arr)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

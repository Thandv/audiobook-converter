"""Backend protocol — every TTS engine implements this contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class BackendError(RuntimeError):
    """Raised when a backend can't fulfill a synthesis request."""


@dataclass
class VoiceConfig:
    """A single voice configuration.

    `voice` is backend-specific:
      - Kokoro: voice ID like 'bm_george'
      - XTTS:   speaker name in the fine-tuned model
    `emotions` (optional) maps emotion label -> overrides for that emotion.
    Anything not in `emotions` uses the top-level voice + speed.
    """

    voice: str
    speed: float = 1.0
    # Per-emotion overrides. Key is emotion label, value is partial dict
    # of {voice?, speed?}.
    emotions: dict[str, dict] = field(default_factory=dict)

    def resolve(self, emotion: str = "neutral") -> "ResolvedVoice":
        """Apply the emotion-specific override (if any) and return the
        concrete voice + speed used for this utterance."""
        if emotion and emotion in self.emotions:
            o = self.emotions[emotion]
            return ResolvedVoice(
                voice=str(o.get("voice", self.voice)),
                speed=float(o.get("speed", self.speed)),
                emotion=emotion,
            )
        return ResolvedVoice(voice=self.voice, speed=self.speed, emotion=emotion or "neutral")


@dataclass(frozen=True)
class ResolvedVoice:
    voice: str
    speed: float
    emotion: str


class Backend(Protocol):
    """TTS backend interface.

    Implementations must return float32 mono audio at `sample_rate`.
    """

    sample_rate: int

    def synthesize(self, text: str, voice: VoiceConfig, emotion: str = "neutral") -> np.ndarray:
        """Render `text` with the given voice + emotion."""
        ...

    @property
    def name(self) -> str:
        """Short identifier (e.g. 'kokoro', 'xtts')."""
        ...

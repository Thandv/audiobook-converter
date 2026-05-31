"""XTTS v2 backend — uses a fine-tuned model from `audiobook train`.

The fine-tuned model knows a fixed set of speakers (characters) and
emotions, with reference clips stored under `model_dir/reference_clips/`.
This backend looks up the right reference clip for a given (voice, emotion)
pair and calls XTTS for inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Backend, BackendError, VoiceConfig


SAMPLE_RATE = 24_000


class XTTSBackend:
    """Loads a fine-tuned XTTS v2 checkpoint and synthesizes from it.

    Requires the `[training]` extras installed:
        pip install -e ".[training]"
    """

    name = "xtts"
    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        try:
            from audiobook.training.infer import FineTunedXTTSSynth  # lazy
        except ImportError as e:
            raise BackendError(
                "XTTS backend requires the training extras. "
                "Run: pip install -e \".[training]\""
            ) from e
        self._synth = FineTunedXTTSSynth(self._model_dir)

    def synthesize(self, text: str, voice: VoiceConfig, emotion: str = "neutral") -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        resolved = voice.resolve(emotion)
        # XTTS speaker name comes from `voice.voice` (we reuse the field).
        return self._synth.synthesize(
            text=text,
            speaker=resolved.voice,
            emotion=resolved.emotion,
            speed=resolved.speed,
        )

    @property
    def available_speakers(self) -> list[str]:
        return self._synth.available_speakers()

    @property
    def available_emotions(self) -> dict[str, list[str]]:
        return {sp: self._synth.available_emotions(sp) for sp in self.available_speakers}

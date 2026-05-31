"""Chatterbox TTS backend (Resemble AI, open source).

Chatterbox is a zero-shot voice-cloning TTS with an explicit emotion
*exaggeration* parameter that scales the model's expressiveness from
calm (0.0) to over-the-top (2.0). Unlike XTTS, you only need ONE
reference clip per character — emotion intensity comes from the knob
plus a per-emotion exaggeration map we define below.

Library layout: same as the cloning backend.
  voices/<character>/neutral.wav   (used as the voice template)

Emotion handling: detected emotion -> exaggeration intensity. For
characters that have an emotion-specific reference clip in the library,
we still prefer that clip (better prosody match than the knob alone).

Install: pip install chatterbox-tts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..voice_library import VoiceLibrary
from .base import Backend, BackendError, VoiceConfig


TARGET_SAMPLE_RATE = 24_000


# Emotion -> (exaggeration, cfg_weight). Chatterbox's exaggeration is roughly
# emotion intensity (0=flat, 1=normal, 2=theatrical). cfg_weight ~0.2-0.7
# controls speed/cadence: lower = faster + more chaotic, higher = steadier.
EMOTION_TO_KNOBS: dict[str, tuple[float, float]] = {
    "neutral":    (0.50, 0.50),
    "calm":       (0.30, 0.60),
    "happy":      (0.90, 0.45),
    "sad":        (0.70, 0.55),
    "angry":      (1.30, 0.35),
    "fearful":    (1.10, 0.40),
    "surprised":  (1.20, 0.40),
    "disgusted":  (1.00, 0.50),
    "whispered":  (0.40, 0.65),
    "excited":    (1.10, 0.35),
}


class ChatterboxBackend:
    """Chatterbox-based emotional voice cloning."""

    name = "chatterbox"
    sample_rate = TARGET_SAMPLE_RATE

    def __init__(self, library_root: Path) -> None:
        self._library = VoiceLibrary(Path(library_root))
        if not self._library.root.exists() or not self._library.list_characters():
            raise BackendError(
                f"Voice library at {self._library.root} is empty. Add clips first."
            )
        self._model = None
        self._device = "cpu"

    # ---------------------------------------------------------------- API

    def synthesize(self, text: str, voice: VoiceConfig, emotion: str = "neutral") -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        resolved = voice.resolve(emotion)
        ref = self._pick_reference(resolved.voice, resolved.emotion)
        model = self._ensure_loaded()

        exag, cfg_w = EMOTION_TO_KNOBS.get(resolved.emotion, EMOTION_TO_KNOBS["neutral"])

        # Chatterbox API: model.generate(text, audio_prompt_path=..., exaggeration=..., cfg_weight=...)
        # Returns torch.Tensor of audio at model.sr.
        wav_tensor = model.generate(
            text,
            audio_prompt_path=str(ref),
            exaggeration=exag,
            cfg_weight=cfg_w,
        )
        try:
            wav = wav_tensor.cpu().numpy()
        except AttributeError:
            wav = np.asarray(wav_tensor)
        if wav.ndim > 1:
            wav = wav.squeeze()
        wav = wav.astype(np.float32)

        native_sr = int(getattr(model, "sr", TARGET_SAMPLE_RATE))
        if native_sr != TARGET_SAMPLE_RATE:
            wav = _resample(wav, native_sr, TARGET_SAMPLE_RATE)
        # Apply user-specified speed via simple resampling.
        if abs(resolved.speed - 1.0) > 1e-3:
            wav = _change_speed(wav, resolved.speed)
        return wav

    # --------------------------------------------------------- internals

    def _pick_reference(self, character: str, emotion: str) -> Path:
        """Prefer emotion-matched clip; fall back to neutral, then any clip."""
        clip = self._library.find_clip(character, emotion)
        if clip is not None:
            return clip
        clip = self._library.find_clip(character, "neutral")
        if clip is not None:
            return clip
        # Any clip for this character.
        for info in self._library.list_clips(character):
            return info.path
        # Narrator fallback.
        for cand_emo in ("neutral", emotion, "calm"):
            clip = self._library.find_clip("narrator", cand_emo)
            if clip is not None:
                return clip
        all_clips = self._library.list_clips()
        if all_clips:
            return all_clips[0].path
        raise BackendError(f"No reference clips available for {character}/{emotion}")

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from chatterbox.tts import ChatterboxTTS  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendError(
                "Chatterbox backend requires the chatterbox extras. "
                'Run: pip install -e ".[chatterbox]"'
            ) from e

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        self._device = device
        self._model = ChatterboxTTS.from_pretrained(device=device)
        return self._model


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(src_sr, dst_sr)
        return resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        # Linear interpolation fallback.
        ratio = dst_sr / src_sr
        n = int(len(wav) * ratio)
        xs = np.linspace(0, len(wav) - 1, n)
        idx = xs.astype(int)
        return wav[idx].astype(np.float32)


def _change_speed(wav: np.ndarray, factor: float) -> np.ndarray:
    """Resample-based speed change (also affects pitch slightly)."""
    if abs(factor - 1.0) < 1e-3:
        return wav
    target_len = max(1, int(len(wav) / factor))
    xs = np.linspace(0, len(wav) - 1, target_len)
    return wav[xs.astype(int)].astype(np.float32)

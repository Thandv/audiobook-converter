"""Zero-shot voice cloning backend (XTTS v2).

This is the "voice library" backend — at synthesis time, picks the right
reference clip for (character, emotion) from a VoiceLibrary on disk and
asks XTTS v2 to clone the voice and copy the emotional prosody.

No fine-tuning required. The user just needs:
  1. Training extras installed: pip install -e ".[training]"
  2. A populated voice library:    audiobook voices import-ravdess <path>
                                   or audiobook voices import --character X --emotion Y --file Z

Fallback behavior:
  - emotion not in library for character -> try "neutral" for same character
  - character not in library             -> use library's "narrator" voice
  - "narrator" not in library either     -> raise BackendError
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..voice_library import VoiceLibrary
from .base import Backend, BackendError, VoiceConfig


TARGET_SAMPLE_RATE = 24_000
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


class CloningBackend:
    """XTTS v2 voice-cloning backend driven by a VoiceLibrary.

    Loaded lazily — heavy ML deps are not imported until the first
    `synthesize()` call.
    """

    name = "cloning"
    sample_rate = TARGET_SAMPLE_RATE

    def __init__(self, library_root: Path) -> None:
        self._library = VoiceLibrary(Path(library_root))
        if not self._library.root.exists():
            raise BackendError(
                f"Voice library directory does not exist: {self._library.root}\n"
                f"Run `audiobook voices import-ravdess <ravdess_path>` to populate it."
            )
        if not self._library.list_characters():
            raise BackendError(
                f"Voice library at {self._library.root} is empty. Add clips with "
                "`audiobook voices import-ravdess <path>` or "
                "`audiobook voices import --character X --emotion Y --file Z`."
            )
        # Heavy state, populated lazily.
        self._model = None
        self._native_sr = TARGET_SAMPLE_RATE
        self._latents_cache: dict[tuple[str, str], dict] = {}

    # ---------------------------------------------------------------- API

    def synthesize(self, text: str, voice: VoiceConfig, emotion: str = "neutral") -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        resolved = voice.resolve(emotion)
        ref_clip, used_character, used_emotion = self._pick_reference(resolved.voice, resolved.emotion)
        model = self._ensure_loaded()

        latents = self._latents_for(used_character, used_emotion, ref_clip, model)
        out = model.inference(
            text=text,
            language="en",
            gpt_cond_latent=latents["gpt_cond_latent"],
            speaker_embedding=latents["speaker_embedding"],
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=2.0,
            top_k=50,
            top_p=0.85,
            speed=float(resolved.speed),
            enable_text_splitting=True,
        )
        wav = np.asarray(out["wav"], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1).astype(np.float32)
        if self._native_sr != TARGET_SAMPLE_RATE:
            wav = _resample(wav, self._native_sr, TARGET_SAMPLE_RATE)
        return wav

    # --------------------------------------------------------- internals

    def _pick_reference(self, character: str, emotion: str) -> tuple[Path, str, str]:
        """Return (clip_path, character_used, emotion_used) with fallback."""
        clip = self._library.find_clip(character, emotion)
        if clip is not None:
            return clip, character, emotion

        # Try same character with "neutral" emotion.
        clip = self._library.find_clip(character, "neutral")
        if clip is not None:
            return clip, character, "neutral"

        # Try ANY emotion for this character.
        chars = self._library.list_characters()
        if character in chars:
            for emo_clip in self._library.list_clips(character):
                return emo_clip.path, character, emo_clip.emotion

        # Fallback: narrator voice (any emotion).
        narrator_clip = self._library.find_clip("narrator", emotion)
        if narrator_clip is not None:
            return narrator_clip, "narrator", emotion
        narrator_clip = self._library.find_clip("narrator", "neutral")
        if narrator_clip is not None:
            return narrator_clip, "narrator", "neutral"

        # Last resort: ANY clip in the library.
        all_clips = self._library.list_clips()
        if all_clips:
            c = all_clips[0]
            return c.path, c.character, c.emotion

        raise BackendError(
            f"No reference clips found in {self._library.root}. "
            "Library is empty."
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # type: ignore[import-not-found]
            from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore[import-not-found]
            from TTS.tts.models.xtts import Xtts  # type: ignore[import-not-found]
            from TTS.utils.manage import ModelManager  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendError(
                "Cloning backend requires the training extras. "
                'Run: pip install -e ".[training]"'
            ) from e

        # Download / locate the base XTTS v2 model (idempotent — cached).
        manager = ModelManager()
        model_path, config_path, _item = manager.download_model(XTTS_MODEL_NAME)
        base_dir = Path(model_path)
        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        vocab_path = base_dir / "vocab.json"
        model.load_checkpoint(
            config,
            checkpoint_dir=str(base_dir),
            vocab_path=str(vocab_path) if vocab_path.exists() else None,
            eval=True,
            strict=False,
        )

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        model.to(device)
        try:
            self._native_sr = int(config.audio.output_sample_rate)
        except AttributeError:
            self._native_sr = 24_000
        self._model = model
        return model

    def _latents_for(self, character: str, emotion: str, ref: Path, model) -> dict:
        key = (character, emotion)
        if key in self._latents_cache:
            return self._latents_cache[key]
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(ref)]
        )
        cached = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }
        self._latents_cache[key] = cached
        return cached


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    try:
        import torch
        import torchaudio
    except ImportError as e:
        raise BackendError("torchaudio required for resampling") from e
    t = torch.from_numpy(wav).unsqueeze(0)
    return torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)(t).squeeze(0).numpy().astype(np.float32)

"""Inference with a fine-tuned XTTS v2 model.

Provides :class:`FineTunedXTTSSynth`, which mirrors the surface area of
``audiobook.synth.KokoroSynth`` so it can drop into the audiobook pipeline
as an alternative backend.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


TARGET_SAMPLE_RATE = 24_000  # what the pipeline (Kokoro) expects


class FineTunedXTTSSynth:
    """Inference wrapper around a fine-tuned XTTS v2 checkpoint.

    Parameters
    ----------
    model_dir:
        Directory produced by
        :func:`audiobook.training.train.fine_tune_xtts`. Must contain
        ``best_model.pth``, ``config.json``, and a ``reference_clips/`` tree
        of ``<speaker>/<emotion>/*.wav``.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {self.model_dir}")
        self._checkpoint = self.model_dir / "best_model.pth"
        self._config_path = self.model_dir / "config.json"
        if not self._checkpoint.exists():
            raise FileNotFoundError(f"Missing best_model.pth in {self.model_dir}")
        if not self._config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {self.model_dir}")

        self._refs_root = self.model_dir / "reference_clips"
        self._refs = self._index_references(self._refs_root)
        self._native_sr: int = 24_000

        # Lazy-loaded heavy state.
        self._model: Any | None = None
        self._gpt_cond_cache: dict[tuple[str, str], Any] = {}
        self._inference_manifest = self._load_inference_manifest()

    # ------------------------------------------------------------------ API

    def available_speakers(self) -> list[str]:
        """Speakers for which we have at least one reference clip."""
        return sorted(self._refs.keys())

    def available_emotions(self, speaker: str) -> list[str]:
        """Emotions available for ``speaker`` (empty list if unknown speaker)."""
        return sorted(self._refs.get(speaker, {}).keys())

    def synthesize(
        self,
        text: str,
        speaker: str,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> np.ndarray:
        """Render ``text`` as float32 mono audio at 24 kHz.

        If ``emotion`` is not available for ``speaker``, falls back to
        ``"neutral"`` (and then to the speaker's first available emotion) with
        a logged warning. Raises :class:`KeyError` if the speaker is unknown.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if speaker not in self._refs:
            raise KeyError(
                f"Unknown speaker '{speaker}'. Available: {self.available_speakers()}"
            )

        emotion_used = self._resolve_emotion(speaker, emotion)
        ref_clip = self._pick_reference(speaker, emotion_used)
        model = self._ensure_loaded()

        # Cache GPT conditioning latents per (speaker, emotion) — they're the
        # expensive part of XTTS inference.
        latents = self._get_cond_latents(speaker, emotion_used, ref_clip)
        gpt_cond_latent = latents["gpt_cond_latent"]
        speaker_embedding = latents["speaker_embedding"]

        # XTTS native inference: returns dict with 'wav' (numpy float32).
        out = model.inference(
            text=text,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
            length_penalty=1.0,
            repetition_penalty=2.0,
            top_k=50,
            top_p=0.85,
            speed=float(speed),
            enable_text_splitting=True,
        )
        wav = np.asarray(out["wav"], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1).astype(np.float32)

        if self._native_sr != TARGET_SAMPLE_RATE:
            wav = self._resample(wav, self._native_sr, TARGET_SAMPLE_RATE)
        return wav

    # ----------------------------------------------------------- internals

    def _load_inference_manifest(self) -> dict[str, Any]:
        path = self.model_dir / "inference.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Could not parse {}", path)
            return {}

    @staticmethod
    def _index_references(refs_root: Path) -> dict[str, dict[str, list[Path]]]:
        """Build ``{speaker: {emotion: [wav_paths]}}`` from the reference tree."""
        index: dict[str, dict[str, list[Path]]] = {}
        if not refs_root.exists():
            return index
        for sp_dir in sorted(p for p in refs_root.iterdir() if p.is_dir()):
            speaker = sp_dir.name
            for emo_dir in sorted(p for p in sp_dir.iterdir() if p.is_dir()):
                wavs = sorted(emo_dir.glob("*.wav"))
                if wavs:
                    index.setdefault(speaker, {})[emo_dir.name] = wavs
        return index

    def _resolve_emotion(self, speaker: str, requested: str) -> str:
        emotions = self._refs[speaker]
        if requested in emotions:
            return requested
        if "neutral" in emotions:
            logger.warning(
                "Speaker '{}' has no '{}' clips; falling back to 'neutral'.",
                speaker,
                requested,
            )
            return "neutral"
        fallback = next(iter(emotions))
        logger.warning(
            "Speaker '{}' has no '{}' or 'neutral' clips; falling back to '{}'.",
            speaker,
            requested,
            fallback,
        )
        return fallback

    def _pick_reference(self, speaker: str, emotion: str) -> Path:
        clips = self._refs[speaker][emotion]
        # Deterministic-ish pick: shortest reasonable clip first for stability.
        return clips[0] if len(clips) == 1 else random.choice(clips)

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch  # type: ignore[import-not-found]
            from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore[import-not-found]
            from TTS.tts.models.xtts import Xtts  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "coqui-tts is required for inference. Install training extras: "
                'pip install -e ".[training]"'
            ) from exc

        config = XttsConfig()
        config.load_json(str(self._config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(self._checkpoint),
            vocab_path=str(self.model_dir / "vocab.json")
            if (self.model_dir / "vocab.json").exists()
            else None,
            eval=True,
            strict=False,
        )
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        model.to(device)
        # XTTS native output rate (24kHz by default; reread from config if set).
        try:
            self._native_sr = int(config.audio.output_sample_rate)
        except AttributeError:
            self._native_sr = 24_000
        logger.info("Loaded fine-tuned XTTS on {} (native {} Hz)", device, self._native_sr)
        self._model = model
        return model

    def _get_cond_latents(
        self, speaker: str, emotion: str, ref_clip: Path
    ) -> dict[str, Any]:
        key = (speaker, emotion)
        cached = self._gpt_cond_cache.get(key)
        if cached is not None:
            return cached
        model = self._model
        if model is None:  # pragma: no cover - _ensure_loaded already called
            raise RuntimeError("Model not loaded.")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(ref_clip)]
        )
        latents = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }
        self._gpt_cond_cache[key] = latents
        return latents

    @staticmethod
    def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return wav
        try:
            import torch  # type: ignore[import-not-found]
            import torchaudio  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torchaudio is required for inference resampling. "
                'Install training extras: pip install -e ".[training]"'
            ) from exc
        tensor = torch.from_numpy(wav).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
        return resampler(tensor).squeeze(0).numpy().astype(np.float32)

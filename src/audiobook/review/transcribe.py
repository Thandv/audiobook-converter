"""Whisper transcription wrapper.

Uses faster-whisper (CTranslate2 backend) — much faster than openai-whisper
on CPU, with the same quality. Apple Silicon runs on Metal via int8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Default Whisper model size. "base" is the sweet spot of accuracy vs speed
# for review purposes (we mostly need to know "what words did Whisper hear").
DEFAULT_MODEL = "base.en"


@dataclass
class WordHit:
    """One word Whisper transcribed, with timestamps."""

    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class ChapterTranscript:
    """The transcript of one chapter."""

    chapter_audio_path: Path
    full_text: str
    words: list[WordHit] = field(default_factory=list)
    duration_seconds: float = 0.0
    language: str = "en"


class Transcriber:
    """Lazy-loaded faster-whisper wrapper.

    Use a single instance to avoid reloading the model per chapter.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Whisper transcription requires the [review] extras. "
                'Run: pip install -e ".[review]"'
            ) from e
        # Device + compute_type auto: pick the best for the host.
        device, compute_type = self._resolve_device_compute()
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
        )
        return self._model

    def _resolve_device_compute(self) -> tuple[str, str]:
        if self.device != "auto":
            return self.device, self.compute_type or "default"
        # Try CUDA first; otherwise CPU with int8 quantization for speed.
        try:
            import torch  # type: ignore[import-not-found]
            if torch.cuda.is_available():
                return "cuda", self.compute_type or "float16"
        except ImportError:
            pass
        return "cpu", self.compute_type or "int8"

    def transcribe(self, audio_path: Path) -> ChapterTranscript:
        """Transcribe one chapter audio file."""
        model = self._load()
        # word_timestamps=True so we can align word-by-word.
        # vad_filter speeds things up a lot by skipping silences.
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=1,
            word_timestamps=True,
            vad_filter=True,
            language="en",
        )
        all_words: list[WordHit] = []
        text_pieces: list[str] = []
        for seg in segments:
            text_pieces.append(seg.text)
            if seg.words is None:
                continue
            for w in seg.words:
                all_words.append(WordHit(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(getattr(w, "probability", 0.0) or 0.0),
                ))
        return ChapterTranscript(
            chapter_audio_path=audio_path,
            full_text=" ".join(text_pieces).strip(),
            words=all_words,
            duration_seconds=float(info.duration),
            language=info.language or "en",
        )

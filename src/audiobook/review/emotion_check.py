"""Audio-side emotion verification.

Loads a wav2vec2-based speech emotion classifier and compares its prediction
to what the text analyzer INTENDED for the same audio span. Sentence-level
predictions are noisy, so we aggregate to scene level: if a scene's intended
emotion was 'angry' but the model votes for it < 30% of sentences,
something didn't translate to the audio.

Model: superb/wav2vec2-base-superb-er — small (~360 MB), 4-emotion output
(angry, happy, sad, neutral). We map its labels to our 10-emotion vocabulary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import Finding, FindingKind, FindingSeverity


MODEL_ID = "superb/wav2vec2-base-superb-er"
TARGET_SR = 16_000   # model expects 16 kHz mono

# wav2vec2 superb emotion labels -> our vocabulary.
MODEL_TO_OURS = {
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral",
    # Some checkpoints expose full label names; handle both.
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
}


@dataclass
class SegmentPrediction:
    start: float
    end: float
    predicted: str
    confidence: float


class AudioEmotionClassifier:
    """Lazy-loaded wav2vec2 audio emotion classifier."""

    def __init__(self) -> None:
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Audio emotion check requires the [review] extras. "
                'Run: pip install -e ".[review]"'
            ) from e
        self._pipeline = pipeline(
            "audio-classification",
            model=MODEL_ID,
            top_k=None,
        )
        return self._pipeline

    def classify_segment(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[str, float, dict[str, float]]:
        """Return (predicted_label, confidence, all_scores)."""
        if audio.size == 0:
            return "neutral", 0.0, {}
        # Resample to 16k mono if needed.
        if sample_rate != TARGET_SR:
            audio = _resample(audio, sample_rate, TARGET_SR)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        pipeline = self._load()
        results = pipeline({"array": audio.astype(np.float32), "sampling_rate": TARGET_SR})
        if not results:
            return "neutral", 0.0, {}
        scores = {
            MODEL_TO_OURS.get(r["label"].lower(), r["label"].lower()): float(r["score"])
            for r in results
        }
        # Drop unmapped labels.
        scores = {k: v for k, v in scores.items() if k in {"angry", "happy", "sad", "neutral"}}
        if not scores:
            return "neutral", 0.0, {}
        top = max(scores, key=scores.get)
        return top, scores[top], scores


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple resample using polyphase if scipy is available, else linear."""
    if src_sr == dst_sr:
        return wav
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(src_sr, dst_sr)
        return resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        ratio = dst_sr / src_sr
        n = int(len(wav) * ratio)
        idx = np.linspace(0, len(wav) - 1, n).astype(np.int64)
        return wav[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling strategy
# ---------------------------------------------------------------------------
#
# Running the classifier on every sentence-worth of audio is expensive AND
# noisy. We sample evenly across each chapter (default 8 samples of ~6s each)
# and aggregate to a per-chapter emotion distribution. Compare to the text
# analyzer's per-chapter distribution.


def sample_chapter(
    audio_path: Path, n_samples: int = 8, sample_seconds: float = 6.0,
) -> list[tuple[float, float, np.ndarray, int]]:
    """Read N evenly-spaced segments from a chapter audio file.

    Returns list of (start_s, end_s, audio, sample_rate).
    """
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as snd:
        sr = snd.samplerate
        total_frames = len(snd)
        duration = total_frames / sr if sr else 0.0
        if duration < sample_seconds * 2:
            # Whole chapter is one sample.
            data = snd.read(dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return [(0.0, duration, data, sr)]

        out: list[tuple[float, float, np.ndarray, int]] = []
        chunk_frames = int(sample_seconds * sr)
        for i in range(n_samples):
            # Pick midpoints evenly across the chapter, avoiding edges.
            t = duration * (0.1 + 0.8 * i / max(1, n_samples - 1))
            start_frame = max(0, int(t * sr) - chunk_frames // 2)
            end_frame = min(total_frames, start_frame + chunk_frames)
            snd.seek(start_frame)
            data = snd.read(end_frame - start_frame, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            out.append((start_frame / sr, end_frame / sr, data, sr))
        return out


def emotion_findings_for_chapter(
    audio_path: Path,
    chapter_number: int,
    chapter_title: str,
    intended_emotions: dict[str, int],
    classifier: AudioEmotionClassifier,
    n_samples: int = 8,
) -> list[Finding]:
    """Check whether the chapter's audio carries the emotions the text intended.

    `intended_emotions` is a per-emotion count from the text analyzer
    (e.g. {'angry': 12, 'sad': 5, 'neutral': 80}).

    We only fire a finding when there's a meaningful intended non-neutral
    emotion (>=10% of sentences) AND the audio classifier doesn't see it.
    """
    total_intended = sum(intended_emotions.values())
    if total_intended == 0:
        return []

    # Sample + classify.
    samples = sample_chapter(audio_path, n_samples=n_samples)
    predicted_counts: Counter[str] = Counter()
    for _start, _end, audio, sr in samples:
        label, _conf, _scores = classifier.classify_segment(audio, sr)
        predicted_counts[label] += 1
    total_predicted = sum(predicted_counts.values())
    if total_predicted == 0:
        return []

    findings: list[Finding] = []
    for emo in ("angry", "happy", "sad"):
        intended_share = intended_emotions.get(emo, 0) / total_intended
        predicted_share = predicted_counts.get(emo, 0) / total_predicted
        if intended_share >= 0.10 and predicted_share < intended_share * 0.5:
            findings.append(Finding(
                kind=FindingKind.EMOTION_MISMATCH,
                severity=FindingSeverity.SUGGESTION,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                summary=(
                    f"Intended {emo} {intended_share:.0%} of sentences but "
                    f"audio classifier hears {emo} only {predicted_share:.0%} "
                    f"of sampled segments."
                ),
                intended_emotion=emo,
                predicted_emotion=predicted_counts.most_common(1)[0][0],
                confidence=min(1.0, 0.5 + 2 * (intended_share - predicted_share)),
                fix_action="flag_only",  # we don't auto-fix audio emotion
                fix_payload={
                    "intended_emotions": dict(intended_emotions),
                    "predicted_distribution": dict(predicted_counts),
                },
            ))
    return findings

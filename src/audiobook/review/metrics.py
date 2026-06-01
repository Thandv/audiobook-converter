"""DSP-based quality metrics — pace, volume, silence, clipping.

These don't need a model. Just numpy + soundfile (both base deps). librosa
would be slicker but for our purposes raw soundfile data is enough.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Outlier threshold: a chapter is flagged if its metric is >= 1.5 standard
# deviations from the book's mean for that metric.
OUTLIER_SD = 1.5

# Silence detection: anything below -45 dBFS for >= 1.5 seconds is "long silence"
# at chapter scale (we EXPECT chapter heading + scene breaks so don't fire
# under that count).
SILENCE_DBFS = -45.0
SILENCE_MIN_SECONDS = 4.0

# Clipping: peak sample >= this is considered clipped audio.
CLIP_THRESHOLD = 0.999


@dataclass
class ChapterMetrics:
    audio_path: Path
    duration_seconds: float
    word_count: int
    wpm: float                       # words per minute (source word count / duration)
    rms_dbfs: float                  # mean RMS energy in dB FS
    peak_dbfs: float                 # max sample amplitude in dB FS
    long_silence_segments: list[tuple[float, float]]  # (start, end) seconds
    clip_seconds: float              # total seconds with clipping
    sample_rate: int

    def to_dict(self) -> dict:
        return {
            "audio_path": str(self.audio_path),
            "duration_seconds": round(self.duration_seconds, 2),
            "word_count": self.word_count,
            "wpm": round(self.wpm, 1),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "long_silence_segments": [
                [round(a, 2), round(b, 2)] for a, b in self.long_silence_segments
            ],
            "clip_seconds": round(self.clip_seconds, 3),
            "sample_rate": self.sample_rate,
        }


def _dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * float(np.log10(value))


def compute_chapter_metrics(
    audio_path: Path, source_text: str
) -> ChapterMetrics:
    """Crunch one chapter's DSP metrics."""
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as snd:
        sr = snd.samplerate
        data = snd.read(dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

    duration = len(data) / sr if sr else 0.0
    word_count = len((source_text or "").split())
    wpm = (word_count / duration * 60.0) if duration > 0 else 0.0

    abs_data = np.abs(data)
    peak = float(abs_data.max()) if abs_data.size else 0.0
    # RMS over the whole chapter.
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if data.size else 0.0
    rms_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(peak)

    long_silences = _detect_long_silences(data, sr)
    clip_seconds = float((abs_data >= CLIP_THRESHOLD).sum() / sr)

    return ChapterMetrics(
        audio_path=audio_path,
        duration_seconds=duration,
        word_count=word_count,
        wpm=wpm,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        long_silence_segments=long_silences,
        clip_seconds=clip_seconds,
        sample_rate=sr,
    )


def _detect_long_silences(
    data: np.ndarray, sr: int,
    threshold_dbfs: float = SILENCE_DBFS,
    min_seconds: float = SILENCE_MIN_SECONDS,
) -> list[tuple[float, float]]:
    """Find chunks of audio quieter than threshold_dbfs for longer than min_seconds."""
    if sr <= 0 or data.size == 0:
        return []
    # Smooth via RMS over a 50 ms window.
    win = max(1, int(0.05 * sr))
    sq = data.astype(np.float64) ** 2
    # Cumulative sum trick for fast windowed mean.
    cum = np.concatenate(([0.0], np.cumsum(sq)))
    rms = np.sqrt((cum[win:] - cum[:-win]) / win)
    # Threshold to "silent" boolean.
    threshold = 10 ** (threshold_dbfs / 20.0)
    quiet = rms < threshold
    # Find runs of True longer than min_seconds.
    out: list[tuple[float, float]] = []
    n_min = int(min_seconds * sr)
    start_idx = None
    for i, q in enumerate(quiet):
        if q and start_idx is None:
            start_idx = i
        elif not q and start_idx is not None:
            run_len = i - start_idx
            if run_len >= n_min:
                out.append((start_idx / sr, i / sr))
            start_idx = None
    if start_idx is not None:
        run_len = len(quiet) - start_idx
        if run_len >= n_min:
            out.append((start_idx / sr, len(quiet) / sr))
    return out


def find_metric_outliers(
    metrics_by_chapter: dict[int, ChapterMetrics],
    chapter_titles: dict[int, str],
) -> list:
    """Flag chapters whose pace / volume are >= OUTLIER_SD from the book mean."""
    from .types import Finding, FindingKind, FindingSeverity

    if not metrics_by_chapter:
        return []
    findings: list[Finding] = []

    # Pace outliers (excluding very short headings).
    long_enough = {
        n: m for n, m in metrics_by_chapter.items() if m.duration_seconds > 60
    }
    if len(long_enough) >= 3:
        wpms = np.array([m.wpm for m in long_enough.values()], dtype=float)
        mean, sd = float(wpms.mean()), float(wpms.std())
        if sd > 0:
            for n, m in long_enough.items():
                z = (m.wpm - mean) / sd
                if abs(z) >= OUTLIER_SD:
                    direction = "fast" if z > 0 else "slow"
                    findings.append(Finding(
                        kind=FindingKind.PACE_OUTLIER,
                        severity=FindingSeverity.SUGGESTION,
                        chapter_number=n,
                        chapter_title=chapter_titles.get(n, ""),
                        summary=(
                            f"Chapter pace is {direction}: {m.wpm:.0f} WPM "
                            f"vs book mean {mean:.0f} (z={z:+.2f})"
                        ),
                        metric_value=m.wpm,
                        metric_baseline=mean,
                        fix_action="adjust_voice_speed",
                        fix_payload={"direction": direction, "z_score": z},
                    ))

    # Volume outliers.
    rmss = np.array([m.rms_dbfs for m in metrics_by_chapter.values()], dtype=float)
    mean_rms, sd_rms = float(rmss.mean()), float(rmss.std())
    if sd_rms > 0:
        for n, m in metrics_by_chapter.items():
            z = (m.rms_dbfs - mean_rms) / sd_rms
            if abs(z) >= OUTLIER_SD:
                findings.append(Finding(
                    kind=FindingKind.VOLUME_OUTLIER,
                    severity=FindingSeverity.SUGGESTION,
                    chapter_number=n,
                    chapter_title=chapter_titles.get(n, ""),
                    summary=(
                        f"Chapter volume is off: RMS {m.rms_dbfs:.1f} dBFS "
                        f"vs book mean {mean_rms:.1f} (z={z:+.2f})"
                    ),
                    metric_value=m.rms_dbfs,
                    metric_baseline=mean_rms,
                    fix_action="flag_only",
                ))

    # Long-silence findings (don't compare to mean; absolute).
    for n, m in metrics_by_chapter.items():
        for start, end in m.long_silence_segments:
            # Ignore silence in the first 5 seconds (chapter heading pause).
            if start < 5.0:
                continue
            findings.append(Finding(
                kind=FindingKind.LONG_SILENCE,
                severity=FindingSeverity.INFO,
                chapter_number=n,
                chapter_title=chapter_titles.get(n, ""),
                summary=(
                    f"Long silence in {chapter_titles.get(n, str(n))} "
                    f"from {start:.1f}s to {end:.1f}s ({end - start:.1f}s)"
                ),
                start_seconds=start,
                end_seconds=end,
                metric_value=end - start,
                fix_action="flag_only",
            ))

    # Clipping.
    for n, m in metrics_by_chapter.items():
        if m.clip_seconds >= 0.05:  # 50 ms of clipping is real
            findings.append(Finding(
                kind=FindingKind.CLIPPING,
                severity=FindingSeverity.HIGH,
                chapter_number=n,
                chapter_title=chapter_titles.get(n, ""),
                summary=f"Clipping detected: {m.clip_seconds:.2f}s of audio over threshold",
                metric_value=m.clip_seconds,
                fix_action="flag_only",
            ))
    return findings

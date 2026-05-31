"""Dataset preparation for emotion-aware TTS fine-tuning.

Manifest format (CSV with header):

    audio_path,text,speaker,emotion

Emotion vocabulary is fixed to :data:`audiobook.training.EMOTIONS`. Audio
paths in the manifest may be relative; ``validate_manifest`` resolves them
against the manifest's parent directory.
"""
from __future__ import annotations

import csv
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from . import EMOTIONS, MANIFEST_COLUMNS


# RAVDESS Speech filename: 03-01-EE-II-SS-RR-AA.wav
# Field 1 = modality (03 = audio), field 2 = vocal channel (01 = speech),
# field 3 = emotion (01-08), field 4 = intensity, field 5 = statement,
# field 6 = repetition, field 7 = actor (01-24; odd = male, even = female).
# Statements are fixed:
RAVDESS_STATEMENTS: dict[str, str] = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

RAVDESS_EMOTION_MAP: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

# ESD English speaker IDs are 0011-0020 in the canonical release.
ESD_ENGLISH_SPEAKERS: frozenset[str] = frozenset(
    {f"{i:04d}" for i in range(11, 21)}
)

# ESD subdirectory names (English release). Lowercased for our schema.
ESD_EMOTION_MAP: dict[str, str] = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "angry",
    "Surprise": "surprised",
}

# XTTS v2 wants 22.05 kHz mono training audio, clipped to <= 11s.
XTTS_SAMPLE_RATE: int = 22_050
XTTS_MAX_SECONDS: float = 11.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_manifest(path: Path) -> list[str]:
    """Validate a manifest CSV. Returns a list of human-readable issues."""
    issues: list[str] = []
    if not path.exists():
        return [f"Manifest not found: {path}"]

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return ["Empty manifest file (no header row)."]
        missing = [c for c in MANIFEST_COLUMNS if c not in reader.fieldnames]
        if missing:
            issues.append(f"Missing required columns: {', '.join(missing)}")
            return issues

        base = path.parent
        seen_paths: set[str] = set()
        for lineno, row in enumerate(reader, start=2):
            audio_raw = (row.get("audio_path") or "").strip()
            text = (row.get("text") or "").strip()
            speaker = (row.get("speaker") or "").strip()
            emotion = (row.get("emotion") or "").strip().lower()

            if not audio_raw:
                issues.append(f"line {lineno}: empty audio_path")
                continue
            if audio_raw in seen_paths:
                issues.append(f"line {lineno}: duplicate audio_path '{audio_raw}'")
            seen_paths.add(audio_raw)

            audio_path = Path(audio_raw)
            if not audio_path.is_absolute():
                audio_path = (base / audio_path).resolve()
            if not audio_path.exists():
                issues.append(f"line {lineno}: audio file does not exist: {audio_path}")

            if not text:
                issues.append(f"line {lineno}: empty text")
            if not speaker:
                issues.append(f"line {lineno}: empty speaker")
            if emotion not in EMOTIONS:
                issues.append(
                    f"line {lineno}: unknown emotion '{emotion}' "
                    f"(allowed: {', '.join(EMOTIONS)})"
                )
    return issues


# ---------------------------------------------------------------------------
# Manifest writer helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRow:
    """Single row of our canonical manifest."""

    audio_path: str
    text: str
    speaker: str
    emotion: str


def _write_manifest(rows: Iterable[ManifestRow], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "text": row.text,
                    "speaker": row.speaker,
                    "emotion": row.emotion,
                }
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# RAVDESS ingestion
# ---------------------------------------------------------------------------


_RAVDESS_RE = re.compile(
    r"^(?P<mod>\d{2})-(?P<vc>\d{2})-(?P<emo>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<stmt>\d{2})-(?P<rep>\d{2})-(?P<actor>\d{2})\.wav$"
)


def ingest_ravdess(ravdess_root: Path, out_manifest: Path) -> int:
    """Ingest the RAVDESS Speech dataset into our manifest format.

    Walks ``ravdess_root`` looking for ``Actor_NN/03-01-EE-II-SS-RR-AA.wav``
    files. Writes a manifest CSV to ``out_manifest`` and returns the row count.
    """
    if not ravdess_root.exists():
        raise FileNotFoundError(f"RAVDESS root not found: {ravdess_root}")

    rows: list[ManifestRow] = []
    for wav in sorted(ravdess_root.rglob("*.wav")):
        match = _RAVDESS_RE.match(wav.name)
        if not match:
            continue
        if match["mod"] != "03" or match["vc"] != "01":
            # Skip video/song modalities.
            continue
        emotion = RAVDESS_EMOTION_MAP.get(match["emo"])
        if emotion is None:
            logger.warning("Unknown RAVDESS emotion code {} in {}", match["emo"], wav.name)
            continue
        text = RAVDESS_STATEMENTS.get(match["stmt"], "")
        if not text:
            logger.warning("Unknown RAVDESS statement code {} in {}", match["stmt"], wav.name)
            continue
        actor = int(match["actor"])
        # Odd actor IDs are male, even are female (RAVDESS convention).
        gender = "M" if actor % 2 == 1 else "F"
        speaker = f"RAVDESS_{gender}{actor:02d}"
        try:
            rel = wav.resolve().relative_to(out_manifest.parent.resolve())
            audio_str = rel.as_posix()
        except ValueError:
            audio_str = str(wav.resolve())
        rows.append(
            ManifestRow(audio_path=audio_str, text=text, speaker=speaker, emotion=emotion)
        )

    count = _write_manifest(rows, out_manifest)
    logger.info("Wrote {} RAVDESS rows to {}", count, out_manifest)
    return count


# ---------------------------------------------------------------------------
# ESD ingestion
# ---------------------------------------------------------------------------


def _load_esd_transcripts(speaker_dir: Path) -> dict[str, str]:
    """Parse the per-speaker ``<speaker>.txt`` transcript file.

    The file is tab-separated: ``<utt_id>\\t<text>\\t<emotion>`` (the emotion
    column is occasionally absent in older releases — we ignore it).
    """
    transcripts: dict[str, str] = {}
    speaker_id = speaker_dir.name
    txt = speaker_dir / f"{speaker_id}.txt"
    if not txt.exists():
        # Fallback: any .txt in the speaker dir.
        candidates = list(speaker_dir.glob("*.txt"))
        if not candidates:
            return transcripts
        txt = candidates[0]
    for raw in txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
        utt_id, text = parts[0].strip(), parts[1].strip()
        transcripts[utt_id] = text
    return transcripts


def ingest_esd(esd_root: Path, out_manifest: Path) -> int:
    """Ingest the ESD (English speakers only) dataset into our manifest format.

    Expected layout::

        esd_root/
            0011/
                0011.txt
                Angry/
                    0011_000351.wav
                    ...
                Happy/...
                Neutral/...
                Sad/...
                Surprise/...
            0012/...

    Only speakers in :data:`ESD_ENGLISH_SPEAKERS` are imported. Returns the
    number of rows written.
    """
    if not esd_root.exists():
        raise FileNotFoundError(f"ESD root not found: {esd_root}")

    rows: list[ManifestRow] = []
    for speaker_dir in sorted(p for p in esd_root.iterdir() if p.is_dir()):
        speaker_id = speaker_dir.name
        if speaker_id not in ESD_ENGLISH_SPEAKERS:
            continue
        transcripts = _load_esd_transcripts(speaker_dir)
        if not transcripts:
            logger.warning("No transcripts found for ESD speaker {}", speaker_id)
            continue
        for emo_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
            emotion = ESD_EMOTION_MAP.get(emo_dir.name)
            if emotion is None:
                continue
            for wav in sorted(emo_dir.rglob("*.wav")):
                utt_id = wav.stem
                text = transcripts.get(utt_id)
                if not text:
                    # ESD utt IDs sometimes include the speaker prefix.
                    text = transcripts.get(utt_id.split("_")[-1], "")
                if not text:
                    continue
                try:
                    rel = wav.resolve().relative_to(out_manifest.parent.resolve())
                    audio_str = rel.as_posix()
                except ValueError:
                    audio_str = str(wav.resolve())
                rows.append(
                    ManifestRow(
                        audio_path=audio_str,
                        text=text,
                        speaker=f"ESD_{speaker_id}",
                        emotion=emotion,
                    )
                )

    count = _write_manifest(rows, out_manifest)
    logger.info("Wrote {} ESD rows to {}", count, out_manifest)
    return count


# ---------------------------------------------------------------------------
# Audio prep for XTTS
# ---------------------------------------------------------------------------


def _load_torchaudio() -> tuple[object, object]:
    """Lazy import torch + torchaudio so the base package stays slim."""
    try:
        import torch  # type: ignore[import-not-found]
        import torchaudio  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "torchaudio is required for dataset preparation. Install training "
            'extras: pip install -e ".[training]"'
        ) from exc
    return torch, torchaudio


def _resample_and_clip(
    src_wav: Path,
    dst_wav: Path,
    *,
    target_sr: int = XTTS_SAMPLE_RATE,
    max_seconds: float = XTTS_MAX_SECONDS,
) -> float:
    """Resample ``src_wav`` to mono ``target_sr`` Hz, clip to ``max_seconds``.

    Returns the resulting duration in seconds.
    """
    torch, torchaudio = _load_torchaudio()
    waveform, sr = torchaudio.load(str(src_wav))  # type: ignore[attr-defined]
    # Downmix to mono if stereo+.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)  # type: ignore[attr-defined]
        waveform = resampler(waveform)
    max_samples = int(max_seconds * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst_wav), waveform, target_sr)  # type: ignore[attr-defined]
    return float(waveform.shape[1]) / target_sr


def prepare_for_xtts(manifest: Path, out_dir: Path) -> Path:
    """Convert our manifest into XTTS v2 fine-tuning layout.

    Output structure::

        out_dir/
            metadata_train.csv   # LJSpeech-style: <wav_id>|<text>|<text>
            metadata_eval.csv
            wavs/<wav_id>.wav     # 22.05 kHz mono, <= 11s
            speakers.csv          # wav_id,speaker,emotion (for reference picking)
            reference_clips/<speaker>/<emotion>/<wav_id>.wav

    A small held-out eval split (~5%, capped at 100 rows) is taken from the
    tail of the manifest for validation. Returns ``out_dir``.
    """
    issues = validate_manifest(manifest)
    if issues:
        raise ValueError(
            "Manifest has issues; fix before preparing:\n  - " + "\n  - ".join(issues)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = out_dir / "wavs"
    refs_dir = out_dir / "reference_clips"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    base = manifest.parent
    rows: list[tuple[str, str, str, str, Path]] = []  # (wav_id, text, speaker, emo, src_path)
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            audio_raw = row["audio_path"].strip()
            src = Path(audio_raw)
            if not src.is_absolute():
                src = (base / src).resolve()
            wav_id = f"utt_{i:07d}"
            rows.append(
                (
                    wav_id,
                    row["text"].strip(),
                    row["speaker"].strip(),
                    row["emotion"].strip().lower(),
                    src,
                )
            )

    if not rows:
        raise ValueError("Manifest contains zero rows.")

    # Resample + clip each clip into wavs/.
    speakers_csv_path = out_dir / "speakers.csv"
    refs_picked: dict[tuple[str, str], int] = Counter()
    refs_per_combo_target = 2

    metadata_lines: list[tuple[str, str, str]] = []
    with speakers_csv_path.open("w", encoding="utf-8", newline="") as sfh:
        speakers_writer = csv.writer(sfh)
        speakers_writer.writerow(["wav_id", "speaker", "emotion", "duration_seconds"])
        for wav_id, text, speaker, emotion, src in rows:
            dst = wavs_dir / f"{wav_id}.wav"
            if dst.exists():
                # Resumable: skip work already done.
                duration = _wav_duration(dst)
            else:
                try:
                    duration = _resample_and_clip(src, dst)
                except Exception as exc:  # noqa: BLE001 - we re-raise summary
                    logger.error("Failed to process {}: {}", src, exc)
                    continue
            speakers_writer.writerow([wav_id, speaker, emotion, f"{duration:.3f}"])
            metadata_lines.append((wav_id, speaker, emotion))

            # Save up to N reference clips per (speaker, emotion).
            combo = (speaker, emotion)
            if refs_picked[combo] < refs_per_combo_target and duration >= 3.0:
                ref_dst = refs_dir / speaker / emotion / f"{wav_id}.wav"
                if not ref_dst.exists():
                    ref_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dst, ref_dst)
                refs_picked[combo] += 1

    # If some (speaker, emotion) combos have no >=3s clip, fall back to whatever
    # is longest.
    _backfill_references(out_dir, refs_picked, refs_per_combo_target)

    # LJSpeech metadata: wav_id|normalised_text|raw_text.
    train_rows, eval_rows = _split_train_eval(rows)
    _write_ljspeech_metadata(out_dir / "metadata_train.csv", train_rows)
    _write_ljspeech_metadata(out_dir / "metadata_eval.csv", eval_rows)

    logger.info(
        "Prepared {} train / {} eval clips into {}",
        len(train_rows),
        len(eval_rows),
        out_dir,
    )
    return out_dir


def _wav_duration(path: Path) -> float:
    """Return WAV duration in seconds (lightweight, no decode)."""
    import wave

    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate) if rate else 0.0


def _split_train_eval(
    rows: list[tuple[str, str, str, str, Path]],
) -> tuple[list[tuple[str, str, str, str, Path]], list[tuple[str, str, str, str, Path]]]:
    """Hold out ~5% (cap 100) from the tail for validation."""
    eval_size = min(100, max(1, len(rows) // 20)) if len(rows) > 20 else 0
    if eval_size == 0:
        return rows, []
    return rows[:-eval_size], rows[-eval_size:]


def _write_ljspeech_metadata(
    path: Path, rows: list[tuple[str, str, str, str, Path]]
) -> None:
    """LJSpeech CSV is ``id|text|text`` separated by ``|``."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for wav_id, text, _speaker, _emotion, _src in rows:
            writer.writerow([wav_id, text, text])


def _backfill_references(
    out_dir: Path, picked: Counter[tuple[str, str]], target: int
) -> None:
    """For (speaker, emotion) combos with < ``target`` refs, copy the longest
    available clip from ``wavs/`` as a fallback reference.
    """
    speakers_csv = out_dir / "speakers.csv"
    refs_dir = out_dir / "reference_clips"
    wavs_dir = out_dir / "wavs"

    # Group wav_ids by (speaker, emotion) with their durations.
    by_combo: dict[tuple[str, str], list[tuple[str, float]]] = {}
    with speakers_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            combo = (row["speaker"], row["emotion"])
            by_combo.setdefault(combo, []).append(
                (row["wav_id"], float(row["duration_seconds"]))
            )

    for combo, items in by_combo.items():
        deficit = target - picked.get(combo, 0)
        if deficit <= 0:
            continue
        speaker, emotion = combo
        ref_subdir = refs_dir / speaker / emotion
        ref_subdir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem for p in ref_subdir.glob("*.wav")}
        # Sort by duration desc, pick clips we haven't already saved.
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        for wav_id, _dur in items_sorted:
            if deficit <= 0:
                break
            if wav_id in existing:
                continue
            src = wavs_dir / f"{wav_id}.wav"
            if not src.exists():
                continue
            shutil.copy2(src, ref_subdir / f"{wav_id}.wav")
            deficit -= 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def dataset_stats(manifest: Path) -> dict[str, object]:
    """Count clips per emotion and per speaker and sum durations."""
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    per_emotion: Counter[str] = Counter()
    per_speaker: Counter[str] = Counter()
    per_combo: Counter[tuple[str, str]] = Counter()
    total_rows = 0
    total_duration = 0.0
    base = manifest.parent

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            emotion = (row.get("emotion") or "").strip().lower()
            speaker = (row.get("speaker") or "").strip()
            per_emotion[emotion] += 1
            per_speaker[speaker] += 1
            per_combo[(speaker, emotion)] += 1
            total_rows += 1

            audio_raw = (row.get("audio_path") or "").strip()
            audio_path = Path(audio_raw)
            if not audio_path.is_absolute():
                audio_path = (base / audio_path).resolve()
            if audio_path.exists():
                try:
                    total_duration += _wav_duration(audio_path)
                except (OSError, EOFError, ValueError):
                    # Non-WAV or unreadable; skip duration counting.
                    continue

    return {
        "total_rows": total_rows,
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_hms": _hms(total_duration),
        "per_emotion": dict(per_emotion),
        "per_speaker": dict(per_speaker),
        "per_speaker_emotion": {
            f"{sp}/{em}": n for (sp, em), n in sorted(per_combo.items())
        },
    }


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

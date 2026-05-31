"""Voice reference clip library.

Manages a tree of emotion-tagged reference clips per character::

    voices/                           (library root)
      Gael/
        neutral.wav
        angry.wav
        whispered.wav
        sad.wav
      Sera/
        neutral.wav
        sad.wav
      narrator/
        calm.wav
        neutral.wav

These reference clips are consumed by voice-cloning backends (XTTS v2,
Chatterbox) which copy the voice identity AND emotional prosody of the
clip into newly-synthesized text. No model training required.

Recommended clip characteristics for XTTS v2:
    - 6 to 30 seconds
    - 22050 Hz or higher sample rate (we resample at load)
    - Mono preferred (stereo is downmixed)
    - Clean recording: minimal background noise
    - The actor genuinely expressing the target emotion
"""
from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from .emotion import EMOTIONS


# Recommended bounds; we warn but don't reject outside these.
MIN_CLIP_SECONDS = 4.0
MAX_CLIP_SECONDS = 30.0
PREFERRED_SAMPLE_RATE = 22_050

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class ClipInfo:
    """Metadata for one reference clip in the library."""

    character: str
    emotion: str
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    issues: tuple[str, ...]    # empty tuple if all checks pass

    @property
    def ok(self) -> bool:
        return not self.issues


class VoiceLibrary:
    """Manages the on-disk reference-clip tree.

    Library root defaults to ``<project>/voices/``. Pass any directory
    to ``VoiceLibrary(root)`` to use a different location.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # --------------------------------------------------------------- paths

    def character_dir(self, character: str) -> Path:
        return self.root / character

    def clip_path(self, character: str, emotion: str, ext: str = ".wav") -> Path:
        if emotion not in EMOTIONS:
            raise ValueError(
                f"Unknown emotion {emotion!r}. Allowed: {', '.join(EMOTIONS)}"
            )
        if not ext.startswith("."):
            ext = "." + ext
        return self.character_dir(character) / f"{emotion}{ext}"

    def has_clip(self, character: str, emotion: str) -> bool:
        return self.find_clip(character, emotion) is not None

    def find_clip(self, character: str, emotion: str) -> Path | None:
        """Return the first existing clip for (character, emotion), trying
        all supported formats in priority order (wav > flac > mp3 > ogg > m4a).
        """
        char_dir = self.character_dir(character)
        if not char_dir.exists():
            return None
        for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            candidate = char_dir / f"{emotion}{ext}"
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------ listing

    def list_characters(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def list_clips(self, character: str | None = None) -> list[ClipInfo]:
        """List every (character, emotion) clip with metadata."""
        if not self.root.exists():
            return []
        characters = [character] if character else self.list_characters()
        out: list[ClipInfo] = []
        for ch in characters:
            char_dir = self.character_dir(ch)
            if not char_dir.exists():
                continue
            for emotion in EMOTIONS:
                clip = self.find_clip(ch, emotion)
                if clip is None:
                    continue
                info = _inspect_clip(clip, character=ch, emotion=emotion)
                out.append(info)
        return out

    def coverage(self, characters: list[str]) -> dict[str, dict[str, bool]]:
        """For each character, which emotions are covered?"""
        return {
            ch: {emo: self.has_clip(ch, emo) for emo in EMOTIONS}
            for ch in characters
        }

    def missing(self, characters: list[str], required_emotions: list[str]) -> list[tuple[str, str]]:
        """Return (character, emotion) pairs that lack a clip."""
        out: list[tuple[str, str]] = []
        for ch in characters:
            for emo in required_emotions:
                if not self.has_clip(ch, emo):
                    out.append((ch, emo))
        return out

    # ------------------------------------------------------------ import

    def import_clip(
        self,
        source: Path,
        character: str,
        emotion: str,
        *,
        overwrite: bool = False,
        normalize: bool = True,
    ) -> Path:
        """Copy a clip from ``source`` into the library.

        If ``normalize=True``, resamples to PREFERRED_SAMPLE_RATE mono WAV.
        Otherwise the file is copied as-is (extension preserved).

        Raises FileExistsError if a clip already exists and overwrite=False.
        """
        if emotion not in EMOTIONS:
            raise ValueError(
                f"Unknown emotion {emotion!r}. Allowed: {', '.join(EMOTIONS)}"
            )
        if not source.exists():
            raise FileNotFoundError(f"Source clip not found: {source}")
        if source.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {source.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        existing = self.find_clip(character, emotion)
        if existing is not None and not overwrite:
            raise FileExistsError(
                f"Clip already exists at {existing}. Pass overwrite=True to replace."
            )
        if existing is not None and overwrite:
            existing.unlink()

        char_dir = self.character_dir(character)
        char_dir.mkdir(parents=True, exist_ok=True)

        if normalize:
            dst = char_dir / f"{emotion}.wav"
            _resample_to_wav(source, dst, PREFERRED_SAMPLE_RATE)
        else:
            dst = char_dir / f"{emotion}{source.suffix.lower()}"
            shutil.copy2(source, dst)
        return dst

    def delete_clip(self, character: str, emotion: str) -> bool:
        clip = self.find_clip(character, emotion)
        if clip is None:
            return False
        clip.unlink()
        # Remove empty character directory.
        char_dir = self.character_dir(character)
        if char_dir.exists() and not any(char_dir.iterdir()):
            char_dir.rmdir()
        return True

    # ------------------------------------------------------------ validate

    def validate(self, character: str | None = None) -> list[ClipInfo]:
        """Return ClipInfo for every clip, including those with issues."""
        return [c for c in self.list_clips(character)]


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def _inspect_clip(path: Path, *, character: str, emotion: str) -> ClipInfo:
    """Inspect a clip and report duration, sample rate, channels, issues."""
    duration = 0.0
    sample_rate = 0
    channels = 0
    issues: list[str] = []

    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                channels = w.getnchannels()
                sample_rate = w.getframerate()
                frames = w.getnframes()
                duration = frames / float(sample_rate) if sample_rate else 0.0
        except (wave.Error, EOFError) as e:
            issues.append(f"unreadable WAV: {e}")
    else:
        # Use soundfile for non-WAV (already a base dep).
        try:
            import soundfile as sf
            with sf.SoundFile(str(path)) as snd:
                channels = snd.channels
                sample_rate = snd.samplerate
                duration = len(snd) / float(sample_rate) if sample_rate else 0.0
        except Exception as e:  # noqa: BLE001 - libsndfile errors
            issues.append(f"unreadable {suffix}: {e}")

    if duration < MIN_CLIP_SECONDS and duration > 0:
        issues.append(
            f"too short ({duration:.1f}s; min {MIN_CLIP_SECONDS:.0f}s recommended)"
        )
    if duration > MAX_CLIP_SECONDS:
        issues.append(
            f"too long ({duration:.1f}s; max {MAX_CLIP_SECONDS:.0f}s recommended — "
            f"will be clipped at synthesis time)"
        )
    if sample_rate and sample_rate < 16000:
        issues.append(
            f"low sample rate ({sample_rate} Hz; >= 22050 Hz recommended)"
        )
    if channels > 2:
        issues.append(f"unusual channel count ({channels})")

    return ClipInfo(
        character=character,
        emotion=emotion,
        path=path,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        issues=tuple(issues),
    )


# ---------------------------------------------------------------------------
# Audio resampling (used by import + RAVDESS bulk-import)
# ---------------------------------------------------------------------------


def _resample_to_wav(src: Path, dst: Path, target_sr: int = PREFERRED_SAMPLE_RATE) -> None:
    """Resample any audio file to mono WAV at target_sr.

    Uses soundfile + numpy for WAV in/out (both base deps). For non-WAV
    inputs that soundfile can't read (e.g. some MP3s on macOS), falls back
    to ffmpeg if present.
    """
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    except RuntimeError:
        # libsndfile can't read this file — try ffmpeg fallback.
        _ffmpeg_to_wav(src, dst, target_sr)
        return

    # Downmix to mono.
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample if needed using a simple polyphase resampler.
    if sr != target_sr:
        data = _resample_simple(data, sr, target_sr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, target_sr, subtype="PCM_16")


def _resample_simple(data, src_sr: int, dst_sr: int):
    """Simple resampling via scipy if available; ffmpeg fallback otherwise."""
    try:
        from scipy.signal import resample_poly  # optional dep
        from math import gcd
        g = gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return resample_poly(data, up, down).astype("float32")
    except ImportError:
        import numpy as np
        # Cheap linear interpolation fallback.
        ratio = dst_sr / src_sr
        n = int(len(data) * ratio)
        xs = np.linspace(0, len(data) - 1, n)
        idx = xs.astype(int)
        return data[idx].astype("float32")


def _ffmpeg_to_wav(src: Path, dst: Path, target_sr: int) -> None:
    """Fallback resampler using ffmpeg subprocess."""
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Cannot read {src.suffix} files without ffmpeg. "
            f"Install with: brew install ffmpeg"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(target_sr),
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)

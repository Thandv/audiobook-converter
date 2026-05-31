"""M4B packaging.

Stitch per-chapter audio files into a single .m4b (audiobook) with
chapter markers, metadata, and optional cover art. Uses ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .stitch import ChapterRenderResult


@dataclass
class BookMetadata:
    title: str
    author: str
    cover_path: Path | None = None


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with: `brew install ffmpeg`."
        )


def _write_ffmetadata(
    metadata: BookMetadata, results: list[ChapterRenderResult], path: Path
) -> None:
    """Write an ffmpeg metadata file with chapter markers (timestamps in ms)."""
    lines = [
        ";FFMETADATA1",
        f"title={metadata.title}",
        f"artist={metadata.author}",
        f"album={metadata.title}",
        f"album_artist={metadata.author}",
        "genre=Audiobook",
        "",
    ]
    cursor_ms = 0
    for r in results:
        start_ms = cursor_ms
        end_ms = cursor_ms + int(r.duration_seconds * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={r.chapter.display_title}",
            "",
        ]
        cursor_ms = end_ms
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_concat_list(results: list[ChapterRenderResult], path: Path) -> None:
    """Write an ffmpeg concat demuxer list referencing each chapter file."""
    lines: list[str] = []
    for r in results:
        # Escape single quotes for the concat demuxer.
        escaped = str(r.audio_path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_m4b(
    results: list[ChapterRenderResult],
    metadata: BookMetadata,
    output_path: Path,
    bitrate: str = "64k",
) -> Path:
    """Combine chapter files into an .m4b audiobook with chapter markers."""
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="audiobook_") as td:
        td_path = Path(td)
        list_file = td_path / "concat.txt"
        meta_file = td_path / "chapters.ffmetadata"
        _write_concat_list(results, list_file)
        _write_ffmetadata(metadata, results, meta_file)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-i", str(meta_file),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", bitrate,
            "-ac", "1",
            "-movflags", "+faststart",
        ]
        if metadata.cover_path and metadata.cover_path.exists():
            cmd += [
                "-i", str(metadata.cover_path),
                "-map", "0:a",
                "-map", "2",
                "-c:v", "copy",
                "-disposition:v:0", "attached_pic",
            ]
        cmd += [str(output_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}):\n{proc.stderr}"
            )

    return output_path

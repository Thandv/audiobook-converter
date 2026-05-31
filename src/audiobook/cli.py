"""Command-line interface for the manuscript-to-audiobook pipeline.

Running `audiobook` with no arguments launches an interactive guided
flow. Otherwise see the subcommands below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .attribution import attribute_book, attribution_stats
from .package import BookMetadata, package_m4b
from .parser import book_stats, parse_manuscript
from .pronounce import load_pronunciations
from .stitch import ChapterRenderResult, RenderConfig, render_book
from .synth import load_voice_cast


console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_book_filename(title: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip()
    return out.replace(" ", "_") or "audiobook"


@click.group(invoke_without_command=True)
@click.version_option(package_name="audiobook-converter")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Manuscript markdown -> M4B audiobook (Kokoro TTS / XTTS).

    Run with no subcommand to launch the interactive walkthrough.
    """
    if ctx.invoked_subcommand is None:
        # Launch interactive flow.
        from .interactive import run_interactive
        run_interactive()


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
def inspect(manuscript: Path) -> None:
    """Parse the manuscript and print structure stats (no audio)."""
    book = parse_manuscript(manuscript)
    stats = book_stats(book)

    table = Table(title=f"{book.title}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Title", book.title)
    table.add_row("Author", book.author or "—")
    for k, v in stats.items():
        table.add_row(k.capitalize(), f"{v:,}")
    estimated_minutes = stats["words"] / 150
    table.add_row("Estimated audio (min)", f"{estimated_minutes:.0f}")
    table.add_row("Estimated audio (hr:mm)",
                  f"{int(estimated_minutes // 60)}:{int(estimated_minutes % 60):02d}")
    console.print(table)

    ch_table = Table(title="Chapters")
    ch_table.add_column("#", justify="right")
    ch_table.add_column("Title")
    ch_table.add_column("Subtitle")
    ch_table.add_column("Scenes", justify="right")
    ch_table.add_column("Words", justify="right")
    for ch in book.chapters:
        words = sum(
            len(p.plain_text().split()) for s in ch.scenes for p in s.paragraphs
        )
        ch_table.add_row(
            str(ch.number), ch.title, ch.subtitle or "", str(len(ch.scenes)), f"{words:,}"
        )
    console.print(ch_table)


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("--voices", "voices_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out", "output", type=click.Path(path_type=Path), default=None)
def attribute(manuscript: Path, voices_path: Path | None, output: Path | None) -> None:
    """Run dialogue attribution and dump JSON for inspection / editing."""
    voices_path = voices_path or (_project_root() / "config" / "voices.yaml")
    output = output or (_project_root() / "output" / "attribution.json")

    book = parse_manuscript(manuscript)
    voice_cast = load_voice_cast(voices_path)
    attribution = attribute_book(book, voice_cast.cast.keys())

    out: list[dict] = []
    for chapter, ch_paras in zip(book.chapters, attribution):
        out.append({
            "chapter": chapter.display_title,
            "paragraphs": [
                [u.to_dict() for u in para] for para in ch_paras
            ],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    stats = attribution_stats(attribution)
    table = Table(title="Dialogue utterances by speaker")
    table.add_column("Speaker", style="cyan")
    table.add_column("Lines", justify="right")
    for speaker, count in stats.items():
        table.add_row(speaker, f"{count:,}")
    console.print(table)
    console.print(f"\nFull attribution written to [bold]{output}[/bold]")


def _backend_options(f):
    f = click.option(
        "--backend", type=click.Choice(["kokoro", "xtts"]), default="kokoro",
        help="TTS backend. xtts requires a fine-tuned model directory."
    )(f)
    f = click.option(
        "--model-dir", "backend_model_dir",
        type=click.Path(exists=True, path_type=Path), default=None,
        help="Path to fine-tuned XTTS model (required if --backend xtts)."
    )(f)
    return f


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("--mode", type=click.Choice(["single", "multi"]), default="single")
@click.option("--chapter", "chapters", type=int, multiple=True,
              help="Chapter number(s) to sample. Default: 1.")
@click.option("--paragraphs", type=int, default=8,
              help="Paragraphs to render from the chapter start.")
@click.option("--voices", "voices_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--pronunciations", "pron_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out", "output", type=click.Path(path_type=Path), default=None)
@_backend_options
def sample(
    manuscript: Path, mode: str, chapters: tuple[int, ...], paragraphs: int,
    voices_path: Path | None, pron_path: Path | None, output: Path | None,
    backend: str, backend_model_dir: Path | None,
) -> None:
    """Render a short sample to verify setup and voice choices."""
    voices_path = voices_path or (_project_root() / "config" / "voices.yaml")
    pron_path = pron_path or (_project_root() / "config" / "pronunciations.yaml")
    output = output or (_project_root() / "samples")
    chapter_nums = chapters or (1,)

    book = parse_manuscript(manuscript)
    voice_cast = load_voice_cast(voices_path)
    pronouncer = load_pronunciations(pron_path) if pron_path.exists() else None

    from .parser import Book, Scene
    from copy import copy
    trimmed = Book(title=book.title, author=book.author, chapters=[])
    for n in chapter_nums:
        if n < 1 or n > len(book.chapters):
            console.print(f"[yellow]Skipping chapter {n}: out of range.[/yellow]")
            continue
        src = book.chapters[n - 1]
        flat = [p for s in src.scenes for p in s.paragraphs][:paragraphs]
        trimmed_chap = copy(src)
        trimmed_chap.scenes = [Scene(paragraphs=flat)]
        trimmed.chapters.append(trimmed_chap)

    cfg = RenderConfig(
        mode=mode, output_dir=output / mode,
        pronouncer=pronouncer, voices=voice_cast,
        backend_name=backend, backend_model_dir=backend_model_dir,
    )
    results = render_book(trimmed, cfg)
    console.print(f"\n[green]Wrote {len(results)} sample file(s) to {output / mode}[/green]")
    for r in results:
        console.print(f"  {r.audio_path}  ({r.duration_seconds:.1f}s)")


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("--mode", type=click.Choice(["single", "multi", "both"]), default="single")
@click.option("--voices", "voices_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--pronunciations", "pron_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out", "output", type=click.Path(path_type=Path), default=None)
@click.option("--cover", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--no-m4b", is_flag=True, help="Skip M4B packaging step.")
@click.option("--bitrate", default="64k", help="M4B audio bitrate.")
@_backend_options
def render(
    manuscript: Path, mode: str,
    voices_path: Path | None, pron_path: Path | None,
    output: Path | None, cover: Path | None, no_m4b: bool, bitrate: str,
    backend: str, backend_model_dir: Path | None,
) -> None:
    """Render the full book — per-chapter MP3s + optional .m4b."""
    voices_path = voices_path or (_project_root() / "config" / "voices.yaml")
    pron_path = pron_path or (_project_root() / "config" / "pronunciations.yaml")
    output = output or (_project_root() / "output")

    book = parse_manuscript(manuscript)
    voice_cast = load_voice_cast(voices_path)
    pronouncer = load_pronunciations(pron_path) if pron_path.exists() else None

    modes = ["single", "multi"] if mode == "both" else [mode]
    for m in modes:
        console.rule(f"[bold]{m.upper()} narrator mode  ({backend})")
        cfg = RenderConfig(
            mode=m, output_dir=output / m,
            pronouncer=pronouncer, voices=voice_cast,
            backend_name=backend, backend_model_dir=backend_model_dir,
        )
        results = render_book(book, cfg)
        total_sec = sum(r.duration_seconds for r in results)
        hh = int(total_sec // 3600)
        mm = int((total_sec % 3600) // 60)
        console.print(f"[green]Rendered {len(results)} chapters ({hh}h {mm:02d}m)[/green]")
        if not no_m4b:
            try:
                m4b_path = output / m / f"{_safe_book_filename(book.title)}.m4b"
                package_m4b(
                    results,
                    BookMetadata(title=book.title, author=book.author, cover_path=cover),
                    m4b_path, bitrate=bitrate,
                )
                console.print(f"[green]M4B audiobook: {m4b_path}[/green]")
            except RuntimeError as e:
                console.print(f"[red]M4B packaging skipped: {e}[/red]")


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("--chapters-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--cover", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out", "output", type=click.Path(path_type=Path), default=None)
@click.option("--bitrate", default="64k")
def repackage(
    manuscript: Path, chapters_dir: Path, cover: Path | None,
    output: Path | None, bitrate: str,
) -> None:
    """Re-package existing chapter audio files into an M4B."""
    import soundfile as sf
    book = parse_manuscript(manuscript)
    files = sorted(chapters_dir.glob("*.mp3")) or sorted(chapters_dir.glob("*.wav"))
    if len(files) != len(book.chapters):
        console.print(
            f"[yellow]Warning: {len(files)} audio files vs {len(book.chapters)} chapters."
            f" Pairing by sorted order.[/yellow]"
        )
    results: list[ChapterRenderResult] = []
    for ch, path in zip(book.chapters, files):
        with sf.SoundFile(str(path)) as snd:
            duration = len(snd) / snd.samplerate
        results.append(ChapterRenderResult(ch, path, duration))

    out_path = output or (chapters_dir.parent / f"{_safe_book_filename(book.title)}.m4b")
    package_m4b(
        results,
        BookMetadata(title=book.title, author=book.author, cover_path=cover),
        out_path, bitrate=bitrate,
    )
    console.print(f"[green]M4B written: {out_path}[/green]")


# Lazy import + register the training subcommand group. Keeping the
# import lazy means the heavy training deps don't fail-fast for normal
# rendering use.
try:
    from .training.cli import train as _train_group  # noqa: F401
    cli.add_command(_train_group)
except ImportError:
    # Training extras not installed — that's fine, training subcommand
    # simply won't appear in `audiobook --help`.
    pass


if __name__ == "__main__":
    cli()

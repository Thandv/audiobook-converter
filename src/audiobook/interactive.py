"""Interactive CLI flow — `audiobook` with no args runs this.

Walks the user through manuscript selection, mode selection, backend
choice, and starts the render. Uses Rich for the prompts so it's
keyboard-navigable and pretty.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .package import BookMetadata, package_m4b
from .parser import book_stats, parse_manuscript
from .pronounce import load_pronunciations
from .stitch import RenderConfig, render_book
from .synth import load_voice_cast


console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick_manuscript() -> Path:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 1 / 4[/bold cyan]  Manuscript",
        border_style="cyan",
    ))
    while True:
        raw = Prompt.ask(
            "[bold]Path to your manuscript[/bold] (Markdown .md file)",
            default=str(Path.home() / "Downloads" / "manuscript.md"),
        )
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            continue
        if path.suffix.lower() != ".md":
            console.print(f"[yellow]Warning: file isn't a .md ({path.suffix}). Continue anyway?[/yellow]")
            if not Confirm.ask("Continue?", default=False):
                continue
        return path


def _show_parse_preview(manuscript: Path) -> None:
    book = parse_manuscript(manuscript)
    stats = book_stats(book)
    table = Table(title=f"[bold]{book.title}[/bold]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Author", book.author or "—")
    for k, v in stats.items():
        table.add_row(k.capitalize(), f"{v:,}")
    est_min = stats["words"] / 150
    table.add_row("Estimated audio", f"{int(est_min // 60)}h {int(est_min % 60):02d}m")
    console.print(table)


def _pick_mode() -> str:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 2 / 4[/bold cyan]  Narration mode",
        border_style="cyan",
    ))
    console.print("[dim]single[/dim]  — one narrator does everything (traditional audiobook)")
    console.print("[dim]multi [/dim]  — different voice per character via auto-attribution")
    console.print("[dim]both  [/dim]  — render both modes back-to-back")
    while True:
        choice = Prompt.ask("[bold]Mode[/bold]", choices=["single", "multi", "both"], default="single")
        return choice


def _pick_backend() -> tuple[str, Path | None]:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 3 / 4[/bold cyan]  TTS backend",
        border_style="cyan",
    ))
    console.print("[dim]kokoro[/dim] — free, local, fast, 50+ built-in voices (default)")
    console.print("[dim]xtts  [/dim] — use a fine-tuned XTTS v2 model you trained with `audiobook train`")
    while True:
        choice = Prompt.ask(
            "[bold]Backend[/bold]", choices=["kokoro", "xtts"], default="kokoro"
        )
        if choice == "kokoro":
            return "kokoro", None
        # XTTS — ask for model dir.
        default_dir = str(_project_root() / "models")
        raw = Prompt.ask(
            "[bold]Path to fine-tuned XTTS model directory[/bold]",
            default=default_dir,
        )
        model_dir = Path(raw).expanduser().resolve()
        if not model_dir.exists():
            console.print(f"[red]Model directory not found: {model_dir}[/red]")
            console.print("[dim]Train one first with: audiobook train run --data ... --out ...[/dim]")
            if not Confirm.ask("Try a different path?", default=True):
                console.print("[yellow]Falling back to kokoro.[/yellow]")
                return "kokoro", None
            continue
        return "xtts", model_dir


def _pick_output_options() -> tuple[Path, Path | None, bool, str]:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 4 / 4[/bold cyan]  Output & packaging",
        border_style="cyan",
    ))
    default_out = str(_project_root() / "output")
    raw = Prompt.ask("[bold]Output directory[/bold]", default=default_out)
    out_dir = Path(raw).expanduser().resolve()

    cover: Path | None = None
    if Confirm.ask("Embed a cover image in the M4B?", default=False):
        cover_raw = Prompt.ask("[bold]Path to cover image (jpg/png)[/bold]")
        candidate = Path(cover_raw).expanduser().resolve()
        if candidate.exists():
            cover = candidate
        else:
            console.print(f"[yellow]Cover not found, continuing without it.[/yellow]")

    package = Confirm.ask("Package the result as an M4B audiobook? (recommended)", default=True)
    bitrate = "64k"
    if package:
        bitrate = Prompt.ask("[bold]M4B bitrate[/bold]", default="64k")
    return out_dir, cover, package, bitrate


def _confirm_and_render(
    manuscript: Path, mode: str, backend: str, backend_model_dir: Path | None,
    out_dir: Path, cover: Path | None, package: bool, bitrate: str,
) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Ready to render[/bold cyan]",
        border_style="cyan",
    ))
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan", justify="right")
    summary.add_column()
    summary.add_row("Manuscript:", str(manuscript))
    summary.add_row("Mode:",        mode)
    summary.add_row("Backend:",     backend + (f" (model: {backend_model_dir})" if backend_model_dir else ""))
    summary.add_row("Output:",      str(out_dir))
    summary.add_row("Cover:",       str(cover) if cover else "—")
    summary.add_row("Package M4B:", "yes" if package else "no")
    if package:
        summary.add_row("M4B bitrate:", bitrate)
    console.print(summary)

    console.print()
    console.print("[yellow]This may take several hours on CPU. Press Ctrl+C to cancel at any time.[/yellow]")
    if not Confirm.ask("[bold]Start rendering?[/bold]", default=True):
        console.print("[dim]Cancelled.[/dim]")
        return

    book = parse_manuscript(manuscript)
    pron_path = _project_root() / "config" / "pronunciations.yaml"
    voice_path = _project_root() / "config" / "voices.yaml"
    pronouncer = load_pronunciations(pron_path) if pron_path.exists() else None
    voices = load_voice_cast(voice_path)

    modes = ["single", "multi"] if mode == "both" else [mode]
    for m in modes:
        console.rule(f"[bold]{m.upper()} narrator mode")
        cfg = RenderConfig(
            mode=m,
            output_dir=out_dir / m,
            pronouncer=pronouncer,
            voices=voices,
            backend_name=backend,
            backend_model_dir=backend_model_dir,
        )
        results = render_book(book, cfg)
        total_sec = sum(r.duration_seconds for r in results)
        hh, mm = int(total_sec // 3600), int((total_sec % 3600) // 60)
        console.print(f"[green]Rendered {len(results)} chapters ({hh}h {mm:02d}m)[/green]")
        if package:
            from .cli import _safe_book_filename
            m4b_path = out_dir / m / f"{_safe_book_filename(book.title)}.m4b"
            try:
                package_m4b(
                    results,
                    BookMetadata(title=book.title, author=book.author, cover_path=cover),
                    m4b_path,
                    bitrate=bitrate,
                )
                console.print(f"[green]M4B audiobook: {m4b_path}[/green]")
            except RuntimeError as e:
                console.print(f"[red]M4B packaging failed: {e}[/red]")


def run_interactive() -> None:
    """Main interactive entrypoint."""
    console.print()
    console.print(Panel.fit(
        "[bold]Audiobook Converter[/bold]\n"
        "[dim]Markdown manuscript -> M4B audiobook with realistic voices[/dim]",
        border_style="bold cyan",
    ))

    manuscript = _pick_manuscript()
    _show_parse_preview(manuscript)

    mode = _pick_mode()
    backend, model_dir = _pick_backend()
    out_dir, cover, package, bitrate = _pick_output_options()

    _confirm_and_render(
        manuscript=manuscript,
        mode=mode,
        backend=backend,
        backend_model_dir=model_dir,
        out_dir=out_dir,
        cover=cover,
        package=package,
        bitrate=bitrate,
    )

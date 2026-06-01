"""`audiobook review ...` — reviewer/observer + auto-fixer CLI.

Subcommands:
  run        Run all review passes on a rendered book, emit findings.json
  show       Pretty-print a findings.json
  apply      Apply auto-fixable findings to config + schedule re-render
  iterate    Loop: render -> review -> apply -> render again, up to N rounds
"""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .review.fixer import apply_findings
from .review.reviewer import review_book
from .review.types import (
    FindingKind,
    FindingSeverity,
    ReviewReport,
)


console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@click.group()
def review() -> None:
    """Reviewer/observer: listen to the audiobook, find issues, propose fixes."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@review.command("run")
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--chapters-dir", "chapters_dir",
    type=click.Path(exists=True, path_type=Path), required=True,
    help="Directory of rendered chapter audio files.",
)
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--whisper-model", default="base.en",
    help="Whisper model size. tiny.en is fastest, small.en is more accurate.",
)
@click.option("--skip-transcribe", is_flag=True)
@click.option("--skip-metrics", is_flag=True)
@click.option("--skip-emotion", is_flag=True)
@click.option(
    "--analyzer", type=click.Choice(["content", "content+ml"]), default="content",
    help="Which text emotion analyzer's predictions to compare audio against.",
)
def run(
    manuscript: Path, chapters_dir: Path, out_path: Path | None,
    whisper_model: str, skip_transcribe: bool, skip_metrics: bool,
    skip_emotion: bool, analyzer: str,
) -> None:
    """Run all review passes on a rendered book."""
    out_path = out_path or (_project_root() / "output" / "review.json")
    voices_yaml = _project_root() / "config" / "voices.yaml"

    report = review_book(
        manuscript_path=manuscript,
        chapters_dir=chapters_dir,
        voices_yaml=voices_yaml,
        do_transcribe=not skip_transcribe,
        do_metrics=not skip_metrics,
        do_emotion_check=not skip_emotion,
        use_ml_analyzer=(analyzer == "content+ml"),
        whisper_model=whisper_model,
    )
    report.save(out_path)

    _print_summary(report)
    console.print(f"\n[green]Full report -> {out_path}[/green]")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@review.command("show")
@click.option(
    "--report", "report_path",
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--severity", type=click.Choice(["info", "suggestion", "high"]), default=None,
    help="Filter to findings of this severity or higher.",
)
@click.option(
    "--kind",
    type=click.Choice([k.value for k in FindingKind]),
    default=None,
    help="Filter to one kind of finding.",
)
@click.option("--chapter", type=int, default=None)
def show(report_path: Path, severity: str | None, kind: str | None, chapter: int | None) -> None:
    """Pretty-print findings from a saved review report."""
    report = ReviewReport.load(report_path)
    findings = report.findings

    if severity:
        thresh = {"info": 0, "suggestion": 1, "high": 2}[severity]
        rank = {FindingSeverity.INFO: 0, FindingSeverity.SUGGESTION: 1, FindingSeverity.HIGH: 2}
        findings = [f for f in findings if rank[f.severity] >= thresh]
    if kind:
        findings = [f for f in findings if f.kind.value == kind]
    if chapter is not None:
        findings = [f for f in findings if f.chapter_number == chapter]

    _print_summary(report)

    if not findings:
        console.print("\n[dim]No findings match the filters.[/dim]")
        return

    table = Table(title=f"{len(findings)} finding(s)")
    table.add_column("Ch", justify="right")
    table.add_column("Kind")
    table.add_column("Sev")
    table.add_column("Summary")
    table.add_column("Auto", justify="center")
    for f in findings:
        sev_color = {"high": "red", "suggestion": "yellow", "info": "dim"}.get(f.severity.value, "")
        table.add_row(
            str(f.chapter_number),
            f.kind.value,
            f"[{sev_color}]{f.severity.value}[/{sev_color}]",
            f.summary[:90],
            "✓" if f.auto_fixable else "",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@review.command("apply")
@click.option(
    "--report", "report_path",
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--chapters-dir", "chapters_dir",
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option(
    "--pronunciations", "pronunciations_yaml",
    type=click.Path(path_type=Path), default=None,
)
@click.option(
    "--apply-all", is_flag=True,
    help="Also apply suggestion-level fixes (default: only auto-fixable ones).",
)
@click.option("--dry-run", is_flag=True)
def apply(
    report_path: Path, chapters_dir: Path, pronunciations_yaml: Path | None,
    apply_all: bool, dry_run: bool,
) -> None:
    """Apply auto-fixable findings and delete chapter files that need re-render."""
    pronunciations_yaml = pronunciations_yaml or (
        _project_root() / "config" / "pronunciations.yaml"
    )
    report = ReviewReport.load(report_path)
    plan = apply_findings(
        report,
        pronunciations_yaml=pronunciations_yaml,
        chapters_dir=chapters_dir,
        apply_all=apply_all,
        dry_run=dry_run,
    )

    panel_lines = []
    if dry_run:
        panel_lines.append("[yellow]DRY RUN — no changes written.[/yellow]")
    panel_lines.extend(plan.summary_lines())
    if plan.pronunciations_added:
        panel_lines.append("\nPronunciations added:")
        for k, v in plan.pronunciations_added.items():
            panel_lines.append(f"  • {k} -> {v}")
    if plan.chapters_to_rerender:
        panel_lines.append(
            "\nChapters that need re-rendering: "
            + ", ".join(str(n) for n in sorted(plan.chapters_to_rerender))
        )
    if plan.proposed:
        panel_lines.append(
            "\n[dim]Findings the fixer left for your review "
            f"({len(plan.proposed)}):[/dim]"
        )
        for f in plan.proposed[:10]:
            panel_lines.append(
                f"  • Ch {f.chapter_number}: {f.kind.value} — {f.summary[:80]}"
            )
        if len(plan.proposed) > 10:
            panel_lines.append(f"  ... ({len(plan.proposed) - 10} more)")
    console.print(Panel("\n".join(panel_lines), title="Fix plan", border_style="cyan"))

    if plan.chapters_to_rerender and not dry_run:
        console.print(
            "\n[dim]Re-render the affected chapters with:[/dim]\n"
            f"  audiobook render <book.md> --mode single --resume --emotion-analyzer content"
        )


# ---------------------------------------------------------------------------
# iterate (the full cycle)
# ---------------------------------------------------------------------------


@review.command("iterate")
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--chapters-dir", "chapters_dir",
    type=click.Path(exists=True, path_type=Path), required=True,
)
@click.option("--mode", type=click.Choice(["single", "multi"]), default="single")
@click.option("--rounds", type=int, default=2)
@click.option("--whisper-model", default="base.en")
@click.option(
    "--apply-all", is_flag=True,
    help="Apply suggestion-level fixes too (not just auto-fixable ones).",
)
@click.option(
    "--no-emotion-check", is_flag=True,
    help="Skip the audio emotion classifier (faster, but no emotion findings).",
)
@click.option(
    "--max-new-pronunciations", type=int, default=20,
    help="Cap how many pronunciations a single round can auto-add (sanity check).",
)
def iterate(
    manuscript: Path, chapters_dir: Path, mode: str, rounds: int,
    whisper_model: str, apply_all: bool, no_emotion_check: bool,
    max_new_pronunciations: int,
) -> None:
    """Loop: review -> apply fixes -> re-render -> review again.

    Stops early when a round produces zero auto-fixable findings.
    """
    voices_yaml = _project_root() / "config" / "voices.yaml"
    pronunciations_yaml = _project_root() / "config" / "pronunciations.yaml"
    output_root = _project_root() / "output"

    for round_idx in range(1, rounds + 1):
        console.rule(f"[bold cyan]Iteration {round_idx} / {rounds}")
        report = review_book(
            manuscript_path=manuscript,
            chapters_dir=chapters_dir,
            voices_yaml=voices_yaml,
            do_transcribe=True,
            do_metrics=True,
            do_emotion_check=not no_emotion_check,
            whisper_model=whisper_model,
            round_number=round_idx,
        )
        report_path = output_root / f"review_round_{round_idx}.json"
        report.save(report_path)
        _print_summary(report)

        auto_fixable = report.auto_fixable()
        # Cap auto-pron count.
        pronunciation_fixes = [f for f in auto_fixable if f.kind == FindingKind.PRONUNCIATION]
        if len(pronunciation_fixes) > max_new_pronunciations:
            console.print(
                f"[yellow]Limiting to {max_new_pronunciations} pronunciation fixes "
                f"(found {len(pronunciation_fixes)}).[/yellow]"
            )
            keep = set(id(f) for f in pronunciation_fixes[:max_new_pronunciations])
            report.findings = [
                f for f in report.findings
                if f.kind != FindingKind.PRONUNCIATION or id(f) in keep
            ]

        if not report.auto_fixable():
            console.print("[green]No auto-fixable findings remain. Loop converged.[/green]")
            break

        plan = apply_findings(
            report,
            pronunciations_yaml=pronunciations_yaml,
            chapters_dir=chapters_dir,
            apply_all=apply_all,
            dry_run=False,
        )
        for line in plan.summary_lines():
            console.print(f"  {line}")

        if not plan.chapters_to_rerender:
            console.print("[dim]No chapters need re-render. Stopping.[/dim]")
            break

        # Re-render only the affected chapters by leveraging --resume.
        console.print(
            f"[dim]Re-rendering {len(plan.chapters_to_rerender)} chapter(s)...[/dim]"
        )
        # Run the render in-process so iterate is self-contained.
        from .parser import parse_manuscript
        from .pronounce import load_pronunciations
        from .stitch import RenderConfig, render_book
        from .synth import load_voice_cast

        book = parse_manuscript(manuscript)
        voice_cast = load_voice_cast(voices_yaml)
        pronouncer = load_pronunciations(pronunciations_yaml) if pronunciations_yaml.exists() else None
        cfg = RenderConfig(
            mode=mode,
            output_dir=chapters_dir.parent,
            pronouncer=pronouncer,
            voices=voice_cast,
            backend_name="kokoro",
            emotion_analyzer="content",
            resume=True,
        )
        render_book(book, cfg)

    console.print("[bold green]Iteration complete.[/bold green]")


# ---------------------------------------------------------------------------
# shared: summary print
# ---------------------------------------------------------------------------


def _print_summary(report: ReviewReport) -> None:
    counts = report.summary_table()
    total = sum(counts.values())
    auto = len(report.auto_fixable())
    rerender = len(report.chapters_needing_rerender())
    panel = Panel(
        f"Round [bold]{report.round_number}[/bold]   "
        f"Findings: [bold]{total}[/bold]   "
        f"Auto-fixable: [bold green]{auto}[/bold green]   "
        f"Chapters needing re-render: [bold yellow]{rerender}[/bold yellow]",
        title="Review summary",
        border_style="cyan",
    )
    console.print(panel)
    if counts:
        breakdown = Table.grid(padding=(0, 2))
        breakdown.add_column(style="cyan")
        breakdown.add_column(justify="right")
        for kind, n in counts.items():
            breakdown.add_row(kind, str(n))
        console.print(breakdown)

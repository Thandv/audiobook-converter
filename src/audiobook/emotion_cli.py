"""`audiobook emotions ...` — inspect what the analyzer thinks per sentence.

Subcommands:
  analyze   Run the analyzer over the manuscript, dump per-sentence JSON.
  stats     Aggregate emotion distribution across the book.
  preview   Render a few sample sentences at different emotions to verify
            (uses the chosen backend; useful for tuning the cast).
  lexicon   Show the bundled lexicon (sizes per emotion).
"""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .attribution import SceneState
from .emotion import EMOTIONS
from .emotion_analyzer import (
    AnalysisContext,
    EmotionAnalyzer,
    emotion_distribution,
    split_sentences,
)
from .emotion_lexicon import lexicon_size, words_per_emotion


console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@click.group()
def emotions() -> None:
    """Inspect and tune the per-sentence emotion analyzer."""


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@emotions.command("analyze")
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--analyzer", type=click.Choice(["content", "content+ml"]), default="content",
)
@click.option(
    "--mode", type=click.Choice(["single", "multi"]), default="multi",
    help="Run dialogue attribution before scoring (multi) or not (single).",
)
@click.option(
    "--chapter", "chapters", type=int, multiple=True,
    help="Only analyze listed chapters (1-based). Default: all.",
)
@click.option(
    "--out", "output", type=click.Path(path_type=Path), default=None,
)
def analyze(
    manuscript: Path, analyzer: str, mode: str,
    chapters: tuple[int, ...], output: Path | None,
) -> None:
    """Run the analyzer and write per-sentence emotion JSON for inspection."""
    from .parser import parse_manuscript
    from .pronounce import load_pronunciations
    from .synth import load_voice_cast
    from .attribution import attribute_paragraph, DEFAULT_PRONOUNS

    out = output or (_project_root() / "output" / "emotion_analysis.json")
    book = parse_manuscript(manuscript)
    voices = load_voice_cast(_project_root() / "config" / "voices.yaml")
    cast_names = set(voices.cast.keys())
    eng = EmotionAnalyzer(use_ml=(analyzer == "content+ml"))

    selected = set(chapters) if chapters else None
    out_data: list[dict] = []

    for i, chapter in enumerate(book.chapters):
        if selected is not None and (i + 1) not in selected:
            continue
        chap_entry: dict = {
            "chapter_number": chapter.number,
            "chapter": chapter.display_title,
            "scenes": [],
        }
        for s_i, scene in enumerate(chapter.scenes):
            eng.reset_scene()
            state = SceneState.fresh()
            scene_entry: dict = {"scene_index": s_i, "sentences": []}
            for paragraph in scene.paragraphs:
                if mode == "multi":
                    utts = attribute_paragraph(
                        paragraph.plain_text(), cast_names, state, DEFAULT_PRONOUNS
                    )
                else:
                    from .attribution import Utterance
                    utts = [Utterance("NARRATOR", paragraph.plain_text(), False)]
                surround_narration = " ".join(u.text for u in utts if not u.is_dialogue)
                for utt in utts:
                    for sent in split_sentences(utt.text):
                        ctx = AnalysisContext(
                            speaker=utt.speaker,
                            surrounding_narration=surround_narration if utt.is_dialogue else "",
                            is_dialogue=utt.is_dialogue,
                        )
                        result = eng.analyze(sent, ctx)
                        scene_entry["sentences"].append(result.to_dict())
            chap_entry["scenes"].append(scene_entry)
        out_data.append(chap_entry)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    # Print a small summary table.
    total = sum(len(s["sentences"]) for c in out_data for s in c["scenes"])
    console.print(f"[green]Analyzed {total:,} sentences -> {out}[/green]")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@emotions.command("stats")
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("--analyzer", type=click.Choice(["content", "content+ml"]), default="content")
@click.option("--mode", type=click.Choice(["single", "multi"]), default="multi")
def stats(manuscript: Path, analyzer: str, mode: str) -> None:
    """Show emotion distribution + top speakers per emotion."""
    from .parser import parse_manuscript
    from .synth import load_voice_cast
    from .attribution import attribute_paragraph, DEFAULT_PRONOUNS, Utterance

    book = parse_manuscript(manuscript)
    voices = load_voice_cast(_project_root() / "config" / "voices.yaml")
    cast_names = set(voices.cast.keys())
    eng = EmotionAnalyzer(use_ml=(analyzer == "content+ml"))

    counts: dict[str, int] = {}
    speaker_emo: dict[tuple[str, str], int] = {}
    total = 0

    for chapter in book.chapters:
        for scene in chapter.scenes:
            eng.reset_scene()
            state = SceneState.fresh()
            for paragraph in scene.paragraphs:
                if mode == "multi":
                    utts = attribute_paragraph(
                        paragraph.plain_text(), cast_names, state, DEFAULT_PRONOUNS
                    )
                else:
                    utts = [Utterance("NARRATOR", paragraph.plain_text(), False)]
                surround_narration = " ".join(u.text for u in utts if not u.is_dialogue)
                for utt in utts:
                    for sent in split_sentences(utt.text):
                        ctx = AnalysisContext(
                            speaker=utt.speaker,
                            surrounding_narration=surround_narration if utt.is_dialogue else "",
                            is_dialogue=utt.is_dialogue,
                        )
                        r = eng.analyze(sent, ctx)
                        counts[r.emotion] = counts.get(r.emotion, 0) + 1
                        speaker_emo[(r.speaker, r.emotion)] = (
                            speaker_emo.get((r.speaker, r.emotion), 0) + 1
                        )
                        total += 1

    overall = Table(title=f"Emotion distribution ({total:,} sentences)")
    overall.add_column("Emotion", style="cyan")
    overall.add_column("Sentences", justify="right")
    overall.add_column("Share", justify="right")
    for emo in EMOTIONS:
        n = counts.get(emo, 0)
        if n == 0:
            continue
        overall.add_row(emo, f"{n:,}", f"{(100*n/total):.1f}%")
    console.print(overall)

    # Speaker x emotion breakdown.
    speakers = sorted({sp for sp, _ in speaker_emo.keys()})
    sp_table = Table(title="Speaker x emotion (top emotion per speaker)")
    sp_table.add_column("Speaker", style="cyan")
    sp_table.add_column("Top emotion")
    sp_table.add_column("Top count", justify="right")
    sp_table.add_column("Total", justify="right")
    rows: list[tuple[str, str, int, int]] = []
    for sp in speakers:
        sp_counts = {e: c for (s, e), c in speaker_emo.items() if s == sp}
        if not sp_counts:
            continue
        top_emo, top_n = max(sp_counts.items(), key=lambda kv: kv[1])
        rows.append((sp, top_emo, top_n, sum(sp_counts.values())))
    rows.sort(key=lambda r: -r[3])
    for sp, emo, n, t in rows[:25]:
        sp_table.add_row(sp, emo, f"{n:,}", f"{t:,}")
    console.print(sp_table)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


@emotions.command("preview")
@click.option("--character", required=True)
@click.option(
    "--emotions", "emos",
    default="neutral,angry,sad,whispered,excited",
    help="Comma-separated emotions to render.",
)
@click.option(
    "--text", default="The stones held through the night. We walked the perimeter at dawn.",
)
@click.option("--backend", type=click.Choice(["kokoro", "cloning", "chatterbox"]), default="kokoro")
@click.option("--library", "backend_library_root", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None)
def preview(
    character: str, emos: str, text: str,
    backend: str, backend_library_root: Path | None,
    out_dir: Path | None,
) -> None:
    """Render the same line in different emotions to verify your config."""
    import soundfile as sf
    from .synth import load_voice_cast, make_backend

    out_dir = out_dir or (_project_root() / "samples" / "emotion_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    voices = load_voice_cast(_project_root() / "config" / "voices.yaml")
    voice = voices.for_speaker(character)
    be = make_backend(backend, library_root=backend_library_root)

    emos_list = [e.strip() for e in emos.split(",") if e.strip()]
    for emo in emos_list:
        console.print(f"[dim]Rendering [{emo}]...[/dim]")
        audio = be.synthesize(text, voice, emotion=emo)
        path = out_dir / f"{character}_{emo}.wav"
        sf.write(str(path), audio, be.sample_rate)
        console.print(f"  [green]{path}[/green]  ({audio.size / be.sample_rate:.1f}s)")


# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------


@emotions.command("lexicon")
def show_lexicon() -> None:
    """Show the bundled lexicon's per-emotion coverage."""
    table = Table(title=f"Bundled lexicon ({lexicon_size()} words)")
    table.add_column("Emotion", style="cyan")
    table.add_column("Word count", justify="right")
    for emo, n in words_per_emotion().items():
        table.add_row(emo, f"{n}")
    console.print(table)
    console.print(
        "\n[dim]Words can carry signal for multiple emotions, "
        "so totals exceed the raw lexicon size.[/dim]"
    )

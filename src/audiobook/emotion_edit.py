"""Interactive emotion override tool.

Walks the analyzer's per-sentence labels and lets the user override any
they disagree with. Overrides are saved to a JSON file the renderer reads
at render time — the override replaces the analyzer's emotion for that
exact sentence.

Override key format:  `<chapter_number>:<scene_index>:<sentence_idx_in_scene>`

The sentence index is the order in which sentences are encountered while
walking the scene's paragraphs in source order. Both the editor and the
renderer use the same walk, so indices line up.
"""
from __future__ import annotations

import json
import sys
import termios
import tty
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .attribution import SceneState, Utterance, attribute_paragraph, DEFAULT_PRONOUNS
from .emotion import EMOTIONS
from .emotion_analyzer import (
    AnalysisContext,
    EmotionAnalyzer,
    SentenceEmotion,
    split_sentences,
)
from .parser import Book, parse_manuscript
from .synth import load_voice_cast


console = Console()


# Single-key shortcuts for emotion selection.
KEY_TO_EMOTION = {
    "1": "neutral",
    "2": "happy",
    "3": "sad",
    "4": "angry",
    "5": "fearful",
    "6": "surprised",
    "7": "disgusted",
    "8": "whispered",
    "9": "excited",
    "0": "calm",
}


@dataclass
class SentenceItem:
    """One walkable sentence with its analyzer result and context."""

    key: str                      # "<chapter>:<scene>:<idx>"
    chapter_number: int
    chapter_title: str
    scene_index: int
    sentence_idx_in_scene: int
    speaker: str
    text: str
    is_dialogue: bool
    analyzer_emotion: str
    analyzer_confidence: float
    surrounding_narration: str    # what the tag-detector saw
    # Filled in by the editor:
    override_emotion: str | None = None


@dataclass
class OverridesFile:
    version: int = 1
    overrides: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "OverridesFile":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            console.print(f"[yellow]Could not parse {path}; starting fresh.[/yellow]")
            return cls()
        return cls(
            version=int(data.get("version", 1)),
            overrides=dict(data.get("overrides") or {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "overrides": self.overrides,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set(self, key: str, *, emotion: str, text: str, speaker: str) -> None:
        self.overrides[key] = {
            "emotion": emotion,
            "text": text,
            "speaker": speaker,
        }

    def clear(self, key: str) -> bool:
        if key in self.overrides:
            del self.overrides[key]
            return True
        return False

    def get(self, key: str) -> str | None:
        entry = self.overrides.get(key)
        return entry["emotion"] if entry else None


# ---------------------------------------------------------------------------
# Walk the manuscript to build a flat list of SentenceItem
# ---------------------------------------------------------------------------


def walk_sentences(
    manuscript: Path,
    *,
    use_ml: bool = False,
    mode: str = "multi",
) -> list[SentenceItem]:
    """Produce one SentenceItem per sentence, using the same analyzer as render."""
    book = parse_manuscript(manuscript)
    voices = load_voice_cast(Path(__file__).resolve().parents[2] / "config" / "voices.yaml")
    cast = set(voices.cast.keys())
    analyzer = EmotionAnalyzer(use_ml=use_ml)

    items: list[SentenceItem] = []
    for chapter in book.chapters:
        for s_i, scene in enumerate(chapter.scenes):
            analyzer.reset_scene()
            attribution_state = SceneState.fresh()
            sent_idx = 0
            for paragraph in scene.paragraphs:
                if mode == "multi":
                    utts = attribute_paragraph(
                        paragraph.plain_text(), cast, attribution_state, DEFAULT_PRONOUNS
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
                        result = analyzer.analyze(sent, ctx)
                        key = f"{chapter.number}:{s_i}:{sent_idx}"
                        items.append(SentenceItem(
                            key=key,
                            chapter_number=chapter.number,
                            chapter_title=chapter.display_title,
                            scene_index=s_i,
                            sentence_idx_in_scene=sent_idx,
                            speaker=result.speaker,
                            text=sent,
                            is_dialogue=utt.is_dialogue,
                            analyzer_emotion=result.emotion,
                            analyzer_confidence=result.confidence,
                            surrounding_narration=surround_narration if utt.is_dialogue else "",
                        ))
                        sent_idx += 1
    return items


# ---------------------------------------------------------------------------
# Keypress reader (raw mode so we don't need the Enter key)
# ---------------------------------------------------------------------------


def _read_key() -> str:
    """Read one key press from stdin. Returns the literal char, or
    a multi-char string for known special keys."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


HELP_TEXT = """\
[bold]Keys[/bold]
  [cyan]Enter / space[/cyan]  accept current emotion, advance
  [cyan]j / →[/cyan]          next sentence (skip)
  [cyan]k / ←[/cyan]          previous sentence
  [cyan]r[/cyan]              clear override for this sentence
  [cyan]n[/cyan]              jump to next sentence whose analyzer emotion ≠ neutral
  [cyan]l[/cyan]              jump to next low-confidence sentence (conf < 0.6)
  [cyan]?[/cyan]              show help
  [cyan]w[/cyan]              save now (without quitting)
  [cyan]q[/cyan]              save and quit

[bold]Override the emotion:[/bold]
  [cyan]1[/cyan]=neutral  [cyan]2[/cyan]=happy  [cyan]3[/cyan]=sad      [cyan]4[/cyan]=angry  [cyan]5[/cyan]=fearful
  [cyan]6[/cyan]=surprised [cyan]7[/cyan]=disgusted [cyan]8[/cyan]=whispered [cyan]9[/cyan]=excited [cyan]0[/cyan]=calm
"""


def _render_screen(
    item: SentenceItem, override: str | None,
    position: int, total: int,
    prev_items: list[SentenceItem], next_items: list[SentenceItem],
) -> None:
    """Draw the current sentence + 3 lines of context above and below."""
    console.clear()

    # Header.
    header = Panel.fit(
        f"[bold]Emotion editor[/bold]   "
        f"Chapter {item.chapter_number}, scene {item.scene_index + 1}   "
        f"[dim]{position + 1} / {total}[/dim]",
        border_style="cyan",
    )
    console.print(header)
    console.print(f"[bold cyan]{item.chapter_title}[/bold cyan]")
    console.print()

    # Context before.
    for prev in prev_items:
        emo = prev.override_emotion or prev.analyzer_emotion
        marker = "[dim yellow]*[/dim yellow]" if prev.override_emotion else " "
        console.print(
            f"  [dim]{prev.speaker:<10s}[/dim] "
            f"[dim]{emo:<10s}[/dim] {marker} "
            f"[dim]{prev.text[:90]}[/dim]"
        )
    console.print()

    # The current sentence — highlighted.
    final_emotion = override or item.analyzer_emotion
    star = "[yellow]*[/yellow]" if override else " "
    conf_color = "green" if item.analyzer_confidence >= 0.6 else "yellow"
    console.print(
        f"[bold white on blue]> {item.speaker:<10s}  "
        f"{final_emotion:<10s} {star}  "
        f"[/bold white on blue]"
    )
    console.print()
    console.print(f"  [white]{item.text}[/white]")
    console.print()
    console.print(
        f"  [dim]analyzer: {item.analyzer_emotion} "
        f"([{conf_color}]conf {item.analyzer_confidence:.2f}[/{conf_color}])"
        + (f"  override -> [yellow]{override}[/yellow]" if override else "")
        + (f"  [dim](dialogue, tag context: \"{item.surrounding_narration[:60]}\")[/dim]"
           if item.is_dialogue and item.surrounding_narration else "")
        + "[/dim]"
    )
    console.print()

    # Context after.
    for nxt in next_items:
        emo = nxt.override_emotion or nxt.analyzer_emotion
        marker = "[dim yellow]*[/dim yellow]" if nxt.override_emotion else " "
        console.print(
            f"  [dim]{nxt.speaker:<10s}[/dim] "
            f"[dim]{emo:<10s}[/dim] {marker} "
            f"[dim]{nxt.text[:90]}[/dim]"
        )
    console.print()
    console.print(
        "[dim]Enter=accept  1-9,0=override emotion  j/k=next/prev  "
        "n=next non-neutral  l=next low-conf  r=clear  w=save  q=save&quit  ?=help[/dim]"
    )


def edit_interactive(
    manuscript: Path, overrides_path: Path,
    *, use_ml: bool = False, mode: str = "multi",
    start_filter: str | None = None,
) -> None:
    """Run the interactive editor over the manuscript."""
    if not sys.stdin.isatty():
        console.print(
            "[red]The editor needs an interactive terminal "
            "(stdin is not a TTY).[/red]"
        )
        raise SystemExit(1)

    console.print("[dim]Analyzing manuscript (this may take a moment)...[/dim]")
    items = walk_sentences(manuscript, use_ml=use_ml, mode=mode)
    if not items:
        console.print("[red]No sentences found.[/red]")
        return

    overrides = OverridesFile.load(overrides_path)
    # Seed override_emotion on items from existing overrides file.
    for item in items:
        existing = overrides.get(item.key)
        if existing:
            item.override_emotion = existing

    # Optional starting filter — skip to the first item matching it.
    position = 0
    if start_filter == "non-neutral":
        for i, it in enumerate(items):
            if it.analyzer_emotion != "neutral":
                position = i
                break
    elif start_filter == "low-confidence":
        for i, it in enumerate(items):
            if it.analyzer_confidence < 0.6:
                position = i
                break
    elif start_filter == "dialogue":
        for i, it in enumerate(items):
            if it.is_dialogue:
                position = i
                break

    show_help = True
    while True:
        item = items[position]
        prev_items = items[max(0, position - 3): position]
        next_items = items[position + 1: position + 4]
        _render_screen(item, item.override_emotion, position, len(items),
                       prev_items, next_items)
        if show_help:
            console.print(HELP_TEXT)
            show_help = False

        key = _read_key()

        # Quit.
        if key == "q":
            overrides.save(overrides_path)
            console.print(f"\n[green]Saved {len(overrides.overrides)} override(s) -> {overrides_path}[/green]")
            return
        # Save without quitting.
        if key == "w":
            overrides.save(overrides_path)
            console.print(f"\n[green]Saved (still editing). Continue with any key.[/green]")
            _read_key()
            continue
        # Help.
        if key == "?":
            show_help = True
            continue
        # Number keys = emotion override.
        if key in KEY_TO_EMOTION:
            emo = KEY_TO_EMOTION[key]
            item.override_emotion = emo
            overrides.set(item.key, emotion=emo, text=item.text, speaker=item.speaker)
            position = min(position + 1, len(items) - 1)
            continue
        # Clear override.
        if key == "r":
            item.override_emotion = None
            overrides.clear(item.key)
            continue
        # Accept (Enter / space).
        if key in ("\r", "\n", " "):
            position = min(position + 1, len(items) - 1)
            continue
        # Next sentence (j / right arrow).
        if key in ("j", "\x1b[C"):
            position = min(position + 1, len(items) - 1)
            continue
        # Previous sentence (k / left arrow).
        if key in ("k", "\x1b[D"):
            position = max(position - 1, 0)
            continue
        # Jump: next non-neutral.
        if key == "n":
            for i in range(position + 1, len(items)):
                if items[i].analyzer_emotion != "neutral":
                    position = i
                    break
            continue
        # Jump: next low-confidence.
        if key == "l":
            for i in range(position + 1, len(items)):
                if items[i].analyzer_confidence < 0.6:
                    position = i
                    break
            continue
        # Anything else: ignore.


# ---------------------------------------------------------------------------
# Renderer integration: lookup helper
# ---------------------------------------------------------------------------


def load_overrides(path: Path) -> dict[str, str]:
    """Return key -> emotion mapping for renderer consumption."""
    f = OverridesFile.load(path)
    return {k: v["emotion"] for k, v in f.overrides.items()}

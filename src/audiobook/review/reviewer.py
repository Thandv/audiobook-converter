"""Top-level orchestration: review a whole rendered book.

Reads chapter audio + source text, runs:
  - Whisper transcription + alignment (pronunciation, dropped, hallucinated)
  - DSP metrics (pace, volume, silence, clipping)
  - Audio emotion classifier (intended vs heard)

Emits a ReviewReport with Findings and per-chapter metric snapshots.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.console import Console

from ..attribution import SceneState, Utterance, attribute_paragraph, DEFAULT_PRONOUNS
from ..emotion_analyzer import (
    AnalysisContext,
    EmotionAnalyzer,
    split_sentences,
)
from ..parser import Book, parse_manuscript
from ..synth import load_voice_cast
from .align import align_chapter
from .metrics import (
    ChapterMetrics,
    compute_chapter_metrics,
    find_metric_outliers,
)
from .transcribe import Transcriber
from .types import ReviewReport


console = Console()


def _chapter_source_text(book: Book, chapter_idx: int) -> str:
    """Flatten all paragraphs from one chapter into a single source-text string."""
    ch = book.chapters[chapter_idx]
    parts: list[str] = []
    for sc in ch.scenes:
        for p in sc.paragraphs:
            parts.append(p.plain_text())
    return "\n".join(parts)


def _intended_emotions_per_chapter(
    book: Book, voices_yaml: Path, use_ml: bool = False,
) -> dict[int, dict[str, int]]:
    """Run the text analyzer over the book; return per-chapter emotion counts."""
    voices = load_voice_cast(voices_yaml)
    cast = set(voices.cast.keys())
    analyzer = EmotionAnalyzer(use_ml=use_ml)
    out: dict[int, dict[str, int]] = {}
    for chapter in book.chapters:
        per_chapter: Counter[str] = Counter()
        for scene in chapter.scenes:
            analyzer.reset_scene()
            state = SceneState.fresh()
            for paragraph in scene.paragraphs:
                utts = attribute_paragraph(
                    paragraph.plain_text(), cast, state, DEFAULT_PRONOUNS
                )
                surround = " ".join(u.text for u in utts if not u.is_dialogue)
                for utt in utts:
                    for sent in split_sentences(utt.text):
                        ctx = AnalysisContext(
                            speaker=utt.speaker,
                            surrounding_narration=surround if utt.is_dialogue else "",
                            is_dialogue=utt.is_dialogue,
                        )
                        r = analyzer.analyze(sent, ctx)
                        per_chapter[r.emotion] += 1
        out[chapter.number] = dict(per_chapter)
    return out


def _audio_path_for_chapter(chapters_dir: Path, idx: int, title: str) -> Path | None:
    """Find the audio file for this chapter (mirrors the renderer naming)."""
    from ..stitch import _safe_filename
    base = f"{idx + 1:02d}_{_safe_filename(title)}"
    for ext in (".mp3", ".wav"):
        p = chapters_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def review_book(
    manuscript_path: Path,
    chapters_dir: Path,
    *,
    voices_yaml: Path,
    do_transcribe: bool = True,
    do_metrics: bool = True,
    do_emotion_check: bool = True,
    use_ml_analyzer: bool = False,
    whisper_model: str = "base.en",
    round_number: int = 1,
) -> ReviewReport:
    """Run the full review pipeline. Returns a ReviewReport."""
    book = parse_manuscript(manuscript_path)
    report = ReviewReport(
        manuscript_path=str(manuscript_path),
        chapters_dir=str(chapters_dir),
        round_number=round_number,
    )

    # Map chapter number -> (idx, title, source text, audio path).
    chapter_inputs: list[tuple[int, int, str, str, Path]] = []
    for idx, ch in enumerate(book.chapters):
        audio = _audio_path_for_chapter(chapters_dir, idx, ch.display_title)
        if audio is None:
            console.print(f"[yellow]No audio file for chapter {ch.number}: {ch.display_title}[/yellow]")
            continue
        chapter_inputs.append((ch.number, idx, ch.display_title, _chapter_source_text(book, idx), audio))

    chapter_titles = {n: t for n, _, t, _, _ in chapter_inputs}

    # 1. Metrics pass — cheap, do first so we can also compute book-wide stats.
    metrics_by_chapter: dict[int, ChapterMetrics] = {}
    if do_metrics:
        console.print(f"[bold]Pass 1/3: DSP metrics ({len(chapter_inputs)} chapters)[/bold]")
        for n, _, title, src, audio in chapter_inputs:
            m = compute_chapter_metrics(audio, src)
            metrics_by_chapter[n] = m
            report.chapter_metrics[str(n)] = m.to_dict()
        report.findings.extend(find_metric_outliers(metrics_by_chapter, chapter_titles))

    # 2. Transcription + alignment pass.
    if do_transcribe:
        console.print(f"[bold]Pass 2/3: Whisper transcription + alignment[/bold]")
        transcriber = Transcriber(model_size=whisper_model)
        for i, (n, _, title, src, audio) in enumerate(chapter_inputs, 1):
            console.print(f"  [{i}/{len(chapter_inputs)}] {title}")
            transcript = transcriber.transcribe(audio)
            findings = align_chapter(
                source_text=src, transcript=transcript,
                chapter_number=n, chapter_title=title,
            )
            report.findings.extend(findings)

    # 3. Audio emotion classifier (per-chapter aggregate vs intended).
    if do_emotion_check:
        console.print(f"[bold]Pass 3/3: audio emotion verification[/bold]")
        from .emotion_check import AudioEmotionClassifier, emotion_findings_for_chapter
        intended_by_chapter = _intended_emotions_per_chapter(
            book, voices_yaml, use_ml=use_ml_analyzer
        )
        classifier = AudioEmotionClassifier()
        for n, _, title, _src, audio in chapter_inputs:
            intended = intended_by_chapter.get(n, {})
            findings = emotion_findings_for_chapter(
                audio_path=audio,
                chapter_number=n,
                chapter_title=title,
                intended_emotions=intended,
                classifier=classifier,
            )
            report.findings.extend(findings)

    return report

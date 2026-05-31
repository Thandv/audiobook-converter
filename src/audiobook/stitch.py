"""Audio assembly.

Renders a parsed (and optionally attributed) Book into per-chapter audio
files. Adds calibrated pauses so chapters / scenes / paragraphs breathe.
Picks the TTS backend (kokoro / xtts) per RenderConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .attribution import SceneState, Utterance, attribute_paragraph, DEFAULT_PRONOUNS
from .backends.base import Backend
from .emotion import detect_emotion
from .emotion_analyzer import (
    AnalysisContext,
    EmotionAnalyzer,
    SentenceEmotion,
    split_sentences,
)
from .parser import Book, Chapter, Paragraph, Scene
from .pronounce import PronunciationMap
from .synth import SAMPLE_RATE, VoiceCast, make_backend


console = Console()

# Pause durations in seconds.
PAUSE_INTRA_PARAGRAPH = 0.25
PAUSE_BETWEEN_PARAGRAPHS = 0.7
PAUSE_SCENE_BREAK = 1.8
PAUSE_CHAPTER_HEAD = 1.5
PAUSE_END_OF_CHAPTER = 2.0


@dataclass
class RenderConfig:
    mode: str                          # "single" or "multi"
    output_dir: Path
    pronouncer: PronunciationMap | None
    voices: VoiceCast
    backend_name: str = "kokoro"           # kokoro | xtts | cloning | chatterbox
    backend_model_dir: Path | None = None  # required for xtts
    backend_library_root: Path | None = None  # required for cloning / chatterbox
    # Emotion analyzer:
    #   "tag"       - dialogue-tag detection only (the original behavior)
    #   "content"   - tag + lexicon content analysis with consistency
    #   "content+ml" - all three layers (requires [ml] extras)
    emotion_analyzer: str = "tag"
    # Per-sentence emotion overrides keyed by "<chapter>:<scene>:<idx_in_scene>".
    # Only consulted when emotion_analyzer != "tag".
    emotion_overrides: dict[str, str] = field(default_factory=dict)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _emphasize_italic_text(text: str, italic_phrases: list[str]) -> str:
    """Light prosodic emphasis on italic spans.

    Kokoro doesn't take SSML; the best we can do is insert a soft em-dash
    that nudges its prosody to a slightly stressed delivery. Intentionally
    subtle to avoid sounding artificial.
    """
    out = text
    for phrase in italic_phrases:
        phrase = phrase.strip()
        if not phrase or len(phrase) < 2:
            continue
        if phrase in out:
            out = out.replace(phrase, f"— {phrase} —", 1)
    return out


def _paragraph_to_utterances(
    paragraph: Paragraph, mode: str, state: SceneState, cast: set[str]
) -> list[Utterance]:
    plain = paragraph.plain_text()
    italics = [s.text for s in paragraph.spans if s.italic and s.text.strip()]
    if italics:
        plain = _emphasize_italic_text(plain, italics)
    if mode == "single":
        return [Utterance("NARRATOR", plain, False)] if plain else []
    return attribute_paragraph(plain, cast, state, DEFAULT_PRONOUNS)


def _emotion_for(utt: Utterance, neighbors: list[Utterance]) -> str:
    """Pick an emotion for an utterance using the dialogue and any
    adjacent NARRATOR utterances (i.e. dialogue tags) in the same paragraph.
    """
    if not utt.is_dialogue:
        return "neutral"
    surround = " ".join(
        n.text for n in neighbors if not n.is_dialogue and n is not utt
    )
    return detect_emotion(utt.text, surround)


@dataclass
class EmotionSegment:
    """One render-able chunk: a contiguous run of text that shares
    speaker AND emotion. Used by the sentence-aware render path.
    """

    speaker: str
    text: str
    emotion: str
    is_dialogue: bool


def _paragraph_to_emotion_segments(
    paragraph: Paragraph,
    *,
    mode: str,
    attribution_state: SceneState,
    cast: set[str],
    analyzer: EmotionAnalyzer,
    chapter_number: int,
    scene_index: int,
    scene_sentence_counter: list[int],   # single-element box for mutation
    overrides: dict[str, str],
) -> list[EmotionSegment]:
    """Sentence-aware split: per-sentence emotion, merging consecutive
    sentences that share (speaker, emotion) into one segment to avoid
    audio whiplash. Consults overrides keyed by chapter:scene:sent_idx."""
    utts = _paragraph_to_utterances(paragraph, mode, attribution_state, cast)
    surround_narration = " ".join(
        u.text for u in utts if not u.is_dialogue
    )

    segments: list[EmotionSegment] = []
    for utt in utts:
        sentences = split_sentences(utt.text)
        if not sentences:
            sentences = [utt.text]
        for sent in sentences:
            ctx = AnalysisContext(
                speaker=utt.speaker,
                surrounding_narration=surround_narration if utt.is_dialogue else "",
                is_dialogue=utt.is_dialogue,
            )
            result = analyzer.analyze(sent, ctx)
            sent_idx = scene_sentence_counter[0]
            scene_sentence_counter[0] += 1
            key = f"{chapter_number}:{scene_index}:{sent_idx}"
            emo = overrides.get(key, result.emotion)
            # Merge with previous segment if (speaker, emotion) match —
            # this is the consistency-preserving step at the audio boundary.
            if (
                segments
                and segments[-1].speaker == utt.speaker
                and segments[-1].emotion == emo
                and segments[-1].is_dialogue == utt.is_dialogue
            ):
                segments[-1] = EmotionSegment(
                    speaker=utt.speaker,
                    text=segments[-1].text + " " + sent,
                    emotion=emo,
                    is_dialogue=utt.is_dialogue,
                )
            else:
                segments.append(EmotionSegment(
                    speaker=utt.speaker,
                    text=sent,
                    emotion=emo,
                    is_dialogue=utt.is_dialogue,
                ))
    return segments


def _render_segment(
    backend: Backend, seg: EmotionSegment, voices: VoiceCast,
    pronouncer: PronunciationMap | None,
) -> np.ndarray:
    text = seg.text
    if pronouncer is not None:
        text = pronouncer.apply(text)
    voice = voices.for_speaker(seg.speaker)
    return backend.synthesize(text, voice, emotion=seg.emotion)


def _render_utterance(
    backend: Backend, utt: Utterance, voices: VoiceCast,
    pronouncer: PronunciationMap | None, emotion: str,
) -> np.ndarray:
    text = utt.text
    if pronouncer is not None:
        text = pronouncer.apply(text)
    voice = voices.for_speaker(utt.speaker)
    return backend.synthesize(text, voice, emotion=emotion)


def _safe_filename(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_ " else "_" for c in s).strip()
    return out.replace(" ", "_") or "untitled"


def write_chapter_audio(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(path), audio, SAMPLE_RATE, format="MP3")
    except (RuntimeError, ValueError):
        # libsndfile without MP3 support — write WAV instead.
        wav_path = path.with_suffix(".wav")
        sf.write(str(wav_path), audio, SAMPLE_RATE)


@dataclass
class ChapterRenderResult:
    chapter: Chapter
    audio_path: Path
    duration_seconds: float


def render_book(book: Book, config: RenderConfig) -> list[ChapterRenderResult]:
    backend = make_backend(
        config.backend_name,
        model_dir=config.backend_model_dir,
        library_root=config.backend_library_root,
    )
    cast_names = set(config.voices.cast.keys())
    results: list[ChapterRenderResult] = []

    # Set up the content-aware emotion analyzer, or None for tag-only mode.
    use_content_analyzer = config.emotion_analyzer in ("content", "content+ml")
    analyzer: EmotionAnalyzer | None = None
    if use_content_analyzer:
        analyzer = EmotionAnalyzer(use_ml=(config.emotion_analyzer == "content+ml"))

    chapters_dir = config.output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    total_units = sum(
        sum(len(s.paragraphs) for s in ch.scenes) + 1
        for ch in book.chapters
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Rendering ({backend.name}, {config.mode})", total=total_units
        )

        for idx, chapter in enumerate(book.chapters):
            progress.update(task, description=f"[bold]{chapter.display_title}")
            chunks: list[np.ndarray] = []

            heading_text = chapter.title
            if chapter.subtitle:
                heading_text = f"{chapter.title}. {chapter.subtitle}."
            heading_utt = Utterance("NARRATOR", heading_text, False)
            chunks.append(_render_utterance(backend, heading_utt, config.voices, config.pronouncer, "calm"))
            if chapter.dateline:
                chunks.append(silence(0.6))
                dateline_utt = Utterance("NARRATOR", chapter.dateline, False)
                chunks.append(_render_utterance(backend, dateline_utt, config.voices, config.pronouncer, "calm"))
            chunks.append(silence(PAUSE_CHAPTER_HEAD))
            progress.advance(task)

            for s_i, scene in enumerate(chapter.scenes):
                state = SceneState.fresh()
                if analyzer is not None:
                    analyzer.reset_scene()
                # Mutable counter for sentence index within this scene.
                scene_sent_counter = [0]
                for p_i, paragraph in enumerate(scene.paragraphs):
                    if analyzer is not None:
                        # Sentence-aware path: per-sentence emotion with
                        # consistency filter, contiguous (speaker, emotion)
                        # runs merged into one segment.
                        segments = _paragraph_to_emotion_segments(
                            paragraph,
                            mode=config.mode,
                            attribution_state=state,
                            cast=cast_names,
                            analyzer=analyzer,
                            chapter_number=chapter.number,
                            scene_index=s_i,
                            scene_sentence_counter=scene_sent_counter,
                            overrides=config.emotion_overrides,
                        )
                        for seg_i, seg in enumerate(segments):
                            audio = _render_segment(backend, seg, config.voices, config.pronouncer)
                            if audio.size:
                                chunks.append(audio)
                            if seg_i < len(segments) - 1:
                                chunks.append(silence(PAUSE_INTRA_PARAGRAPH))
                    else:
                        # Paragraph-level tag-only path (original behavior).
                        utts = _paragraph_to_utterances(
                            paragraph, config.mode, state, cast_names
                        )
                        for u_i, u in enumerate(utts):
                            emotion = _emotion_for(u, utts)
                            audio = _render_utterance(backend, u, config.voices, config.pronouncer, emotion)
                            if audio.size:
                                chunks.append(audio)
                            if u_i < len(utts) - 1:
                                chunks.append(silence(PAUSE_INTRA_PARAGRAPH))
                    if p_i < len(scene.paragraphs) - 1:
                        chunks.append(silence(PAUSE_BETWEEN_PARAGRAPHS))
                    progress.advance(task)
                if s_i < len(chapter.scenes) - 1:
                    chunks.append(silence(PAUSE_SCENE_BREAK))

            chunks.append(silence(PAUSE_END_OF_CHAPTER))

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            duration = audio.size / SAMPLE_RATE

            filename = f"{idx + 1:02d}_{_safe_filename(chapter.display_title)}.mp3"
            audio_path = chapters_dir / filename
            write_chapter_audio(audio, audio_path)
            if not audio_path.exists():
                audio_path = audio_path.with_suffix(".wav")

            results.append(ChapterRenderResult(chapter, audio_path, duration))

    return results

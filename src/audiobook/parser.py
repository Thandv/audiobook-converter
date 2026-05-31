"""Markdown manuscript parser.

Produces a structured Book of Chapters -> Scenes -> Paragraphs -> Spans,
preserving italics and dialogue boundaries that downstream stages need.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Headings that begin a new top-level audiobook "chapter" entry.
# We treat PROLOGUE, CHAPTER N, INTERLUDE, CODA, and PART markers as chapters
# (each with their own chapter marker in the final M4B).
CHAPTER_HEAD_RE = re.compile(
    r"^# +(PROLOGUE|CHAPTER +\d+|INTERLUDE|CODA|PART +[A-Z]+|EPILOGUE)\b",
    re.IGNORECASE,
)
# Book title and "End of Book" / appendix-style top headings to skip.
SKIP_HEAD_RE = re.compile(
    r"^# +(THE +.+|A +NOTE +ON|END +OF +BOOK|FROM +.+JOURNAL)",
    re.IGNORECASE,
)
# Sub-heading lines that we treat as the subtitle / dateline of a chapter.
SUBHEAD_RE = re.compile(r"^#{2,4} +(.+)")
# Horizontal rule scene break.
SCENE_BREAK_RE = re.compile(r"^-{3,}\s*$")

# Markdown italic spans. Match `*...*` or `_..._` but not bold `**...**`.
# Non-greedy, no line breaks inside.
ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)|(?<!_)_(?!_)([^_\n]+?)_(?!_)")


@dataclass
class Span:
    """A run of text with a style tag (plain | italic)."""

    text: str
    italic: bool = False


@dataclass
class Paragraph:
    spans: list[Span] = field(default_factory=list)

    def plain_text(self) -> str:
        return "".join(s.text for s in self.spans).strip()


@dataclass
class Scene:
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Chapter:
    number: int            # 1-based ordinal within the book
    title: str             # e.g. "Chapter 1", "Prologue"
    subtitle: str = ""     # e.g. "First Watch"
    dateline: str = ""     # e.g. "Early spring 3307..."
    scenes: list[Scene] = field(default_factory=list)

    @property
    def display_title(self) -> str:
        bits = [self.title]
        if self.subtitle:
            bits.append(self.subtitle)
        return " — ".join(bits)


@dataclass
class Book:
    title: str
    author: str
    chapters: list[Chapter] = field(default_factory=list)


def _strip_md_inline(text: str) -> str:
    """Remove inline markdown that shouldn't be spoken (bold markers, links)."""
    # Bold ** ** -> just the inner text.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Inline code `...` -> drop backticks.
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Markdown links [text](url) -> text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _spans_from_line(line: str) -> list[Span]:
    """Split a line into italic / plain spans."""
    line = _strip_md_inline(line)
    spans: list[Span] = []
    last = 0
    for m in ITALIC_RE.finditer(line):
        if m.start() > last:
            spans.append(Span(line[last : m.start()], italic=False))
        inner = m.group(1) or m.group(2) or ""
        if inner:
            spans.append(Span(inner, italic=True))
        last = m.end()
    if last < len(line):
        spans.append(Span(line[last:], italic=False))
    return [s for s in spans if s.text]


def _normalize_paragraph_text(buf: list[str]) -> str:
    """Join buffered paragraph lines into a single string."""
    return " ".join(s.strip() for s in buf if s.strip())


def _classify_chapter_title(raw: str) -> str:
    """Normalize a chapter heading for the chapter marker."""
    raw = raw.strip()
    # `# CHAPTER 1` -> `Chapter 1`
    return re.sub(r"\s+", " ", raw.title())


def parse_manuscript(md_path: Path) -> Book:
    """Parse a manuscript markdown file into a Book.

    Recognizes:
      - `# CHAPTER N`, `# PROLOGUE`, `# INTERLUDE`, `# CODA`, `# PART X` as chapters
      - `### subtitle` and `*dateline*` immediately under the heading
      - `---` as scene breaks
      - Blank lines as paragraph separators
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Extract book title + author from the very top of the file.
    book_title = md_path.stem
    author = ""
    for ln in lines[:15]:
        ln = ln.strip()
        if ln.startswith("# ") and not CHAPTER_HEAD_RE.match(ln):
            book_title = _strip_md_inline(ln[2:]).strip()
            break
    for ln in lines[:15]:
        m = re.match(r"^#{2,4}\s+(.+)", ln)
        if m and "book" not in m.group(1).lower():
            author = _strip_md_inline(m.group(1)).strip()
            if author and not author.lower().startswith("book"):
                break

    chapters: list[Chapter] = []
    current: Chapter | None = None
    current_scene: Scene | None = None
    para_buf: list[str] = []
    expect_subtitle = False
    expect_dateline = False

    def flush_paragraph() -> None:
        nonlocal para_buf
        if not para_buf or current is None or current_scene is None:
            para_buf = []
            return
        text = _normalize_paragraph_text(para_buf)
        if text:
            current_scene.paragraphs.append(Paragraph(_spans_from_line(text)))
        para_buf = []

    def new_scene() -> None:
        nonlocal current_scene
        if current is None:
            return
        flush_paragraph()
        current_scene = Scene()
        current.scenes.append(current_scene)

    for raw in lines:
        line = raw.rstrip()

        # Chapter heading?
        m_chap = CHAPTER_HEAD_RE.match(line)
        if m_chap:
            flush_paragraph()
            current = Chapter(
                number=len(chapters) + 1,
                title=_classify_chapter_title(m_chap.group(1)),
            )
            chapters.append(current)
            current_scene = Scene()
            current.scenes.append(current_scene)
            expect_subtitle = True
            expect_dateline = False
            continue

        # Skip top-of-file or appendix headings we don't render.
        if line.startswith("# ") and SKIP_HEAD_RE.match(line):
            current = None
            current_scene = None
            continue

        # Inside a chapter: handle subtitle / dateline.
        if current is not None and expect_subtitle:
            m_sub = SUBHEAD_RE.match(line)
            if m_sub:
                current.subtitle = _strip_md_inline(m_sub.group(1)).strip()
                expect_subtitle = False
                expect_dateline = True
                continue
            if line.strip() == "":
                continue
            # No subtitle present — fall through to dateline check.
            expect_subtitle = False
            expect_dateline = True

        if current is not None and expect_dateline:
            stripped = line.strip()
            # Dateline is typically a single italic line like `*Early spring 3307...*`
            if stripped.startswith("*") and stripped.endswith("*") and stripped.count("*") == 2:
                current.dateline = stripped.strip("* ").strip()
                expect_dateline = False
                continue
            if stripped == "" or SCENE_BREAK_RE.match(stripped):
                continue
            # Some lead in directly with prose — fall through.
            expect_dateline = False

        if current is None:
            continue

        # Scene break.
        if SCENE_BREAK_RE.match(line):
            new_scene()
            continue

        if line.strip() == "":
            flush_paragraph()
            continue

        # Skip stray sub-headings that appear mid-chapter (e.g. block quotes).
        if line.startswith("#"):
            flush_paragraph()
            continue

        para_buf.append(line)

    flush_paragraph()

    # Drop empty scenes that come from back-to-back breaks.
    for ch in chapters:
        ch.scenes = [s for s in ch.scenes if s.paragraphs]
        # Drop chapters that ended up empty (e.g. parsing edge cases).
    chapters = [c for c in chapters if c.scenes]

    return Book(title=book_title, author=author, chapters=chapters)


def iter_paragraphs(book: Book) -> Iterable[tuple[Chapter, Scene, Paragraph]]:
    """Yield every paragraph with its chapter/scene context."""
    for ch in book.chapters:
        for sc in ch.scenes:
            for p in sc.paragraphs:
                yield ch, sc, p


def book_stats(book: Book) -> dict[str, int]:
    """Quick stats for sanity-checking a parse."""
    words = 0
    paragraphs = 0
    scenes = 0
    for ch in book.chapters:
        for sc in ch.scenes:
            scenes += 1
            for p in sc.paragraphs:
                paragraphs += 1
                words += len(p.plain_text().split())
    return {
        "chapters": len(book.chapters),
        "scenes": scenes,
        "paragraphs": paragraphs,
        "words": words,
    }

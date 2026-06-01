"""Finding + ReviewReport types — shared across reviewer / fixer / CLI."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FindingKind(str, Enum):
    # Pronunciation: Whisper transcribed a source word as something different.
    # Auto-fixable by adding to pronunciations.yaml.
    PRONUNCIATION = "pronunciation"
    # A source word is missing from the audio entirely.
    DROPPED_WORD = "dropped_word"
    # Audio contains words/phrase that don't exist in source — TTS hallucination.
    HALLUCINATED = "hallucinated"
    # The audio emotion classifier disagrees with our intended emotion for a
    # whole scene's worth of sentences. Suggests emotion override.
    EMOTION_MISMATCH = "emotion_mismatch"
    # Pace (WPM) for this chapter is significantly off from the book mean.
    PACE_OUTLIER = "pace_outlier"
    # RMS / peak volume off from the book mean.
    VOLUME_OUTLIER = "volume_outlier"
    # Long unexpected silence inside a chapter.
    LONG_SILENCE = "long_silence"
    # Clipping / overdriven audio.
    CLIPPING = "clipping"


class FindingSeverity(str, Enum):
    INFO = "info"           # Worth noting, no action needed.
    SUGGESTION = "suggestion"  # Should consider; user reviews.
    HIGH = "high"           # Real defect; auto-fix if possible.


@dataclass
class Finding:
    """One observation produced by the reviewer."""

    kind: FindingKind
    severity: FindingSeverity
    chapter_number: int                  # 1-based, matches Book.chapters[i].number
    chapter_title: str
    summary: str                         # Human-readable one-liner.

    # Optional details, populated by the specific reviewer that produced it.
    scene_index: int | None = None
    sentence_index_in_scene: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    source_text: str | None = None
    audio_text: str | None = None
    intended_emotion: str | None = None
    predicted_emotion: str | None = None
    confidence: float | None = None
    metric_value: float | None = None
    metric_baseline: float | None = None

    # Whether the fixer can apply this without user review.
    auto_fixable: bool = False
    # Concrete fix action — set by the reviewer to direct the fixer.
    # One of: "add_pronunciation", "add_emotion_override",
    # "adjust_voice_speed", "rerender_chapter", "flag_only"
    fix_action: str | None = None
    fix_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        data = dict(data)
        data["kind"] = FindingKind(data["kind"])
        data["severity"] = FindingSeverity(data["severity"])
        return cls(**data)


@dataclass
class ReviewReport:
    """Top-level review output for the whole book."""

    manuscript_path: str
    chapters_dir: str
    findings: list[Finding] = field(default_factory=list)
    chapter_metrics: dict[str, dict] = field(default_factory=dict)
    round_number: int = 1                # Which iterate round produced this.

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manuscript_path": self.manuscript_path,
            "chapters_dir": self.chapters_dir,
            "round_number": self.round_number,
            "chapter_metrics": self.chapter_metrics,
            "findings": [f.to_dict() for f in self.findings],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReviewReport":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            manuscript_path=data["manuscript_path"],
            chapters_dir=data["chapters_dir"],
            round_number=int(data.get("round_number", 1)),
            chapter_metrics=data.get("chapter_metrics") or {},
            findings=[Finding.from_dict(f) for f in data.get("findings") or []],
        )

    # Convenience filters / accessors.

    def by_kind(self, kind: FindingKind) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def by_chapter(self, n: int) -> list[Finding]:
        return [f for f in self.findings if f.chapter_number == n]

    def auto_fixable(self) -> list[Finding]:
        return [f for f in self.findings if f.auto_fixable]

    def chapters_needing_rerender(self) -> list[int]:
        """Chapter numbers where at least one finding requires audio re-render."""
        out: set[int] = set()
        for f in self.findings:
            if f.fix_action in {
                "rerender_chapter", "add_pronunciation",
                "add_emotion_override", "adjust_voice_speed",
            }:
                out.add(f.chapter_number)
        return sorted(out)

    def summary_table(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind.value] = counts.get(f.kind.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

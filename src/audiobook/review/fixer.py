"""Apply findings: mutate config + plan re-render.

Auto-applies safe fixes:
  - PRONUNCIATION: append to config/pronunciations.yaml
  - DROPPED_WORD with severity HIGH: schedule chapter for re-render

Proposes (but does NOT auto-apply) risky fixes:
  - EMOTION_MISMATCH: would need an emotion override JSON (require user OK)
  - PACE_OUTLIER: would change config/voices.yaml speed (require user OK)
  - VOLUME_OUTLIER: would suggest normalization step (require user OK)
  - HALLUCINATED / CLIPPING / LONG_SILENCE: reported only

The output FixPlan tells the orchestrator what was changed and which chapter
audio files should be deleted so the next render with --resume regenerates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from .types import Finding, FindingKind, ReviewReport


console = Console()


@dataclass
class FixPlan:
    pronunciations_added: dict[str, str] = field(default_factory=dict)
    overrides_added: dict[str, str] = field(default_factory=dict)
    speed_changes: dict[str, float] = field(default_factory=dict)  # speaker -> new speed
    chapters_to_rerender: set[int] = field(default_factory=set)
    proposed: list[Finding] = field(default_factory=list)  # not applied; for user review

    def summary_lines(self) -> list[str]:
        out: list[str] = []
        if self.pronunciations_added:
            out.append(f"Added {len(self.pronunciations_added)} pronunciation override(s).")
        if self.overrides_added:
            out.append(f"Added {len(self.overrides_added)} emotion override(s).")
        if self.speed_changes:
            out.append(f"Changed {len(self.speed_changes)} character speed(s).")
        if self.chapters_to_rerender:
            out.append(f"Scheduled {len(self.chapters_to_rerender)} chapter(s) for re-render.")
        if self.proposed:
            out.append(f"{len(self.proposed)} finding(s) need user review (use --apply-all).")
        if not out:
            out.append("No fixes applied.")
        return out


# ---------------------------------------------------------------------------
# Pronunciation YAML mutation
# ---------------------------------------------------------------------------


def _update_pronunciations_yaml(
    path: Path, new_entries: dict[str, str],
) -> dict[str, str]:
    """Append new pronunciation entries to the YAML file.

    Returns the merged dict that was actually written (so we know what's new
    vs. what was already present).
    """
    import yaml

    existing: dict[str, str] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing = {str(k): str(v) for k, v in loaded.items()}

    merged = dict(existing)
    truly_new: dict[str, str] = {}
    for k, v in new_entries.items():
        # Don't overwrite an existing manual mapping.
        if k in merged:
            continue
        # Don't replace a name with the same name (would be a no-op).
        if v.lower().strip() == k.lower().strip():
            continue
        merged[k] = v
        truly_new[k] = v

    if truly_new:
        # Append rather than rewrite, to preserve user comments.
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n# --- added by `audiobook review apply` ---\n")
            for k, v in truly_new.items():
                fh.write(f'"{k}": "{v}"\n')
    return truly_new


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_findings(
    report: ReviewReport,
    *,
    pronunciations_yaml: Path,
    chapters_dir: Path,
    apply_all: bool = False,
    dry_run: bool = False,
) -> FixPlan:
    """Apply auto-fixable findings; collect proposed ones for user review.

    If `apply_all` is True, ALSO apply the "propose" findings (still skipping
    the strictly informational ones like LONG_SILENCE).
    """
    plan = FixPlan()

    # 1. Auto-fixable: pronunciations.
    pron_proposed: dict[str, str] = {}
    for f in report.findings:
        if f.kind != FindingKind.PRONUNCIATION:
            continue
        if not f.auto_fixable and not apply_all:
            plan.proposed.append(f)
            continue
        original = f.fix_payload.get("original")
        phonetic = f.fix_payload.get("phonetic")
        if not original or not phonetic:
            continue
        # Skip if phonetic is essentially the same word (just casing/whitespace).
        if phonetic.lower().strip() == original.lower().strip():
            continue
        pron_proposed[original] = phonetic
        plan.chapters_to_rerender.add(f.chapter_number)

    if pron_proposed and not dry_run:
        added = _update_pronunciations_yaml(pronunciations_yaml, pron_proposed)
        plan.pronunciations_added = added
    elif pron_proposed and dry_run:
        plan.pronunciations_added = pron_proposed

    # 2. Dropped words: schedule re-render if HIGH severity. No config change.
    for f in report.findings:
        if f.kind == FindingKind.DROPPED_WORD and f.fix_action == "rerender_chapter":
            plan.chapters_to_rerender.add(f.chapter_number)

    # 3. Findings the user must review (unless apply_all is set).
    for f in report.findings:
        if f.kind in {
            FindingKind.EMOTION_MISMATCH,
            FindingKind.PACE_OUTLIER,
        }:
            plan.proposed.append(f)
            # When apply_all, schedule chapter re-render so the next pass
            # picks up any config tweaks we made.
            if apply_all:
                plan.chapters_to_rerender.add(f.chapter_number)

    # Delete the affected chapter audio files so --resume re-renders them.
    if not dry_run:
        from ..stitch import _safe_filename
        from ..parser import parse_manuscript
        book = parse_manuscript(Path(report.manuscript_path))
        for n in plan.chapters_to_rerender:
            idx = n - 1
            if idx < 0 or idx >= len(book.chapters):
                continue
            title = book.chapters[idx].display_title
            base = f"{idx + 1:02d}_{_safe_filename(title)}"
            for ext in (".mp3", ".wav"):
                p = chapters_dir / f"{base}{ext}"
                if p.exists():
                    p.unlink()
                    console.print(f"  [dim]deleted {p.name}[/dim]")

    return plan

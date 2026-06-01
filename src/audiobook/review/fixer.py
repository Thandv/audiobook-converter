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
    pronunciations_skipped: dict[str, str] = field(default_factory=dict)  # name -> reason
    overrides_added: dict[str, str] = field(default_factory=dict)
    speed_changes: dict[str, float] = field(default_factory=dict)  # speaker -> new speed
    chapters_to_rerender: set[int] = field(default_factory=set)
    proposed: list[Finding] = field(default_factory=list)  # not applied; for user review

    def summary_lines(self) -> list[str]:
        out: list[str] = []
        if self.pronunciations_added:
            out.append(f"Added {len(self.pronunciations_added)} pronunciation override(s).")
        if self.pronunciations_skipped:
            out.append(
                f"Skipped {len(self.pronunciations_skipped)} unsafe pronunciation finding(s) "
                "(truncation / already-mapped / no-op)."
            )
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


def _load_existing_pronunciations(path: Path) -> dict[str, str]:
    """Load the current pronunciations.yaml as a case-insensitive dict."""
    import yaml

    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return {}
    # Case-insensitive key lookup.
    return {str(k).lower(): str(v) for k, v in loaded.items()}


def _is_safe_pronunciation_fix(
    original: str, phonetic: str, existing: dict[str, str],
) -> tuple[bool, str]:
    """Decide whether (original -> phonetic) should be auto-applied.

    Returns (safe, reason). 'safe' is True if it's a real, useful fix.

    Filters out:
      1. Already-mapped originals (don't override the user's manual mapping)
      2. Same string (no-op)
      3. Truncations (Whisper cut off the name mid-word)
      4. Prefix/suffix matches (just chopped)
      5. Already-substituted forms (the existing mapping ALREADY produces
         a similar phonetic to what Whisper heard)
      6. Very high character overlap (probably the same pronunciation,
         different spelling — Whisper artifact)
    """
    o, p = original.strip(), phonetic.strip()
    if not o or not p:
        return False, "empty"
    if p.lower() == o.lower():
        return False, "no-op (same string)"
    if o.lower() in existing:
        return False, f"already mapped to '{existing[o.lower()]}'"

    # Truncation: a very short phonetic versus a longer original.
    if len(p) <= 3 and len(o) >= 5:
        return False, f"likely truncation ({len(p)}-char Whisper transcript)"

    # Prefix/suffix: phonetic is just a chunk of the original.
    o_lower = o.lower()
    p_lower = p.lower()
    if o_lower.startswith(p_lower) and len(p_lower) < len(o_lower) * 0.6:
        return False, "prefix-only truncation"
    if o_lower.endswith(p_lower) and len(p_lower) < len(o_lower) * 0.6:
        return False, "suffix-only truncation"

    # Single letter or extremely short.
    if len(p) <= 2:
        return False, "phonetic too short"

    # Possessive form (X's -> y's): if the base form is already mapped,
    # the possessive will inherit the fix at render time. Skip.
    if o.endswith("'s") and o[:-2].lower() in existing:
        return False, f"possessive of already-mapped '{o[:-2]}'"

    # Character-set overlap heuristic: if 80%+ of chars are shared,
    # they probably sound the same — Whisper just picked a different spelling.
    o_set = set(o_lower.replace("'", "").replace("-", ""))
    p_set = set(p_lower.replace("'", "").replace("-", ""))
    if o_set and p_set:
        overlap = len(o_set & p_set) / max(len(o_set), len(p_set))
        if overlap >= 0.85:
            return False, f"high char overlap ({overlap:.0%}) — likely same sound"

    return True, "looks like a real mispronunciation"


def _update_pronunciations_yaml(
    path: Path, new_entries: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Append new pronunciation entries to the YAML file.

    Returns (added, skipped_with_reason). The fixer uses both — `added`
    drives the re-render set; `skipped_with_reason` is shown to the user.
    """
    existing = _load_existing_pronunciations(path)

    truly_new: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for k, v in new_entries.items():
        safe, reason = _is_safe_pronunciation_fix(k, v, existing)
        if not safe:
            skipped[k] = reason
            continue
        truly_new[k] = v
        existing[k.lower()] = v  # avoid in-session duplicates

    if truly_new:
        # Append rather than rewrite, to preserve user comments.
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n# --- added by `audiobook review apply` ---\n")
            for k, v in truly_new.items():
                fh.write(f'"{k}": "{v}"\n')
    return truly_new, skipped


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

    # 1. Auto-fixable: pronunciations. Collect first, decide which are safe.
    pron_proposed: dict[str, str] = {}
    # Track which chapter(s) each candidate appeared in, so we can decide
    # rerender per chapter based on whether the fix was actually applied.
    pron_chapters: dict[str, set[int]] = {}
    # Sum total occurrences across the book per (original) — a one-off
    # mistake is probably Whisper noise; a repeated mistake is a real Kokoro
    # pattern. We require >= 3 total occurrences to auto-apply.
    pron_occ: dict[str, int] = {}
    for f in report.findings:
        if f.kind != FindingKind.PRONUNCIATION:
            continue
        if not f.auto_fixable and not apply_all:
            plan.proposed.append(f)
            continue
        original = f.fix_payload.get("original")
        phonetic = f.fix_payload.get("phonetic")
        occ = int(f.fix_payload.get("occurrences", 0))
        if not original or not phonetic:
            continue
        pron_proposed[original] = phonetic
        pron_chapters.setdefault(original, set()).add(f.chapter_number)
        pron_occ[original] = pron_occ.get(original, 0) + occ

    # Filter: require >= 3 total Kokoro occurrences across the book.
    # One-off mistakes are typically Whisper noise, not real Kokoro patterns.
    rare = {k for k, n in pron_occ.items() if n < 3}
    for k in rare:
        plan.pronunciations_skipped[k] = f"only {pron_occ[k]} occurrences (need >= 3)"
    pron_proposed = {k: v for k, v in pron_proposed.items() if k not in rare}

    existing_prons = _load_existing_pronunciations(pronunciations_yaml)
    if dry_run:
        # In dry-run we still want to filter, but never touch the file.
        added: dict[str, str] = {}
        skipped: dict[str, str] = {}
        for k, v in pron_proposed.items():
            safe, reason = _is_safe_pronunciation_fix(k, v, existing_prons)
            if safe:
                added[k] = v
            else:
                skipped[k] = reason
        plan.pronunciations_added = added
        plan.pronunciations_skipped = skipped
    elif pron_proposed:
        added, skipped = _update_pronunciations_yaml(pronunciations_yaml, pron_proposed)
        plan.pronunciations_added = added
        plan.pronunciations_skipped = skipped

    # Only schedule re-render for chapters whose pronunciation finding was
    # actually applied (not skipped).
    for name in plan.pronunciations_added:
        for ch_num in pron_chapters.get(name, set()):
            plan.chapters_to_rerender.add(ch_num)

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

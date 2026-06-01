"""Align Whisper transcript to source manuscript text.

Uses difflib's sequence matcher to find runs of equal/replace/delete/insert
tokens. Each operation maps to a Finding kind:
    - equal:   no finding
    - replace: PRONUNCIATION (source word X transcribed as Y)
    - delete:  DROPPED_WORD (source word X missing from audio)
    - insert:  HALLUCINATED (audio word Y not in source)

Heuristics suppress noise:
    - whisper's punctuation/capitalization differences are ignored
    - very short tokens ("a", "the") are de-emphasized
    - we only flag REPLACE if the source word is a "name-shaped" proper noun
      (capitalized in source, not a sentence start), because flagging every
      tiny mistranscription would drown the report
    - DROPPED_WORD only fires for runs of >= 3 missing words (small gaps are
      often Whisper missing function words; not actually missing in audio)
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass

from .transcribe import ChapterTranscript
from .types import Finding, FindingKind, FindingSeverity


# Things that don't count as words.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _is_name_shaped(token: str, prev_token: str | None) -> bool:
    """Heuristic: a likely proper noun (worth flagging on mismatch).

    Capitalized, length >= 3, and either it's not preceded by sentence-end
    punctuation OR previous token is not a normal sentence-ender.
    """
    if not token or not token[0].isupper() or len(token) < 3:
        return False
    if token.upper() == token:
        return True  # ALL CAPS — probably emphasis or a name like 'NOW'
    # Sentence-start capitalization is fine, ignore.
    if prev_token is None:
        return False
    if prev_token.endswith((".", "!", "?")):
        return False
    return True


def _normalize_for_match(tok: str) -> str:
    """Casefold + strip apostrophes so 'Gael' == 'gael' and "didn't" == 'didnt'."""
    return tok.lower().replace("'", "").replace("-", "")


@dataclass
class _AlignOp:
    """Internal: one operation from the sequence matcher."""
    op: str            # equal | replace | delete | insert
    src_range: tuple[int, int]
    trn_range: tuple[int, int]


def _align(src_tokens: list[str], trn_tokens: list[str]) -> list[_AlignOp]:
    """Return diff operations between source and transcript tokens."""
    src_norm = [_normalize_for_match(t) for t in src_tokens]
    trn_norm = [_normalize_for_match(t) for t in trn_tokens]
    matcher = difflib.SequenceMatcher(a=src_norm, b=trn_norm, autojunk=False)
    return [
        _AlignOp(op=op, src_range=(i1, i2), trn_range=(j1, j2))
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
    ]


def align_chapter(
    source_text: str,
    transcript: ChapterTranscript,
    *,
    chapter_number: int,
    chapter_title: str,
) -> list[Finding]:
    """Produce findings from a single chapter alignment."""
    src_tokens = _tokenize(source_text)
    trn_tokens = _tokenize(transcript.full_text)
    if not src_tokens or not trn_tokens:
        return []

    ops = _align(src_tokens, trn_tokens)
    findings: list[Finding] = []

    # Track pronunciation mismatches by (source_word -> [(transcribed_word, count)]).
    # If a name is mis-transcribed the same way multiple times in the chapter,
    # we know the TTS is reliably saying the wrong thing — strong signal for
    # an auto-fixable pronunciation entry.
    name_misses: dict[str, Counter[str]] = {}

    for op in ops:
        if op.op == "equal":
            continue

        if op.op == "replace":
            for k in range(op.src_range[1] - op.src_range[0]):
                if op.trn_range[0] + k >= op.trn_range[1]:
                    break
                src_word = src_tokens[op.src_range[0] + k]
                trn_word = trn_tokens[op.trn_range[0] + k]
                prev = src_tokens[op.src_range[0] + k - 1] if (op.src_range[0] + k) > 0 else None
                if _is_name_shaped(src_word, prev):
                    name_misses.setdefault(src_word, Counter())[trn_word.lower()] += 1

        elif op.op == "delete":
            n = op.src_range[1] - op.src_range[0]
            if n >= 3:
                start = op.src_range[0]
                end = op.src_range[1]
                snippet = " ".join(src_tokens[start:end])
                findings.append(Finding(
                    kind=FindingKind.DROPPED_WORD,
                    severity=FindingSeverity.HIGH if n >= 6 else FindingSeverity.SUGGESTION,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    summary=(
                        f"{n} source words missing from audio: \"{snippet[:80]}\""
                        + ("…" if len(snippet) > 80 else "")
                    ),
                    source_text=snippet,
                    fix_action="rerender_chapter" if n >= 6 else "flag_only",
                ))

        elif op.op == "insert":
            n = op.trn_range[1] - op.trn_range[0]
            if n >= 5:
                trn_snippet = " ".join(trn_tokens[op.trn_range[0]:op.trn_range[1]])
                findings.append(Finding(
                    kind=FindingKind.HALLUCINATED,
                    severity=FindingSeverity.SUGGESTION,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    summary=(
                        f"{n} audio words have no source match: \"{trn_snippet[:80]}\""
                        + ("…" if len(trn_snippet) > 80 else "")
                    ),
                    audio_text=trn_snippet,
                    fix_action="flag_only",
                ))

    # Convert reliable name mis-pronunciations into findings.
    for name, replacements in name_misses.items():
        most_common, count = replacements.most_common(1)[0]
        if count < 2:
            continue
        findings.append(Finding(
            kind=FindingKind.PRONUNCIATION,
            severity=FindingSeverity.HIGH,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            summary=f"'{name}' transcribed as '{most_common}' x{count}",
            source_text=name,
            audio_text=most_common,
            confidence=min(0.95, 0.6 + 0.1 * count),
            auto_fixable=True,
            fix_action="add_pronunciation",
            fix_payload={
                "original": name,
                "phonetic": most_common,
                "occurrences": count,
            },
        ))

    return findings

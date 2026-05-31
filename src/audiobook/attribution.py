"""Dialogue speaker attribution for multi-voice mode.

The job: given a parsed Book, produce, for every paragraph, a list of
(speaker, text, is_dialogue) Utterances. Speaker is either the literal
string "NARRATOR" or a known character name.

This is rule-based and pragmatic — fiction is messy. We:
  1. Recognize explicit dialogue tags ("...,"   X said. / X said, "...").
  2. Carry forward the last-named speaker for unattributed lines that
     start a back-and-forth.
  3. Track the two most recent speakers in a scene and alternate between
     them for further unattributed dialogue lines.
  4. Fall back to NARRATOR for anything still ambiguous.

Output is JSON-serializable so a user can hand-edit before synthesis.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .parser import Book, Chapter, Paragraph, Scene


# Normalize curly quotes to straight ones for matching.
QUOTE_PAIRS = [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]

DIALOGUE_VERBS = (
    "said|asked|replied|answered|whispered|muttered|shouted|called|cried|"
    "yelled|breathed|murmured|snapped|told|sighed|grunted|hissed|added|"
    "continued|began|barked|laughed|repeated|nodded|noted|admitted|"
    "agreed|countered|insisted|interrupted|offered|warned|growled|"
    "objected|prompted|drawled|conceded|observed|remarked|wondered"
)
DV_RE = rf"(?:{DIALOGUE_VERBS})"


@dataclass
class Utterance:
    speaker: str             # "NARRATOR" or a character first name
    text: str
    is_dialogue: bool        # True if this came from inside quote marks

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_quotes(text: str) -> str:
    for a, b in QUOTE_PAIRS:
        text = text.replace(a, b)
    return text


def _split_quoted_segments(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, is_quoted) chunks.

    Naïve double-quote tokenizer: alternates inside/outside quotes.
    Apostrophes in contractions are unaffected (we only split on ").
    """
    parts: list[tuple[str, bool]] = []
    buf: list[str] = []
    inside = False
    for ch in text:
        if ch == '"':
            if buf:
                parts.append(("".join(buf), inside))
                buf = []
            inside = not inside
        else:
            buf.append(ch)
    if buf:
        parts.append(("".join(buf), inside))
    return parts


def _candidate_speakers(cast: set[str]) -> str:
    """Regex alternation of character first names, longest first."""
    names = sorted(cast, key=len, reverse=True)
    return "|".join(re.escape(n) for n in names) or "_NEVER_"


def _find_tag_speaker(
    fragment: str, cast: set[str], pronoun_to_speaker: dict[str, str]
) -> str | None:
    """Find a speaker named in a dialogue-tag fragment.

    A "fragment" here is the bit of text adjacent to a quoted utterance.
    e.g. for `"X," she said.` the fragment is `, she said.`
    """
    if not fragment.strip():
        return None
    names = _candidate_speakers(cast)

    # 1. Named speaker with verb: "X said" / "said X" / "X asked".
    m = re.search(rf"\b({names})\b\s+{DV_RE}\b", fragment)
    if m:
        return m.group(1)
    m = re.search(rf"\b{DV_RE}\b\s+({names})\b", fragment)
    if m:
        return m.group(1)

    # 2. Pronoun tag: "he said" / "she asked" -> map via context.
    m = re.search(rf"\b(he|she|they)\b\s+{DV_RE}\b", fragment, re.IGNORECASE)
    if m:
        pronoun = m.group(1).lower()
        return pronoun_to_speaker.get(pronoun)
    m = re.search(rf"\b{DV_RE}\b\s+(he|she|they)\b", fragment, re.IGNORECASE)
    if m:
        pronoun = m.group(1).lower()
        return pronoun_to_speaker.get(pronoun)

    # 3. Bare named subject ("X frowned. ...") — only if it's at the
    #    very start of the fragment (right after the quote).
    m = re.match(rf"\s*(?:,?\s*)?\b({names})\b", fragment)
    if m:
        return m.group(1)

    return None


def _gender_pronoun(name: str, pronouns: dict[str, str]) -> str | None:
    """Return the pronoun (he/she/they) for a character, if known."""
    return pronouns.get(name)


@dataclass
class SceneState:
    """Per-scene attribution state — reset between scenes."""

    last_speaker_history: list[str]
    pronoun_to_speaker: dict[str, str]
    # Distinct speakers named via explicit tags, in order of first appearance.
    named_speakers: list[str]
    # Speaker of the previous paragraph's last dialogue utterance.
    # None if the previous paragraph was pure narration or didn't exist.
    last_paragraph_dialogue_speaker: str | None = None

    @classmethod
    def fresh(cls) -> "SceneState":
        return cls(
            last_speaker_history=[],
            pronoun_to_speaker={},
            named_speakers=[],
            last_paragraph_dialogue_speaker=None,
        )


def _alternate_speaker(state: SceneState) -> str | None:
    """Given the scene state, infer the alternation partner for an
    unattributed dialogue paragraph that follows another dialogue paragraph.

    Returns:
      - The other known speaker if we have ≥2 named speakers in the scene
      - "_OTHER" placeholder if we only know one speaker so far
      - None if we don't even know one speaker
    """
    last = state.last_paragraph_dialogue_speaker
    distinct = [s for s in state.named_speakers if s != "_OTHER"]
    if len(distinct) >= 2:
        most_recent = last if last in distinct else distinct[-1]
        for prior in reversed(distinct):
            if prior != most_recent:
                return prior
        return most_recent
    if len(distinct) == 1:
        # We know one named speaker; the previous paragraph spoke as them,
        # so this one is the other side of the conversation.
        if last == distinct[0]:
            return "_OTHER"
        return distinct[0]
    return None


def attribute_paragraph(
    text: str,
    cast: set[str],
    state: SceneState,
    pronouns: dict[str, str],
) -> list[Utterance]:
    """Return utterances for one paragraph. Mutates `state`."""
    text = _normalize_quotes(text)
    segments = _split_quoted_segments(text)

    # If there are zero quoted segments, the whole paragraph is narration.
    # Crucially we do NOT clear last_paragraph_dialogue_speaker — a short
    # narrative beat between two dialogue lines is normal in fiction and
    # the alternation context should survive it.
    if not any(q for _, q in segments):
        return [Utterance("NARRATOR", text.strip(), False)] if text.strip() else []

    utterances: list[Utterance] = []
    last_dialogue_speaker_this_para: str | None = None

    for i, (seg, quoted) in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue
        if not quoted:
            utterances.append(Utterance("NARRATOR", seg, False))
            continue

        # Quoted: find a tag in the adjacent non-quoted segment.
        speaker: str | None = None
        explicit = False
        if i + 1 < len(segments) and not segments[i + 1][1]:
            speaker = _find_tag_speaker(
                segments[i + 1][0], cast, state.pronoun_to_speaker
            )
            if speaker is not None:
                explicit = True
        if speaker is None and i > 0 and not segments[i - 1][1]:
            speaker = _find_tag_speaker(
                segments[i - 1][0], cast, state.pronoun_to_speaker
            )
            if speaker is not None:
                explicit = True

        if speaker is None:
            # If this paragraph is purely dialogue (no narration segments),
            # alternation kicks in — the previous paragraph established who
            # was just speaking.
            is_pure_dialogue = all(q for _, q in segments if _.strip())
            if is_pure_dialogue and state.last_paragraph_dialogue_speaker is not None:
                speaker = _alternate_speaker(state)
            if speaker is None:
                # Fall back: most recent named speaker, then narrator.
                if state.last_speaker_history:
                    speaker = state.last_speaker_history[-1]
                else:
                    speaker = "NARRATOR"

        utterances.append(Utterance(speaker, seg, True))
        last_dialogue_speaker_this_para = speaker

        if speaker != "NARRATOR":
            state.last_speaker_history.append(speaker)
            if explicit and speaker not in state.named_speakers:
                state.named_speakers.append(speaker)
            # Update pronoun map.
            p = _gender_pronoun(speaker, pronouns)
            if p:
                state.pronoun_to_speaker[p] = speaker
                state.pronoun_to_speaker.setdefault("they", speaker)

    state.last_paragraph_dialogue_speaker = last_dialogue_speaker_this_para
    return utterances


# Default gender map for the cast in this manuscript. Used to resolve
# "he said" / "she said" by looking up the most recent named speaker
# of the right gender. Names not in this map default to "they".
DEFAULT_PRONOUNS = {
    # men
    "Gael": "he", "Aldren": "he", "Kerrin": "he", "Corvin": "he",
    "Edric": "he", "Hamund": "he", "Orath": "he", "Deran": "he",
    "Tarrin": "he", "Brask": "he", "Hessel": "he", "Rellen": "he",
    "Aven": "he", "Garrick": "he", "Drevan": "he", "Kessler": "he",
    "Reske": "he", "Breck": "he", "Orren": "he",
    # women
    "Sera": "she", "Mira": "she", "Sevet": "she", "Brenneth": "she",
    "Dorsa": "she", "Laren": "she", "Mara": "she", "Vorell": "she",
    "Elleth": "she",
}


def attribute_book(book: Book, cast: Iterable[str]) -> list[list[list[Utterance]]]:
    """Run attribution over the whole book.

    Returns a nested list: chapters[].scenes[].paragraphs[].utterances[].
    Each scene resets attribution state (alternation, pronoun map).
    """
    cast_set = set(cast)
    pronouns = dict(DEFAULT_PRONOUNS)

    result: list[list[list[Utterance]]] = []
    for chapter in book.chapters:
        ch_paras: list[list[Utterance]] = []
        for scene in chapter.scenes:
            state = SceneState.fresh()
            scene_paras: list[list[Utterance]] = []
            for p in scene.paragraphs:
                utts = attribute_paragraph(
                    p.plain_text(), cast_set, state, pronouns
                )
                scene_paras.append(utts)
            ch_paras.extend(scene_paras)
        result.append(ch_paras)
    return result


def attribution_stats(attribution: list[list[list[Utterance]]]) -> dict[str, int]:
    """Speakers x utterance counts, for sanity-checking attribution quality."""
    counts: dict[str, int] = {}
    for chapter in attribution:
        for paragraph in chapter:
            for u in paragraph:
                if u.is_dialogue:
                    counts[u.speaker] = counts.get(u.speaker, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

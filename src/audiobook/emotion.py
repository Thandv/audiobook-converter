"""Emotion detection from dialogue tags.

Fiction writers convey emotion through dialogue tags ("he whispered",
"she shouted", "growled") and punctuation/casing patterns (ALL CAPS for
shouting, italics for emphasis, trailing ! for excitement). This module
extracts an emotion label per dialogue utterance which the synthesis
layer maps to per-emotion voice overrides.

Emotion labels — kept aligned with the training subsystem's labels:
  neutral, happy, sad, angry, fearful, surprised,
  disgusted, whispered, excited, calm
"""
from __future__ import annotations

import re


EMOTIONS = (
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "whispered", "excited", "calm",
)


# Verb -> emotion. Order matters — more specific verbs win. The lookup
# is case-insensitive, matched as whole words.
TAG_VERB_TO_EMOTION: dict[str, str] = {
    # whispered
    "whispered": "whispered", "murmured": "whispered", "breathed": "whispered",
    "hissed": "whispered", "mouthed": "whispered",
    # angry
    "shouted": "angry", "yelled": "angry", "barked": "angry", "snarled": "angry",
    "growled": "angry", "snapped": "angry", "spat": "angry", "roared": "angry",
    "thundered": "angry",
    # sad
    "sighed": "sad", "sobbed": "sad", "wept": "sad", "moaned": "sad",
    "lamented": "sad", "mourned": "sad",
    # happy
    "laughed": "happy", "chuckled": "happy", "giggled": "happy", "beamed": "happy",
    "grinned": "happy",
    # surprised
    "gasped": "surprised", "exclaimed": "surprised", "stammered": "surprised",
    "blurted": "surprised",
    # fearful
    "trembled": "fearful", "quavered": "fearful", "whimpered": "fearful",
    # excited
    "cried": "excited", "called": "excited", "blurted": "excited",
    # calm
    "said quietly": "calm", "noted": "calm", "observed": "calm", "remarked": "calm",
    "mused": "calm", "drawled": "calm",
}

# Build a single alternation pattern, longest first to avoid prefix collisions.
_VERBS = sorted(TAG_VERB_TO_EMOTION.keys(), key=len, reverse=True)
_VERB_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _VERBS) + r")\b",
    re.IGNORECASE,
)


def detect_emotion(
    dialogue_text: str,
    surrounding_narration: str = "",
) -> str:
    """Return an emotion label for a dialogue utterance.

    Looks at:
      1. Adjacent narration / dialogue tag verbs (highest priority)
      2. ALL-CAPS in the dialogue text -> angry
      3. Trailing exclamation marks -> excited
      4. Otherwise neutral
    """
    surround = (surrounding_narration or "").lower()
    m = _VERB_RE.search(surround)
    if m:
        verb = m.group(1).lower()
        if verb in TAG_VERB_TO_EMOTION:
            return TAG_VERB_TO_EMOTION[verb]
        # Fall back to longer multi-word match.
        for phrase, emo in TAG_VERB_TO_EMOTION.items():
            if " " in phrase and phrase in surround:
                return emo

    text = (dialogue_text or "").strip()
    if not text:
        return "neutral"

    # Heuristic 2: ALL CAPS dialogue (with letters >= 4) => shouting.
    letters_only = re.sub(r"[^A-Za-z]", "", text)
    if len(letters_only) >= 4 and letters_only.isupper():
        return "angry"

    # Heuristic 3: Two or more "!" => excited.
    if text.count("!") >= 2 or text.endswith("!!"):
        return "excited"

    # Heuristic 4: Single "!" => excited (mild).
    if text.endswith("!"):
        return "excited"

    return "neutral"


def emotion_summary(emotions: list[str]) -> dict[str, int]:
    """Count emotions seen across an utterance stream."""
    out: dict[str, int] = {}
    for e in emotions:
        out[e] = out.get(e, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

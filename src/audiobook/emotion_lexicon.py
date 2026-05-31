"""Bundled emotion lexicon.

A curated list of high-signal English emotion words. Each entry maps a
word to an emotion label with a weight (1.0 = strongly diagnostic,
0.5 = mild signal). The analyzer aggregates per-sentence by summing
weights, handles negation (`not happy` -> not happy) and intensifiers
(`very angry` -> stronger angry), and picks the dominant emotion.

This is intentionally small (~600 words) so it stays fast and stays
in the repo without licensing problems. For higher recall, install the
ML extras to enable the transformers-based analyzer alongside.

Format:
    LEXICON: dict[str, dict[str, float]]
             word -> { emotion_label: weight }

Words can carry signal for multiple emotions (e.g. "wept" = sad + fearful,
weighted differently).
"""
from __future__ import annotations


# Words that flip the emotion of the next 1-3 content words.
NEGATIONS: frozenset[str] = frozenset({
    "not", "no", "never", "nor", "neither", "without", "nothing",
    "nobody", "barely", "hardly", "scarcely", "rarely", "seldom",
    "isn't", "wasn't", "weren't", "aren't", "won't", "wouldn't",
    "don't", "didn't", "doesn't", "can't", "couldn't", "shouldn't",
    "haven't", "hasn't", "hadn't",
})

# Multipliers applied to the next emotion word.
INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "really": 1.4, "quite": 1.2, "extremely": 1.8, "utterly": 1.7,
    "absolutely": 1.7, "completely": 1.5, "totally": 1.5, "deeply": 1.5,
    "profoundly": 1.6, "intensely": 1.7, "incredibly": 1.6, "terribly": 1.5,
    "horribly": 1.6, "awfully": 1.4, "so": 1.3, "such": 1.3, "too": 1.2,
}
DAMPENERS: dict[str, float] = {
    "slightly": 0.5, "somewhat": 0.6, "a_bit": 0.5, "a_little": 0.5,
    "rather": 0.7, "fairly": 0.7, "kind_of": 0.6, "sort_of": 0.6,
    "almost": 0.7, "nearly": 0.7,
}


# -------------------- the lexicon --------------------
#
# Words below are picked for fiction narration (verbs of action / states /
# physical reaction) more than abstract feeling words. The analyzer's job
# is to read what the prose is *showing*, not just what characters are
# *thinking*.


def _w(*entries: tuple[str, float]) -> dict[str, float]:
    """Helper to build emotion weight dicts: _w(('angry', 1.0), ('fearful', 0.3))."""
    return dict(entries)


LEXICON: dict[str, dict[str, float]] = {
    # ---- happy ----
    "happy": _w(("happy", 1.0)),
    "joy": _w(("happy", 1.0)),
    "joyful": _w(("happy", 1.0)),
    "joyous": _w(("happy", 1.0)),
    "smile": _w(("happy", 0.8)),
    "smiled": _w(("happy", 0.8)),
    "smiling": _w(("happy", 0.8)),
    "grin": _w(("happy", 0.9)),
    "grinned": _w(("happy", 0.9)),
    "grinning": _w(("happy", 0.9)),
    "laugh": _w(("happy", 0.9)),
    "laughed": _w(("happy", 0.9)),
    "laughing": _w(("happy", 0.9)),
    "laughter": _w(("happy", 0.9)),
    "chuckle": _w(("happy", 0.8)),
    "chuckled": _w(("happy", 0.8)),
    "giggle": _w(("happy", 0.8)),
    "giggled": _w(("happy", 0.8)),
    "beamed": _w(("happy", 0.9)),
    "beaming": _w(("happy", 0.9)),
    "delight": _w(("happy", 1.0)),
    "delighted": _w(("happy", 1.0)),
    "glad": _w(("happy", 0.7)),
    "pleased": _w(("happy", 0.7)),
    "cheerful": _w(("happy", 0.9)),
    "cheer": _w(("happy", 0.8)),
    "cheered": _w(("happy", 0.8)),
    "bright": _w(("happy", 0.4)),
    "brightened": _w(("happy", 0.6)),
    "warm": _w(("happy", 0.3), ("calm", 0.3)),
    "warmth": _w(("happy", 0.4), ("calm", 0.3)),
    "love": _w(("happy", 0.6)),
    "loved": _w(("happy", 0.6)),
    "loving": _w(("happy", 0.6)),
    "kissed": _w(("happy", 0.5)),
    "embraced": _w(("happy", 0.6)),
    "hugged": _w(("happy", 0.6)),
    "celebrated": _w(("happy", 0.9), ("excited", 0.4)),
    "ecstatic": _w(("happy", 1.0), ("excited", 0.6)),
    "thrilled": _w(("happy", 0.6), ("excited", 0.9)),

    # ---- sad ----
    "sad": _w(("sad", 1.0)),
    "sadly": _w(("sad", 1.0)),
    "sadness": _w(("sad", 1.0)),
    "sorrow": _w(("sad", 1.0)),
    "sorrowful": _w(("sad", 1.0)),
    "grief": _w(("sad", 1.0)),
    "grieve": _w(("sad", 1.0)),
    "grieved": _w(("sad", 1.0)),
    "grieving": _w(("sad", 1.0)),
    "mourn": _w(("sad", 1.0)),
    "mourned": _w(("sad", 1.0)),
    "mourning": _w(("sad", 1.0)),
    "weep": _w(("sad", 1.0)),
    "wept": _w(("sad", 1.0)),
    "weeping": _w(("sad", 1.0)),
    "cry": _w(("sad", 0.6)),  # ambiguous; could be "cry out"
    "cried": _w(("sad", 0.6)),
    "crying": _w(("sad", 0.8)),
    "tear": _w(("sad", 0.5)),
    "tears": _w(("sad", 0.7)),
    "sob": _w(("sad", 1.0)),
    "sobbed": _w(("sad", 1.0)),
    "sobbing": _w(("sad", 1.0)),
    "sigh": _w(("sad", 0.5)),
    "sighed": _w(("sad", 0.5)),
    "ache": _w(("sad", 0.6)),
    "ached": _w(("sad", 0.6)),
    "aching": _w(("sad", 0.6)),
    "lonely": _w(("sad", 0.9)),
    "loneliness": _w(("sad", 0.9)),
    "alone": _w(("sad", 0.4)),
    "lost": _w(("sad", 0.4)),
    "empty": _w(("sad", 0.5)),
    "hollow": _w(("sad", 0.6)),
    "broken": _w(("sad", 0.5)),
    "regret": _w(("sad", 0.8)),
    "regretted": _w(("sad", 0.8)),
    "buried": _w(("sad", 0.4)),
    "funeral": _w(("sad", 0.8)),
    "died": _w(("sad", 0.5)),
    "dead": _w(("sad", 0.4)),
    "death": _w(("sad", 0.5)),
    "killed": _w(("sad", 0.3)),
    "miss": _w(("sad", 0.4)),
    "missed": _w(("sad", 0.4)),
    "missing": _w(("sad", 0.4)),
    "abandoned": _w(("sad", 0.8)),
    "forsaken": _w(("sad", 0.9)),

    # ---- angry ----
    "angry": _w(("angry", 1.0)),
    "anger": _w(("angry", 1.0)),
    "angrily": _w(("angry", 1.0)),
    "furious": _w(("angry", 1.0)),
    "fury": _w(("angry", 1.0)),
    "rage": _w(("angry", 1.0)),
    "raged": _w(("angry", 1.0)),
    "raging": _w(("angry", 1.0)),
    "wrath": _w(("angry", 1.0)),
    "snarl": _w(("angry", 1.0)),
    "snarled": _w(("angry", 1.0)),
    "snarling": _w(("angry", 1.0)),
    "growl": _w(("angry", 0.9)),
    "growled": _w(("angry", 0.9)),
    "growling": _w(("angry", 0.9)),
    "snap": _w(("angry", 0.6)),
    "snapped": _w(("angry", 0.6)),
    "snapping": _w(("angry", 0.6)),
    "bark": _w(("angry", 0.6)),
    "barked": _w(("angry", 0.7)),
    "scowl": _w(("angry", 0.9)),
    "scowled": _w(("angry", 0.9)),
    "glare": _w(("angry", 0.9)),
    "glared": _w(("angry", 0.9)),
    "glaring": _w(("angry", 0.9)),
    "glower": _w(("angry", 0.9)),
    "glowered": _w(("angry", 0.9)),
    "frown": _w(("angry", 0.4)),
    "frowned": _w(("angry", 0.4)),
    "shout": _w(("angry", 0.7), ("excited", 0.3)),
    "shouted": _w(("angry", 0.7), ("excited", 0.3)),
    "shouting": _w(("angry", 0.7), ("excited", 0.3)),
    "yell": _w(("angry", 0.7), ("excited", 0.3)),
    "yelled": _w(("angry", 0.7), ("excited", 0.3)),
    "yelling": _w(("angry", 0.7), ("excited", 0.3)),
    "roar": _w(("angry", 0.9)),
    "roared": _w(("angry", 0.9)),
    "thunder": _w(("angry", 0.6)),
    "thundered": _w(("angry", 0.8)),
    "smash": _w(("angry", 0.7)),
    "smashed": _w(("angry", 0.7)),
    "slam": _w(("angry", 0.6)),
    "slammed": _w(("angry", 0.6)),
    "punch": _w(("angry", 0.6)),
    "punched": _w(("angry", 0.6)),
    "strike": _w(("angry", 0.4)),
    "struck": _w(("angry", 0.4)),
    "spit": _w(("angry", 0.6)),
    "spat": _w(("angry", 0.7)),
    "curse": _w(("angry", 0.7)),
    "cursed": _w(("angry", 0.7)),
    "swore": _w(("angry", 0.7)),
    "betrayed": _w(("angry", 0.5), ("sad", 0.5)),
    "betrayal": _w(("angry", 0.5), ("sad", 0.5)),
    "hatred": _w(("angry", 1.0)),
    "hate": _w(("angry", 0.9)),
    "hated": _w(("angry", 0.9)),
    "despise": _w(("angry", 0.9)),
    "despised": _w(("angry", 0.9)),
    "contempt": _w(("angry", 0.8)),
    "outrage": _w(("angry", 1.0)),
    "outraged": _w(("angry", 1.0)),

    # ---- fearful ----
    "fear": _w(("fearful", 1.0)),
    "feared": _w(("fearful", 1.0)),
    "fearing": _w(("fearful", 1.0)),
    "fearful": _w(("fearful", 1.0)),
    "afraid": _w(("fearful", 1.0)),
    "scared": _w(("fearful", 1.0)),
    "terror": _w(("fearful", 1.0)),
    "terrified": _w(("fearful", 1.0)),
    "terrifying": _w(("fearful", 1.0)),
    "panic": _w(("fearful", 1.0)),
    "panicked": _w(("fearful", 1.0)),
    "panicking": _w(("fearful", 1.0)),
    "dread": _w(("fearful", 1.0)),
    "dreaded": _w(("fearful", 1.0)),
    "dreading": _w(("fearful", 1.0)),
    "horror": _w(("fearful", 1.0)),
    "horrified": _w(("fearful", 1.0)),
    "trembled": _w(("fearful", 0.7)),
    "trembling": _w(("fearful", 0.7)),
    "shook": _w(("fearful", 0.4)),
    "shaking": _w(("fearful", 0.5)),
    "shudder": _w(("fearful", 0.6)),
    "shuddered": _w(("fearful", 0.6)),
    "shivered": _w(("fearful", 0.5)),
    "froze": _w(("fearful", 0.6)),
    "frozen": _w(("fearful", 0.5)),
    "petrified": _w(("fearful", 1.0)),
    "danger": _w(("fearful", 0.6)),
    "dangerous": _w(("fearful", 0.6)),
    "threat": _w(("fearful", 0.7)),
    "threatened": _w(("fearful", 0.7)),
    "nightmare": _w(("fearful", 0.9)),
    "doom": _w(("fearful", 0.8)),
    "doomed": _w(("fearful", 0.9)),
    "creeped": _w(("fearful", 0.7)),
    "spooky": _w(("fearful", 0.7)),
    "haunted": _w(("fearful", 0.7)),
    "haunting": _w(("fearful", 0.6)),
    "ominous": _w(("fearful", 0.7)),
    "menacing": _w(("fearful", 0.7)),
    "ghastly": _w(("fearful", 0.8)),
    "gasped": _w(("fearful", 0.5), ("surprised", 0.7)),
    "stiffened": _w(("fearful", 0.5)),

    # ---- surprised ----
    "surprised": _w(("surprised", 1.0)),
    "surprise": _w(("surprised", 1.0)),
    "astonished": _w(("surprised", 1.0)),
    "astonishment": _w(("surprised", 1.0)),
    "amazed": _w(("surprised", 0.9)),
    "amazement": _w(("surprised", 0.9)),
    "stunned": _w(("surprised", 1.0)),
    "shocked": _w(("surprised", 0.9), ("fearful", 0.3)),
    "shock": _w(("surprised", 0.7), ("fearful", 0.3)),
    "startled": _w(("surprised", 1.0)),
    "blinked": _w(("surprised", 0.4)),
    "gasp": _w(("surprised", 0.7), ("fearful", 0.3)),
    "wow": _w(("surprised", 0.9)),
    "stared": _w(("surprised", 0.4)),
    "staring": _w(("surprised", 0.3)),
    "stammered": _w(("surprised", 0.8)),
    "blurted": _w(("surprised", 0.7), ("excited", 0.4)),
    "sudden": _w(("surprised", 0.4)),
    "suddenly": _w(("surprised", 0.4)),
    "abrupt": _w(("surprised", 0.4)),
    "abruptly": _w(("surprised", 0.4)),

    # ---- disgusted ----
    "disgust": _w(("disgusted", 1.0)),
    "disgusted": _w(("disgusted", 1.0)),
    "disgusting": _w(("disgusted", 1.0)),
    "revolted": _w(("disgusted", 1.0)),
    "revolting": _w(("disgusted", 1.0)),
    "repulsed": _w(("disgusted", 1.0)),
    "repulsive": _w(("disgusted", 1.0)),
    "sicken": _w(("disgusted", 0.9)),
    "sickened": _w(("disgusted", 0.9)),
    "sickening": _w(("disgusted", 0.9)),
    "nauseated": _w(("disgusted", 1.0)),
    "nauseous": _w(("disgusted", 1.0)),
    "vomit": _w(("disgusted", 0.9)),
    "vomited": _w(("disgusted", 0.9)),
    "retched": _w(("disgusted", 1.0)),
    "gag": _w(("disgusted", 0.9)),
    "gagged": _w(("disgusted", 0.9)),
    "foul": _w(("disgusted", 0.8)),
    "vile": _w(("disgusted", 0.9)),
    "rotten": _w(("disgusted", 0.8)),
    "putrid": _w(("disgusted", 1.0)),
    "stench": _w(("disgusted", 0.9)),
    "stink": _w(("disgusted", 0.8)),
    "filthy": _w(("disgusted", 0.8)),
    "grotesque": _w(("disgusted", 0.9)),
    "vulgar": _w(("disgusted", 0.7)),
    "loathed": _w(("disgusted", 0.9)),
    "loathing": _w(("disgusted", 0.9)),

    # ---- whispered ----
    "whisper": _w(("whispered", 1.0)),
    "whispered": _w(("whispered", 1.0)),
    "whispering": _w(("whispered", 1.0)),
    "murmur": _w(("whispered", 0.8)),
    "murmured": _w(("whispered", 0.8)),
    "murmuring": _w(("whispered", 0.8)),
    "mutter": _w(("whispered", 0.7)),
    "muttered": _w(("whispered", 0.7)),
    "muttering": _w(("whispered", 0.7)),
    "hushed": _w(("whispered", 0.9)),
    "hush": _w(("whispered", 0.9)),
    "quiet": _w(("whispered", 0.3), ("calm", 0.4)),
    "quietly": _w(("whispered", 0.5), ("calm", 0.4)),
    "softly": _w(("whispered", 0.5), ("calm", 0.4)),
    "low": _w(("whispered", 0.3)),
    "lowered": _w(("whispered", 0.3)),
    "silent": _w(("calm", 0.4), ("whispered", 0.3)),
    "silently": _w(("calm", 0.4), ("whispered", 0.3)),
    "breathed": _w(("whispered", 0.7)),
    "mouthed": _w(("whispered", 1.0)),
    "secret": _w(("whispered", 0.5)),
    "secrecy": _w(("whispered", 0.6)),
    "confided": _w(("whispered", 0.7)),

    # ---- excited ----
    "excited": _w(("excited", 1.0)),
    "excitement": _w(("excited", 1.0)),
    "exciting": _w(("excited", 0.8)),
    "exhilarated": _w(("excited", 1.0)),
    "eager": _w(("excited", 0.7)),
    "eagerly": _w(("excited", 0.7)),
    "thrill": _w(("excited", 0.8), ("happy", 0.4)),
    "thrilled": _w(("excited", 0.8), ("happy", 0.4)),
    "rushed": _w(("excited", 0.5)),
    "ran": _w(("excited", 0.3)),
    "raced": _w(("excited", 0.6)),
    "racing": _w(("excited", 0.6)),
    "leapt": _w(("excited", 0.6)),
    "leaped": _w(("excited", 0.6)),
    "dash": _w(("excited", 0.5)),
    "dashed": _w(("excited", 0.6)),
    "darted": _w(("excited", 0.6)),
    "sprang": _w(("excited", 0.7)),
    "jumped": _w(("excited", 0.5)),
    "burst": _w(("excited", 0.7)),
    "exclaimed": _w(("excited", 0.8), ("surprised", 0.4)),
    "energetic": _w(("excited", 0.8)),
    "enthusiastic": _w(("excited", 0.9)),
    "alive": _w(("excited", 0.5), ("happy", 0.3)),
    "electric": _w(("excited", 0.7)),

    # ---- calm ----
    "calm": _w(("calm", 1.0)),
    "calmly": _w(("calm", 1.0)),
    "calmed": _w(("calm", 1.0)),
    "peaceful": _w(("calm", 1.0)),
    "peace": _w(("calm", 0.9)),
    "serene": _w(("calm", 1.0)),
    "serenity": _w(("calm", 1.0)),
    "tranquil": _w(("calm", 1.0)),
    "still": _w(("calm", 0.4)),
    "stillness": _w(("calm", 0.7)),
    "steady": _w(("calm", 0.7)),
    "steadily": _w(("calm", 0.7)),
    "deliberate": _w(("calm", 0.6)),
    "patient": _w(("calm", 0.7)),
    "patiently": _w(("calm", 0.7)),
    "gentle": _w(("calm", 0.7)),
    "gently": _w(("calm", 0.7)),
    "slow": _w(("calm", 0.4)),
    "slowly": _w(("calm", 0.5)),
    "breath": _w(("calm", 0.3)),
    "breathing": _w(("calm", 0.4)),
    "settled": _w(("calm", 0.6)),
    "relaxed": _w(("calm", 0.9)),
    "rested": _w(("calm", 0.7)),
    "composed": _w(("calm", 0.8)),
    "centered": _w(("calm", 0.7)),
    "balanced": _w(("calm", 0.6)),
    "easy": _w(("calm", 0.4)),
    "soothed": _w(("calm", 0.9)),
    "soothing": _w(("calm", 0.9)),
}


def lookup(word: str) -> dict[str, float]:
    """Look up emotion weights for a single (lowercased) word.

    Returns {} if the word carries no emotion signal.
    """
    return LEXICON.get(word.lower(), {})


def is_negation(word: str) -> bool:
    return word.lower() in NEGATIONS


def intensifier_weight(word: str) -> float:
    """Return the multiplier for the next emotion word, or 1.0 if not an intensifier."""
    w = word.lower()
    if w in INTENSIFIERS:
        return INTENSIFIERS[w]
    if w in DAMPENERS:
        return DAMPENERS[w]
    return 1.0


# ----- introspection helpers -----


def lexicon_size() -> int:
    return len(LEXICON)


def words_per_emotion() -> dict[str, int]:
    """For quick visibility into lexicon coverage."""
    out: dict[str, int] = {}
    for entries in LEXICON.values():
        for emo in entries:
            out[emo] = out.get(emo, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

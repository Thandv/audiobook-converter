"""Multi-layer per-sentence emotion analyzer with consistency filter.

Pipeline per sentence:

   raw text
      |
      v
   +-------------------+      +------------------+
   | tag detector      |----->| ensemble scorer  |
   | (whispered, etc.) |      | weights each     |
   +-------------------+      | source by its    |
   +-------------------+      | confidence       |
   | lexicon scorer    |----->|                  |
   | (bundled words)   |      |                  |
   +-------------------+      |                  |
   +-------------------+      |                  |
   | ML classifier     |----->|                  |
   | (optional)        |      |                  |
   +-------------------+      +------------------+
                                       |
                                       v
                            +-----------------------+
                            | consistency filter    |
                            | per-speaker state,    |
                            | scene baseline,       |
                            | smoothing window,     |
                            | confidence threshold  |
                            +-----------------------+
                                       |
                                       v
                            list[SentenceEmotion]

Per-speaker, per-scene state means we don't whiplash between emotions —
once a character is sad, they stay sad until the evidence to change is
strong.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from .emotion import EMOTIONS, detect_emotion
from . import emotion_lexicon as lex


# Window over which we smooth emotion votes for a single speaker.
SMOOTHING_WINDOW = 3

# Minimum confidence required to switch emotions. Below this, speaker
# keeps the emotion they had.
TRANSITION_THRESHOLD = 0.55

# Confidence baseline for the lexicon and ML. Tags get higher baselines.
TAG_CONFIDENCE = 0.95
LEXICON_BASE_CONFIDENCE = 0.45
ML_BASE_CONFIDENCE = 0.65

# Negation window: a negation flips the next N content tokens' emotion.
NEGATION_WINDOW = 3

# Decay: after this many sentences without reinforcement, a speaker's
# emotion fades back toward "neutral" / scene baseline.
EMOTION_DECAY_AFTER = 6


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SentenceEmotion:
    """The analyzer's output for one sentence."""

    text: str
    speaker: str
    emotion: str
    confidence: float
    sources: tuple[str, ...]          # which analyzer layers fired
    evidence: tuple[str, ...]         # the actual cues (words, tag verbs)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "confidence": round(self.confidence, 3),
            "sources": list(self.sources),
            "evidence": list(self.evidence),
            "raw_scores": {k: round(v, 3) for k, v in self.raw_scores.items()},
        }


@dataclass
class AnalysisContext:
    """Information the analyzer needs about where it is in the text."""

    speaker: str
    surrounding_narration: str
    is_dialogue: bool
    scene_baseline: str | None = None


# ---------------------------------------------------------------------------
# Per-layer analyzers
# ---------------------------------------------------------------------------


class Analyzer(Protocol):
    name: str

    def score(self, text: str, context: AnalysisContext) -> dict[str, float]:
        """Return per-emotion scores in [0, 1]. Sum need not equal 1."""
        ...

    def evidence(self) -> tuple[str, ...]:
        """Return cues that triggered the last score (debug/explain)."""
        ...


class TagAnalyzer:
    """Wraps the existing dialogue-tag + ALL-CAPS + bang detector."""

    name = "tag"

    def __init__(self) -> None:
        self._evidence: tuple[str, ...] = ()

    def score(self, text: str, context: AnalysisContext) -> dict[str, float]:
        emo = detect_emotion(text, context.surrounding_narration)
        self._evidence = (f"tag:{emo}",) if emo != "neutral" else ()
        if emo == "neutral":
            return {}
        return {emo: TAG_CONFIDENCE}

    def evidence(self) -> tuple[str, ...]:
        return self._evidence


_WORD_RE = re.compile(r"[A-Za-z']+")


class LexiconAnalyzer:
    """Bundled-word-list scorer with negation + intensifier handling."""

    name = "lexicon"

    def __init__(self) -> None:
        self._evidence: list[str] = []

    def score(self, text: str, context: AnalysisContext) -> dict[str, float]:
        self._evidence = []
        words = _WORD_RE.findall(text.lower())
        if not words:
            return {}

        scores: dict[str, float] = {}
        # Walk the sentence tracking negation + intensifier state for a window.
        negate_until = -1
        multiplier = 1.0
        for i, word in enumerate(words):
            if lex.is_negation(word):
                negate_until = i + NEGATION_WINDOW
                self._evidence.append(f"neg:{word}")
                multiplier = 1.0
                continue
            inten = lex.intensifier_weight(word)
            if inten != 1.0:
                multiplier = inten
                self._evidence.append(f"x{inten}:{word}")
                continue
            entries = lex.lookup(word)
            if not entries:
                multiplier = 1.0
                continue
            negated = i <= negate_until
            for emo, w in entries.items():
                target = _negation_swap(emo) if negated else emo
                if target not in EMOTIONS:
                    continue
                scores[target] = scores.get(target, 0.0) + w * multiplier
                self._evidence.append(f"{target}:{word}")
            multiplier = 1.0

        if not scores:
            return {}
        # Normalize: scale so the dominant emotion sits in the
        # [LEXICON_BASE_CONFIDENCE, 0.9] band based on its margin over the
        # runner-up. Strong signals get higher confidence.
        sorted_emos = sorted(scores.items(), key=lambda kv: -kv[1])
        top_score = sorted_emos[0][1]
        runner_up = sorted_emos[1][1] if len(sorted_emos) > 1 else 0.0
        margin = (top_score - runner_up) / max(top_score, 1.0)
        confidence = LEXICON_BASE_CONFIDENCE + 0.45 * margin
        confidence = max(LEXICON_BASE_CONFIDENCE, min(0.9, confidence))

        # Return ALL emotions found, scaled so the top one == confidence
        # and others scaled proportionally. Downstream ensemble combines.
        scale = confidence / top_score if top_score > 0 else 0
        return {emo: s * scale for emo, s in scores.items()}

    def evidence(self) -> tuple[str, ...]:
        return tuple(self._evidence)


def _negation_swap(emotion: str) -> str:
    """When negated, an emotion word becomes... what? Mostly: not-X = neutral.

    A few semantic swaps make sense ("not happy" => sad), but most negations
    should just be discarded as noise. We default to neutral so negated
    matches don't accidentally vote.
    """
    swap = {
        "happy": "sad",
        "sad": "happy",
        "angry": "calm",
        "calm": "angry",
        "excited": "calm",
        "fearful": "calm",
    }
    return swap.get(emotion, "neutral")


class MLAnalyzer:
    """Optional Hugging Face transformers-based emotion classifier.

    Lazy-loads `j-hartmann/emotion-english-distilroberta-base` on first call.
    Maps the model's 7 emotions (anger, disgust, fear, joy, neutral,
    sadness, surprise) to our 10-emotion vocabulary.

    Install with: pip install -e ".[ml]" (adds transformers + torch).
    """

    name = "ml"

    # Model emotion -> our emotion. Note: our 10 includes whispered/excited/
    # calm which the model doesn't directly predict; we don't map those here
    # and let other layers contribute them.
    MAP = {
        "anger": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "joy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "surprise": "surprised",
    }

    def __init__(self) -> None:
        self._classifier = None
        self._evidence: tuple[str, ...] = ()

    def _load(self):
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "MLAnalyzer requires the [ml] extras. "
                'Run: pip install -e ".[ml]"'
            ) from e
        # top_k=None returns scores for ALL labels (we want the distribution).
        self._classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        return self._classifier

    def score(self, text: str, context: AnalysisContext) -> dict[str, float]:
        text = (text or "").strip()
        if not text:
            self._evidence = ()
            return {}
        try:
            clf = self._load()
        except RuntimeError:
            # Extras not installed — silently skip this layer.
            self._evidence = ()
            return {}
        # Pipeline returns [[{label, score}, ...]] for batched input.
        raw = clf(text[:512])
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        scores: dict[str, float] = {}
        for entry in raw:
            label = self.MAP.get(entry["label"].lower())
            if label is None:
                continue
            scores[label] = float(entry["score"])
        # Scale so the dominant emotion sits at ML_BASE_CONFIDENCE * its raw score.
        if not scores:
            return {}
        top = max(scores.values())
        scale = ML_BASE_CONFIDENCE / max(top, 0.01)
        scaled = {k: min(0.95, v * scale) for k, v in scores.items()}
        # Drop near-zero predictions for a cleaner ensemble.
        evidence: list[str] = []
        for k, v in sorted(scaled.items(), key=lambda kv: -kv[1])[:3]:
            evidence.append(f"ml:{k}={v:.2f}")
        self._evidence = tuple(evidence)
        return scaled

    def evidence(self) -> tuple[str, ...]:
        return self._evidence


# ---------------------------------------------------------------------------
# Ensemble + consistency
# ---------------------------------------------------------------------------


@dataclass
class _SpeakerState:
    """Per-speaker rolling state used by the consistency filter."""

    history: deque[str] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    current_emotion: str = "neutral"
    current_confidence: float = 0.0
    sentences_since_change: int = 0


class EmotionAnalyzer:
    """Top-level analyzer.

    Usage:
        analyzer = EmotionAnalyzer(use_ml=False)
        for sentence in sentences:
            result = analyzer.analyze(sentence, context)
        analyzer.reset_scene()  # between scenes
    """

    def __init__(self, *, use_ml: bool = False) -> None:
        self.layers: list[Analyzer] = [TagAnalyzer(), LexiconAnalyzer()]
        if use_ml:
            self.layers.append(MLAnalyzer())
        self._speakers: dict[str, _SpeakerState] = {}
        self._scene_baseline: str | None = None
        self._scene_baseline_lock_after = 5  # lock baseline after first N sentences
        self._sentences_in_scene = 0

    # --------------------------------------------------------- public API

    def reset_scene(self) -> None:
        """Reset per-scene state. Call between scenes."""
        self._speakers.clear()
        self._scene_baseline = None
        self._sentences_in_scene = 0

    def analyze(self, sentence: str, context: AnalysisContext) -> SentenceEmotion:
        """Return per-sentence emotion with consistency filter applied."""
        # Run all layers; combine their scores into an ensemble.
        per_layer_scores: list[dict[str, float]] = []
        per_layer_evidence: list[str] = []
        per_layer_sources: list[str] = []
        for layer in self.layers:
            try:
                s = layer.score(sentence, context)
            except Exception:  # noqa: BLE001 - safety net
                s = {}
            if s:
                per_layer_scores.append(s)
                per_layer_sources.append(layer.name)
                per_layer_evidence.extend(layer.evidence())

        # Combine: sum of layer scores (each layer's max already represents
        # its confidence) — tag layer dominates when present.
        combined: dict[str, float] = {}
        for s in per_layer_scores:
            for emo, score in s.items():
                combined[emo] = combined.get(emo, 0.0) + score

        # Pick raw dominant.
        if combined:
            raw_top = max(combined, key=combined.get)
            raw_top_score = combined[raw_top]
        else:
            raw_top = "neutral"
            raw_top_score = 0.0

        # Apply consistency filter.
        state = self._speakers.setdefault(context.speaker, _SpeakerState())
        final, final_confidence = self._consistency_filter(
            raw_top=raw_top,
            raw_score=raw_top_score,
            combined=combined,
            state=state,
            context=context,
        )

        # Update state.
        state.history.append(final)
        if final == state.current_emotion:
            state.sentences_since_change += 1
        else:
            state.current_emotion = final
            state.current_confidence = final_confidence
            state.sentences_since_change = 0

        # Possibly set scene baseline from the first few sentences.
        self._sentences_in_scene += 1
        if (
            self._scene_baseline is None
            and self._sentences_in_scene >= self._scene_baseline_lock_after
        ):
            self._scene_baseline = self._compute_scene_baseline()

        return SentenceEmotion(
            text=sentence,
            speaker=context.speaker,
            emotion=final,
            confidence=final_confidence,
            sources=tuple(per_layer_sources),
            evidence=tuple(per_layer_evidence[:6]),
            raw_scores={k: round(v, 3) for k, v in sorted(
                combined.items(), key=lambda kv: -kv[1]
            )[:5]},
        )

    # ----------------------------------------------------- internals

    def _consistency_filter(
        self,
        *,
        raw_top: str,
        raw_score: float,
        combined: dict[str, float],
        state: _SpeakerState,
        context: AnalysisContext,
    ) -> tuple[str, float]:
        """Apply the consistency rules to the raw ensemble result."""
        # Rule 1: if speaker has no prior emotion, accept the raw result
        # (or scene baseline if confidence is low).
        if state.current_emotion == "neutral" and not state.history:
            if raw_score >= TRANSITION_THRESHOLD:
                return raw_top, min(0.95, raw_score)
            baseline = self._scene_baseline or context.scene_baseline or "neutral"
            return baseline, max(0.4, raw_score)

        # Rule 2: same emotion as before — reinforce.
        if raw_top == state.current_emotion:
            return state.current_emotion, max(state.current_confidence, raw_score)

        # Rule 3: transition requires confidence above threshold.
        if raw_score >= TRANSITION_THRESHOLD:
            return raw_top, min(0.95, raw_score)

        # Rule 4: window smoothing — if the last K sentences have voted for
        # `raw_top` (including this one), accept it even at lower confidence.
        votes = list(state.history) + [raw_top]
        if votes.count(raw_top) >= max(2, len(votes) // 2):
            return raw_top, max(0.5, raw_score)

        # Rule 5: emotion decay — if the speaker has been in the same emotion
        # for a long time, gently allow drift back toward the scene baseline.
        if state.sentences_since_change >= EMOTION_DECAY_AFTER:
            target = self._scene_baseline or "neutral"
            if target != state.current_emotion and raw_score >= 0.3:
                return target, 0.5

        # Default: keep the current emotion.
        return state.current_emotion, state.current_confidence * 0.95

    def _compute_scene_baseline(self) -> str:
        """Determine the scene's overall emotional tone from the
        first sentences' results across all speakers."""
        votes: dict[str, int] = {}
        for state in self._speakers.values():
            for e in state.history:
                if e == "neutral":
                    continue
                votes[e] = votes.get(e, 0) + 1
        if not votes:
            return "neutral"
        return max(votes, key=votes.get)


# ---------------------------------------------------------------------------
# Convenience: analyze a whole book
# ---------------------------------------------------------------------------


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"'A-Z])")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    bits = _SENT_SPLIT.split(text)
    return [b.strip() for b in bits if b.strip()]


def analyze_paragraph(
    text: str,
    *,
    analyzer: EmotionAnalyzer,
    speaker: str,
    surrounding_narration: str = "",
    is_dialogue: bool = False,
) -> list[SentenceEmotion]:
    """Run per-sentence analysis on one paragraph's worth of text."""
    results: list[SentenceEmotion] = []
    for sent in split_sentences(text):
        ctx = AnalysisContext(
            speaker=speaker,
            surrounding_narration=surrounding_narration,
            is_dialogue=is_dialogue,
        )
        results.append(analyzer.analyze(sent, ctx))
    return results


def emotion_distribution(results: Iterable[SentenceEmotion]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.emotion] = out.get(r.emotion, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

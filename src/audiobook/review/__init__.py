"""Review subsystem: analyze a rendered audiobook, propose improvements.

Pipeline:
    chapter audio + source text
        |
        v
    transcribe (Whisper)
        |  +-- dsp metrics (librosa)
        |  +-- audio emotion classifier (wav2vec2)
        v
    align + score -> Findings list (JSON)
        |
        v
    fixer: apply safe fixes, propose risky ones, plan re-render
        |
        v
    delete affected chapters + re-render with --resume
"""
from .types import (
    Finding,
    FindingKind,
    FindingSeverity,
    ReviewReport,
)

__all__ = [
    "Finding",
    "FindingKind",
    "FindingSeverity",
    "ReviewReport",
]

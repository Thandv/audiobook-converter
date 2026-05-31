"""TTS backend implementations.

Each backend implements the `Backend` protocol from `base.py`.
Available backends:
  - kokoro: local, free, no training required (default)
  - xtts:   uses a fine-tuned XTTS v2 model (requires training/install of `[training]` extra)
"""
from .base import Backend, BackendError

__all__ = ["Backend", "BackendError"]

"""brain — a tiny Python decision-layer framework.

Query -> interpret -> AIML -> memory -> local Qwen -> cloud,
with a relevance classifier that promotes good answers back into AIML.

Stdlib only. Low-spec by design.
"""
from .brain import Brain, Answer
from .config import BrainConfig

__all__ = ["Brain", "Answer", "BrainConfig"]
__version__ = "0.1.0"

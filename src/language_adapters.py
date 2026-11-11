"""Language-specific tokenisation/normalisation adapters.

The neuro-symbolic fusion model should consume a language-independent 24-d semantic
vector.  This module isolates language-specific tokenisation from the ontology engine.

The Vietnamese adapter preserves the behaviour of the original repository by using
PyVi.  The English adapter deliberately has no external NLP dependency so it can run
reliably in Kaggle notebooks.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List


class BaseLanguageAdapter(ABC):
    """Minimal interface expected by ``OntologyEngine``."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Return normalized lexical tokens used by longest-match ontology lookup."""
        raise NotImplementedError

    @staticmethod
    def normalize_token(token: str) -> str:
        return re.sub(r"\s+", " ", str(token).lower().strip().replace("_", " ")).strip()


class VietnameseAdapter(BaseLanguageAdapter):
    """Original Vietnamese behaviour using PyVi word segmentation."""

    def __init__(self) -> None:
        try:
            from pyvi import ViTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error is explicit
            raise ImportError(
                "PyVi is required for VietnameseAdapter. Install with `pip install pyvi`."
            ) from exc
        self._tokenizer = ViTokenizer

    def tokenize(self, text: str) -> List[str]:
        segmented = self._tokenizer.tokenize(str(text).lower())
        return [self.normalize_token(t) for t in segmented.split() if self.normalize_token(t)]


class EnglishAdapter(BaseLanguageAdapter):
    """Lightweight English adapter for lexicon matching.

    It lowercases text, normalises curly apostrophes, expands common negative
    contractions to an explicit ``not`` token, removes punctuation while keeping
    hyphenated/apostrophe-bearing words, and returns whitespace-separated tokens.

    The transformer branch still uses the Hugging Face model tokenizer.  This adapter
    is only for the symbolic branch.
    """

    _NEG_PATTERNS = [
        (re.compile(r"\b(can)['’]t\b", re.I), r"\1 not"),
        (re.compile(r"\b(won)['’]t\b", re.I), "will not"),
        (re.compile(r"\b(shan)['’]t\b", re.I), "shall not"),
        (re.compile(r"n['’]t\b", re.I), " not"),
    ]

    def tokenize(self, text: str) -> List[str]:
        s = str(text).lower().replace("’", "'").replace("`", "'")
        for pat, repl in self._NEG_PATTERNS:
            s = pat.sub(repl, s)

        # Keep English letters, digits, apostrophes and hyphens.  We intentionally do
        # not stem/lemmatise, because the NRC lexicon is already word-level and we want
        # a transparent lexical trace.
        s = re.sub(r"[^a-z0-9'\-\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return [self.normalize_token(t) for t in s.split() if self.normalize_token(t)]


def build_language_adapter(name: str) -> BaseLanguageAdapter:
    """Factory used by experiment scripts."""
    key = str(name).strip().lower()
    if key in {"vi", "vie", "vietnamese"}:
        return VietnameseAdapter()
    if key in {"en", "eng", "english"}:
        return EnglishAdapter()
    raise ValueError(f"Unsupported language adapter: {name}")

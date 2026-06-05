"""Synonym expansion for the refinement service.

Wraps NLTK's WordNet corpus:

    from services.refinement.app.synonyms import SynonymExpander
    exp = SynonymExpander()
    print(exp.expand_token("car"))   # -> ["auto", "automobile", ...]
    print(exp.expand("fast car"))    # -> "fast car auto automobile motorcar"

WordNet is the Princeton lexical database of English (155,000 words
organized into 117,000 synsets). It ships as a NLTK data package
(``wordnet``) and was downloaded in Phase 0.

For each **non-stopword** token we pull the first N lemmas from each
POS (noun, verb, adj, adv) and dedupe. Multi-word synonyms are
intentionally dropped -- they break the downstream "space-joined
expanded_query" format used by Phase 5's hybrid retriever.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as wn

from services.refinement.app.config import SYNONYM_SKIP, WORDNET_POS

__all__ = ["SynonymExpander", "build_synonym_expander"]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _is_wordnet_loaded() -> bool:
    """Return True if WordNet is available. The download happened in Phase 0."""
    try:
        # ``wn.synsets("")`` raises if wordnet isn't downloaded.
        wn.synsets("test")
        return True
    except LookupError:
        return False


def build_synonym_expander() -> SynonymExpander:
    """Factory used by the service (and by tests)."""
    if not _is_wordnet_loaded():
        raise LookupError(
            "WordNet not downloaded. Run `python -m nltk.downloader wordnet` "
            "(or `make download-nltk`)."
        )
    return SynonymExpander()


class SynonymExpander:
    """Expands a single token (or a string of tokens) with WordNet synonyms."""

    def __init__(self) -> None:
        # Touch WordNet once so any download error surfaces at startup.
        if not _is_wordnet_loaded():
            raise LookupError("WordNet is not available; see download instructions above.")
        # Use NLTK's English stopword list (the same one preprocess.py uses
        # in step 5). WordNet's ``words()`` is the *vocabulary* of all
        # words, not a stopword list -- a common confusion.
        try:
            self._stop: set[str] = set(nltk_stopwords.words("english"))
        except LookupError:
            self._stop = set()
        self._stop.update(SYNONYM_SKIP)

    def _should_skip(self, token: str) -> bool:
        t = token.lower().strip()
        if not t:
            return True
        if t in self._stop:
            return True
        if not any(c.isalpha() for c in t):
            return True
        return False

    def expand_token(self, token: str, n: int = 2) -> list[str]:
        """Return up to ``n`` synonyms for ``token``. Excludes the token itself."""
        if n <= 0 or self._should_skip(token):
            return []
        t = token.lower()
        seen: set[str] = set()
        results: list[str] = []
        for pos in WORDNET_POS:
            for syn in wn.synsets(t, pos=pos):
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    # WordNet names use "_" for multi-word entries; skip those.
                    if "_" in name:
                        continue
                    if name == t:
                        continue
                    if name in seen:
                        continue
                    seen.add(name)
                    results.append(name)
                    if len(results) >= n:
                        return results
        return results[:n]

    def expand(self, text: str, n: int = 2) -> str:
        """Expand each token in ``text`` with up to ``n`` synonyms.

        Returns a single space-joined string: original tokens + their
        synonyms. The original token is always kept in place; synonyms
        are appended at the end (so the result is ``"w1 w2 ... syn(w1)_1 syn(w1)_2 syn(w2)_1 ..."``).
        """
        if not text or n <= 0:
            return text
        out: list[str] = []
        for raw_word in text.split(" "):
            if not raw_word:
                out.append(raw_word)
                continue
            # Strip punctuation for the lookup; keep the surface form.
            cleaned = "".join(c for c in raw_word if c.isalpha())
            if not cleaned:
                out.append(raw_word)
                continue
            synonyms = self.expand_token(cleaned, n=n)
            out.append(raw_word)
            out.extend(synonyms)
        return " ".join(out)

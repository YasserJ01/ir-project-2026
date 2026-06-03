"""Text preprocessing for the IR system.

This is the **single source of truth** for tokenization. It is imported by:
- `scripts/ingest_dataset_a.py` and `scripts/ingest_dataset_b.py` (Phase 1 ingest)
- `scripts/tokenize_corpus.py` (Phase 1 persist)
- `services/preprocessing/app/pipeline.py` (Phase 1 FastAPI)
- Phase 4 (query refinement) and beyond

Pipeline (in this exact order):
    1. Strip HTML markup  (`<p>...</p>` -> `...`)
    2. NFKC unicode normalize (so ``\ufb01`` -> `fi`, ``\u00e9`` stays ``\u00e9``)
    3. Lowercase
    4. Word tokenize (NLTK `word_tokenize` — handles contractions, punctuation)
    5. Drop NLTK English stopwords
    6. Drop tokens with `len < 2` (drops `,`, `.`, `?`, `a`, `I`, etc.)
    7. Porter-stem each remaining token

Rationale per step lives in `docs/PHASE_1.md §5`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

__all__ = [
    "preprocess",
    "preprocess_batch",
    "strip_html",
    "normalize_unicode",
    "tokenize",
    "remove_stopwords",
    "drop_short",
    "drop_non_alpha",
    "stem_tokens",
    "PIPELINE_STEPS",
]

PIPELINE_STEPS: tuple[str, ...] = (
    "strip_html",
    "normalize_unicode",
    "lowercase",
    "tokenize",
    "remove_stopwords",
    "drop_short",
    "drop_non_alpha",
    "stem",
)

_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9]+")  # used only inside ``remove_stopwords``'s edge cases

# Module-level lazy-init guards
_stemmer: PorterStemmer | None = None
_stopwords: frozenset[str] | None = None
_nltk_ready: bool = False


def _ensure_nltk() -> None:
    """Download NLTK assets the first time we are called.

    Modern NLTK (>= 3.8.2) renamed ``punkt`` to ``punkt_tab`` for the
    pre-tokenized models. We try the modern name first, fall back to the
    legacy one, and only warn (never fail) if both are already present
    on disk. ``stopwords`` and ``wordnet`` are required for
    ``remove_stopwords``.
    """
    global _nltk_ready
    if _nltk_ready:
        return
    needed = []
    # Tokenizer model (one of these will exist after the first download).
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            needed.append("punkt_tab")
    # Stopwords corpus
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        needed.append("stopwords")
    # Wordnet (used later by refinement, but harmless to fetch now)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        needed.append("wordnet")
    # punkt legacy also needed in some NLTK versions for the sent_tokenize fallback
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        if "punkt_tab" not in needed and "punkt" not in needed:
            needed.append("punkt")
    if needed:
        # Quiet download; resources are idempotent.
        nltk.download(needed, quiet=True)
    _nltk_ready = True


def _get_stemmer() -> PorterStemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer


def _get_stopwords() -> frozenset[str]:
    global _stopwords
    if _stopwords is None:
        _stopwords = frozenset(stopwords.words("english"))
    return _stopwords


# ─────────────────────────────────────────────────────────────────────────
# Pipeline steps (each is independently importable + unit-testable)
# ─────────────────────────────────────────────────────────────────────────


def strip_html(text: str) -> str:
    """Remove anything that looks like an HTML tag.

    >>> strip_html("<p>Hello <b>world</b>!</p>")
    'Hello world!'
    """
    return _HTML_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """NFKC normalize: compatibility decomposition + canonical composition.

    Ensures ``\ufb01`` -> ``fi`` (ligature), ``\u00bd`` -> ``1⁄2`` (vulgar
    fractions), and full-width ASCII -> ASCII. Already-NFC text is unchanged.
    """
    return unicodedata.normalize("NFKC", text)


def tokenize(text: str) -> list[str]:
    """NLTK ``word_tokenize`` (handles contractions + punctuation)."""
    _ensure_nltk()
    # ``word_tokenize`` is typed as ``list[str]`` in stubs but mypy infers
    # ``Any`` for the sentence-split fallback path; explicit cast keeps
    # downstream code strictly typed.
    result: list[str] = word_tokenize(text)
    return result


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Drop NLTK's 179 English stopwords."""
    sw = _get_stopwords()
    return [t for t in tokens if t not in sw]


def drop_short(tokens: Iterable[str], min_len: int = 2) -> list[str]:
    """Drop tokens shorter than ``min_len`` (default 2)."""
    return [t for t in tokens if len(t) >= min_len]


def drop_non_alpha(tokens: Iterable[str]) -> list[str]:
    """Drop tokens that contain no alphanumeric character.

    Catches stragglers like ``...`` (NLTK's ellipsis token) and ``--``.
    Stems like ``co2`` and ``xbox1`` survive because they have at least one
    alphanumeric character.
    """
    return [t for t in tokens if any(c.isalnum() for c in t)]


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Porter-stem every token (deterministic, no corpus stats)."""
    stemmer = _get_stemmer()
    return [stemmer.stem(t) for t in tokens]


# ─────────────────────────────────────────────────────────────────────────
# The public pipeline
# ─────────────────────────────────────────────────────────────────────────


def preprocess(text: str) -> list[str]:
    """Run the full preprocessing pipeline on a single string.

    >>> preprocess("The quick brown foxes were running fast.")
    ['quick', 'brown', 'fox', 'run', 'fast']
    """
    if not text:
        return []
    text = strip_html(text)
    text = normalize_unicode(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = drop_short(tokens)
    tokens = drop_non_alpha(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def preprocess_batch(texts: Iterable[str]) -> Iterator[list[str]]:
    """Lazy variant for streaming corpora (never holds the full list)."""
    for text in texts:
        yield preprocess(text)


# Re-export for `lru_cache`-style hot-loop (e.g. query refinement)
@lru_cache(maxsize=4096)
def preprocess_cached(text: str) -> tuple[str, ...]:
    """Cached variant — returns a tuple (hashable) for repeated queries."""
    return tuple(preprocess(text))

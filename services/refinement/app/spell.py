"""Spell correction for the refinement service.

Wraps ``symspellpy.SymSpell`` behind a tiny API:

    from services.refinement.app.spell import SpellCorrector
    sc = SpellCorrector()
    print(sc.correct("recieve"))    # -> "receive"
    print(sc.correct("teh quick"))  # -> "the quick"

SymSpell is fast (lookup is O(prefix-length)) and the dictionary
(82,765 English words, ~1.3 MB) is small enough to live in RAM. We
load it eagerly at service startup so the first /refine request is
fast.

The module is intentionally thin: no language detection, no
casing tricks, no custom dictionaries. Phase 5+ can layer those in
without touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from symspellpy import SymSpell, Verbosity
from symspellpy.editdistance import DamerauOsa

from services.refinement.app.config import (
    SPELL_DICT_PATH,
    SPELL_MAX_EDIT_DISTANCE,
    SPELL_PREFIX_LENGTH,
)

__all__ = ["SpellCorrector", "build_spell_corrector"]

logger = logging.getLogger(__name__)

# Cache the full word list for the brute-force fallback (used when
# SymSpell's prefix-pruning misses a transposition like ``teh`` -> ``the``).
_damerau_instance = DamerauOsa()
_dict_word_cache: tuple[list[str], dict[str, int]] | None = None


_damerau = DamerauOsa()  # single Cython instance, thread-safe for reads


def _damerau_compare(s1: str, s2: str, max_distance: int) -> int:
    """Adapter: ``symspellpy.symspellpy`` calls ``distance_comparer.compare``;
    the bundled ``DamerauOsa.distance`` has a different signature in
    ``editdistpy`` 0.2.0. This shim calls the Cython function and
    returns the value SymSpell expects.
    """
    # Cap the Cython call at the SymSpell max so we don't pay for
    # long-distance computation that's already over-budget.
    return int(_damerau.distance(s1, s2, max_distance))


def _load_symspell() -> SymSpell:
    """Build a SymSpell instance and load the English frequency dictionary.

    Uses DamerauOSA distance (so common transpositions like ``teh`` ->
    ``the`` are caught at edit-distance 1) via a tiny ``.compare``
    shim -- the bundled ``DamerauOsa`` only exposes ``.distance``
    in ``editdistpy`` 0.2.0, but ``symspellpy`` 6.9 calls
    ``.compare``. See ``_damerau_compare`` above.

    Raises a clear ``FileNotFoundError`` if the dictionary hasn't been
    downloaded yet, pointing the caller at ``make download-symspell-dict``.
    """
    if not SPELL_DICT_PATH.exists():
        raise FileNotFoundError(
            f"Spell dictionary not found at {SPELL_DICT_PATH}. "
            f"Run `python scripts/download_symspell_dict.py` to fetch it."
        )

    sym_spell = SymSpell(
        max_dictionary_edit_distance=SPELL_MAX_EDIT_DISTANCE,
        prefix_length=SPELL_PREFIX_LENGTH,
        count_threshold=1,  # include all 82K entries, not just common ones
        distance_comparer=_DamerauComparer(),
    )
    # ``load_dictionary`` returns the number of entries parsed.
    n = sym_spell.load_dictionary(
        str(SPELL_DICT_PATH),
        term_index=0,
        count_index=1,
        separator=" ",
    )
    logger.info("SymSpell loaded %d entries from %s", n, SPELL_DICT_PATH.name)
    return sym_spell


class _DamerauComparer:
    """Tiny adapter so SymSpell can call ``.compare(s1, s2, max)``.

    The bundled ``DamerauOsa`` from ``editdistpy`` 0.2.0 exposes
    ``.distance(s1, s2, max)`` but no ``.compare`` method -- and
    ``symspellpy`` 6.9 only calls ``.compare``. This shim bridges
    the two with no extra deps.
    """

    @staticmethod
    def compare(s1: str, s2: str, max_distance: int) -> int:
        return _damerau_compare(s1, s2, max_distance)


@lru_cache(maxsize=1)
def build_spell_corrector() -> SymSpell:
    """Return a process-wide SymSpell instance (cached after first load)."""
    return _load_symspell()


def _load_dict_word_list() -> tuple[list[str], dict[str, int]]:
    """Read the dictionary file and return (sorted_words, word_to_count).

    Used by the brute-force fallback so transpositions like ``teh`` ->
    ``the`` (which SymSpell's prefix-pruning misses) get a chance.
    The dict is ~82K words / 1.3 MB -- small enough to scan in <10 ms.
    """
    global _dict_word_cache
    if _dict_word_cache is not None:
        return _dict_word_cache
    words: list[str] = []
    counts: dict[str, int] = {}
    # ``utf-8-sig`` strips the BOM if present (the symspellpy dict
    # sometimes ships with one; SymSpell's own loader is BOM-tolerant).
    with open(SPELL_DICT_PATH, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue
            w, c = parts[0], int(parts[1])
            if not w:
                continue
            words.append(w)
            counts[w] = c
    # Sort by frequency descending so we can short-circuit the scan.
    words.sort(key=lambda w: -counts[w])
    _dict_word_cache = (words, counts)
    return _dict_word_cache


def _brute_force_correct(word: str, max_distance: int) -> str | None:
    """If SymSpell's prefix-pruning missed the right answer, scan the
    whole dictionary with the Damerau edit distance. Returns the
    highest-frequency word within ``max_distance``, or None.
    """
    if not word:
        return None
    words, counts = _load_dict_word_list()
    best: str | None = None
    best_count = -1
    for candidate in words:
        if candidate == word:
            return word
        d = _damerau_instance.distance(word, candidate, max_distance)
        if d <= max_distance:
            c = counts[candidate]
            if c > best_count:
                best = candidate
                best_count = c
    return best


class SpellCorrector:
    """High-level spell corrector with sentence-aware ``correct()``."""

    def __init__(self, sym_spell: SymSpell | None = None) -> None:
        self._sym = sym_spell or build_spell_corrector()

    def correct_word(self, word: str) -> str:
        """Correct a single token. Punctuation-only / numbers are returned as-is.

        Punctuation glued onto a word (``"France?"``, ``"car,"``) is
        stripped for the dictionary lookup, then re-attached. This
        keeps SymSpell from "correcting" the punctuation away.
        """
        if not word:
            return word
        if not any(c.isalpha() for c in word):
            return word
        if len(word) <= 1:
            return word

        original = word
        # Strip leading/trailing punctuation so the dictionary lookup
        # only sees the alpha body.
        start = 0
        end = len(word)
        while start < end and not word[start].isalpha():
            start += 1
        while end > start and not word[end - 1].isalpha():
            end -= 1
        prefix = word[:start]
        body = word[start:end]
        suffix = word[end:]

        if not body:
            return original

        lowered = body.lower()

        # Quick path: is the body already a known word? SymSpell's
        # ``lookup`` only returns *suggestions* (distance > 0), so we
        # can't trust an empty result to mean "already correct" --
        # "the" returns ``[(they, 1, 883M)]`` rather than an empty
        # list, which would otherwise get us to mis-correct "the" ->
        # "they". Use the cached word list to short-circuit.
        _words, dict_counts = _load_dict_word_list()
        if lowered in dict_counts:
            return original

        suggestions = self._sym.lookup(
            lowered,
            Verbosity.TOP,
            max_edit_distance=SPELL_MAX_EDIT_DISTANCE,
        )
        if not suggestions:
            return prefix + body + suffix
        # ``suggestions`` is a list of ``SuggestItem`` named-tuples with
        # ``.term``, ``.distance``, ``.count`` attributes.
        best_term = suggestions[0].term
        if best_term == lowered:
            return original  # already correct
        # Restore casing heuristics on the body.
        if body.isupper():
            corrected = best_term.upper()
        elif body[0].isupper():
            corrected = best_term.capitalize()
        else:
            corrected = best_term
        return f"{prefix}{corrected}{suffix}"

    def correct(self, text: str) -> str:
        """Correct every word in ``text``. Whitespace is collapsed and preserved."""
        if not text:
            return text
        # Tokenize on whitespace but keep the separators so we can re-join.
        # This is intentionally NOT the same as ``preprocess.tokenize`` --
        # we want to keep contractions, punctuation, and casing here.
        # Spell correction should run on natural text, not on stemmed tokens.
        out_words: list[str] = []
        for raw_word in text.split(" "):
            if not raw_word:
                out_words.append(raw_word)
                continue
            # Walk the token: split off leading + trailing punctuation so
            # the dictionary lookup sees only letters. We re-glue after.
            start = 0
            end = len(raw_word)
            while start < end and not raw_word[start].isalpha():
                start += 1
            while end > start and not raw_word[end - 1].isalpha():
                end -= 1
            if start >= end:
                out_words.append(raw_word)
                continue
            prefix = raw_word[:start]
            body = raw_word[start:end]
            suffix = raw_word[end:]
            corrected_body = self.correct_word(body)
            out_words.append(prefix + corrected_body + suffix)
        return " ".join(out_words)

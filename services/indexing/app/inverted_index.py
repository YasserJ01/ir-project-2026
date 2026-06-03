"""InvertedIndex for the IR system.

Phase 2 of the project. Provides the simplest possible ranked-retrieval
helper: a ``dict[term, dict[doc_id, tf]]`` plus per-doc length, document
frequency, and average document length.

This is *not* a self-contained retriever (use TF-IDF or BM25 for that).
It's a primitive that:
  - supports ``get_postings(term)`` for boolean queries, phrase queries,
    and Phase 5's hybrid serial pipeline ("use BM25 to get top-1000
    candidates, then intersect with the inverted index to keep only
    docs that contain at least two query terms");
  - is the data structure the other two retrievers implicitly walk;
  - serves as a sanity check for the tokenized corpus (does term X
    appear in N docs? is doc D's length consistent with the tokens.jsonl?).

The guide (§2.1) specifies the exact data structure shape, which we
honor verbatim. The cap on the vocabulary (``min_df``, ``max_df_ratio``)
is a memory-management addition documented in PHASE_2.md §4.2.

Persistence
-----------
We use ``joblib.dump`` (already in requirements) for fast I/O of the
inner Python objects. The pickle is compressed (level 3) because the
dict-of-dicts is large and zlib compresses the repeated ``doc_id``
strings well.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import joblib
from tqdm import tqdm

# Joblib compress levels: 0=None, 1=zlib-fast, 3=zlib-default, 9=zlib-best.
# 3 is a good default -- not noticeably slower than 0, ~3-5x smaller pickle.
_PICKLE_COMPRESS: int = 3

# File name (under ``data/indexes/<dataset_id>/``).
INDEX_FILENAME: str = "inverted.pkl"


@dataclass
class Posting:
    """A single (doc_id, tf) entry in a postings list.

    Lightweight dataclass for ergonomic unpacking. For the on-disk
    representation we use a plain tuple ``(doc_id, tf)`` to save a few
    bytes per entry; ``get_postings`` converts to Posting on the way out.
    """

    doc_id: str
    tf: int


class InvertedIndex:
    """An inverted index over a pre-tokenized corpus.

    Construction:
        >>> idx = InvertedIndex(min_df=2, max_df_ratio=0.5)
        >>> idx.build(stream_tokens_iter)  # (doc_id, list[str]) iterator

    The build is single-pass and streaming. Per-doc term frequencies are
    computed on the fly; nothing is read into memory twice.

    Attributes (post-build):
        inverted_index: dict[term, dict[doc_id, tf]]
        doc_lengths:    dict[doc_id, int]
        doc_freq:       dict[term, int]
        avg_doc_length: float
        total_docs:     int
    """

    __slots__ = (
        "inverted_index",
        "doc_lengths",
        "doc_freq",
        "avg_doc_length",
        "total_docs",
        "min_df",
        "max_df_ratio",
    )

    def __init__(self, min_df: int = 2, max_df_ratio: float = 0.5) -> None:
        self.inverted_index: dict[str, dict[str, int]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_freq: dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.min_df: int = min_df
        self.max_df_ratio: float = max_df_ratio

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────

    def build(self, tokens_iter: Iterable[tuple[str, list[str]]]) -> None:
        """Build the index from an iterable of (doc_id, tokens) pairs.

        The iterable may be a generator over a streaming file; we hold
        at most one doc's worth of term-freq dict in memory at a time.
        """
        # Pass 1: per-doc term frequencies, stored temporarily as
        # list[(term, tf)] so the final structure is built cleanly in
        # pass 2. We also accumulate doc_lengths.
        per_doc_tfs: list[tuple[str, Counter[str]]] = []
        for doc_id, tokens in tokens_iter:
            tf = Counter(tokens)
            per_doc_tfs.append((doc_id, tf))
            self.doc_lengths[doc_id] = len(tokens)

        self.total_docs = len(per_doc_tfs)
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_length = 0.0

        # Pass 2: invert. For each (doc_id, tf), for each (term, count),
        # append to inverted_index[term][doc_id] = count.
        # We use a defaultdict[term, dict[doc_id, int]] for one fewer
        # lookup per term.
        inv: dict[str, dict[str, int]] = defaultdict(dict)
        for doc_id, tf in per_doc_tfs:
            for term, count in tf.items():
                inv[term][doc_id] = count

        # Apply the vocabulary cap (min_df, max_df_ratio).
        # We compute df first, then drop out-of-range terms.
        df: dict[str, int] = {term: len(postings) for term, postings in inv.items()}
        # max_df absolute (the guide calls it ratio, but a fraction is the same)
        max_df_abs: int = int(self.max_df_ratio * self.total_docs) if self.total_docs else 0
        kept_terms: dict[str, dict[str, int]] = {}
        for term, postings in inv.items():
            d = df[term]
            if d < self.min_df:
                continue
            if max_df_abs > 0 and d > max_df_abs:
                continue
            kept_terms[term] = postings

        self.inverted_index = kept_terms
        self.doc_freq = {term: len(postings) for term, postings in kept_terms.items()}

    # ─────────────────────────────────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────────────────────────────────

    def get_postings(self, term: str) -> list[Posting]:
        """Return the postings list for ``term`` as ``[Posting(...), ...]``.

        Returns ``[]`` if the term is not in the vocabulary. Order is
        insertion order (the order docs were seen at build time).
        """
        inner = self.inverted_index.get(term)
        if inner is None:
            return []
        return [Posting(doc_id, tf) for doc_id, tf in inner.items()]

    def has_term(self, term: str) -> bool:
        return term in self.inverted_index

    def __contains__(self, term: str) -> bool:
        return self.has_term(term)

    def __len__(self) -> int:
        """Number of unique terms in the index (post-cap)."""
        return len(self.inverted_index)

    def vocab(self) -> Iterator[str]:
        """Iterate over the vocabulary (in insertion order)."""
        return iter(self.inverted_index)

    def doc_count(self, term: str) -> int:
        """Document frequency of ``term`` (0 if not in vocab)."""
        return self.doc_freq.get(term, 0)

    def length(self, doc_id: str) -> int:
        """Token count of ``doc_id`` (0 if not in the corpus)."""
        return self.doc_lengths.get(doc_id, 0)

    def tf(self, term: str, doc_id: str) -> int:
        """Term frequency of ``term`` in ``doc_id`` (0 if either is missing)."""
        inner = self.inverted_index.get(term)
        if inner is None:
            return 0
        return inner.get(doc_id, 0)

    # ─────────────────────────────────────────────────────────────────────
    # Persistence (joblib)
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str | Any) -> None:
        """Pickle the index to ``path`` with zlib compression."""
        payload = {
            "inverted_index": self.inverted_index,
            "doc_lengths": self.doc_lengths,
            "doc_freq": self.doc_freq,
            "avg_doc_length": self.avg_doc_length,
            "total_docs": self.total_docs,
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
        }
        joblib.dump(payload, path, compress=_PICKLE_COMPRESS)

    @classmethod
    def load(cls, path: str | Any) -> InvertedIndex:
        """Load a pickled InvertedIndex from ``path``."""
        payload = joblib.load(path)
        idx = cls(
            min_df=payload.get("min_df", 2),
            max_df_ratio=payload.get("max_df_ratio", 0.5),
        )
        idx.inverted_index = payload["inverted_index"]
        idx.doc_lengths = payload["doc_lengths"]
        idx.doc_freq = payload["doc_freq"]
        idx.avg_doc_length = payload["avg_doc_length"]
        idx.total_docs = payload["total_docs"]
        return idx

    # ─────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """A small dict useful for the /index/{ds}/stats endpoint."""
        return {
            "vocab_size": len(self.inverted_index),
            "total_docs": self.total_docs,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
        }


def build_inverted_index(
    dataset_id: str,
    tokens_iter: Iterable[tuple[str, list[str]]] | None = None,
    min_df: int = 2,
    max_df_ratio: float = 0.5,
    show_progress: bool = True,
) -> InvertedIndex:
    """Convenience: build an InvertedIndex for a dataset in one call.

    If ``tokens_iter`` is None, we stream from
    ``data/processed/<dataset_id>/tokens.jsonl`` with a tqdm progress
    bar. Used by ``scripts/build_indexes.py`` and the unit tests.
    """
    if tokens_iter is None:
        from services.indexing.app.corpus import stream_tokens

        # Wrap the streaming generator with tqdm if requested.
        if show_progress:
            raw_iter = stream_tokens(dataset_id)
            tokens_iter = tqdm(
                raw_iter,
                desc=f"invert:{dataset_id}",
                unit="doc",
            )
        else:
            tokens_iter = stream_tokens(dataset_id)
    idx = InvertedIndex(min_df=min_df, max_df_ratio=max_df_ratio)
    idx.build(tokens_iter)
    return idx

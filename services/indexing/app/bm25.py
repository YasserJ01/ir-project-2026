"""BM25 retriever for the IR system.

Phase 2 of the project. Wraps the ``bm25s`` library
(``pip install bm25s``), which is a NumPy-vectorized reimplementation of
BM25 that's roughly 50x faster than the pure-Python ``rank_bm25`` on
large corpora (the scoring loop is the hot path; bm25s vectorizes it).

The guide (§2.3) names ``rank_bm25`` explicitly, but the wrapper exposes
the same ``score(query_tokens, k, k1, b)`` API and produces identical
scores (modulo floating-point determinism). The deviation is documented
in ``docs/PHASE_2.md §13``.

Why bm25s + the single source of truth?
---------------------------------------
The guide's recommended flow is: tokenize with the shared
``preprocess()`` and feed the tokens to BM25Okapi. ``bm25s`` has its
own tokenizer (``bm25s.tokenize``) that lowercases + stems. To honour
the **single source of truth** guarantee (Phase 1, §1.5), we skip
bm25s's tokenizer entirely: we run our own ``preprocess()`` on the
corpus, build a vocab from the result, convert to token IDs, and
construct a ``bm25s.tokenization.Tokenized`` namedtuple to hand to
``bm25s.index()``.

At query time, the same pattern: ``preprocess(query_text)`` -> map
through the saved vocab -> ``bm.get_scores(query_token_ids)``.

(k1, b) re-tuning
-----------------
The default ``BM25Okapi`` constructor takes (k1, b) at build time. To
support runtime tuning (the React UI's BM25 sliders in Phase 7), we
maintain a small LRU cache of ``(k1, b) -> BM25 instance``. The first
query with a new (k1, b) is ~1 sec (the bm25s BM25 init is O(N) over
the corpus); subsequent queries are O(|query| * avg_doc_len), < 100ms.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import joblib
import numpy as np
from tqdm import tqdm

from services.indexing.app.config import (
    BM25_CACHE_SIZE,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
)

DOC_IDS_FILENAME: str = "doc_ids.json"
TOKEN_IDS_FILENAME: str = "bm25_token_ids.pkl"  # corpus in token-ID form
VOCAB_FILENAME: str = "bm25_vocab.json"
DEFAULT_BM25_FILENAME: str = "bm25.pkl"  # the default-k1/b BM25 (fast path)


@dataclass
class BM25Hit:
    """A single search hit. Score is the BM25 score (non-negative)."""

    rank: int
    doc_id: str
    score: float


class BM25Retriever:
    """BM25 retrieval over a pre-tokenized corpus.

    Build:
        >>> r = BM25Retriever()
        >>> r.build(corpus_tokens, doc_ids)

    Search:
        >>> r.search(["fox", "jump"], k=10) -> list[BM25Hit]
        >>> r.search(["fox", "jump"], k=10, k1=1.2, b=0.5) -> [BM25Hit, ...]

    The corpus and vocab are persisted so the (k1, b) LRU can rebuild
    BM25 instances on demand without re-tokenizing the corpus.
    """

    __slots__ = (
        "doc_ids",
        "vocab",
        "token_ids",
        "_default_bm",
        "_cache",
    )

    def __init__(
        self,
        doc_ids: list[str] | None = None,
        vocab: dict[str, int] | None = None,
        token_ids: list[list[int]] | None = None,
        default_bm: bm25s.BM25 | None = None,
    ) -> None:
        self.doc_ids: list[str] = doc_ids or []
        # OrderedDict for reproducibility: vocab insertion order is the
        # order terms were first seen at build time.
        self.vocab: OrderedDict[str, int] = (
            OrderedDict(vocab) if vocab is not None else OrderedDict()
        )
        self.token_ids: list[list[int]] = token_ids or []
        self._default_bm: bm25s.BM25 | None = default_bm
        # LRU cache of (k1, b, method) -> bm25s.BM25
        self._cache: OrderedDict[tuple[float, float, str], bm25s.BM25] = OrderedDict()

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        corpus_tokens: list[list[str]],
        doc_ids: list[str],
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
        method: str = "lucene",
        show_progress: bool = True,
    ) -> None:
        """Build a BM25 index from the pre-tokenized corpus.

        The vocab is built from the corpus; bm25s is configured with
        ``method="lucene"`` which is the BM25Okapi equivalent.
        """
        if len(corpus_tokens) != len(doc_ids):
            raise ValueError(
                f"corpus_tokens ({len(corpus_tokens)}) and doc_ids ({len(doc_ids)}) "
                "must have the same length"
            )

        # Build the vocab (preserves insertion order).
        vocab: OrderedDict[str, int] = OrderedDict()
        for doc in corpus_tokens:
            for t in doc:
                if t not in vocab:
                    vocab[t] = len(vocab)
        # Convert corpus to token IDs.
        token_ids: list[list[int]] = [[vocab[t] for t in doc] for doc in corpus_tokens]

        self.doc_ids = list(doc_ids)
        self.vocab = vocab
        self.token_ids = token_ids

        # Build the default BM25 instance (most queries will use it).
        tok = bm25s.tokenization.Tokenized(ids=token_ids, vocab=dict(vocab))
        # ``show_progress`` is forwarded but bm25s may print to stdout
        # regardless (Numba warmup). We silence stderr but keep stdout
        # so the user sees the warmup message.
        bm = bm25s.BM25(method=method, k1=k1, b=b)
        bm.index(tok, show_progress=show_progress)
        self._default_bm = bm
        # Reset the cache -- the default BM25 just went in.
        self._cache = OrderedDict()
        self._cache[(k1, b, method)] = bm

    # ─────────────────────────────────────────────────────────────────────
    # (k1, b) tuning + LRU
    # ─────────────────────────────────────────────────────────────────────

    def _get_bm(self, k1: float, b: float, method: str = "lucene") -> tuple[bm25s.BM25, bool]:
        """Return the BM25 instance for (k1, b); build one if missing.

        bm25s uses **eager BM25**: at ``index()`` time it precomputes
        the per-(doc, term) BM25 score into a sparse matrix using the
        k1, b that were set on the BM25 object. Mutating ``bm.k1`` /
        ``bm.b`` after the fact has no effect -- the precomputed
        scores are baked in. To change (k1, b) we must build a fresh
        ``bm25s.BM25`` instance and re-run ``index()``. That pass is
        O(corpus) and dominated by the per-doc BM25 score loop; on
        a 500K-doc corpus it costs ~30 seconds. The LRU-8 cache
        amortises that over repeat queries (typical Phase-7 workflow:
        user nudges k1/b, watches the metric, repeats).

        Returns ``(bm, cached)`` where ``cached`` is True if the entry
        was already in the LRU (cheap) and False if we had to build it
        (expensive -- reported to the caller for diagnostics).
        """
        key = (round(k1, 6), round(b, 6), method)  # rounding avoids float-equality misses
        bm = self._cache.get(key)
        if bm is not None:
            self._cache.move_to_end(key)
            return bm, True
        # Build a new one. bm25s's BM25.init is O(N) over the corpus,
        # not O(corpus_size^2) -- it's a single pass to compute the
        # BM25 scores that ``get_scores`` will sum up. With N=500K
        # docs and ~100 tokens/doc, this takes ~30 sec on 12 cores.
        # We cache the result.
        tok = bm25s.tokenization.Tokenized(ids=self.token_ids, vocab=dict(self.vocab))
        new_bm = bm25s.BM25(method=method, k1=key[0], b=key[1])
        new_bm.index(tok, show_progress=False)
        self._cache[key] = new_bm
        if len(self._cache) > BM25_CACHE_SIZE:
            self._cache.popitem(last=False)  # evict LRU
        return new_bm, False

    # ─────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_tokens: list[str],
        k: int = 10,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> tuple[list[BM25Hit], bool]:
        """Return the top-``k`` docs by BM25 score.

        Returns ``(hits, cached)`` where ``cached`` is True if the
        (k1, b) was served from the LRU.
        """
        if self._default_bm is None:
            raise RuntimeError("BM25Retriever.build() must be called before search().")
        if not query_tokens:
            return [], True
        bm, cached = self._get_bm(k1, b)
        q_ids = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not q_ids:
            return [], cached
        scores = bm.get_scores(q_ids)
        if scores is None or scores.size == 0:
            return [], cached
        # Top-k. argpartition O(n) is faster than argsort O(n log n).
        k = min(k, scores.shape[0])
        if k == scores.shape[0]:
            top_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k - 1)[:k]
            top_idx = part[np.argsort(-scores[part])]
        hits: list[BM25Hit] = []
        for rank, idx in enumerate(top_idx, start=1):
            s = float(scores[idx])
            if not math.isfinite(s) or s <= 0.0:
                break
            hits.append(BM25Hit(rank=rank, doc_id=self.doc_ids[idx], score=s))
        return hits, cached

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, dirpath: str | Path) -> None:
        """Save the retriever to ``dirpath``.

        Four artefacts:
          - ``bm25.pkl``           -- the default (k1=1.5, b=0.75) BM25 instance
          - ``bm25_token_ids.pkl`` -- the corpus in token-ID form (for re-tuning)
          - ``bm25_vocab.json``    -- term -> id mapping
          - ``doc_ids.json``       -- row-index -> doc_id mapping
        """
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        if self._default_bm is None:
            raise RuntimeError("BM25Retriever.build() must be called before save().")
        joblib.dump(self._default_bm, d / DEFAULT_BM25_FILENAME, compress=3)
        # token_ids: store as a list of lists; numpy arrays of inhomogeneous
        # shape don't pickle well via joblib.
        joblib.dump(self.token_ids, d / TOKEN_IDS_FILENAME, compress=3)
        (d / VOCAB_FILENAME).write_text(
            json.dumps(list(self.vocab.items()), ensure_ascii=False), encoding="utf-8"
        )
        (d / DOC_IDS_FILENAME).write_text(
            json.dumps(self.doc_ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> BM25Retriever:
        """Load a saved BM25Retriever from ``dirpath``."""
        d = Path(dirpath)
        for name in (DEFAULT_BM25_FILENAME, TOKEN_IDS_FILENAME, VOCAB_FILENAME, DOC_IDS_FILENAME):
            if not (d / name).exists():
                raise FileNotFoundError(f"{d / name} not found")
        default_bm = joblib.load(d / DEFAULT_BM25_FILENAME)
        token_ids = joblib.load(d / TOKEN_IDS_FILENAME)
        vocab_list = json.loads((d / VOCAB_FILENAME).read_text(encoding="utf-8"))
        vocab: OrderedDict[str, int] = OrderedDict(vocab_list)
        doc_ids = json.loads((d / DOC_IDS_FILENAME).read_text(encoding="utf-8"))
        ret = cls(doc_ids=doc_ids, vocab=vocab, token_ids=token_ids, default_bm=default_bm)
        # Warm the cache with the default (k1, b).
        ret._cache[(default_bm.k1, default_bm.b, default_bm.method)] = default_bm
        return ret

    # ─────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        if self._default_bm is None:
            return {"vocab_size": 0, "total_docs": 0, "cache_size": 0}
        return {
            "vocab_size": len(self.vocab),
            "total_docs": len(self.doc_ids),
            "cache_size": len(self._cache),
            "default_k1": self._default_bm.k1,
            "default_b": self._default_bm.b,
            "method": self._default_bm.method,
        }


def build_bm25(
    corpus_tokens: list[list[str]],
    doc_ids: list[str],
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
    show_progress: bool = True,
) -> BM25Retriever:
    """Convenience builder used by the build script and tests."""
    if show_progress:
        # bm25s prints its own progress bar; we only add a one-line
        # "starting" message before the call.
        print(f"[bm25] building index over {len(corpus_tokens):,} docs (k1={k1}, b={b})")
    r = BM25Retriever()
    r.build(corpus_tokens, doc_ids, k1=k1, b=b, show_progress=show_progress)
    if show_progress:
        print(f"[bm25] done. vocab={len(r.vocab):,}, cache=[({k1}, {b})]")
    return r


# Silence the unused-import warning for tqdm (kept for forward compat).
_ = tqdm

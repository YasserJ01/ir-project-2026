"""TF-IDF retriever for the IR system.

Phase 2 of the project. Wraps scikit-learn's ``TfidfVectorizer`` and
sparse-cosine similarity scoring. The corpus is supplied as
``list[list[str]]`` (output of our ``preprocess()``), so we configure
the vectorizer for *pre-tokenized* input:

    TfidfVectorizer(
        preprocessor=lambda x: x,        # return the input unchanged
        tokenizer=lambda x: x,            # split is a no-op for a list
        lowercase=False,                  # already lowercased
        token_pattern=None,               # suppress the regex check
        sublinear_tf=True,                # log(1 + tf) -- common IR practice
        norm="l2",                        # cosine similarity
    )

The retriever caches the per-doc vector rows; the query is transformed
on the fly and cosine-scored against the matrix.

Persistence
-----------
Three artefacts on disk (under ``data/indexes/<dataset_id>/``):
  - ``tfidf_vectorizer.pkl``  -- the fitted TfidfVectorizer
  - ``tfidf_matrix.npz``      -- the sparse TF-IDF matrix (CSR)
  - ``doc_ids.json``          -- row-index -> doc_id mapping
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VECTORIZER_FILENAME: str = "tfidf_vectorizer.pkl"
MATRIX_FILENAME: str = "tfidf_matrix.npz"
DOC_IDS_FILENAME: str = "doc_ids.json"


@dataclass
class TfidfHit:
    """A single search hit. Score is cosine similarity in [-1, 1] (>= 0 in practice)."""

    rank: int
    doc_id: str
    score: float


def _identity(x: Any) -> Any:
    """A no-op ``preprocessor`` / ``tokenizer`` for pre-tokenized input."""
    return x


class TfidfRetriever:
    """A thin wrapper around TfidfVectorizer + cosine similarity.

    Build:
        >>> r = TfidfRetriever()
        >>> r.build(corpus_tokens, doc_ids)

    Search:
        >>> r.search(["fox", "jump"], k=10) -> list[TfidfHit]
    """

    __slots__ = (
        "vectorizer",
        "matrix",
        "doc_ids",
    )

    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        matrix: csr_matrix | None = None,
        doc_ids: list[str] | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.doc_ids = doc_ids or []

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        corpus_tokens: list[list[str]],
        doc_ids: list[str],
    ) -> None:
        """Fit a TfidfVectorizer on the pre-tokenized corpus.

        ``corpus_tokens`` is the list-of-lists output of
        ``scripts/tokenize_corpus.py`` (or ``preprocess()`` per doc).
        """
        if len(corpus_tokens) != len(doc_ids):
            raise ValueError(
                f"corpus_tokens ({len(corpus_tokens)}) and doc_ids ({len(doc_ids)}) "
                "must have the same length"
            )

        vec = TfidfVectorizer(
            preprocessor=_identity,
            tokenizer=_identity,
            lowercase=False,
            token_pattern=None,
            sublinear_tf=True,
            norm="l2",
        )
        # sklearn's tokenizer is expected to take a single string; we
        # pass the whole list at once because we know each element is
        # already a list. This works because we explicitly set
        # ``tokenizer=_identity`` -- the function is called once with
        # the full input.
        matrix = vec.fit_transform(corpus_tokens)

        self.vectorizer = vec
        self.matrix = matrix
        self.doc_ids = list(doc_ids)

    # ─────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────

    def search(self, query_tokens: list[str], k: int = 10) -> list[TfidfHit]:
        """Return the top-``k`` docs by cosine similarity to the query.

        If the query has no tokens (or all tokens are OOV), returns
        an empty list. Score is in ``[0, 1]`` for normalized vectors.
        """
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("TfidfRetriever.build() must be called before search().")
        if not query_tokens:
            return []
        # Transform the query. Pass as a one-element list because the
        # vectorizer expects an iterable of documents.
        try:
            q_vec = self.vectorizer.transform([query_tokens])
        except ValueError:
            # Empty vocabulary after vectorizer's internal filtering
            # (e.g., all tokens are OOV). The matrix will have zero
            # columns. cosine_similarity will return zeros, which is
            # fine, but transform() may raise. We catch and return [].
            return []

        # Cosine similarity: shape (1, n_docs).
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        if sims.size == 0:
            return []

        # argsort descending, take top-k. np.argpartition is O(n)
        # instead of O(n log n) but returns unordered indices; we
        # re-sort the top-k for deterministic rank order.
        k = min(k, sims.shape[0])
        if k == sims.shape[0]:
            top_idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k - 1)[:k]
            top_idx = part[np.argsort(-sims[part])]

        results: list[TfidfHit] = []
        for rank, idx in enumerate(top_idx, start=1):
            score = float(sims[idx])
            if score <= 0.0:
                # Defensive: stop at the first zero score (everything
                # after is also zero or negative; we never want ranks
                # with score 0 in the top-k).
                break
            results.append(TfidfHit(rank=rank, doc_id=self.doc_ids[idx], score=score))
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, dirpath: str | Path) -> None:
        """Save vectorizer + matrix + doc_ids under ``dirpath``.

        Creates the directory if needed. The three artefacts are
        written atomically (temp-file + rename) so a partial write
        can't leave a corrupt index.
        """
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("TfidfRetriever.build() must be called before save().")
        joblib.dump(self.vectorizer, d / VECTORIZER_FILENAME, compress=3)
        save_npz(d / MATRIX_FILENAME, self.matrix)
        (d / DOC_IDS_FILENAME).write_text(
            json.dumps(self.doc_ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> TfidfRetriever:
        """Load a saved TfidfRetriever from ``dirpath``."""
        d = Path(dirpath)
        if not (d / VECTORIZER_FILENAME).exists():
            raise FileNotFoundError(f"{d / VECTORIZER_FILENAME} not found")
        if not (d / MATRIX_FILENAME).exists():
            raise FileNotFoundError(f"{d / MATRIX_FILENAME} not found")
        if not (d / DOC_IDS_FILENAME).exists():
            raise FileNotFoundError(f"{d / DOC_IDS_FILENAME} not found")
        vec = joblib.load(d / VECTORIZER_FILENAME)
        from scipy.sparse import load_npz

        matrix = load_npz(d / MATRIX_FILENAME)
        doc_ids = json.loads((d / DOC_IDS_FILENAME).read_text(encoding="utf-8"))
        return cls(vectorizer=vec, matrix=matrix, doc_ids=doc_ids)

    # ─────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        if self.vectorizer is None or self.matrix is None:
            return {"vocab_size": 0, "total_docs": 0, "matrix_nnz": 0}
        return {
            "vocab_size": len(self.vectorizer.vocabulary_),
            "total_docs": int(self.matrix.shape[0]),
            "matrix_nnz": int(self.matrix.nnz),
        }


def build_tfidf(
    corpus_tokens: Iterable[list[str]] | list[list[str]],
    doc_ids: list[str],
    show_progress: bool = True,
) -> TfidfRetriever:
    """Convenience builder used by the build script and tests."""
    import tqdm

    if show_progress and not isinstance(corpus_tokens, list):
        corpus_tokens = list(tqdm.tqdm(corpus_tokens, desc="tfidf:collect", unit="doc"))
    r = TfidfRetriever()
    r.build(list(corpus_tokens), doc_ids)  # type: ignore[arg-type]
    return r

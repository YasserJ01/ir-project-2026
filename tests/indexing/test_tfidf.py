"""Tests for the TfidfRetriever.

Uses a small in-memory corpus; the tests are fast and don't touch the
disk. For the save/load round-trip we use pytest's ``tmp_path``.
"""

from __future__ import annotations

import pytest

from services.indexing.app.tfidf import TfidfRetriever

# 5 docs, 4 unique terms.
CORPUS: list[list[str]] = [
    ["fox", "fox", "dog"],
    ["cat", "dog"],
    ["fox", "cat"],
    ["dog"],
    ["fox", "fox", "fox", "cat"],
]
DOC_IDS: list[str] = ["d1", "d2", "d3", "d4", "d5"]


def _build() -> TfidfRetriever:
    r = TfidfRetriever()
    r.build(CORPUS, DOC_IDS)
    return r


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────


def test_build_sets_attributes() -> None:
    r = _build()
    assert r.vectorizer is not None
    assert r.matrix is not None
    assert r.doc_ids == DOC_IDS
    # Matrix shape: (5 docs, 3 unique terms)
    assert r.matrix.shape == (5, 3)


def test_build_mismatched_lengths_raises() -> None:
    r = TfidfRetriever()
    with pytest.raises(ValueError, match="must have the same length"):
        r.build(CORPUS, ["d1"])


# ─────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────


def test_search_empty_query_returns_empty() -> None:
    r = _build()
    assert r.search([], k=10) == []


def test_search_returns_relevant_doc() -> None:
    r = _build()
    hits = r.search(["fox"], k=3)
    assert len(hits) > 0
    # d1, d3, d5 contain "fox" -- at least one of them should be top-3.
    top3_ids = {h.doc_id for h in hits}
    assert top3_ids & {"d1", "d3", "d5"}


def test_search_top_doc_is_highest_tf() -> None:
    # "fox" appears 2 in d1, 1 in d3, 3 in d5. d5 should win.
    r = _build()
    hits = r.search(["fox"], k=3)
    assert hits[0].doc_id == "d5"


def test_search_cosine_in_unit_range() -> None:
    r = _build()
    hits = r.search(["fox", "cat", "dog"], k=5)
    for h in hits:
        assert 0.0 <= h.score <= 1.0


def test_search_ranks_1_indexed_and_sequential() -> None:
    r = _build()
    hits = r.search(["fox"], k=3)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_search_k_caps_result_count() -> None:
    r = _build()
    hits = r.search(["fox"], k=1)
    assert len(hits) == 1


def test_search_oov_query_returns_empty() -> None:
    r = _build()
    # All OOV terms: vectorizer will produce an empty vector.
    hits = r.search(["nonexistent_term_xyz"], k=5)
    assert hits == []


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path) -> None:
    r = _build()
    r.save(tmp_path)
    loaded = TfidfRetriever.load(tmp_path)
    assert loaded.doc_ids == DOC_IDS
    assert loaded.vectorizer is not None
    assert loaded.matrix is not None
    # Shape preserved
    assert loaded.matrix.shape == r.matrix.shape


def test_save_load_round_trip_preserves_search_results(tmp_path) -> None:
    r = _build()
    r.save(tmp_path)
    loaded = TfidfRetriever.load(tmp_path)
    # Same query -> same top-1
    r_hits = r.search(["fox"], k=3)
    l_hits = loaded.search(["fox"], k=3)
    assert [h.doc_id for h in r_hits] == [h.doc_id for h in l_hits]
    # Same scores (within float tolerance)
    for a, b in zip(r_hits, l_hits, strict=True):
        assert a.score == pytest.approx(b.score, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────


def test_stats_after_build() -> None:
    r = _build()
    s = r.stats()
    assert s["vocab_size"] == 3
    assert s["total_docs"] == 5
    assert s["matrix_nnz"] > 0


def test_stats_before_build() -> None:
    r = TfidfRetriever()
    s = r.stats()
    assert s == {"vocab_size": 0, "total_docs": 0, "matrix_nnz": 0}

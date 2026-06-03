"""Tests for the BM25Retriever.

Uses a small in-memory corpus; the tests are fast and don't touch the
disk. For the save/load round-trip we use pytest's ``tmp_path``.

We also verify the (k1, b) sensitivity:
  - k1=0  -> score depends only on IDF
  - b=0   -> length normalization is disabled
  - default (1.5, 0.75) -> standard Lucene BM25Okapi behaviour
"""

from __future__ import annotations

import math

import pytest

from services.indexing.app.bm25 import BM25Retriever

CORPUS: list[list[str]] = [
    ["fox", "fox", "dog"],
    ["cat", "dog"],
    ["fox", "cat"],
    ["dog"],
    ["fox", "fox", "fox", "cat"],
]
DOC_IDS: list[str] = ["d1", "d2", "d3", "d4", "d5"]


def _build(k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    r = BM25Retriever()
    r.build(CORPUS, DOC_IDS, k1=k1, b=b, show_progress=False)
    return r


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────


def test_build_sets_attributes() -> None:
    r = _build()
    assert r.doc_ids == DOC_IDS
    assert r.vocab  # non-empty
    assert r.token_ids  # non-empty
    assert r._default_bm is not None
    # The default (k1, b, method) is in the cache
    assert (1.5, 0.75, "lucene") in r._cache


def test_build_mismatched_lengths_raises() -> None:
    r = BM25Retriever()
    with pytest.raises(ValueError, match="must have the same length"):
        r.build(CORPUS, ["d1"], show_progress=False)


# ─────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────


def test_search_empty_query_returns_empty() -> None:
    r = _build()
    hits, cached = r.search([], k=10)
    assert hits == []
    assert cached is True


def test_search_returns_relevant_doc() -> None:
    r = _build()
    hits, _ = r.search(["fox"], k=3)
    assert len(hits) > 0
    top3_ids = {h.doc_id for h in hits}
    assert top3_ids & {"d1", "d3", "d5"}


def test_search_oov_query_returns_empty() -> None:
    r = _build()
    hits, _ = r.search(["nonexistent_xyz"], k=5)
    assert hits == []


def test_search_k_caps_result_count() -> None:
    r = _build()
    hits, _ = r.search(["fox"], k=1)
    assert len(hits) == 1


def test_search_ranks_1_indexed_and_sequential() -> None:
    r = _build()
    hits, _ = r.search(["fox", "cat", "dog"], k=3)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_search_all_scores_positive_finite() -> None:
    r = _build()
    hits, _ = r.search(["fox", "cat", "dog"], k=5)
    for h in hits:
        assert math.isfinite(h.score)
        assert h.score > 0.0


# ─────────────────────────────────────────────────────────────────────────
# (k1, b) sensitivity
# ─────────────────────────────────────────────────────────────────────────


def test_k1_zero_makes_tf_irrelevant() -> None:
    # With k1=0, BM25 reduces to IDF * (1 if tf > 0 else 0).
    # So the score for a single term is the same regardless of which
    # doc contains it -- only the doc-freq matters.
    r = _build(k1=0.0, b=0.0)
    # Pass k1/b explicitly -- search() defaults are (1.5, 0.75).
    hits, _ = r.search(["fox"], k=3, k1=0.0, b=0.0)
    # d1, d3, d5 contain "fox". All three have the same score.
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores[0] == pytest.approx(scores[1])
    assert scores[1] == pytest.approx(scores[2])


def test_b_zero_disables_length_normalization() -> None:
    # With b=0, doc length has no effect on the score. The default
    # (b=0.75) penalizes longer docs; with b=0 it doesn't.
    r0 = _build(k1=1.5, b=0.0)
    r75 = _build(k1=1.5, b=0.75)
    hits0, _ = r0.search(["fox"], k=5, k1=1.5, b=0.0)
    hits75, _ = r75.search(["fox"], k=5, k1=1.5, b=0.75)
    # d5 (3 occurrences) wins in both cases.
    assert hits0[0].doc_id == "d5"
    assert hits75[0].doc_id == "d5"


def test_cache_hit_on_repeat_query() -> None:
    r = _build()
    _, cached1 = r.search(["fox"], k=3)
    _, cached2 = r.search(["fox"], k=3)
    assert cached1 is True
    assert cached2 is True


def test_cache_miss_on_new_k1_b() -> None:
    r = _build()
    _, cached1 = r.search(["fox"], k=3, k1=1.5, b=0.75)
    _, cached2 = r.search(["fox"], k=3, k1=1.2, b=0.5)
    # First call: default (1.5, 0.75) is in cache.
    assert cached1 is True
    # Second call: (1.2, 0.5) is a new pair -> miss.
    assert cached2 is False
    # Third call: (1.2, 0.5) is now in cache.
    _, cached3 = r.search(["fox"], k=3, k1=1.2, b=0.5)
    assert cached3 is True


def test_cache_lru_eviction() -> None:
    from services.indexing.app.config import BM25_CACHE_SIZE

    r = _build()
    # Fill the cache with BM25_CACHE_SIZE + 1 distinct (k1, b) pairs.
    for i in range(BM25_CACHE_SIZE + 1):
        k1 = 0.5 + i * 0.1
        r.search(["fox"], k=1, k1=k1, b=0.5)
    # Cache size should be capped.
    assert len(r._cache) <= BM25_CACHE_SIZE


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path) -> None:
    r = _build()
    r.save(tmp_path)
    loaded = BM25Retriever.load(tmp_path)
    assert loaded.doc_ids == DOC_IDS
    assert loaded.vocab == r.vocab
    # Search results should be identical for the default (k1, b).
    r_hits, _ = r.search(["fox"], k=3)
    l_hits, _ = loaded.search(["fox"], k=3)
    assert [h.doc_id for h in r_hits] == [h.doc_id for h in l_hits]
    for a, b in zip(r_hits, l_hits, strict=True):
        assert a.score == pytest.approx(b.score, rel=1e-6)


def test_save_load_preserves_default_k1_b(tmp_path) -> None:
    r = _build(k1=1.7, b=0.5)
    r.save(tmp_path)
    loaded = BM25Retriever.load(tmp_path)
    assert loaded._default_bm.k1 == pytest.approx(1.7)
    assert loaded._default_bm.b == pytest.approx(0.5)


def test_save_load_retunes_k1_b(tmp_path) -> None:
    """The whole point of the LRU is that we can rebuild BM25 on load."""
    r = _build()
    r.save(tmp_path)
    loaded = BM25Retriever.load(tmp_path)
    # New (k1, b) should still work -- the (1.2, 0.5) instance is built lazily.
    hits, cached = loaded.search(["fox"], k=3, k1=1.2, b=0.5)
    assert len(hits) > 0
    assert cached is False  # first time we see this pair
    # And a second call hits the cache.
    _, cached2 = loaded.search(["fox"], k=3, k1=1.2, b=0.5)
    assert cached2 is True


# ─────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────


def test_stats_after_build() -> None:
    r = _build()
    s = r.stats()
    assert s["vocab_size"] == 3  # fox, dog, cat
    assert s["total_docs"] == 5
    assert s["cache_size"] == 1
    assert s["default_k1"] == pytest.approx(1.5)
    assert s["default_b"] == pytest.approx(0.75)
    assert s["method"] == "lucene"


def test_stats_before_build() -> None:
    r = BM25Retriever()
    s = r.stats()
    assert s == {"vocab_size": 0, "total_docs": 0, "cache_size": 0}

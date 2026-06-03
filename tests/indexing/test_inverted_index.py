"""Tests for the InvertedIndex primitive.

Uses an in-memory fixture corpus so the tests are fast (< 100 ms total)
and don't touch the disk. The fixture is small enough that
min_df / max_df_ratio caps don't kick in -- we have separate
``test_min_df_drops_singletons`` and ``test_max_df_ratio_drops_common``
for the cap behaviour.
"""

from __future__ import annotations

import pytest

from services.indexing.app.inverted_index import InvertedIndex, Posting

# ─────────────────────────────────────────────────────────────────────────
# Fixture corpus
# ─────────────────────────────────────────────────────────────────────────

# 5 documents, 4 unique terms, clear term-frequency pattern.
# d1: fox, fox, dog          (fox=2, dog=1)
# d2: cat, dog               (cat=1, dog=1)
# d3: fox, cat               (fox=1, cat=1)
# d4: dog                    (dog=1)
# d5: fox, fox, fox, cat     (fox=3, cat=1)
FIXTURE: list[tuple[str, list[str]]] = [
    ("d1", ["fox", "fox", "dog"]),
    ("d2", ["cat", "dog"]),
    ("d3", ["fox", "cat"]),
    ("d4", ["dog"]),
    ("d5", ["fox", "fox", "fox", "cat"]),
]


def _build(cap_min_df: int = 1, cap_max_df_ratio: float = 1.0) -> InvertedIndex:
    idx = InvertedIndex(min_df=cap_min_df, max_df_ratio=cap_max_df_ratio)
    idx.build(FIXTURE)
    return idx


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────


def test_build_empty_corpus() -> None:
    idx = InvertedIndex()
    idx.build([])
    assert idx.total_docs == 0
    assert idx.avg_doc_length == 0.0
    assert len(idx) == 0
    assert idx.doc_lengths == {}


def test_build_single_doc() -> None:
    # min_df=1 here because the production default is 2 (drops singletons);
    # with a 1-doc corpus every term is a singleton.
    idx = InvertedIndex(min_df=1, max_df_ratio=1.0)
    idx.build([("d1", ["fox", "dog"])])
    assert idx.total_docs == 1
    assert idx.avg_doc_length == 2.0
    assert idx.length("d1") == 2
    assert idx.doc_count("fox") == 1
    assert idx.doc_count("dog") == 1
    assert idx.doc_count("missing") == 0


def test_build_multi_doc_tf_accumulation() -> None:
    idx = _build()
    # fox: d1=2, d3=1, d5=3
    assert idx.tf("fox", "d1") == 2
    assert idx.tf("fox", "d3") == 1
    assert idx.tf("fox", "d5") == 3
    # dog: d1=1, d2=1, d4=1
    assert idx.tf("dog", "d1") == 1
    assert idx.tf("dog", "d2") == 1
    assert idx.tf("dog", "d4") == 1
    # cat: d2=1, d3=1, d5=1
    assert idx.tf("cat", "d2") == 1
    assert idx.tf("cat", "d3") == 1
    assert idx.tf("cat", "d5") == 1


def test_build_postings_lengths() -> None:
    idx = _build()
    # df: fox=3, dog=3, cat=3
    assert idx.doc_count("fox") == 3
    assert idx.doc_count("dog") == 3
    assert idx.doc_count("cat") == 3


def test_build_avg_doc_length() -> None:
    idx = _build()
    # (3 + 2 + 2 + 1 + 4) / 5 = 12 / 5 = 2.4
    assert idx.avg_doc_length == pytest.approx(2.4)


# ─────────────────────────────────────────────────────────────────────────
# get_postings
# ─────────────────────────────────────────────────────────────────────────


def test_get_postings_returns_list_of_postings() -> None:
    idx = _build()
    plist = idx.get_postings("fox")
    assert isinstance(plist, list)
    assert all(isinstance(p, Posting) for p in plist)
    # 3 postings: d1, d3, d5 in insertion order
    assert [(p.doc_id, p.tf) for p in plist] == [("d1", 2), ("d3", 1), ("d5", 3)]


def test_get_postings_missing_term_returns_empty() -> None:
    idx = _build()
    assert idx.get_postings("nonexistent") == []


def test_get_postings_via_contains() -> None:
    idx = _build()
    assert "fox" in idx
    assert "nonexistent" not in idx
    assert idx.has_term("cat")
    assert not idx.has_term("nonexistent")


# ─────────────────────────────────────────────────────────────────────────
# Vocabulary cap
# ─────────────────────────────────────────────────────────────────────────


def test_min_df_drops_singletons() -> None:
    # A corpus with one singleton term.
    docs = [
        ("d1", ["a", "b"]),
        ("d2", ["b", "c"]),
        ("d3", ["a", "b", "d"]),
    ]
    idx = InvertedIndex(min_df=2, max_df_ratio=1.0)
    idx.build(docs)
    # a: 2, b: 3, c: 1 (singleton), d: 1 (singleton)
    assert "a" in idx
    assert "b" in idx
    assert "c" not in idx  # dropped (min_df=2)
    assert "d" not in idx  # dropped (min_df=2)


def test_max_df_ratio_drops_common() -> None:
    # A corpus where 'the' appears in 4 of 5 docs (df=4/5=0.8).
    docs = [
        ("d1", ["the", "fox"]),
        ("d2", ["the", "dog"]),
        ("d3", ["the", "cat"]),
        ("d4", ["the", "owl"]),
        ("d5", ["wolf"]),
    ]
    idx = InvertedIndex(min_df=1, max_df_ratio=0.5)
    idx.build(docs)
    # 'the' has df=4 > 0.5 * 5 = 2.5, so dropped.
    assert "the" not in idx
    # 'fox', 'dog', 'cat', 'owl' have df=1 each -- kept.
    assert "fox" in idx
    assert "wolf" in idx


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path) -> None:
    idx = _build()
    path = tmp_path / "inv.pkl"
    idx.save(path)
    assert path.exists()

    loaded = InvertedIndex.load(path)
    assert loaded.total_docs == idx.total_docs
    assert loaded.avg_doc_length == idx.avg_doc_length
    assert loaded.doc_lengths == idx.doc_lengths
    assert loaded.doc_freq == idx.doc_freq
    # inverted_index: dict-of-dicts, exact equality
    assert loaded.inverted_index == idx.inverted_index
    # Cap params round-trip too
    assert loaded.min_df == idx.min_df
    assert loaded.max_df_ratio == idx.max_df_ratio


def test_save_load_preserves_postings_order(tmp_path) -> None:
    idx = _build()
    path = tmp_path / "inv.pkl"
    idx.save(path)
    loaded = InvertedIndex.load(path)
    # Order is the build order
    assert [p.doc_id for p in loaded.get_postings("fox")] == ["d1", "d3", "d5"]


# ─────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────


def test_stats_returns_expected_keys() -> None:
    idx = _build()
    s = idx.stats()
    assert s["vocab_size"] == 3
    assert s["total_docs"] == 5
    assert s["avg_doc_length"] == pytest.approx(2.4)
    assert s["min_df"] == 1
    assert s["max_df_ratio"] == 1.0


def test_len_returns_vocab_size() -> None:
    idx = _build()
    assert len(idx) == 3  # fox, dog, cat


def test_doc_count_and_length_for_missing() -> None:
    idx = _build()
    assert idx.doc_count("missing") == 0
    assert idx.length("missing") == 0
    assert idx.tf("fox", "missing") == 0
    assert idx.tf("missing", "d1") == 0

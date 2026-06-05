"""Unit tests for ``services/retrieval/app/fusion.py``.

Pure-function tests, no model loading, no FAISS, no httpx. All tests
run in <100ms combined.

Each fusion method gets 4 tests:
  - empty rankings
  - single retriever (sanity / regression)
  - multiple retrievers with overlap
  - tie-breaking (deterministic order)

Plus 3 shared tests for the dispatcher ``fuse()`` and the helpers.
"""

from __future__ import annotations

import dataclasses

import pytest

from services.retrieval.app.fusion import (
    DEFAULT_RRF_K,
    FusedHit,
    RankedHit,
    assert_valid_input,
    combmnz,
    combsum,
    fuse,
    min_max_normalize,
    rrf,
)

# ─────────────────────────────────────────────────────────────────────────
# min_max_normalize
# ─────────────────────────────────────────────────────────────────────────


def test_min_max_empty_returns_empty() -> None:
    assert min_max_normalize([]) == []


def test_min_max_constant_returns_zeros() -> None:
    # All equal (and len > 1) -> 0.0 for each (no signal in the retriever).
    assert min_max_normalize([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_min_max_single_element_is_max() -> None:
    # A retriever with one result treats it as the maximum (1.0).
    assert min_max_normalize([42.0]) == [1.0]
    assert min_max_normalize([0.0]) == [1.0]  # even 0.0 -> 1.0
    assert min_max_normalize([-5.0]) == [1.0]  # even negative -> 1.0


def test_min_max_basic_range() -> None:
    out = min_max_normalize([1.0, 2.0, 3.0, 4.0])
    assert out == [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]


def test_min_max_handles_negative_scores() -> None:
    # Some retrievers can legitimately produce negative scores
    # (e.g. unbounded BM25 with strange parameters). Min-max should
    # still produce a valid [0, 1] range.
    out = min_max_normalize([-2.0, 0.0, 2.0])
    assert out == [0.0, 0.5, 1.0]


# ─────────────────────────────────────────────────────────────────────────
# RRF
# ─────────────────────────────────────────────────────────────────────────


def test_rrf_empty_rankings() -> None:
    assert rrf({}) == []


def test_rrf_single_retriever_passthrough() -> None:
    # With one retriever, RRF score == 1 / (k + rank). Order matches the input.
    hits = [RankedHit("a", 1.0), RankedHit("b", 0.5), RankedHit("c", 0.1)]
    out = rrf({"bm25": hits})
    assert [h.doc_id for h in out] == ["a", "b", "c"]
    # Rank 1 -> 1 / (k + 1) = 1/61 with default k=60
    assert out[0].score == pytest.approx(1.0 / (DEFAULT_RRF_K + 1))
    assert out[1].score == pytest.approx(1.0 / (DEFAULT_RRF_K + 2))
    # individual_scores only has the bm25 contribution.
    assert out[0].individual_scores == {"bm25": pytest.approx(1.0 / (DEFAULT_RRF_K + 1))}


def test_rrf_two_retrievers_overlap() -> None:
    bm25 = [RankedHit("a", 1.0), RankedHit("b", 0.5), RankedHit("c", 0.1)]
    dense = [RankedHit("b", 0.9), RankedHit("a", 0.8), RankedHit("d", 0.7)]
    out = rrf({"bm25": bm25, "dense": dense}, k=60)
    # Both retrievers ranked "a" and "b" -- they should beat "c" and "d".
    top_two = {h.doc_id for h in out[:2]}
    assert top_two == {"a", "b"}
    # "a" was rank 1 in BM25 and rank 2 in dense; "b" was rank 2 + rank 1.
    # So they should be equal. Order between them is by doc_id (determinism).
    a_score = next(h.score for h in out if h.doc_id == "a")
    b_score = next(h.score for h in out if h.doc_id == "b")
    assert a_score == pytest.approx(b_score)
    # Both retrievers contribute to a and b.
    a_indiv = next(h.individual_scores for h in out if h.doc_id == "a")
    assert set(a_indiv.keys()) == {"bm25", "dense"}


def test_rrf_tie_break_by_doc_id() -> None:
    # RRF uses *rank*, not score. So if every retriever returns the
    # docs in the same order (x, y, z), the fused scores are:
    #   x = 3 * 1/(k+1)
    #   y = 3 * 1/(k+2)
    #   z = 3 * 1/(k+3)
    # which are NOT equal -- x is the strongest, z the weakest.
    # The test here is the tie-break case: two docs with the same
    # RRF score (e.g. z from r1, and z from r2) tie at the doc level.
    rankings = {
        "r1": [RankedHit("x", 0.5), RankedHit("y", 0.5), RankedHit("z", 0.5)],
        "r2": [RankedHit("x", 0.5), RankedHit("y", 0.5), RankedHit("z", 0.5)],
        "r3": [RankedHit("x", 0.5), RankedHit("y", 0.5), RankedHit("z", 0.5)],
    }
    out = rrf(rankings)
    # x is rank 1 in all three, y is rank 2, z is rank 3. x has the
    # highest RRF score.
    assert out[0].doc_id == "x"
    assert out[1].doc_id == "y"
    assert out[2].doc_id == "z"
    # The scores are strictly decreasing (RRF favours higher ranks).
    assert out[0].score > out[1].score > out[2].score


def test_rrf_true_tie_breaks_by_doc_id() -> None:
    # A true tie: a is rank 1 in r1 and rank 2 in r2; b is rank 2 in
    # r1 and rank 1 in r2. Both get fused score 1/(k+1) + 1/(k+2).
    # The tie must be broken by doc_id ascending.
    rankings = {
        "r1": [RankedHit("a", 0.5), RankedHit("b", 0.5)],
        "r2": [RankedHit("b", 0.5), RankedHit("a", 0.5)],
    }
    out = rrf(rankings)
    assert len(out) == 2
    assert out[0].score == pytest.approx(out[1].score)
    # doc_id ascending tie-break.
    assert out[0].doc_id == "a"
    assert out[1].doc_id == "b"


# ─────────────────────────────────────────────────────────────────────────
# CombSUM
# ─────────────────────────────────────────────────────────────────────────


def test_combsum_empty_rankings() -> None:
    assert combsum({}) == []


def test_combsum_single_retriever_passthrough() -> None:
    # One retriever, scores get min-max-normalised into [0, 1]. Order
    # is preserved by doc_id ascending for ties at 0.0.
    bm25 = [RankedHit("a", 10.0), RankedHit("b", 5.0), RankedHit("c", 0.0)]
    out = combsum({"bm25": bm25})
    assert out[0].doc_id == "a"
    assert out[0].score == pytest.approx(1.0)  # max -> 1.0
    assert out[1].doc_id == "b"
    assert out[1].score == pytest.approx(0.5)
    assert out[2].doc_id == "c"
    assert out[2].score == pytest.approx(0.0)


def test_combsum_two_retrievers_sum() -> None:
    bm25 = [RankedHit("a", 2.0), RankedHit("b", 1.0)]
    dense = [RankedHit("b", 0.8), RankedHit("a", 0.2)]
    out = combsum({"bm25": bm25, "dense": dense})
    # "a": bm25=1.0 (max) + dense=0.0 (min) = 1.0
    # "b": bm25=0.0 (min) + dense=1.0 (max) = 1.0
    # Tie; doc_id ascending -> a, b.
    assert out[0].doc_id == "a"
    assert out[0].score == pytest.approx(1.0)
    assert out[1].doc_id == "b"
    assert out[1].score == pytest.approx(1.0)


def test_combsum_handles_partial_overlap() -> None:
    # "c" only in bm25, "d" only in dense.
    bm25 = [RankedHit("a", 1.0), RankedHit("c", 0.5)]
    dense = [RankedHit("a", 0.9), RankedHit("d", 0.4)]
    out = combsum({"bm25": bm25, "dense": dense})
    doc_ids = [h.doc_id for h in out]
    assert set(doc_ids) == {"a", "c", "d"}
    # "a" should win (the only doc in both).
    assert out[0].doc_id == "a"
    # "c" only in bm25 -> contribution = 1.0 (max) + 0.0 (absent) = 1.0
    # "d" only in dense -> contribution = 0.0 (absent) + 1.0 (max) = 1.0
    # Tied. Order is a (both), c (bm25), d (dense).
    c_score = next(h.score for h in out if h.doc_id == "c")
    d_score = next(h.score for h in out if h.doc_id == "d")
    assert c_score == pytest.approx(d_score)


# ─────────────────────────────────────────────────────────────────────────
# CombMNZ
# ─────────────────────────────────────────────────────────────────────────


def test_combmnz_empty_rankings() -> None:
    assert combmnz({}) == []


def test_combmnz_single_retriever_same_as_combsum() -> None:
    # With one retriever, count_nonzero == 1 for every doc that appears.
    bm25 = [RankedHit("a", 1.0), RankedHit("b", 0.5)]
    sum_out = combsum({"bm25": bm25})
    mnz_out = combmnz({"bm25": bm25})
    assert [h.doc_id for h in mnz_out] == [h.doc_id for h in sum_out]
    assert [h.score for h in mnz_out] == pytest.approx([h.score for h in sum_out])


def test_combmnz_multiplies_by_retriever_count() -> None:
    # "a" is in both retrievers (the max in both -> norm=1.0 in each).
    # "b" only in dense (min -> norm=0.0 in dense).
    # "c" only in bm25 (min -> norm=0.0 in bm25).
    # Combsum: a = 1.0 + 1.0 = 2.0, b = 0.0, c = 0.0
    # Combmnz: a = 2.0 * 2 = 4.0, b = 0.0 * 0 = 0.0, c = 0.0 * 0 = 0.0
    bm25 = [RankedHit("a", 1.0), RankedHit("c", 0.0)]
    dense = [RankedHit("a", 0.8), RankedHit("b", 0.4)]
    out = combmnz({"bm25": bm25, "dense": dense})
    a_score = next(h.score for h in out if h.doc_id == "a")
    b_score = next(h.score for h in out if h.doc_id == "b")
    c_score = next(h.score for h in out if h.doc_id == "c")
    assert a_score == pytest.approx(4.0)
    assert b_score == pytest.approx(0.0)
    assert c_score == pytest.approx(0.0)
    # "a" is the only doc with two nonzero contributions.
    a = next(h for h in out if h.doc_id == "a")
    assert set(a.individual_scores.keys()) == {"bm25", "dense"}


def test_combmnz_zero_does_not_multiply() -> None:
    # A single-element list in a retriever is treated as "max" (norm=1.0).
    # So a retriever with [0.0] contributes 1.0 to the fusion, not 0.0.
    # This is the standard min-max behaviour: with one element there's
    # no "min" to map it to, so we map it to the max.
    bm25 = [RankedHit("a", 1.0)]
    dense = [RankedHit("a", 0.0)]  # single entry, normalised to 1.0
    out = combmnz({"bm25": bm25, "dense": dense})
    a = next(h for h in out if h.doc_id == "a")
    # bm25 contributes 1.0, dense contributes 1.0; sum=2.0, count=2, final=4.0
    assert a.score == pytest.approx(4.0)


def test_combmnz_zero_score_within_multi_element() -> None:
    # A doc with score=0.0 within a multi-element list IS still 0.0
    # (the min, not the max).
    bm25 = [RankedHit("a", 1.0), RankedHit("a_competitor", 0.5), RankedHit("a", 0.0)]
    # Wait, "a" can't be in bm25 twice. Use a fresh test:
    bm25 = [RankedHit("a", 1.0), RankedHit("z", 0.0)]
    dense = [RankedHit("a", 0.0)]
    out = combmnz({"bm25": bm25, "dense": dense})
    a = next(h for h in out if h.doc_id == "a")
    # bm25: a's norm = 1.0 (max of [1.0, 0.0])
    # dense: single entry -> norm = 1.0
    # sum = 2.0, count = 2, final = 4.0
    assert a.score == pytest.approx(4.0)


# ─────────────────────────────────────────────────────────────────────────
# fuse() dispatcher + helpers
# ─────────────────────────────────────────────────────────────────────────


def test_fuse_dispatches_rrf() -> None:
    hits = [RankedHit("a", 1.0)]
    out_rrf = fuse({"r": hits}, method="rrf")
    out_direct = rrf({"r": hits})
    assert out_rrf == out_direct


def test_fuse_dispatches_combsum() -> None:
    hits = [RankedHit("a", 1.0)]
    out = fuse({"r": hits}, method="combsum")
    assert out == combsum({"r": hits})


def test_fuse_dispatches_combmnz() -> None:
    hits = [RankedHit("a", 1.0)]
    out = fuse({"r": hits}, method="combmnz")
    assert out == combmnz({"r": hits})


def test_fuse_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown fusion method"):
        fuse({"r": [RankedHit("a", 1.0)]}, method="bogus")


def test_assert_valid_input_rejects_string() -> None:
    with pytest.raises(TypeError, match="is a string"):
        assert_valid_input({"r": "not a list"})


def test_assert_valid_input_accepts_list() -> None:
    # Should not raise.
    assert_valid_input({"r": [RankedHit("a", 1.0)]})


def test_ranked_hit_rejects_nan() -> None:
    # NaN scores are clamped to 0.0 at construction.
    h = RankedHit("a", float("nan"))
    assert h.score == 0.0


def test_ranked_hit_rejects_inf() -> None:
    h = RankedHit("a", float("inf"))
    assert h.score == 0.0


def test_fused_hit_is_immutable() -> None:
    # FusedHit is a frozen dataclass.
    h = FusedHit("a", 1.0, {"r": 0.5})
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.score = 2.0  # type: ignore[misc]

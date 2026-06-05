"""Score-fusion algorithms for parallel hybrid retrieval.

Three pure functions implementing the three fusion methods called out in
the guide §5.3:

* :func:`rrf` -- Reciprocal Rank Fusion: ``score(d) = sum 1 / (k + rank_i(d))``
* :func:`combsum` -- CombSUM: sum of per-retriever scores, after
  min-max-normalising each retriever's scores into ``[0, 1]``.
* :func:`combmnz` -- CombMNZ: same as CombSUM, then multiplied by the
  number of retrievers that gave a non-zero score.

All three functions take a list of "ranked lists" (one per retriever) and
return a single list of ``(doc_id, score)`` tuples, sorted by descending
score. Doc ids that did not appear in a retriever's output are treated as
having a score of 0 for that retriever (so the fusion is well-defined
even when the candidate sets differ).

This module is **pure** -- no I/O, no global state, no exceptions raised
for edge cases. The smallest possible contract that makes the rest of
the hybrid orchestrator easy to test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# Default k for RRF. The original paper (Cormack et al., 2009) suggests
# k=60 as a near-universal default. TREC experiments have shown k=10 to
# k=60 are roughly equivalent; we use the guide's recommendation.
DEFAULT_RRF_K: int = 60

__all__ = [
    "RankedHit",
    "FusedHit",
    "rrf",
    "combsum",
    "combmnz",
    "min_max_normalize",
    "fuse",
]


@dataclass(frozen=True)
class RankedHit:
    """A single retriever's hit list (in rank order, score descending)."""

    doc_id: str
    score: float

    def __post_init__(self) -> None:
        # NaN / Inf in the input would poison the fusion. We clamp to 0
        # which is the "did not match" semantic.
        if not math.isfinite(self.score):
            object.__setattr__(self, "score", 0.0)


@dataclass(frozen=True)
class FusedHit:
    """A fused hit returned by the fusion functions."""

    doc_id: str
    score: float
    # Per-retriever contribution: {"bm25": 0.85, "dense": 0.81, ...}.
    individual_scores: dict[str, float]


def _clamp_score(score: float) -> float:
    """Defensive: NaN/Inf -> 0 so fusion never produces NaN outputs."""
    if not math.isfinite(score):
        return 0.0
    return float(score)


def min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalise ``scores`` into ``[0, 1]``.

    Edge cases:

    * Empty list -> ``[]``.
    * Single element -> ``[1.0]`` (a retriever with one hit treats that
      hit as the maximum -- it carries the full signal).
    * Multiple equal elements -> ``[0.0, 0.0, ...]`` (no signal in
      a constant retriever).
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:  # all equal (and len > 1)
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rrf(
    rankings: dict[str, list[RankedHit]],
    k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    For each doc_id d, computes::

        score(d) = sum_{r in retrievers} 1 / (k + rank_r(d))

    where ``rank_r(d)`` is 1-based. Docs not in a retriever's output
    are skipped (treated as rank = infinity, contribution 0).

    Parameters
    ----------
    rankings
        ``{retriever_name: [RankedHit, ...]}``. Each list is assumed
        to be in descending score order.
    k
        The smoothing constant. Default 60 per the guide.
    """
    if k < 0:
        raise ValueError(f"RRF k must be non-negative; got {k}")
    fused: dict[str, float] = {}
    indiv: dict[str, dict[str, float]] = {}
    for retriever_name, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit.doc_id
            contribution = 1.0 / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + contribution
            indiv.setdefault(doc_id, {})[retriever_name] = contribution
    return _sort_fused(fused, indiv)


def combsum(
    rankings: dict[str, list[RankedHit]],
) -> list[FusedHit]:
    """CombSUM: sum of min-max-normalised scores across retrievers.

    For each doc_id d, computes::

        normalised_score_r(d) = (score_r(d) - min_r) / (max_r - min_r)
        score(d) = sum_r normalised_score_r(d)

    Docs not in a retriever's output get a contribution of 0.0
    (the post-normalisation minimum).
    """
    return _combn(rankings, multiply_by_count=False)


def combmnz(
    rankings: dict[str, list[RankedHit]],
) -> list[FusedHit]:
    """CombMNZ: CombSUM multiplied by the number of non-zero contributions.

    For each doc_id d, computes::

        score(d) = sum_r normalised_score_r(d) * count_nonzero(d)
    """
    return _combn(rankings, multiply_by_count=True)


def _combn(
    rankings: dict[str, list[RankedHit]],
    multiply_by_count: bool,
) -> list[FusedHit]:
    """Shared body for CombSUM and CombMNZ."""
    fused: dict[str, float] = {}
    indiv: dict[str, dict[str, float]] = {}
    for retriever_name, hits in rankings.items():
        if not hits:
            continue
        scores = [_clamp_score(h.score) for h in hits]
        normed = min_max_normalize(scores)
        for hit, norm_score in zip(hits, normed, strict=True):
            doc_id = hit.doc_id
            fused[doc_id] = fused.get(doc_id, 0.0) + norm_score
            indiv.setdefault(doc_id, {})[retriever_name] = norm_score
    if multiply_by_count:
        for doc_id in list(fused.keys()):
            count = sum(1 for v in indiv[doc_id].values() if v > 0.0)
            fused[doc_id] = fused[doc_id] * count
    return _sort_fused(fused, indiv)


def _sort_fused(
    fused: dict[str, float],
    indiv: dict[str, dict[str, float]],
) -> list[FusedHit]:
    """Return the fused map as a list of ``FusedHit`` sorted by score desc.

    Ties are broken by doc_id (ascending) for determinism -- important
    for the Phase 9 evaluation harness, which compares two TREC runs
    bit-by-bit.
    """
    out = [
        FusedHit(doc_id=d, score=s, individual_scores=indiv.get(d, {})) for d, s in fused.items()
    ]
    out.sort(key=lambda h: (-h.score, h.doc_id))
    return out


def fuse(
    rankings: dict[str, list[RankedHit]],
    method: str = "rrf",
    rrf_k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Dispatch to the right fusion function based on ``method``.

    ``method`` is one of ``"rrf"``, ``"combsum"``, ``"combmnz"``.
    Unknown methods raise ``ValueError`` (the caller -- the
    orchestrator -- validates against the ``FusionMethod`` Literal
    before we get here, so this is a defensive belt-and-braces).
    """
    method_norm = method.strip().lower()
    if method_norm == "rrf":
        return rrf(rankings, k=rrf_k)
    if method_norm == "combsum":
        return combsum(rankings)
    if method_norm == "combmnz":
        return combmnz(rankings)
    raise ValueError(f"Unknown fusion method: {method!r}. Expected one of: rrf, combsum, combmnz.")


def assert_valid_input(rankings: dict[str, Iterable[RankedHit]]) -> None:
    """Defensive check: every entry in ``rankings`` must be a non-string iterable.

    Python 2's ``str`` is iterable, and a typo like ``"bm25"`` instead of
    ``[RankedHit(...)]`` would otherwise silently iterate characters.
    """
    for name, hits in rankings.items():
        if isinstance(hits, (str, bytes)):
            raise TypeError(
                f"ranking for retriever {name!r} is a string; expected a list of RankedHit"
            )

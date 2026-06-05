"""Unit tests for ``services/retrieval/app/hybrid.py``.

The orchestrator depends on two external HTTP clients (indexing
service :8002, refinement service :8004) and a dense search function.
We mock all three:

  * :class:`FakeIndexingClient` -- a drop-in for :class:`IndexingClient`
    that returns canned results.
  * :class:`FakeRefinementClient` -- a drop-in for :class:`RefinementClient`.
  * ``fake_dense`` -- a closure that mimics the dense encoder+FAISS path.

The test corpus is a tiny 5-doc set with deterministic tokens so
assertions are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from services.retrieval.app.hybrid import (
    HybridOrchestrator,
    HybridOrchestratorError,
    _personalization_scalar,
    build_orchestrator,
)
from shared.ir_common.schemas import (
    HybridSearchRequest,
    RefinedToken,
    RefineRequest,
    RefineResponse,
)

# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class FakeIndexingClient:
    """In-memory :class:`IndexingClient` that returns canned results."""

    bm25_results: list[dict[str, Any]] = field(default_factory=list)
    tfidf_results: list[dict[str, Any]] = field(default_factory=list)
    reachable_flag: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def aclose(self) -> None:
        pass

    async def lexical_search(
        self,
        dataset_id: str,
        query_tokens: list[str],
        model: str,
        k: int,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "dataset_id": dataset_id,
                "query_tokens": list(query_tokens),
                "model": model,
                "k": k,
                "k1": k1,
                "b": b,
            }
        )
        if model == "bm25":
            return [h for h in self.bm25_results if h["rank"] <= k]
        if model == "tfidf":
            return [h for h in self.tfidf_results if h["rank"] <= k]
        raise ValueError(f"unknown model {model}")

    async def reachable(self) -> bool:
        return self.reachable_flag


@dataclass
class FakeRefinementClient:
    """In-memory :class:`RefinementClient`."""

    response: RefineResponse | None = None
    reachable_flag: bool = True
    raise_on_call: bool = False
    calls: list[RefineRequest] = field(default_factory=list)

    async def aclose(self) -> None:
        pass

    async def refine(self, req: RefineRequest) -> RefineResponse:
        self.calls.append(req)
        if self.raise_on_call:
            raise HybridOrchestratorError("refinement down", status_code=502)
        assert self.response is not None
        return self.response

    async def reachable(self) -> bool:
        return self.reachable_flag


async def fake_dense(
    query_text: str,
    dataset_id: str,
    k: int,
    model_name: str | None,
) -> tuple[list[float], list[str]]:
    """A deterministic dense search.

    Ranks docs by how many of their tokens appear in the query. This
    is not a real dense encoder but it gives a stable, meaningful
    ordering for the tests.
    """
    # Tiny per-dataset corpus (same for both test datasets so we can
    # run the same assertions for touche2020 and nq).
    corpus: dict[str, list[str]] = {
        "touche2020": {
            "d1": ["fox", "dog"],
            "d2": ["cat", "dog"],
            "d3": ["fox", "cat"],
            "d4": ["dog"],
            "d5": ["fox", "fox", "fox", "cat"],
        },
        "nq": {
            "d1": ["capital", "france"],
            "d2": ["eiffel", "tower"],
            "d3": ["brexit", "timeline"],
            "d4": ["climate", "change"],
            "d5": ["paris", "city"],
        },
    }
    ds_corpus = corpus.get(dataset_id, corpus["touche2020"])
    q_tokens = set(query_text.lower().split())
    scored: list[tuple[float, str]] = []
    for doc_id, tokens in ds_corpus.items():
        overlap = sum(1 for t in tokens if t in q_tokens)
        if overlap > 0:
            scored.append((float(overlap), doc_id))
    scored.sort(key=lambda t: (-t[0], t[1]))
    scores = [s for s, _ in scored[:k]]
    doc_ids = [d for _, d in scored[:k]]
    return scores, doc_ids


def make_orchestrator(
    bm25: list[dict[str, Any]] | None = None,
    tfidf: list[dict[str, Any]] | None = None,
    refine_response: RefineResponse | None = None,
    refine_unreachable: bool = False,
) -> tuple[HybridOrchestrator, FakeIndexingClient, FakeRefinementClient]:
    """Build an orchestrator with the three fakes wired in."""
    idx = FakeIndexingClient(
        bm25_results=bm25 or [],
        tfidf_results=tfidf or [],
    )
    ref = FakeRefinementClient(
        response=refine_response,
        raise_on_call=refine_unreachable,
    )
    orch = HybridOrchestrator(
        dense_search_fn=fake_dense,
        indexing_client=idx,  # type: ignore[arg-type]
        refinement_client=ref,  # type: ignore[arg-type]
    )
    return orch, idx, ref


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def bm25_results() -> list[dict[str, Any]]:
    """BM25: d5 wins, then d1, then d3 (with reasonable scores)."""
    return [
        {"rank": 1, "doc_id": "d5", "score": 4.2},
        {"rank": 2, "doc_id": "d1", "score": 3.1},
        {"rank": 3, "doc_id": "d3", "score": 1.8},
        {"rank": 4, "doc_id": "d2", "score": 0.7},
    ]


@pytest.fixture
def tfidf_results() -> list[dict[str, Any]]:
    return [
        {"rank": 1, "doc_id": "d5", "score": 0.9},
        {"rank": 2, "doc_id": "d3", "score": 0.5},
        {"rank": 3, "doc_id": "d1", "score": 0.3},
    ]


# ─────────────────────────────────────────────────────────────────────────
# _personalization_scalar
# ─────────────────────────────────────────────────────────────────────────


def test_personalization_scalar_no_tokens_returns_one() -> None:
    assert _personalization_scalar([]) == 1.0


def test_personalization_scalar_no_boosts_returns_one() -> None:
    tokens = [RefinedToken(token="foo", weight=1.0), RefinedToken(token="bar", weight=1.0)]
    assert _personalization_scalar(tokens) == 1.0


def test_personalization_scalar_with_boosts() -> None:
    # 3 tokens, one with weight 2.0 (boost of 1.0). Average boost = 1/3.
    # Scalar = 1 + 1/3 = 1.333...
    tokens = [
        RefinedToken(token="foo", weight=1.0),
        RefinedToken(token="eiffel", weight=2.0),
        RefinedToken(token="bar", weight=1.0),
    ]
    assert _personalization_scalar(tokens) == pytest.approx(1.0 + 1.0 / 3.0)


def test_personalization_scalar_multiple_boosts() -> None:
    # All 3 tokens have weight 2.0. Average boost = 1.0. Scalar = 2.0.
    tokens = [
        RefinedToken(token="a", weight=2.0),
        RefinedToken(token="b", weight=2.0),
        RefinedToken(token="c", weight=2.0),
    ]
    assert _personalization_scalar(tokens) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────
# Single-retriever strategies
# ─────────────────────────────────────────────────────────────────────────


async def test_tfidf_dispatches_to_indexing_service(
    bm25_results: list[dict[str, Any]],
    tfidf_results: list[dict[str, Any]],
) -> None:
    orch, idx, _ = make_orchestrator(bm25=bm25_results, tfidf=tfidf_results)
    req = HybridSearchRequest(
        query="fox cat",
        representation="tfidf",
        k=3,
    )
    resp = await orch.search("touche2020", req)
    assert resp.representation == "tfidf"
    assert resp.k == 3
    assert len(resp.results) == 3
    assert [h.doc_id for h in resp.results] == ["d5", "d3", "d1"]
    # The fake indexing client should have been called once with model="tfidf".
    assert len(idx.calls) == 1
    assert idx.calls[0]["model"] == "tfidf"
    assert idx.calls[0]["k"] == 3


async def test_bm25_dispatches_to_indexing_service(
    bm25_results: list[dict[str, Any]],
) -> None:
    orch, idx, _ = make_orchestrator(bm25=bm25_results)
    req = HybridSearchRequest(
        query="fox dog",
        representation="bm25",
        k=4,
        bm25_k1=1.2,
        bm25_b=0.5,
    )
    resp = await orch.search("touche2020", req)
    assert resp.representation == "bm25"
    assert len(resp.results) == 4
    # The k1, b should be echoed in the response.
    assert resp.bm25_k1 == 1.2
    assert resp.bm25_b == 0.5
    # The call should have the same k1, b.
    assert idx.calls[0]["k1"] == 1.2
    assert idx.calls[0]["b"] == 0.5


async def test_embedding_uses_injected_dense() -> None:
    orch, idx, _ = make_orchestrator()
    req = HybridSearchRequest(query="fox cat", representation="embedding", k=2)
    resp = await orch.search("touche2020", req)
    assert resp.representation == "embedding"
    # The fake_dense ranks by token overlap. "fox cat" query has:
    #   d5 (fox*3, cat) -> overlap 4
    #   d3 (fox, cat)  -> overlap 2
    #   d1 (fox, dog)  -> overlap 1
    #   d2 (cat, dog)  -> overlap 1
    # Top-2: d5, d3.
    assert [h.doc_id for h in resp.results] == ["d5", "d3"]
    # No HTTP calls.
    assert idx.calls == []


# ─────────────────────────────────────────────────────────────────────────
# hybrid_serial
# ─────────────────────────────────────────────────────────────────────────


async def test_hybrid_serial_runs_bm25_then_dense(
    bm25_results: list[dict[str, Any]],
) -> None:
    orch, idx, _ = make_orchestrator(bm25=bm25_results)
    req = HybridSearchRequest(
        query="fox dog",
        representation="hybrid_serial",
        k=3,
        candidate_k=20,
    )
    resp = await orch.search("touche2020", req)
    assert resp.representation == "hybrid_serial"
    # The fake_dense ranks by token overlap. "fox dog" -> d5 (overlap 3),
    # d1 (overlap 2), d3/d2/d4 (overlap 1 each, doc_id ascending).
    # Top-3 from dense: d5, d1, d2.
    assert len(resp.results) == 3
    assert [h.doc_id for h in resp.results] == ["d5", "d1", "d2"]
    # Per-retriever timings recorded.
    assert "bm25" in resp.per_retriever_latency_ms
    assert "dense" in resp.per_retriever_latency_ms
    # The BM25 call used candidate_k (20), not k (3).
    assert idx.calls[0]["k"] == 20


# ─────────────────────────────────────────────────────────────────────────
# hybrid_parallel
# ─────────────────────────────────────────────────────────────────────────


async def test_hybrid_parallel_rrf(bm25_results: list[dict[str, Any]]) -> None:
    orch, _, _ = make_orchestrator(bm25=bm25_results)
    req = HybridSearchRequest(
        query="fox dog",
        representation="hybrid_parallel",
        fusion="rrf",
        k=4,
    )
    resp = await orch.search("touche2020", req)
    assert resp.representation == "hybrid_parallel"
    assert resp.fusion == "rrf"
    assert len(resp.results) == 4
    # All 4 hits should have both retrievers (BM25 top-4 = d5,d1,d3,d2;
    # dense top-4 = d5,d1,d2,d3 since d2 and d3 both have overlap 1
    # and tie-break by doc_id -> d2, d3).
    for h in resp.results:
        assert "bm25" in h.individual_scores
        assert "dense" in h.individual_scores


async def test_hybrid_parallel_combsum(bm25_results: list[dict[str, Any]]) -> None:
    orch, _, _ = make_orchestrator(bm25=bm25_results)
    req = HybridSearchRequest(
        query="fox dog",
        representation="hybrid_parallel",
        fusion="combsum",
        k=3,
    )
    resp = await orch.search("touche2020", req)
    assert resp.fusion == "combsum"
    assert len(resp.results) == 3


async def test_hybrid_parallel_combmnz(bm25_results: list[dict[str, Any]]) -> None:
    orch, _, _ = make_orchestrator(bm25=bm25_results)
    req = HybridSearchRequest(
        query="fox dog",
        representation="hybrid_parallel",
        fusion="combmnz",
        k=3,
    )
    resp = await orch.search("touche2020", req)
    assert resp.fusion == "combmnz"
    assert len(resp.results) == 3


# ─────────────────────────────────────────────────────────────────────────
# Mode = with_features
# ─────────────────────────────────────────────────────────────────────────


def _refine_resp(query: str = "eiffel tower height") -> RefineResponse:
    """A canned refinement response with 2 boosted tokens."""
    return RefineResponse(
        query=query,
        refined_query="eiffel tower height",
        expanded_query="eiffel tower height",
        tokens=["eiffel", "tower", "height"],
        weighted_tokens=[
            RefinedToken(token="eiffel", weight=2.0),
            RefinedToken(token="tower", weight=2.0),
            RefinedToken(token="height", weight=1.0),
        ],
        stages={"spell": "", "synonyms": "", "personalization": "boosted: eiffel=2, tower=2"},
        latency_ms=10,
        user_id="user_1",
    )


async def test_mode_with_features_calls_refinement(
    bm25_results: list[dict[str, Any]],
) -> None:
    orch, _, ref = make_orchestrator(
        bm25=bm25_results,
        refine_response=_refine_resp(),
    )
    req = HybridSearchRequest(
        query="eiffel tower height",
        mode="with_features",
        user_id="user_1",
        representation="bm25",
        k=4,
    )
    resp = await orch.search("touche2020", req)
    # Refinement was called.
    assert len(ref.calls) == 1
    assert ref.calls[0].user_id == "user_1"
    # The refined query is echoed.
    assert resp.refined_query == "eiffel tower height"
    # BM25 scores were scaled by the personalization scalar.
    # Tokens: eiffel=2, tower=2, height=1. Boosts: [1.0, 1.0, 0.0].
    # Average boost = 2.0 / 3 ≈ 0.667. Scalar = 1.667.
    # d5 score: 4.2 * 1.667 = 7.0
    # d1 score: 3.1 * 1.667 = 5.17
    # The ranking is preserved (all scores scaled uniformly), so d5 is
    # still rank 1.
    assert resp.results[0].doc_id == "d5"
    assert resp.results[0].score == pytest.approx(4.2 * (1.0 + 2.0 / 3.0))
    assert not resp.refinement_fell_back
    assert "refine" in resp.stages


async def test_mode_with_features_falls_back_when_refinement_down(
    bm25_results: list[dict[str, Any]],
) -> None:
    orch, _, _ = make_orchestrator(
        bm25=bm25_results,
        refine_unreachable=True,
    )
    req = HybridSearchRequest(
        query="fox dog",
        mode="with_features",
        representation="bm25",
        k=4,
    )
    resp = await orch.search("touche2020", req)
    # Search still works, but with the basic-mode path.
    assert len(resp.results) == 4
    assert resp.refinement_fell_back is True
    assert "FALLBACK" in resp.stages.get("refine", "")


# ─────────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────────


async def test_indexing_service_error_propagates() -> None:
    """If :8002 returns 500, the orchestrator raises HybridOrchestratorError."""

    class _BoomIndexing(FakeIndexingClient):
        async def lexical_search(
            self,
            dataset_id: str,
            query_tokens: list[str],
            model: str,
            k: int,
            k1: float | None = None,
            b: float | None = None,
        ) -> list[dict[str, Any]]:
            raise HybridOrchestratorError("indexing down", status_code=502)

    orch = HybridOrchestrator(
        dense_search_fn=fake_dense,
        indexing_client=_BoomIndexing(),  # type: ignore[arg-type]
        refinement_client=FakeRefinementClient(),  # type: ignore[arg-type]
    )
    req = HybridSearchRequest(query="fox", representation="bm25", k=3)
    with pytest.raises(HybridOrchestratorError, match="indexing down"):
        await orch.search("touche2020", req)


# ─────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────


async def test_health_reports_sub_service_status() -> None:
    orch, idx, ref = make_orchestrator()
    idx.reachable_flag = True
    ref.reachable_flag = False
    h = await orch.health("touche2020")
    assert h["bm25_endpoint_reachable"] is True
    assert h["refinement_endpoint_reachable"] is False
    assert h["dataset_id"] == "touche2020"


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def test_build_orchestrator_factory() -> None:
    orch = build_orchestrator(dense_search_fn=fake_dense)
    assert isinstance(orch, HybridOrchestrator)
    assert orch._dense is fake_dense


# ─────────────────────────────────────────────────────────────────────────
# Integration sanity (one test that exercises everything together)
# ─────────────────────────────────────────────────────────────────────────


async def test_full_workflow_basic(
    bm25_results: list[dict[str, Any]],
    tfidf_results: list[dict[str, Any]],
) -> None:
    """All 5 representations on the same query, against the same fakes."""
    orch, idx, _ = make_orchestrator(bm25=bm25_results, tfidf=tfidf_results)
    query = "fox dog"
    for rep in (
        "tfidf",
        "bm25",
        "embedding",
        "hybrid_serial",
        "hybrid_parallel",
    ):
        req = HybridSearchRequest(
            query=query,
            representation=rep,  # type: ignore[arg-type]
            fusion="rrf",
            k=3,
        )
        resp = await orch.search("touche2020", req)
        assert resp.representation == rep
        assert len(resp.results) == 3
        for h in resp.results:
            assert h.rank >= 1
            assert h.doc_id
            assert h.score >= 0.0  # not negative

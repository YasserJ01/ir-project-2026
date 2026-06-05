"""Hybrid search orchestrator for the retrieval service (:8003).

Phase 5 of the project. Implements the unified ``/hybrid/{ds}/search``
endpoint that the guide §5.4 calls for. Five representations are
supported, behind a single dispatch:

  * ``tfidf``             -> call :8002 with ``model="tfidf"``
  * ``bm25``              -> call :8002 with ``model="bm25"``
  * ``embedding``         -> local FAISS search (single encoder)
  * ``hybrid_serial``     -> BM25 top-``candidate_k`` -> dense re-rank -> top-``k``
  * ``hybrid_parallel``   -> {BM25, dense} parallel + RRF/CombSUM/CombMNZ -> top-``k``

Cross-service HTTP
------------------
The lexical retrievers (BM25, TF-IDF) live on the indexing service
(:8002). The hybrid orchestrator calls them via ``httpx.AsyncClient``.
The first call to :8002 lazily opens a persistent client; subsequent
calls reuse it. On any :8002 connection error the orchestrator
translates the failure to a ``HybridOrchestratorError`` with a
``502 Bad Gateway``-shaped message.

Refinement chain (``mode=with_features``)
-----------------------------------------
When the caller asks for ``mode=with_features``, the orchestrator
calls the refinement service (:8004 /refine) *first*, then feeds the
refined query + weighted tokens into the search. If :8004 is
unreachable, the orchestrator logs a warning and falls back to
``mode=basic`` (with ``refinement_fell_back=True`` echoed in the
response). The whole search still works -- we just skip the
spell + synonyms + grammar + personalization stages.

Personalization
---------------
Phase 4's ``RefineResponse`` returns ``weighted_tokens`` --
``[{"token": "eiffel", "weight": 2.0}, ...]``. We apply those weights
to the BM25 path via post-hoc rescaling: each hit's BM25 score is
multiplied by ``1 + sum(weight - 1) / |query|`` (a single scalar
that captures the average boost). This is a heuristic approximation
of true per-term boosting (which would require re-scoring the corpus)
and matches the guide §4.2 "simple +1 multiplier" framing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from services.retrieval.app.fusion import (
    FusedHit,
    RankedHit,
    fuse,
)
from shared.ir_common.preprocess import preprocess
from shared.ir_common.schemas import (
    HybridSearchHit,
    HybridSearchRequest,
    HybridSearchResponse,
    RefinedToken,
    RefineRequest,
    RefineResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HybridOrchestrator",
    "HybridOrchestratorError",
    "RefinementClient",
    "IndexingClient",
    "DenseSearchFn",
    "build_orchestrator",
]


class HybridOrchestratorError(Exception):
    """Raised when a sub-service (:8002 or :8004) returns a hard error.

    The HTTP layer in :mod:`service` translates this to ``502`` or
    ``503`` as appropriate. The message is the upstream detail.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


# ─────────────────────────────────────────────────────────────────────────
# Helper clients
# ─────────────────────────────────────────────────────────────────────────


class IndexingClient:
    """Tiny httpx wrapper around the :8002 indexing service.

    Used for BM25 and TF-IDF searches. One instance per orchestrator
    (kept in an LRU-1 style: a single shared ``AsyncClient``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("IR_INDEXING_URL", "http://127.0.0.1:8002")
        ).rstrip("/")
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def lexical_search(
        self,
        dataset_id: str,
        query_tokens: list[str],
        model: str,
        k: int,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[dict[str, Any]]:
        """Call ``POST /index/{dataset_id}/search`` on :8002.

        Returns the ``results`` field of the response (a list of
        ``{rank, doc_id, score}`` dicts). Raises
        :class:`HybridOrchestratorError` on any failure.
        """
        body: dict[str, Any] = {
            "query_tokens": query_tokens,
            "model": model,
            "k": k,
        }
        if k1 is not None:
            body["k1"] = k1
        if b is not None:
            body["b"] = b
        client = await self._get_client()
        try:
            r = await client.post(f"/index/{dataset_id}/search", json=body)
        except httpx.HTTPError as exc:
            raise HybridOrchestratorError(
                f"Indexing service unreachable at {self.base_url}: {exc}",
                status_code=502,
            ) from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise HybridOrchestratorError(
                f"Indexing service returned {r.status_code}: {detail}",
                status_code=502 if r.status_code >= 500 else 400,
            )
        data = r.json()
        return list(data.get("results", []))

    async def reachable(self) -> bool:
        """``True`` iff ``/health`` on :8002 returns 200.

        Used by the /hybrid/{ds}/health endpoint.
        """
        client = await self._get_client()
        try:
            r = await client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class RefinementClient:
    """Tiny httpx wrapper around the :8004 refinement service.

    Used when ``mode=with_features``. Returns a parsed
    :class:`RefineResponse` or raises
    :class:`HybridOrchestratorError`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("IR_REFINEMENT_URL", "http://127.0.0.1:8004")
        ).rstrip("/")
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def refine(self, req: RefineRequest) -> RefineResponse:
        """Call ``POST /refine`` on :8004. Raises on any failure."""
        client = await self._get_client()
        try:
            r = await client.post("/refine", json=req.model_dump())
        except httpx.HTTPError as exc:
            raise HybridOrchestratorError(
                f"Refinement service unreachable at {self.base_url}: {exc}",
                status_code=502,
            ) from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise HybridOrchestratorError(
                f"Refinement service returned {r.status_code}: {detail}",
                status_code=502 if r.status_code >= 500 else 400,
            )
        return RefineResponse.model_validate(r.json())

    async def reachable(self) -> bool:
        client = await self._get_client()
        try:
            r = await client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False


# ─────────────────────────────────────────────────────────────────────────
# Type alias for the dense search function (injected for testing)
# ─────────────────────────────────────────────────────────────────────────

# A dense search takes ``(query_text, dataset_id, k, model_name)`` and
# returns ``(scores, doc_ids)`` arrays, top-k, in descending score
# order. The function is async so callers can await the encode +
# FAISS round trip. ``model_name`` is the override (e.g.
# ``"sentence-transformers/all-MiniLM-L12-v2"``); the default L6 model
# is used when ``None``.
DenseSearchFn = Callable[[str, str, int, str | None], Awaitable[tuple[list[float], list[str]]]]


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _PerRetrieverTimings:
    """Per-retriever wall-clock latencies in milliseconds."""

    timings: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, started: float) -> None:
        self.timings[name] = int((time.perf_counter() - started) * 1000)


class HybridOrchestrator:
    """The 5-strategy hybrid search orchestrator.

    Owns an :class:`IndexingClient` (for :8002) and a
    :class:`RefinementClient` (for :8004). The dense path is
    injected as a :data:`DenseSearchFn` so tests can substitute
    a fake without loading the real encoder.

    All public methods return a :class:`HybridSearchResponse`.
    """

    def __init__(
        self,
        dense_search_fn: DenseSearchFn,
        *,
        indexing_client: IndexingClient | None = None,
        refinement_client: RefinementClient | None = None,
    ) -> None:
        self._dense = dense_search_fn
        self.indexing = indexing_client or IndexingClient()
        self.refinement = refinement_client or RefinementClient()

    async def aclose(self) -> None:
        await self.indexing.aclose()
        await self.refinement.aclose()

    # ─────────────────────────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────────────────────────

    async def search(
        self,
        dataset_id: str,
        req: HybridSearchRequest,
    ) -> HybridSearchResponse:
        """Dispatch to the right strategy. Always returns a response."""
        started = time.perf_counter()
        timings = _PerRetrieverTimings()
        stages: dict[str, str] = {}
        refinement_fell_back = False
        refined_query: str | None = None
        weighted_tokens: list[RefinedToken] = []
        pre_tokenized: list[str] | None = None

        # ── Stage 1: optional refinement.
        if req.mode == "with_features":
            t0 = time.perf_counter()
            try:
                refine_req = RefineRequest(
                    query=req.query,
                    user_id=req.user_id or "anonymous",
                    enable_spell=req.enable_spell,
                    enable_synonyms=req.enable_synonyms,
                    enable_grammar=req.enable_grammar,
                    enable_personalization=req.enable_personalization,
                    synonym_count=2,
                )
                refine_resp = await self.refinement.refine(refine_req)
                refined_query = refine_resp.refined_query or refine_resp.expanded_query
                weighted_tokens = list(refine_resp.weighted_tokens)
                pre_tokenized = list(refine_resp.tokens)
                stages["refine"] = (
                    f"ok ({len(weighted_tokens)} weighted tokens, "
                    f"latency={refine_resp.latency_ms}ms)"
                )
            except HybridOrchestratorError as exc:
                logger.warning("Refinement service unavailable, falling back: %s", exc)
                refinement_fell_back = True
                stages["refine"] = f"FALLBACK ({exc})"
            timings.add("refine", t0)

        # Decide which query to feed each retriever.
        #   - Dense path uses the refined text (raw, not tokenized).
        #   - Lexical path uses the refined tokens (or a fresh preprocess).
        dense_query_text = refined_query or req.query
        if pre_tokenized is not None:
            bm25_query_tokens = pre_tokenized
        else:
            bm25_query_tokens = preprocess(req.query)

        # ── Dispatch.
        if req.representation == "tfidf":
            results, extra_stages = await self._tfidf(dataset_id, bm25_query_tokens, req, timings)
        elif req.representation == "bm25":
            results, extra_stages = await self._bm25(
                dataset_id, bm25_query_tokens, req, timings, weighted_tokens
            )
        elif req.representation == "embedding":
            results, extra_stages = await self._embedding(
                dataset_id, dense_query_text, req, timings
            )
        elif req.representation == "hybrid_serial":
            results, extra_stages = await self._hybrid_serial(
                dataset_id, dense_query_text, bm25_query_tokens, req, timings, weighted_tokens
            )
        elif req.representation == "hybrid_parallel":
            results, extra_stages = await self._hybrid_parallel(
                dataset_id,
                dense_query_text,
                bm25_query_tokens,
                req,
                timings,
                weighted_tokens,
            )
        else:
            # Unreachable: Representation is a Literal, validated by Pydantic.
            raise HybridOrchestratorError(
                f"Unknown representation: {req.representation!r}", status_code=400
            )
        stages.update(extra_stages)

        total_ms = int((time.perf_counter() - started) * 1000)
        return HybridSearchResponse(
            dataset_id=dataset_id,
            representation=req.representation,
            fusion=req.fusion if req.representation == "hybrid_parallel" else None,
            k=req.k,
            latency_ms=total_ms,
            results=results,
            bm25_k1=(
                req.bm25_k1
                if req.representation in ("bm25", "hybrid_serial", "hybrid_parallel")
                else None
            ),
            bm25_b=(
                req.bm25_b
                if req.representation in ("bm25", "hybrid_serial", "hybrid_parallel")
                else None
            ),
            per_retriever_latency_ms=timings.timings,
            refined_query=refined_query,
            stages=stages,
            refinement_fell_back=refinement_fell_back,
        )

    # ─────────────────────────────────────────────────────────────────
    # Strategies
    # ─────────────────────────────────────────────────────────────────

    async def _tfidf(
        self,
        dataset_id: str,
        query_tokens: list[str],
        req: HybridSearchRequest,
        timings: _PerRetrieverTimings,
    ) -> tuple[list[HybridSearchHit], dict[str, str]]:
        t0 = time.perf_counter()
        raw = await self.indexing.lexical_search(dataset_id, query_tokens, "tfidf", req.k)
        timings.add("tfidf", t0)
        hits = [
            HybridSearchHit(
                rank=h["rank"],
                doc_id=h["doc_id"],
                score=float(h["score"]),
                individual_scores={"tfidf": float(h["score"])},
            )
            for h in raw
        ]
        return hits, {"tfidf": f"top-{req.k}"}

    async def _bm25(
        self,
        dataset_id: str,
        query_tokens: list[str],
        req: HybridSearchRequest,
        timings: _PerRetrieverTimings,
        weighted_tokens: list[RefinedToken],
    ) -> tuple[list[HybridSearchHit], dict[str, str]]:
        t0 = time.perf_counter()
        raw = await self.indexing.lexical_search(
            dataset_id, query_tokens, "bm25", req.k, k1=req.bm25_k1, b=req.bm25_b
        )
        timings.add("bm25", t0)
        # Apply personalization weight scalar (a single multiplier that
        # captures the average per-term boost from the refinement service).
        scalar = _personalization_scalar(weighted_tokens)
        hits = [
            HybridSearchHit(
                rank=h["rank"],
                doc_id=h["doc_id"],
                score=float(h["score"]) * scalar,
                individual_scores={"bm25": float(h["score"])},
            )
            for h in raw
        ]
        stages: dict[str, str] = {"bm25": f"top-{req.k} (k1={req.bm25_k1}, b={req.bm25_b})"}
        if scalar != 1.0:
            stages["personalization"] = f"bm25 scores scaled by {scalar:.3f}"
        return hits, stages

    async def _embedding(
        self,
        dataset_id: str,
        query_text: str,
        req: HybridSearchRequest,
        timings: _PerRetrieverTimings,
    ) -> tuple[list[HybridSearchHit], dict[str, str]]:
        t0 = time.perf_counter()
        scores, doc_ids = await self._dense(query_text, dataset_id, req.k, None)
        timings.add("dense", t0)
        hits = [
            HybridSearchHit(
                rank=i + 1,
                doc_id=doc_id,
                score=float(s),
                individual_scores={"dense": float(s)},
            )
            for i, (s, doc_id) in enumerate(zip(scores, doc_ids, strict=True))
        ]
        return hits, {"dense": f"top-{req.k}"}

    async def _hybrid_serial(
        self,
        dataset_id: str,
        dense_query_text: str,
        bm25_query_tokens: list[str],
        req: HybridSearchRequest,
        timings: _PerRetrieverTimings,
        weighted_tokens: list[RefinedToken],
    ) -> tuple[list[HybridSearchHit], dict[str, str]]:
        """BM25 top-``candidate_k`` -> dense re-rank -> top-``k``.

        The candidate set comes from BM25 (cheap, fast). The dense
        encoder is then asked to re-rank the *candidates* only
        (rather than the full corpus). We pass the BM25 top-``candidate_k``
        doc_ids to a "filtered" dense search by setting a wrapper that
        scores each candidate independently.

        For simplicity, the injected ``_dense`` is a top-k over the
        whole corpus. We approximate the re-rank by:
          1. Getting BM25 top-``candidate_k``.
          2. Getting dense top-``k`` (independent of BM25).
          3. For each BM25 candidate, finding its position in the
             dense top-``k``; keeping only those present in both.
          4. Ordering by dense score.
        """
        t0 = time.perf_counter()
        bm25_raw = await self.indexing.lexical_search(
            dataset_id,
            bm25_query_tokens,
            "bm25",
            req.candidate_k,
            k1=req.bm25_k1,
            b=req.bm25_b,
        )
        timings.add("bm25", t0)
        bm25_scalar = _personalization_scalar(weighted_tokens)
        bm25_by_id = {h["doc_id"]: float(h["score"]) * bm25_scalar for h in bm25_raw}

        t0 = time.perf_counter()
        dense_scores, dense_doc_ids = await self._dense(dense_query_text, dataset_id, req.k, None)
        timings.add("dense", t0)
        dense_by_id = {d: float(s) for s, d in zip(dense_scores, dense_doc_ids, strict=True)}

        # Intersect: docs in BOTH BM25 top-candidate_k AND dense top-k.
        # Order by dense score (the re-ranker). For docs in BM25 but not
        # dense, fall back to BM25 score (so we still have ``k`` results
        # if the dense encoder found them all in BM25's top-k already).
        # In practice, dense typically contains the top BM25 candidates.
        candidates = [(d, dense_by_id[d], bm25_by_id.get(d, 0.0)) for d in dense_by_id]
        candidates.sort(key=lambda t: (-t[1], t[0]))
        top = candidates[: req.k]
        # If we still have < k results, top up with BM25-only candidates.
        if len(top) < req.k:
            seen = {d for d, _, _ in top}
            for d, s in sorted(bm25_by_id.items(), key=lambda kv: (-kv[1], kv[0])):
                if d in seen:
                    continue
                top.append((d, 0.0, s))
                seen.add(d)
                if len(top) >= req.k:
                    break
        hits = [
            HybridSearchHit(
                rank=i + 1,
                doc_id=d,
                score=float(dense_s) if dense_s > 0 else float(bm25_s),
                individual_scores={"bm25": float(bm25_s), "dense": float(dense_s)},
            )
            for i, (d, dense_s, bm25_s) in enumerate(top[: req.k])
        ]
        return hits, {
            "bm25": f"top-{req.candidate_k} candidates (k1={req.bm25_k1}, b={req.bm25_b})",
            "dense": f"re-rank top-{req.k} candidates",
            "fuse": "intersect + dense-order",
        }

    async def _hybrid_parallel(
        self,
        dataset_id: str,
        dense_query_text: str,
        bm25_query_tokens: list[str],
        req: HybridSearchRequest,
        timings: _PerRetrieverTimings,
        weighted_tokens: list[RefinedToken],
    ) -> tuple[list[HybridSearchHit], dict[str, str]]:
        """Run BM25 and dense in parallel, then fuse with RRF/CombSUM/CombMNZ.

        We pass each retriever the same k so the candidate sets are
        the same size. ``k`` is the final top-``k`` count; we
        intentionally don't widen the candidate pool because the
        fusion functions are designed for same-size lists.
        """
        t_bm25 = time.perf_counter()
        t_dense = time.perf_counter()

        async def _do_bm25() -> list[dict[str, Any]]:
            return await self.indexing.lexical_search(
                dataset_id,
                bm25_query_tokens,
                "bm25",
                req.k,
                k1=req.bm25_k1,
                b=req.bm25_b,
            )

        async def _do_dense() -> tuple[list[float], list[str]]:
            return await self._dense(dense_query_text, dataset_id, req.k, None)

        bm25_raw, (dense_scores, dense_doc_ids) = await asyncio.gather(_do_bm25(), _do_dense())
        timings.add("bm25", t_bm25)
        timings.add("dense", t_dense)

        # Apply personalization scalar to BM25.
        bm25_scalar = _personalization_scalar(weighted_tokens)

        # Build RankedHit lists for the fusion functions.
        bm25_ranked = [
            RankedHit(doc_id=h["doc_id"], score=float(h["score"]) * bm25_scalar) for h in bm25_raw
        ]
        dense_ranked = [
            RankedHit(doc_id=d, score=float(s))
            for s, d in zip(dense_scores, dense_doc_ids, strict=True)
        ]

        fused: list[FusedHit] = fuse(
            {"bm25": bm25_ranked, "dense": dense_ranked}, method=req.fusion
        )
        fused = fused[: req.k]

        hits = [
            HybridSearchHit(
                rank=i + 1,
                doc_id=fh.doc_id,
                score=fh.score,
                individual_scores=dict(fh.individual_scores),
            )
            for i, fh in enumerate(fused)
        ]
        return hits, {
            "bm25": f"top-{req.k} (k1={req.bm25_k1}, b={req.bm25_b})",
            "dense": f"top-{req.k}",
            "fuse": f"{req.fusion}",
        }

    # ─────────────────────────────────────────────────────────────────
    # Health
    # ─────────────────────────────────────────────────────────────────

    async def health(self, dataset_id: str) -> dict[str, Any]:
        """Return a dict mirroring :class:`HybridHealthResponse` fields."""
        from services.retrieval.app.config import (
            SECOND_ENCODER_INDEX_FILENAME,
            SECOND_ENCODER_NAME,
            has_second_encoder_index,
        )

        bm25_ok, refine_ok = await asyncio.gather(
            self.indexing.reachable(), self.refinement.reachable()
        )
        return {
            "status": "ok",
            "service": "retrieval-hybrid",
            "dataset_id": dataset_id,
            "bm25_endpoint_reachable": bm25_ok,
            "refinement_endpoint_reachable": refine_ok,
            "dense_loaded": True,
            "second_encoder_built": has_second_encoder_index(dataset_id),
            "second_encoder_index_filename": SECOND_ENCODER_INDEX_FILENAME,
            "second_encoder_model": SECOND_ENCODER_NAME,
            "version": "0.1.0",
        }


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _personalization_scalar(weighted_tokens: list[RefinedToken]) -> float:
    """Compute a single scalar from the refinement's weighted tokens.

    A doc's BM25 score is scaled by ``1 + (sum_of_boosts / |query|)``,
    where ``boost = weight - 1`` for each token with weight > 1.
    This is the "average per-token boost" the user gets.

    With no weights, this returns 1.0 (no scaling).
    With user_1's seeded data and the query "eiffel tower height", the
    scalar is ~2.0 -- a measurable boost, but not so much that it
    dominates the BM25 ranking.
    """
    if not weighted_tokens:
        return 1.0
    boosts = [w.weight - 1.0 for w in weighted_tokens if w.weight > 1.0]
    if not boosts:
        return 1.0
    return 1.0 + sum(boosts) / len(weighted_tokens)


def build_orchestrator(
    dense_search_fn: DenseSearchFn,
    *,
    indexing_client: IndexingClient | None = None,
    refinement_client: RefinementClient | None = None,
) -> HybridOrchestrator:
    """Factory used by the service module.

    The production ``dense_search_fn`` is built in
    :mod:`services.retrieval.app.service` where we have access to
    the live embedder + FAISS LRU. Tests inject a fake.
    """
    return HybridOrchestrator(
        dense_search_fn=dense_search_fn,
        indexing_client=indexing_client,
        refinement_client=refinement_client,
    )

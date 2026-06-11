"""FastAPI gateway service (Phase 6).

Runs on port 8000. The single public entry point for the React UI.
The gateway is **only** a router + translator; all retrieval logic
lives in the backend services. The gateway:

  * Adds CORS so the React app (port 5173 dev / 3000 prod) can call us.
  * Stamps every request with an X-Request-ID and logs latency.
  * Translates ``/api/search`` ``representation`` into the right
    downstream call (no business logic in the gateway itself).
  * Bubbles downstream failures up as structured 502/503 responses
    with a ``GatewayErrorResponse`` body the UI can display.
  * Returns 501 for ``/api/rag/answer`` (the RAG service ships in
    Phase 8).
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Force UTF-8 on Windows before any logging/output.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

from services.gateway.app.clients import (  # noqa: E402
    BackendClientError,
    BackendUnreachable,
    GatewayClients,
)
from services.gateway.app.config import CONFIG  # noqa: E402
from services.gateway.app.middleware import RequestContextMiddleware  # noqa: E402
from services.gateway.app.schemas import (  # noqa: E402
    ClusterSearchRequest,
    DATASET_IDS,
    GatewayErrorResponse,
    GatewayHealthResponse,
    GatewaySearchRequest,
    LogClickRequest,
    MultiEncoderSearchRequest,
    RagRequest,
    RefineRequest,
)

# Services that must be reachable for the gateway to report "ok".
# Clustering is optional — not required for health.
_REQUIRED_SERVICES = {"preprocessing", "indexing", "retrieval", "refinement", "rag"}

__all__ = ["app", "run"]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Lifespan: open/close the httpx clients
# ─────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    clients = GatewayClients(
        preprocessing_url=CONFIG.preprocessing_url,
        indexing_url=CONFIG.indexing_url,
        retrieval_url=CONFIG.retrieval_url,
        refinement_url=CONFIG.refinement_url,
        rag_url=CONFIG.rag_url,
        clustering_url=CONFIG.clustering_url,
        timeout_s=CONFIG.downstream_timeout_s,
    )
    await clients.open()
    # Stash on app.state so handlers can grab it.
    _.state.clients = clients  # type: ignore[attr-defined]
    logger.info(
        "Gateway clients opened. URLs: pre=%s idx=%s ret=%s ref=%s",
        CONFIG.preprocessing_url,
        CONFIG.indexing_url,
        CONFIG.retrieval_url,
        CONFIG.refinement_url,
    )
    try:
        yield
    finally:
        await clients.aclose()
        logger.info("Gateway clients closed.")


app = FastAPI(
    title="IR Project 2026 — API Gateway",
    version="0.6.0",
    description=(
        "Phase 6: SOA gateway. Single public entry on :8000. "
        "Routes to preprocessing (:8001), indexing (:8002), retrieval (:8003), "
        "refinement (:8004), and RAG (:8005)."
    ),
    lifespan=lifespan,
)

# CORS — gateway itself. Per-service CORS is also tightened in this phase.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CONFIG.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Request-id + latency (must be added AFTER CORS so the access log
# sees the final response status).
app.add_middleware(RequestContextMiddleware)


def _clients(request: Request) -> GatewayClients:
    return request.app.state.clients  # type: ignore[no-any-return]


# ─────────────────────────────────────────────────────────────────────────
# Helper: build a 502/503 response from a downstream failure
# ─────────────────────────────────────────────────────────────────────────


def _downstream_error_response(exc: Exception) -> HTTPException:
    if isinstance(exc, BackendUnreachable):
        body = GatewayErrorResponse(
            service=exc.service,
            reachable=False,
            status_code=None,
            detail=str(exc),
        )
        return HTTPException(status_code=503, detail=body.model_dump())
    if isinstance(exc, BackendClientError):
        body = GatewayErrorResponse(
            service=exc.service,
            reachable=True,
            status_code=exc.status_code,
            detail=exc.detail or str(exc),
        )
        # If downstream said 4xx, return the same code (caller bug, not ours).
        # If downstream said 5xx, return 502 (bad gateway).
        code = exc.status_code if 400 <= exc.status_code < 500 else 502
        return HTTPException(status_code=code, detail=body.model_dump())
    # Unknown error type — generic 500.
    return HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────
# Public endpoints
# ─────────────────────────────────────────────────────────────────────────


@app.get("/")
def root() -> dict[str, Any]:
    """Tiny landing page for hand-testing from a browser."""
    return {
        "service": "gateway",
        "version": "0.6.0",
        "phase": 8,
        "endpoints": [
            "GET  /health",
            "GET  /api/datasets",
            "GET  /api/docs/{ds}/{id}",
            "POST /api/search",
            "POST /api/refine",
            "POST /api/log/click",
            "POST /api/multi-encoder/{ds}/search",
            "POST /api/rag/answer",
            "POST /api/cluster/{ds}/search",
        ],
        "downstream": {
            "preprocessing": CONFIG.preprocessing_url,
            "indexing": CONFIG.indexing_url,
            "retrieval": CONFIG.retrieval_url,
            "refinement": CONFIG.refinement_url,
            "rag": CONFIG.rag_url,
            "clustering": CONFIG.clustering_url,
        },
        "see": "docs/PHASE_8.md",
    }


@app.get("/health", response_model=GatewayHealthResponse)
async def health(request: Request) -> GatewayHealthResponse:
    """Probe each backend in parallel; report reachability.

    Always returns 200. ``status`` is ``degraded`` if any backend is
    unreachable (the UI can show a banner but the user can still
    use the retrievers that *are* up).
    """
    clients = _clients(request)
    flags = await clients.reachable()
    required = {k: v for k, v in flags.items()
                if k in _REQUIRED_SERVICES}
    overall = "ok" if all(required.values()) else "degraded"
    return GatewayHealthResponse(status=overall, services=flags)


@app.get("/api/datasets")
def datasets() -> dict[str, list[str]]:
    """The canonical dataset list (matches ``shared.ir_common.schemas.DATASET_IDS``)."""
    return {"datasets": list(DATASET_IDS)}


@app.get("/api/docs/{dataset_id}/{doc_id}")
async def get_doc(dataset_id: str, doc_id: str, request: Request) -> dict[str, str]:
    """Pass-through to :8001 ``/docs/{dataset_id}/{doc_id}``.

    ``dataset_id`` is validated against the canonical list so the UI
    gets a 400 with a helpful message (rather than a cryptic 502).
    """
    if dataset_id not in DATASET_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset_id {dataset_id!r}; must be one of {list(DATASET_IDS)}",
        )
    clients = _clients(request)
    try:
        return await clients.preprocessing.get_doc(dataset_id, doc_id)
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


@app.post("/api/search")
async def search(request: Request, body: GatewaySearchRequest) -> dict[str, Any]:
    """Route a search to the right backend.

    The body is the gateway's stricter :class:`GatewaySearchRequest`
    (both ``query`` and ``dataset_id`` are required; Pydantic returns
    422 on a missing field). The gateway inspects ``representation``
    to pick the downstream:

    * ``tfidf`` / ``bm25``: call :8001 ``/preprocess``, then :8002
      ``/index/{ds}/search`` (model=bm25 or tfidf).
    * ``embedding`` / ``hybrid_serial`` / ``hybrid_parallel``: call
      :8003 ``/hybrid/{ds}/search`` directly (the orchestrator does
      the rest, including talking to :8002 for BM25 candidates and
      :8004 for refinement when ``mode=with_features``).
    """
    clients = _clients(request)
    representation = body.representation
    dataset_id = body.dataset_id
    query = body.query

    try:
        if representation in ("tfidf", "bm25"):
            # Gateway-level tokenisation (the backend would also do
            # this, but keeping the symmetry is helpful for testing).
            tokens = await clients.preprocessing.preprocess(query)
            # For ``tfidf`` we pass model="tfidf" downstream; for
            # ``bm25`` model="bm25". The backend picks the right index.
            # Phase 7: forward the BM25 k1/b sliders from the UI so
            # the user can tune the lexical scoring in real time.
            downstream_model = representation
            result = await clients.indexing.search(
                dataset_id,
                tokens,
                model=downstream_model,
                k=body.k,
                k1=body.bm25_k1,
                b=body.bm25_b,
            )
            return result
        # All other representations go straight to :8003. Build a dict
        # the backend understands (its HybridSearchRequest has the same
        # fields, so a model_dump() round-trips cleanly).
        payload = body.model_dump()
        if payload.get("user_id") is None:
            payload["user_id"] = "anonymous"
        result = await clients.retrieval.hybrid_search(dataset_id, payload)
        return result
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


@app.post("/api/multi-encoder/{dataset_id}/search")
async def multi_encoder_search(
    dataset_id: str, request: Request, body: MultiEncoderSearchRequest
) -> dict[str, Any]:
    """The Phase 5 bonus endpoint: fuse L6 + L12 FAISS indexes.

    The body is the standard :class:`MultiEncoderSearchRequest`. The
    dataset path parameter is validated manually so we can return a
    400 with the canonical dataset list (the backend's path validation
    is the same).
    """
    if dataset_id not in DATASET_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset_id {dataset_id!r}; must be one of {list(DATASET_IDS)}",
        )
    clients = _clients(request)
    try:
        return await clients.retrieval.multi_encoder_search(dataset_id, body.model_dump())
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


@app.post("/api/refine")
async def refine(request: Request, body: RefineRequest) -> dict[str, Any]:
    """Pass-through to :8004 ``/refine``.

    The body is the standard :class:`~ir_common.schemas.RefineRequest`
    (Pydantic validates field types and bounds; refinement features
    are toggled via the request body).
    """
    clients = _clients(request)
    try:
        return await clients.refinement.refine(body.model_dump())
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


@app.post("/api/log/click", status_code=204)
async def log_click(request: Request, body: LogClickRequest) -> None:
    """Pass-through to :8004 ``/log/click``.

    The body matches ``LogClickRequest`` (Pydantic enforces the
    ``user_id`` regex so the gateway can 422 on bad input before
    forwarding). The refinement service does the file write (it's the
    only service that already manages ``data/user_logs/``); the
    gateway just forwards.
    """
    clients = _clients(request)
    try:
        # The backend /log/click endpoint expects a JSON object; the
        # refined Pydantic body round-trips cleanly via model_dump().
        await clients.refinement.log_click(body.model_dump())
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


@app.post("/api/rag/answer")
async def rag_answer(request: Request, body: RagRequest) -> dict[str, Any]:
    """Pass-through to :8005 ``/rag/answer`` (Phase 8).

    The body is validated by :class:`RagRequest` before forwarding.
    The RAG service returns a ``RagResponse`` with ``answer``,
    ``source_doc_ids``, and ``latency_ms``.
    """
    clients = _clients(request)
    try:
        return await clients.rag.answer(body.model_dump())
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


@app.post("/api/rag/answer/stream")
async def rag_answer_stream(request: Request, body: RagRequest):
    """SSE pass-through to :8005 ``/rag/answer/stream``."""
    clients = _clients(request)
    try:
        resp = await clients.rag.answer_stream(body.model_dump())
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc
    return StreamingResponse(
        resp.aiter_bytes(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/cluster/{dataset_id}/search")
async def cluster_search(
    dataset_id: str, request: Request, body: ClusterSearchRequest
) -> dict[str, object]:
    """Cluster-constrained search via :8006.

    Accepts the same body as ``/api/search`` plus ``enable_clustering``
    and ``cluster_boost``. The clustering service applies Mini-Batch
    K-Means to boost results from the nearest cluster centroid.
    """
    if dataset_id not in DATASET_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset_id {dataset_id!r}; must be one of {list(DATASET_IDS)}",
        )
    clients = _clients(request)
    try:
        payload: dict[str, object] = body.model_dump()
        if payload.get("user_id") is None:
            payload["user_id"] = "anonymous"
        return await clients.clustering.search(dataset_id, payload)
    except (BackendUnreachable, BackendClientError) as exc:
        raise _downstream_error_response(exc) from exc


# ─────────────────────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────────────────────


def run() -> None:
    """``python -m services.gateway.app.main`` -- uvicorn entrypoint."""
    import uvicorn

    uvicorn.run(
        "services.gateway.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    run()

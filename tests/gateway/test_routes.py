"""Route tests for the gateway (Phase 6).

In-process FastAPI tests with mocked downstream clients. We exercise
each public route, the request-id middleware, CORS preflight, and the
error-translation paths (BackendUnreachable -> 503,
BackendClientError -> 502/4xx).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.gateway.app.clients import (
    BackendClientError,
    BackendUnreachable,
)

# ─────────────────────────────────────────────────────────────────────────
# /  (landing page)
# ─────────────────────────────────────────────────────────────────────────


def test_root_lists_endpoints(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "gateway"
    assert body["version"] == "0.6.0"
    # All 7 public endpoints listed.
    for ep in (
        "GET  /health",
        "GET  /api/datasets",
        "POST /api/search",
        "POST /api/refine",
        "POST /api/log/click",
        "POST /api/multi-encoder/{ds}/search",
        "POST /api/rag/answer  (501 stub, Phase 8)",
    ):
        assert ep in body["endpoints"]


# ─────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────


def test_health_all_reachable(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["services"] == {
        "preprocessing": True,
        "indexing": True,
        "retrieval": True,
        "refinement": True,
    }


def test_health_some_unreachable(client: TestClient, fake_clients: pytest.FixtureRequest) -> None:
    fake_clients.refinement._reachable = False  # type: ignore[attr-defined]
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["services"]["refinement"] is False
    assert body["services"]["indexing"] is True


# ─────────────────────────────────────────────────────────────────────────
# /api/datasets
# ─────────────────────────────────────────────────────────────────────────


def test_datasets_returns_canonical_list(client: TestClient) -> None:
    r = client.get("/api/datasets")
    assert r.status_code == 200
    assert r.json() == {"datasets": ["touche2020", "nq"]}


# ─────────────────────────────────────────────────────────────────────────
# /api/search
# ─────────────────────────────────────────────────────────────────────────


def test_search_bm25_calls_preprocess_then_indexing(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    fake_clients.preprocessing.canned["preprocess"] = ["abort", "legal"]
    fake_clients.indexing.canned["search"] = {
        "results": [{"doc_id": "d1", "score": 0.9, "rank": 1}],
        "latency_ms": 12,
    }
    r = client.post(
        "/api/search",
        json={
            "query": "abortion legal",
            "k": 5,
            "representation": "bm25",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["doc_id"] == "d1"

    # Preprocess was called with the raw query.
    assert fake_clients.preprocessing.calls == [{"method": "preprocess", "text": "abortion legal"}]
    # Indexing was called with the tokenised query, model=bm25, k=5.
    assert len(fake_clients.indexing.calls) == 1
    call = fake_clients.indexing.calls[0]
    assert call["method"] == "search"
    assert call["dataset_id"] == "touche2020"
    assert call["query_tokens"] == ["abort", "legal"]
    assert call["model"] == "bm25"
    assert call["k"] == 5


def test_search_tfidf_uses_tfidf_model(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    fake_clients.preprocessing.canned["preprocess"] = ["x"]
    fake_clients.indexing.canned["search"] = {"results": [], "latency_ms": 1}
    r = client.post(
        "/api/search",
        json={
            "query": "test",
            "k": 3,
            "representation": "tfidf",
            "dataset_id": "nq",
        },
    )
    assert r.status_code == 200
    assert fake_clients.indexing.calls[0]["model"] == "tfidf"


def test_search_embedding_routes_to_retrieval(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    fake_clients.retrieval.canned["hybrid_search"] = {
        "results": [{"doc_id": "dE", "score": 0.88, "rank": 1}],
        "latency_ms": 9,
        "representation": "embedding",
    }
    r = client.post(
        "/api/search",
        json={
            "query": "open source LLM",
            "k": 3,
            "representation": "embedding",
            "dataset_id": "nq",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["doc_id"] == "dE"
    assert fake_clients.indexing.calls == []  # NOT called
    assert len(fake_clients.retrieval.calls) == 1
    call = fake_clients.retrieval.calls[0]
    assert call["method"] == "hybrid_search"
    assert call["dataset_id"] == "nq"
    assert call["req_body"]["representation"] == "embedding"


def test_search_hybrid_parallel_passes_fusion(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    fake_clients.retrieval.canned["hybrid_search"] = {
        "results": [],
        "latency_ms": 1,
        "fusion": "combsum",
    }
    r = client.post(
        "/api/search",
        json={
            "query": "q",
            "k": 3,
            "representation": "hybrid_parallel",
            "fusion": "combsum",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 200
    body = fake_clients.retrieval.calls[0]["req_body"]
    assert body["fusion"] == "combsum"


def test_search_unknown_dataset_returns_422(client: TestClient) -> None:
    r = client.post(
        "/api/search",
        json={"query": "q", "k": 3, "representation": "bm25", "dataset_id": "fashion"},
    )
    assert r.status_code == 422  # Pydantic: dataset_id Literal rejection


def test_search_missing_query_returns_422(client: TestClient) -> None:
    r = client.post(
        "/api/search",
        json={"k": 3, "representation": "bm25", "dataset_id": "touche2020"},
    )
    assert r.status_code == 422  # Pydantic: `query` is required


def test_search_indexing_unreachable_returns_503(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    """If :8002 is down, the gateway returns 503 + GatewayErrorResponse."""
    fake_clients.preprocessing.canned["preprocess"] = ["x"]
    fake_clients.indexing.raise_for["search"] = BackendUnreachable(
        "indexing", "http://indexing:8000", ConnectionRefusedError("nope")
    )
    r = client.post(
        "/api/search",
        json={
            "query": "q",
            "k": 3,
            "representation": "bm25",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["service"] == "indexing"
    assert detail["reachable"] is False


def test_search_indexing_4xx_returns_400(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    """Downstream 4xx is passed through as the same status code."""
    fake_clients.preprocessing.canned["preprocess"] = ["x"]
    fake_clients.indexing.raise_for["search"] = BackendClientError(
        "indexing", 400, detail="bad request"
    )
    r = client.post(
        "/api/search",
        json={
            "query": "q",
            "k": 3,
            "representation": "bm25",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["service"] == "indexing"


def test_search_indexing_5xx_returns_502(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    """Downstream 5xx is translated to 502 (bad gateway)."""
    fake_clients.preprocessing.canned["preprocess"] = ["x"]
    fake_clients.indexing.raise_for["search"] = BackendClientError("indexing", 500, detail="oops")
    r = client.post(
        "/api/search",
        json={
            "query": "q",
            "k": 3,
            "representation": "bm25",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 502
    assert r.json()["detail"]["status_code"] == 500


# ─────────────────────────────────────────────────────────────────────────
# /api/multi-encoder/{ds}/search
# ─────────────────────────────────────────────────────────────────────────


def test_multi_encoder_calls_retrieval(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    fake_clients.retrieval.canned["multi_encoder_search"] = {
        "results": [{"doc_id": "d_me", "score": 0.7, "rank": 1}],
        "latency_ms": 22,
    }
    r = client.post(
        "/api/multi-encoder/touche2020/search",
        json={"query": "q", "k": 3, "fusion": "rrf"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["doc_id"] == "d_me"
    call = fake_clients.retrieval.calls[0]
    assert call["method"] == "multi_encoder_search"
    assert call["dataset_id"] == "touche2020"
    assert call["req_body"]["fusion"] == "rrf"


def test_multi_encoder_unknown_dataset_returns_400(client: TestClient) -> None:
    r = client.post(
        "/api/multi-encoder/not-real/search",
        json={"query": "q", "k": 3},
    )
    assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────
# /api/refine
# ─────────────────────────────────────────────────────────────────────────


def test_refine_passes_through(client: TestClient, fake_clients: pytest.FixtureRequest) -> None:
    fake_clients.refinement.canned["refine"] = {
        "query": "x",
        "tokens": ["x"],
        "stages": {"spell": "ok"},
    }
    r = client.post(
        "/api/refine",
        json={"query": "x", "user_id": "u1", "enable_spell": True},
    )
    assert r.status_code == 200
    assert r.json()["tokens"] == ["x"]
    # Gateway forwards verbatim -- no field stripping.
    call = fake_clients.refinement.calls[0]
    assert call["body"]["user_id"] == "u1"
    assert call["body"]["enable_spell"] is True


# ─────────────────────────────────────────────────────────────────────────
# /api/log/click
# ─────────────────────────────────────────────────────────────────────────


def test_log_click_forwards_to_refinement(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    r = client.post(
        "/api/log/click",
        json={
            "user_id": "user_1",
            "query": "test",
            "doc_id": "doc_42",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 204
    assert r.content == b""
    call = fake_clients.refinement.calls[0]
    assert call["method"] == "log_click"
    assert call["body"]["user_id"] == "user_1"
    assert call["body"]["doc_id"] == "doc_42"


def test_log_click_invalid_user_id_returns_422(client: TestClient) -> None:
    r = client.post(
        "/api/log/click",
        json={
            "user_id": "bad/../escape",
            "query": "q",
            "doc_id": "d",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 422  # Pydantic regex validation


def test_log_click_refinement_unreachable_returns_503(
    client: TestClient, fake_clients: pytest.FixtureRequest
) -> None:
    fake_clients.refinement.raise_for["log_click"] = BackendUnreachable(
        "refinement", "http://refinement:8000", ConnectionRefusedError("nope")
    )
    r = client.post(
        "/api/log/click",
        json={
            "user_id": "u1",
            "query": "q",
            "doc_id": "d",
            "dataset_id": "touche2020",
        },
    )
    assert r.status_code == 503


# ─────────────────────────────────────────────────────────────────────────
# /api/rag/answer  (501 stub)
# ─────────────────────────────────────────────────────────────────────────


def test_rag_answer_returns_501(client: TestClient) -> None:
    r = client.post("/api/rag/answer", json={"query": "q", "dataset_id": "touche2020"})
    assert r.status_code == 501
    detail = r.json()["detail"]
    assert detail["service"] == "rag"
    assert "Phase 8" in detail["detail"]


# ─────────────────────────────────────────────────────────────────────────
# Request-id middleware
# ─────────────────────────────────────────────────────────────────────────


def test_request_id_generated_when_absent(client: TestClient) -> None:
    r = client.get("/api/datasets")
    assert r.status_code == 200
    rid = r.headers.get("X-Request-ID")
    assert rid is not None
    # UUID4 hex = 32 hex chars
    assert len(rid) == 32
    int(rid, 16)  # parses as hex


def test_request_id_echoed_when_supplied(client: TestClient) -> None:
    r = client.get("/api/datasets", headers={"X-Request-ID": "abc-123"})
    assert r.headers.get("X-Request-ID") == "abc-123"


# ─────────────────────────────────────────────────────────────────────────
# CORS preflight
# ─────────────────────────────────────────────────────────────────────────


def test_cors_preflight_from_allowed_origin(client: TestClient) -> None:
    r = client.options(
        "/api/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200  # CORSMiddleware returns 200 for preflight
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_preflight_from_disallowed_origin(client: TestClient) -> None:
    r = client.options(
        "/api/search",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # No CORS headers echoed -> browser will block.
    assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"


# ─────────────────────────────────────────────────────────────────────────
# 404 / unknown routes
# ─────────────────────────────────────────────────────────────────────────


def test_unknown_route_returns_404(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404

"""Unit tests for the gateway's backend-service clients (Phase 6).

These test the httpx wrappers in isolation, using a real
``httpx.MockTransport`` so we exercise the actual error-translation
code path (not the fakes from conftest.py).
"""

from __future__ import annotations

import json

import httpx
import pytest

from services.gateway.app.clients import (
    BackendClientError,
    BackendUnreachable,
    IndexingClient,
    PreprocessingClient,
    RefinementClient,
    RetrievalClient,
)


@pytest.mark.asyncio
async def test_preprocessing_client_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/preprocess"
        return httpx.Response(200, json={"tokens": ["hello", "world"]})

    transport = httpx.MockTransport(handler)
    c = PreprocessingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        # Swap the inner client's transport for the mock.
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        result = await c.preprocess("Hello, World!")
        assert result == ["hello", "world"]
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_indexing_client_passes_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [], "latency_ms": 5})

    transport = httpx.MockTransport(handler)
    c = IndexingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        await c.search("touche2020", ["abortion", "legal"], model="bm25", k=10, k1=1.5, b=0.75)
        assert captured["path"] == "/index/touche2020/search"
        assert captured["body"]["query_tokens"] == ["abortion", "legal"]
        assert captured["body"]["model"] == "bm25"
        assert captured["body"]["k"] == 10
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_indexing_client_4xx_raises_backend_client_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "missing tokens"})

    transport = httpx.MockTransport(handler)
    c = IndexingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        with pytest.raises(BackendClientError) as exc_info:
            await c.search("touche2020", [], model="bm25", k=10, k1=1.5, b=0.75)
        assert exc_info.value.status_code == 400
        assert "missing tokens" in exc_info.value.detail
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_indexing_client_5xx_raises_backend_client_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    c = IndexingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        with pytest.raises(BackendClientError) as exc_info:
            await c.search("touche2020", ["x"], k=10)
        assert exc_info.value.status_code == 500
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_connection_refused_raises_backend_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    c = IndexingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        with pytest.raises(BackendUnreachable) as exc_info:
            await c.search("touche2020", ["x"], k=10)
        assert exc_info.value.service == "indexing"
        assert exc_info.value.base_url == "http://x"
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_retrieval_client_hybrid_search() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"results": [{"doc_id": "d", "score": 0.5, "rank": 1}], "latency_ms": 3}
        )

    transport = httpx.MockTransport(handler)
    c = RetrievalClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        result = await c.hybrid_search("nq", {"query": "q", "k": 3, "representation": "embedding"})
        assert captured["path"] == "/hybrid/nq/search"
        assert captured["body"]["representation"] == "embedding"
        assert result["results"][0]["doc_id"] == "d"
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_retrieval_client_multi_encoder_path() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(handler)
    c = RetrievalClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        await c.multi_encoder_search("touche2020", {"query": "q", "k": 3, "fusion": "rrf"})
        assert captured["path"] == "/multi-encoder/touche2020/search"
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_refinement_client_refine() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"tokens": ["x"], "stages": {}})

    transport = httpx.MockTransport(handler)
    c = RefinementClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        result = await c.refine({"query": "x", "user_id": "u1"})
        assert captured["path"] == "/refine"
        assert captured["body"]["user_id"] == "u1"
        assert result["tokens"] == ["x"]
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_refinement_client_log_click_no_body() -> None:
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    c = RefinementClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        await c.log_click({"user_id": "u1", "query": "q", "doc_id": "d", "dataset_id": "x"})
        assert seen == ["POST"]
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_reachable_true_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    c = PreprocessingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        assert await c.reachable() is True
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_reachable_false_on_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    transport = httpx.MockTransport(handler)
    c = PreprocessingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        assert await c.reachable() is False
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_reachable_false_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    transport = httpx.MockTransport(handler)
    c = PreprocessingClient(base_url="http://x", timeout_s=1.0)
    await c.open()
    try:
        c._client = httpx.AsyncClient(base_url="http://x", transport=transport)  # type: ignore[assignment]
        assert await c.reachable() is False
    finally:
        await c.aclose()

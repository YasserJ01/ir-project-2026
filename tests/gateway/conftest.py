"""Conftest for the gateway test suite (Phase 6).

We don't want to spin up the real backend services for unit tests,
so we patch the gateway's ``GatewayClients`` to point at a
``FakeGatewayClients`` that records calls and returns canned
responses. This keeps the test fast and deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest

from services.gateway.app import main as main_mod


class FakeBaseClient:
    """Replaces a real ``_BaseClient`` for tests.

    The fake doesn't speak HTTP at all. Each method has a configurable
    canned return value (set on the instance before the test) and
    appends a record to ``self.calls`` so the test can assert what
    was sent.
    """

    service_name = "fake"

    def __init__(self, base_url: str = "http://fake", timeout_s: float = 30.0) -> None:
        self.base_url = base_url
        self._timeout = timeout_s
        self._client = None
        self.calls: list[dict[str, Any]] = []
        # ``canned`` is a dict of method-name -> return value.
        self.canned: dict[str, Any] = {}
        # ``raise_for`` is a dict of method-name -> Exception class to
        # raise. Useful for testing the 502/503 error paths.
        self.raise_for: dict[str, Exception] = {}

    async def open(self) -> None:
        self._client = object()  # truthy

    async def aclose(self) -> None:
        self._client = None

    def _require(self) -> Any:
        if self._client is None:
            raise RuntimeError("FakeClient not opened")
        return self._client

    async def reachable(self) -> bool:
        # Default reachable=True; tests can override.
        return getattr(self, "_reachable", True)

    async def _record(self, method_name: str, **kwargs: Any) -> None:
        self.calls.append({"method": method_name, **kwargs})

    async def _dispatch(self, method_name: str, **kwargs: Any) -> Any:
        await self._record(method_name, **kwargs)
        if method_name in self.raise_for:
            raise self.raise_for[method_name]
        return self.canned.get(method_name)


class FakePreprocessingClient(FakeBaseClient):
    service_name = "preprocessing"

    async def preprocess(self, text: str) -> list[str]:
        result = await self._dispatch("preprocess", text=text)
        return result if isinstance(result, list) else []


class FakeIndexingClient(FakeBaseClient):
    service_name = "indexing"

    async def search(
        self,
        dataset_id: str,
        query_tokens: list[str],
        *,
        model: str = "bm25",
        k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[str, Any]:
        return await self._dispatch(
            "search",
            dataset_id=dataset_id,
            query_tokens=query_tokens,
            model=model,
            k=k,
            k1=k1,
            b=b,
        )


class FakeRetrievalClient(FakeBaseClient):
    service_name = "retrieval"

    async def hybrid_search(self, dataset_id: str, req_body: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch("hybrid_search", dataset_id=dataset_id, req_body=req_body)

    async def multi_encoder_search(
        self, dataset_id: str, req_body: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._dispatch(
            "multi_encoder_search", dataset_id=dataset_id, req_body=req_body
        )


class FakeRefinementClient(FakeBaseClient):
    service_name = "refinement"

    async def refine(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._dispatch("refine", body=body)

    async def log_click(self, body: dict[str, Any]) -> None:
        await self._dispatch("log_click", body=body)


class FakeGatewayClients:
    """Drop-in replacement for the real ``GatewayClients`` container."""

    def __init__(self) -> None:
        self.preprocessing = FakePreprocessingClient()
        self.indexing = FakeIndexingClient()
        self.retrieval = FakeRetrievalClient()
        self.refinement = FakeRefinementClient()
        self.rag_url = "http://fake-rag:8005"

    async def open(self) -> None:
        await self.preprocessing.open()
        await self.indexing.open()
        await self.retrieval.open()
        await self.refinement.open()

    async def aclose(self) -> None:
        await self.preprocessing.aclose()
        await self.indexing.aclose()
        await self.retrieval.aclose()
        await self.refinement.aclose()

    async def reachable(self) -> dict[str, bool]:
        return {
            "preprocessing": await self.preprocessing.reachable(),
            "indexing": await self.indexing.reachable(),
            "retrieval": await self.retrieval.reachable(),
            "refinement": await self.refinement.reachable(),
        }


@pytest.fixture
def fake_clients(monkeypatch: pytest.MonkeyPatch) -> FakeGatewayClients:
    """Patch the gateway app to use a ``FakeGatewayClients`` instance.

    The fixture patches ``services.gateway.app.main.GatewayClients``
    so the gateway's lifespan opens the fakes instead of real httpx
    clients.
    """
    fakes = FakeGatewayClients()

    def _factory(**_kwargs: Any) -> FakeGatewayClients:
        return fakes

    monkeypatch.setattr(main_mod, "GatewayClients", _factory)
    return fakes


@pytest.fixture
def client(fake_clients: FakeGatewayClients) -> Any:
    """A FastAPI test client wired up with the fake clients.

    We use the ``with`` form so the lifespan context runs — that
    opens the (faked) httpx clients and stashes them on
    ``app.state.clients`` so handlers can find them.
    """
    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as c:
        yield c

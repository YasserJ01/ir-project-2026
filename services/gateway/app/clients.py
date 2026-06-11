"""Backend-service clients for the gateway (Phase 6).

Each client owns a long-lived ``httpx.AsyncClient`` (created in
``open()``, closed in ``aclose()``). Each client surfaces 2-3
domain methods + a ``reachable()`` probe for the ``/health`` endpoint.

Error model
-----------
- 2xx: returns the parsed JSON dict.
- 4xx: raises ``BackendClientError`` with the status code + the
  response body's ``detail`` field if any.
- 5xx: raises ``BackendClientError`` (caller may treat as 502 in the
  HTTP response).
- Connection refused / timeout: raises ``BackendUnreachable`` (caller
  returns 503 with the structured error body).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class BackendClientError(RuntimeError):
    """A downstream service returned a 4xx/5xx response."""

    def __init__(self, service: str, status_code: int, detail: str = "", body: Any = None) -> None:
        super().__init__(f"{service} returned {status_code}: {detail}")
        self.service = service
        self.status_code = status_code
        self.detail = detail
        self.body = body


class BackendUnreachable(RuntimeError):
    """A downstream service is down (connection refused, timeout, DNS)."""

    def __init__(self, service: str, base_url: str, exc: Exception) -> None:
        super().__init__(f"{service} unreachable at {base_url}: {exc!r}")
        self.service = service
        self.base_url = base_url
        self.exc = exc


class _BaseClient:
    """Shared lifecycle + error translation for the 5 service clients."""

    service_name: str = "unknown"

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        # Short connect timeout: a slow backend is worse than a fast
        # 503 because it ties up the gateway's worker.
        timeout = httpx.Timeout(self._timeout, connect=min(2.0, self._timeout))
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(f"{self.service_name} client not opened; call open() first")
        return self._client

    async def reachable(self) -> bool:
        """True if the downstream's /health returns 200 within the timeout."""
        try:
            r = await self._require().get("/health")
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
            return False
        except Exception:  # noqa: BLE001
            return False

    async def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json("POST", path, json_body=body)

    async def _post_stream(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST and return the raw ``httpx.Response`` (for SSE proxying).

        The caller is responsible for consuming the response body
        (e.g. via ``response.aiter_bytes()``). Error handling follows
        the same pattern as ``_request_json``.
        """
        client = self._require()
        try:
            r = await client.request("POST", path, json=body)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            raise BackendUnreachable(self.service_name, self.base_url, exc) from exc
        except httpx.HTTPError as exc:
            raise BackendUnreachable(self.service_name, self.base_url, exc) from exc
        if r.status_code >= 400:
            detail = ""
            try:
                body_data = r.json()
                if isinstance(body_data, dict) and "detail" in body_data:
                    detail = str(body_data["detail"])
            except Exception:  # noqa: BLE001
                detail = r.text[:200]
            raise BackendClientError(self.service_name, r.status_code, detail)
        return r

    async def _get_json(self, path: str) -> dict[str, Any]:
        return await self._request_json("GET", path)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self._require()
        try:
            r = await client.request(method, path, json=json_body)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            raise BackendUnreachable(self.service_name, self.base_url, exc) from exc
        except httpx.HTTPError as exc:
            # Other httpx-level errors (read error, protocol error, etc.)
            raise BackendUnreachable(self.service_name, self.base_url, exc) from exc

        if r.status_code >= 400:
            # Try to surface the FastAPI ``detail`` field, else the raw text.
            detail = ""
            body: Any = None
            try:
                body = r.json()
                if isinstance(body, dict) and "detail" in body:
                    detail = str(body["detail"])
                else:
                    detail = r.text[:500]
            except Exception:  # noqa: BLE001
                detail = r.text[:500]
            raise BackendClientError(self.service_name, r.status_code, detail=detail, body=body)
        # 204 No Content
        if r.status_code == 204:
            return {}
        try:
            return r.json()
        except Exception as exc:  # noqa: BLE001
            raise BackendClientError(
                self.service_name,
                r.status_code,
                detail=f"Non-JSON response: {exc!r}",
            ) from exc


# ─────────────────────────────────────────────────────────────────────────
# Per-service clients
# ─────────────────────────────────────────────────────────────────────────


class PreprocessingClient(_BaseClient):
    service_name = "preprocessing"

    async def preprocess(self, text: str) -> list[str]:
        result = await self._post_json("/preprocess", {"text": text})
        return list(result.get("tokens", []))

    async def get_doc(self, dataset_id: str, doc_id: str) -> dict[str, str]:
        return await self._get_json(f"/docs/{dataset_id}/{doc_id}")


class IndexingClient(_BaseClient):
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
        return await self._post_json(
            f"/index/{dataset_id}/search",
            {
                "query_tokens": query_tokens,
                "model": model,
                "k": k,
                "k1": k1,
                "b": b,
            },
        )


class RetrievalClient(_BaseClient):
    service_name = "retrieval"

    async def hybrid_search(self, dataset_id: str, req_body: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json(f"/hybrid/{dataset_id}/search", req_body)

    async def multi_encoder_search(
        self, dataset_id: str, req_body: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._post_json(f"/multi-encoder/{dataset_id}/search", req_body)


class RefinementClient(_BaseClient):
    service_name = "refinement"

    async def refine(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/refine", body)

    async def log_click(self, body: dict[str, Any]) -> None:
        # 204 No Content; we don't care about the body.
        await self._post_json("/log/click", body)


class ClusteringClient(_BaseClient):
    service_name = "clustering"

    async def search(self, dataset_id: str, req_body: dict[str, object]) -> dict[str, object]:
        return await self._post_json(f"/cluster/{dataset_id}/search", req_body)


class RagClient(_BaseClient):
    service_name = "rag"

    async def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/rag/answer", body)

    async def answer_stream(self, body: dict[str, Any]) -> httpx.Response:
        """POST ``/rag/answer/stream`` and return the raw SSE response.

        The caller (``main.py``) wraps this in a
        ``fastapi.responses.StreamingResponse``.
        """
        return await self._post_stream("/rag/answer/stream", body)


# ─────────────────────────────────────────────────────────────────────────
# Lifecycle: open + close all 5 clients at once
# ─────────────────────────────────────────────────────────────────────────


class GatewayClients:
    """Container that owns the 6 backend clients and their lifecycle."""

    def __init__(
        self,
        preprocessing_url: str,
        indexing_url: str,
        retrieval_url: str,
        refinement_url: str,
        rag_url: str,
        clustering_url: str,
        timeout_s: float,
    ) -> None:
        self.preprocessing = PreprocessingClient(preprocessing_url, timeout_s)
        self.indexing = IndexingClient(indexing_url, timeout_s)
        self.retrieval = RetrievalClient(retrieval_url, timeout_s)
        self.refinement = RefinementClient(refinement_url, timeout_s)
        self.rag = RagClient(rag_url, timeout_s)
        self.clustering = ClusteringClient(clustering_url, timeout_s)
        self.rag_url = rag_url
        self._all = [
            self.preprocessing,
            self.indexing,
            self.retrieval,
            self.refinement,
            self.rag,
            self.clustering,
        ]

    async def open(self) -> None:
        for c in self._all:
            await c.open()

    async def aclose(self) -> None:
        for c in self._all:
            await c.aclose()

    async def reachable(self) -> dict[str, bool]:
        """Run all 4 reachability probes in parallel."""
        import asyncio

        names = [c.service_name for c in self._all]
        flags = await asyncio.gather(*(c.reachable() for c in self._all))
        return dict(zip(names, flags, strict=True))

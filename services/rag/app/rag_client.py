from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRIEVAL_URL = os.environ.get("RAG_RETRIEVAL_URL", "http://localhost:8003")
PREPROCESSING_URL = os.environ.get("RAG_PREPROCESSING_URL", "http://localhost:8001")
TIMEOUT_S = float(os.environ.get("RAG_CLIENT_TIMEOUT", "120"))


class RagClientError(RuntimeError):
    pass


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=TIMEOUT_S)


def search_retrieval(dataset_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
    """Call the retrieval service's hybrid search (BM25 path) for speed."""
    body = {
        "query": query,
        "dataset_id": dataset_id,
        "representation": "bm25",
        "k": k,
    }
    with _client(RETRIEVAL_URL) as c:
        r = c.post(f"/hybrid/{dataset_id}/search", json=body)
        if r.status_code >= 400:
            raise RagClientError(f"retrieval returned {r.status_code}: {r.text[:200]}")
        data = r.json()
    return data.get("results", [])


def fetch_doc_text(dataset_id: str, doc_id: str) -> dict[str, str]:
    """Fetch a single document's text from the preprocessing service."""
    with _client(PREPROCESSING_URL) as c:
        r = c.get(f"/docs/{dataset_id}/{doc_id}")
        if r.status_code == 404:
            return {"id": doc_id, "text": ""}
        if r.status_code >= 400:
            raise RagClientError(f"preprocessing returned {r.status_code}: {r.text[:200]}")
        return r.json()

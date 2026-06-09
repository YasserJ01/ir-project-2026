from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRIEVAL_URL = os.environ.get("RAG_RETRIEVAL_URL", "http://localhost:8003")
PREPROCESSING_URL = os.environ.get("RAG_PREPROCESSING_URL", "http://localhost:8001")
REFINEMENT_URL = os.environ.get("RAG_REFINEMENT_URL", "http://localhost:8004")
TIMEOUT_S = float(os.environ.get("RAG_CLIENT_TIMEOUT", "120"))
SHORT_TIMEOUT_S = float(os.environ.get("RAG_SHORT_TIMEOUT", "5"))


class RagClientError(RuntimeError):
    pass


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=TIMEOUT_S)


def search_retrieval(
    dataset_id: str,
    query: str,
    k: int = 5,
    representation: str = "embedding",
) -> list[dict[str, Any]]:
    """Call the retrieval service's hybrid search."""
    body = {
        "query": query,
        "dataset_id": dataset_id,
        "representation": representation,
        "k": k,
    }
    with _client(RETRIEVAL_URL) as c:
        r = c.post(f"/hybrid/{dataset_id}/search", json=body)
        if r.status_code >= 400:
            raise RagClientError(f"retrieval returned {r.status_code}: {r.text[:200]}")
        data = r.json()
    return data.get("results", [])


def refine_query(query: str) -> str:
    """Call the refinement service for spell + synonym expansion.

    Returns the expanded query on success, or the original query
    if the service is unreachable or returns an error (graceful
    degradation).
    """
    body = {
        "query": query,
        "user_id": "rag",
        "enable_spell": True,
        "enable_synonyms": True,
        "enable_grammar": False,
        "enable_personalization": False,
    }
    try:
        with httpx.Client(base_url=REFINEMENT_URL, timeout=SHORT_TIMEOUT_S) as c:
            r = c.post("/refine", json=body)
            if r.status_code >= 400:
                logger.warning("refinement returned %s; using raw query", r.status_code)
                return query
            data = r.json()
            expanded = data.get("expanded_query", "").strip()
            return expanded if expanded else query
    except httpx.RequestError as exc:
        logger.warning("refinement unreachable (%s); using raw query", exc)
        return query


def fetch_doc_text(dataset_id: str, doc_id: str) -> dict[str, str]:
    """Fetch a single document's text from the preprocessing service."""
    with _client(PREPROCESSING_URL) as c:
        r = c.get(f"/docs/{dataset_id}/{doc_id}")
        if r.status_code == 404:
            return {"id": doc_id, "text": ""}
        if r.status_code >= 400:
            raise RagClientError(f"preprocessing returned {r.status_code}: {r.text[:200]}")
        return r.json()

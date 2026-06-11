from __future__ import annotations

import logging
import sys
import time
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from services.clustering.app.clusterer import (
    is_built,
    load_clusterer,
    load_doc_id_map,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = logging.getLogger(__name__)

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

RETRIEVAL_URL = "http://localhost:8003"
DOWNSTREAM_TIMEOUT = 180.0

DATASET_IDS = ("touche2020", "nq")

_httpx: httpx.AsyncClient | None = None


async def get_httpx() -> httpx.AsyncClient:
    global _httpx
    if _httpx is None:
        _httpx = httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT)
    return _httpx


# ─────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────


class ClusterSearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(..., min_length=1, max_length=2048)
    k: int = Field(default=10, ge=1, le=200)
    representation: Literal["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"] = "embedding"
    mode: Literal["basic", "with_features"] = "basic"
    fusion: Literal["rrf", "combsum", "combmnz"] = "rrf"
    user_id: str | None = None
    enable_grammar: bool = False
    bm25_k1: float = Field(default=1.5, ge=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    enable_clustering: bool = True
    cluster_boost: float = Field(default=1.5, ge=1.0, le=5.0)


class ClusterStatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    built: bool
    n_clusters: int = 0
    per_cluster_counts: list[int] = []
    inertia: float = 0.0
    total_docs: int = 0


# ─────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Clustering Service",
    version="0.1.0",
    description="Mini-Batch K-Means clustering over FAISS embeddings.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


async def _encode_query(query: str) -> list[float]:
    """Encode query via the retrieval service's ``/retrieval/embed`` endpoint."""
    client = await get_httpx()
    resp = await client.post(
        f"{RETRIEVAL_URL}/retrieval/embed",
        json={"texts": [query]},
    )
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Embedding error: {resp.text[:200]}",
        )
    data = resp.json()
    vectors = data.get("vectors") or data.get("embedding") or []
    if isinstance(vectors, list) and len(vectors) > 0:
        if isinstance(vectors[0], list):
            return vectors[0]
        return vectors
    return []


async def _downstream_search(
    dataset_id: str,
    req: ClusterSearchRequest,
) -> dict[str, Any]:
    """Call the appropriate downstream service for the raw search."""
    client = await get_httpx()
    payload = {
        "query": req.query,
        "k": req.k,
        "representation": req.representation,
        "mode": req.mode,
        "fusion": req.fusion,
        "user_id": req.user_id or "anonymous",
        "enable_grammar": req.enable_grammar,
        "bm25_k1": req.bm25_k1,
        "bm25_b": req.bm25_b,
    }
    resp = await client.post(
        f"{RETRIEVAL_URL}/hybrid/{dataset_id}/search",
        json=payload,
    )
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Downstream error: {resp.text[:200]}",
        )
    return resp.json()


def _boost_and_rerank(
    results: list[dict[str, Any]],
    nearest_cluster: int,
    doc_id_map: dict[str, int],
    boost: float,
) -> list[dict[str, Any]]:
    boosted = []
    for hit in results:
        doc_id = hit.get("doc_id", "")
        cluster_id = doc_id_map.get(doc_id, -1)
        score = float(hit.get("score", 0.0))
        if cluster_id == nearest_cluster:
            score *= boost
        boosted.append({**hit, "score": score, "_cluster_id": cluster_id})
    boosted.sort(key=lambda h: (-h["score"], h.get("doc_id", "")))
    return boosted


# ─────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "clustering"}


@app.get("/cluster/{dataset_id}/stats", response_model=ClusterStatsResponse)
def cluster_stats(dataset_id: str) -> ClusterStatsResponse:
    if dataset_id not in DATASET_IDS:
        raise HTTPException(400, f"Unknown dataset_id {dataset_id!r}")
    if not is_built(dataset_id):
        return ClusterStatsResponse(dataset_id=dataset_id, built=False)
    cc = load_clusterer(dataset_id)
    return ClusterStatsResponse(
        dataset_id=dataset_id,
        built=True,
        n_clusters=cc.n_clusters,
        per_cluster_counts=cc.cluster_sizes(),
        inertia=cc.inertia_,
        total_docs=int(sum(cc.cluster_sizes())),
    )


@app.post("/cluster/{dataset_id}/search")
async def cluster_search(
    dataset_id: str, req: ClusterSearchRequest
) -> dict[str, Any]:
    t0 = time.perf_counter()

    if dataset_id not in DATASET_IDS:
        raise HTTPException(400, f"Unknown dataset_id {dataset_id!r}")
    if not is_built(dataset_id):
        raise HTTPException(
            503,
            detail=f"Clusters not built for {dataset_id}. Run scripts/build_clusters.py first.",
        )

    if not req.enable_clustering:
        result = await _downstream_search(dataset_id, req)
        elapsed = (time.perf_counter() - t0) * 1000
        result["latency_ms"] = elapsed
        result["nearest_cluster_id"] = -1
        result["cluster_centroid_distance"] = 0.0
        return result

    cc = load_clusterer(dataset_id)
    doc_id_map = load_doc_id_map(dataset_id)

    import numpy as np
    query_embedding = await _encode_query(req.query)
    query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    nearest, dist = cc.predict(query_vec)

    raw = await _downstream_search(dataset_id, req)
    raw_hits = raw.get("results") or raw.get("hits") or []
    boosted = _boost_and_rerank(raw_hits, nearest, doc_id_map, req.cluster_boost)

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "results": boosted[: req.k],
        "query": req.query,
        "dataset_id": dataset_id,
        "latency_ms": round(elapsed, 1),
        "nearest_cluster_id": nearest,
        "cluster_centroid_distance": round(dist, 4),
        "cluster_sizes": cc.cluster_sizes(),
        "representation": req.representation,
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "services.clustering.app.service:app",
        host="0.0.0.0",
        port=8006,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()

"""Dense-retrieval service -- FastAPI on port 8003.

Phase 3 of the project. Mirrors the surface of the indexing service
(port 8002) but the underlying primitive is a sentence-transformer
encoder + a FAISS ``IndexFlatIP``.

    GET  /health
    GET  /retrieval/{dataset_id}/exists
    GET  /retrieval/{dataset_id}/stats
    POST /retrieval/{dataset_id}/build
    POST /retrieval/{dataset_id}/load
    POST /retrieval/{dataset_id}/search   body: {query, k, model_name?}
    POST /retrieval/embed                 body: {texts[], model_name?}

Memory strategy
---------------
  * The model (~400 MB) is loaded on first /search or /embed and held
    in an LRU-1 cache. Switching models evicts the old one.
  * The FAISS index (~1.4 GB per dataset) is loaded on /load and held
    in an LRU-1 cache. Only one dataset is "hot" at a time.

Build is async (BackgroundTasks) because encoding 500K docs takes
~20 minutes on CPU. The /stats endpoint reports the build timestamp
so callers can poll for completion.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Any

# Force UTF-8 on Windows before any logging/output.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.retrieval.app import embedder as embedder_mod
from services.retrieval.app import vector_store as vector_store_mod
from services.retrieval.app.config import (
    SECOND_ENCODER_EMBEDDINGS_FILENAME,
    SECOND_ENCODER_INDEX_FILENAME,
    docs_path,
    has_second_encoder_index,
    index_dir,
)
from services.retrieval.app.hybrid import (
    HybridOrchestrator,
    HybridOrchestratorError,
    IndexingClient,
    RefinementClient,
    build_orchestrator,
)
from services.retrieval.app.multi_encoder import (
    MultiEncoderError,
    MultiEncoderRunner,
    build_default_multi_encoder_search,
)
from shared.ir_common.schemas import (
    DATASET_IDS,
    BuildResponse,
    DenseBuildRequest,
    DenseEmbedRequest,
    DenseEmbedResponse,
    DenseSearchHit,
    DenseSearchResponse,
    DenseStatsResponse,
    HybridHealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    MultiEncoderSearchRequest,
    RetrievalHealthResponse,
)

# Permissive CORS for local dev. Tightened in Phase 6.
app = FastAPI(
    title="IR Dense Retrieval Service",
    version="0.1.0",
    description=(
        "Sentence-transformer embeddings + FAISS IndexFlatIP behind a "
        "FastAPI service. Phase 3 of the IR project."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ─────────────────────────────────────────────────────────────────────────
# In-process LRU caches (size 1 each)
# ─────────────────────────────────────────────────────────────────────────

_EMBEDDER: embedder_mod.Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()

_FAISS_CACHE_2: OrderedDict[tuple[str, str], vector_store_mod.DenseIndex] = OrderedDict()
_FAISS_LOCK = threading.Lock()
_FAISS_CACHE_SIZE = 2  # one slot per encoder (L6 + L12)

_LOADED_MODEL_NAME: str = ""
_LOADED_DATASET: str | None = None


def _embedder() -> embedder_mod.Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = embedder_mod.Embedder()
    return _EMBEDDER


def _is_known(dataset_id: str) -> bool:
    return dataset_id in DATASET_IDS


def _load_faiss(
    dataset_id: str,
    *,
    index_filename: str | None = None,
    embeddings_filename: str | None = None,
) -> vector_store_mod.DenseIndex:
    """Return the FAISS index for ``dataset_id``; load if not in LRU.

    Optional ``index_filename`` / ``embeddings_filename`` override the
    default L6 names -- used to load the L12 (2nd-encoder) index in
    Phase 5. The LRU cache is keyed on ``(dataset_id, index_filename)``
    so a dataset can have BOTH the L6 and L12 indexes resident
    simultaneously.
    """
    cache_key = (dataset_id, index_filename or vector_store_mod.INDEX_FILENAME)
    cached = _FAISS_CACHE_2.get(cache_key)
    if cached is not None:
        _FAISS_CACHE_2.move_to_end(cache_key)
        return cached
    # Evict cold entries if at capacity.
    while len(_FAISS_CACHE_2) >= _FAISS_CACHE_SIZE:
        _FAISS_CACHE_2.popitem(last=False)
    d = index_dir(dataset_id)
    faiss_path = d / (index_filename or vector_store_mod.INDEX_FILENAME)
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index for '{dataset_id}' not found at {faiss_path}. "
            "Run `make build-dense` (L6) or `make build-dense-2` (L12) first."
        )
    idx = vector_store_mod.DenseIndex.load(
        d,
        index_filename=index_filename or vector_store_mod.INDEX_FILENAME,
        embeddings_filename=embeddings_filename or vector_store_mod.EMBEDDINGS_FILENAME,
    )
    _FAISS_CACHE_2[cache_key] = idx
    return idx


# ─────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=RetrievalHealthResponse)
def health() -> RetrievalHealthResponse:
    emb = _embedder()
    loaded_models = emb.loaded_models()
    return RetrievalHealthResponse(
        status="ok",
        service="retrieval",
        loaded_dataset=_LOADED_DATASET,
        model_loaded=bool(loaded_models),
        model_name=_LOADED_MODEL_NAME or emb.default_model_name,
        version="0.1.0",
    )


# ─────────────────────────────────────────────────────────────────────────
# Stats + exists
# ─────────────────────────────────────────────────────────────────────────


@app.get("/retrieval/{dataset_id}/exists")
def exists(dataset_id: str) -> dict[str, bool]:
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    d = index_dir(dataset_id)
    return {
        "exists": (d / vector_store_mod.INDEX_FILENAME).exists(),
        "second_encoder_exists": has_second_encoder_index(dataset_id),
    }


@app.get("/retrieval/{dataset_id}/stats", response_model=DenseStatsResponse)
def stats(dataset_id: str) -> DenseStatsResponse:
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    d = index_dir(dataset_id)
    faiss_path = d / vector_store_mod.INDEX_FILENAME
    if not faiss_path.exists():
        return DenseStatsResponse(dataset_id=dataset_id, exists=False)
    # Read sizes from disk (cheap).
    size_mb = 0.0
    for name in (
        vector_store_mod.INDEX_FILENAME,
        vector_store_mod.EMBEDDINGS_FILENAME,
        vector_store_mod.DOC_IDS_FILENAME,
    ):
        p = d / name
        if p.exists():
            size_mb += p.stat().st_size / (1024 * 1024)
    # Read build_meta.json (no joblib, no numpy load).
    meta_path = d / "build_meta.json"
    build_at = ""
    build_seconds = 0.0
    num_vectors = 0
    dim = 0
    model_name = ""
    index_type = "IndexFlatIP"
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            build_at = str(m.get("built_at", ""))
            build_seconds = float(m.get("elapsed_seconds", 0.0))
            num_vectors = int(m.get("num_vectors", 0))
            dim = int(m.get("embedding_dim", 0))
            model_name = str(m.get("model_name", ""))
            index_type = str(m.get("index_type", "IndexFlatIP"))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    loaded = any(k[0] == dataset_id for k in _FAISS_CACHE_2.keys())
    return DenseStatsResponse(
        dataset_id=dataset_id,
        exists=True,
        loaded=loaded,
        num_vectors=num_vectors,
        dim=dim,
        index_type=index_type,
        model_name=model_name,
        build_seconds=build_seconds,
        build_at=build_at,
        size_mb=round(size_mb, 1),
    )


# ─────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────


@app.post("/retrieval/{dataset_id}/load", response_model=DenseStatsResponse)
def load(dataset_id: str) -> DenseStatsResponse:
    global _LOADED_DATASET
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    _load_faiss(dataset_id)  # raises FileNotFoundError -> 404 below
    _LOADED_DATASET = dataset_id
    return stats(dataset_id)


# ─────────────────────────────────────────────────────────────────────────
# Build (async via BackgroundTasks)
# ─────────────────────────────────────────────────────────────────────────


def _do_build(dataset_id: str, model_name: str, batch_size: int, job_id: str) -> None:
    """The actual encode + FAISS write. Runs in BackgroundTasks.

    Writes a sentinel ``build_meta.json`` on success or failure so the
    caller can see what happened.
    """
    started = time.time()
    d = index_dir(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "dataset_id": dataset_id,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "in_progress",
        "job_id": job_id,
        "model_name": model_name,
    }
    meta_path = d / "build_meta.json"
    try:
        # 1. Stream raw docs.
        dp = docs_path(dataset_id)
        if not dp.exists():
            raise FileNotFoundError(
                f"docs.jsonl for '{dataset_id}' not found at {dp}. "
                "Run `make ingest-{a,b}` first."
            )
        doc_ids: list[str] = []
        texts: list[str] = []
        with dp.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                doc_ids.append(str(row["id"]))
                texts.append(str(row["text"]))
        if not texts:
            raise ValueError(f"No documents found in {dp}")

        # 2. Load model.
        emb = _embedder()
        emb.batch_size = batch_size
        emb.warm_up(model_name)

        # 3. Encode.
        t0 = time.time()
        vectors = emb.encode_documents(texts, model_name=model_name, show_progress=True)
        encode_seconds = time.time() - t0
        dim = int(vectors.shape[1])

        # 4. Build FAISS.
        t0 = time.time()
        idx = vector_store_mod.DenseIndex()
        idx.add(vectors, doc_ids)
        idx.save(d)
        save_seconds = time.time() - t0

        # 5. Persist build_meta.json.
        elapsed = time.time() - started
        size_mb = 0.0
        for name in (
            vector_store_mod.INDEX_FILENAME,
            vector_store_mod.EMBEDDINGS_FILENAME,
            vector_store_mod.DOC_IDS_FILENAME,
        ):
            p = d / name
            if p.exists():
                size_mb += p.stat().st_size / (1024 * 1024)
        meta.update(
            {
                "status": "ok",
                "num_vectors": int(vectors.shape[0]),
                "embedding_dim": dim,
                "index_type": "IndexFlatIP",
                "elapsed_seconds": round(elapsed, 2),
                "encode_seconds": round(encode_seconds, 2),
                "save_seconds": round(save_seconds, 2),
                "batch_size": batch_size,
                "size_mb": round(size_mb, 1),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(
            f"[build] {dataset_id} done: {vectors.shape[0]:,} vectors x {dim}-dim, "
            f"{elapsed:.1f}s, {size_mb:.1f} MB",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "error"
        meta["error"] = str(exc)
        meta["traceback"] = traceback.format_exc()
        meta["elapsed_seconds"] = round(time.time() - started, 2)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[build] {dataset_id} FAILED: {exc}", flush=True)


@app.post("/retrieval/{dataset_id}/build", response_model=BuildResponse)
def build(dataset_id: str, body: DenseBuildRequest, bg: BackgroundTasks) -> BuildResponse:
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    job_id = uuid.uuid4().hex[:12]
    bg.add_task(_do_build, dataset_id, body.model_name, body.batch_size, job_id)
    return BuildResponse(
        dataset_id=dataset_id,
        started=True,
        job_id=job_id,
        message=(
            f"Build started in background. Poll /retrieval/{dataset_id}/stats to " "see completion."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────


@app.post("/retrieval/{dataset_id}/search", response_model=DenseSearchResponse)
def search(dataset_id: str, body: dict) -> DenseSearchResponse:
    """Dense retrieval over the loaded FAISS index.

    Body: ``{"query": str, "k": int = 10, "model_name": str | None}``
    """
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    query = body.get("query") if isinstance(body, dict) else None
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(
            status_code=422,
            detail="'query' is required and must be a non-empty string for dense search.",
        )
    k = int(body.get("k", 10)) if isinstance(body, dict) else 10
    if k < 1 or k > 1000:
        raise HTTPException(status_code=422, detail="'k' must be in [1, 1000].")
    model_name = body.get("model_name") if isinstance(body, dict) else None
    if model_name is not None and not isinstance(model_name, str):
        raise HTTPException(status_code=422, detail="'model_name' must be a string.")

    try:
        with _FAISS_LOCK:
            idx = _load_faiss(dataset_id)
        emb = _embedder()
        t0 = time.time()
        q_vec = emb.encode_query(query, model_name=model_name)
        scores, ids = idx.search(q_vec, k)
        latency_ms = int((time.time() - t0) * 1000)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"search failed: {exc}") from exc

    used_model = model_name or emb.default_model_name
    hits = [
        DenseSearchHit(rank=r + 1, doc_id=idx.doc_ids[int(i)], score=float(s))
        for r, (s, i) in enumerate(zip(scores, ids, strict=True))
        if int(i) >= 0
    ]
    return DenseSearchResponse(
        dataset_id=dataset_id,
        model_name=used_model,
        k=k,
        latency_ms=latency_ms,
        results=hits,
        cached=model_name in emb.loaded_models(),
    )


# ─────────────────────────────────────────────────────────────────────────
# One-shot embed
# ─────────────────────────────────────────────────────────────────────────


@app.post("/retrieval/embed", response_model=DenseEmbedResponse)
def embed(body: DenseEmbedRequest) -> DenseEmbedResponse:
    emb = _embedder()
    t0 = time.time()
    try:
        vectors = emb.encode_documents(
            body.texts,
            model_name=body.model_name,
            show_progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"embed failed: {exc}") from exc
    latency_ms = int((time.time() - t0) * 1000)
    used_model = body.model_name or emb.default_model_name
    return DenseEmbedResponse(
        model_name=used_model,
        dim=int(vectors.shape[1]),
        vectors=vectors.tolist(),
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────
# Hybrid (Phase 5)
# ─────────────────────────────────────────────────────────────────────────
#
# Production closures: built once at module load, capture the live
# Embedder and the FAISS LRU-2 cache via _load_faiss. The orchestrator
# and the multi-encoder runner hold references to these closures.
#
# We lazily instantiate the dense closure (which itself calls
# _embedder() / _load_faiss() on each call) so the first search pays
# the model-load cost.


_DENSE_CLOSURE: Any | None = None
_MULTI_ENCODER_CLOSURE: MultiEncoderRunner | None = None
_ORCHESTRATOR: HybridOrchestrator | None = None


def _dense_search_closure() -> Any:
    """Build the production dense-search closure.

    Signature: ``async (query_text, dataset_id, model_name, k) -> (scores, ids)``.
    """
    global _DENSE_CLOSURE
    if _DENSE_CLOSURE is None:
        from services.retrieval.app import multi_encoder as me_mod

        async def _fn(
            query_text: str,
            dataset_id: str,
            k: int,
            model_name: str | None,
        ) -> tuple[list[float], list[str]]:
            """Closure signature: ``(query_text, dataset_id, k, model_name)``.

            Matches the :data:`HybridOrchestrator.DenseSearchFn` type alias
            (and the test fake in tests/retrieval/test_hybrid.py).
            ``model_name=None`` means "use the default L6 encoder".
            """
            emb = _embedder()
            t0 = time.time()
            q_vec = emb.encode_query(query_text, model_name=model_name)
            _ = int((time.time() - t0) * 1000)  # encode_ms (unused)
            # Load the index for the requested model. L12 uses the 2nd
            # encoder filenames; L6 (or None) uses the default.
            if model_name == me_mod.DEFAULT_ENCODER_2:
                idx = _load_faiss(
                    dataset_id,
                    index_filename=SECOND_ENCODER_INDEX_FILENAME,
                    embeddings_filename=SECOND_ENCODER_EMBEDDINGS_FILENAME,
                )
            else:
                idx = _load_faiss(dataset_id)
            scores, ids = idx.search(q_vec, k)
            scores_list = [float(s) for s in scores]
            doc_ids_list = [idx.doc_ids[int(i)] for i in ids if int(i) >= 0]
            n = min(len(scores_list), len(doc_ids_list))
            return scores_list[:n], doc_ids_list[:n]

        _DENSE_CLOSURE = _fn
    return _DENSE_CLOSURE


def _multi_encoder_runner() -> MultiEncoderRunner:
    """Return the singleton multi-encoder runner."""
    global _MULTI_ENCODER_CLOSURE
    if _MULTI_ENCODER_CLOSURE is None:
        search_fn = build_default_multi_encoder_search()
        _MULTI_ENCODER_CLOSURE = MultiEncoderRunner(search_fn=search_fn)
    return _MULTI_ENCODER_CLOSURE


def _orchestrator() -> HybridOrchestrator:
    """Return the singleton orchestrator (lazy: opens httpx clients)."""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = build_orchestrator(
            dense_search_fn=_dense_search_closure(),
            indexing_client=IndexingClient(),
            refinement_client=RefinementClient(),
        )
    return _ORCHESTRATOR


def _validate_dataset(dataset_id: str) -> None:
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")


def _is_refinement_reachable() -> bool:  # kept for back-compat; unused by Phase 5
    """Quick health probe for the refinement service (:8004)."""
    import os

    import httpx

    base = os.environ.get("IR_REFINEMENT_URL", "http://127.0.0.1:8004")
    try:
        r = httpx.get(f"{base}/health", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False


@app.post(
    "/hybrid/{dataset_id}/search",
    response_model=HybridSearchResponse,
)
async def hybrid_search(dataset_id: str, body: HybridSearchRequest) -> HybridSearchResponse:
    """Hybrid / multi-representation search endpoint.

    Body shape is :class:`HybridSearchRequest`. Routes to one of the 5
    representations via the in-process orchestrator.
    """
    _validate_dataset(dataset_id)
    try:
        orch = _orchestrator()
        return await orch.search(dataset_id, body)
    except HybridOrchestratorError as exc:
        # 502: downstream service (indexing or refinement) unreachable
        # and not safely fall-back-able.
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"hybrid search failed: {exc}") from exc


@app.post(
    "/multi-encoder/{dataset_id}/search",
    response_model=HybridSearchResponse,
)
async def multi_encoder_search(
    dataset_id: str, body: MultiEncoderSearchRequest
) -> HybridSearchResponse:
    """Multi-encoder (L6 + L12) search endpoint.

    Body shape is :class:`MultiEncoderSearchRequest`. Returns a 503 if
    the L12 FAISS index is not built yet (build in progress) and a
    400 if both encoders resolve to the same model.
    """
    _validate_dataset(dataset_id)
    try:
        runner = _multi_encoder_runner()
        return await runner.search(dataset_id, body)
    except MultiEncoderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"multi-encoder search failed: {exc}") from exc


@app.get(
    "/hybrid/{dataset_id}/health",
    response_model=HybridHealthResponse,
)
def hybrid_health(dataset_id: str) -> HybridHealthResponse:
    """Per-dataset availability of the 5 representations.

    Reports:
      * ``dense_loaded`` -- L6 FAISS index exists on disk
      * ``second_encoder_built`` -- L12 FAISS index exists on disk
      * ``bm25_endpoint_reachable`` -- indexing (:8002) responds to /health
      * ``refinement_endpoint_reachable`` -- refinement (:8004) responds to /health
    """
    _validate_dataset(dataset_id)
    d = index_dir(dataset_id)
    dense_loaded = (d / vector_store_mod.INDEX_FILENAME).exists()
    second_built = has_second_encoder_index(dataset_id)
    bm25_ok, refinement_ok = _probe_upstreams()
    return HybridHealthResponse(
        dataset_id=dataset_id,
        dense_loaded=dense_loaded,
        second_encoder_built=second_built,
        bm25_endpoint_reachable=bm25_ok,
        refinement_endpoint_reachable=refinement_ok,
        second_encoder_model=("sentence-transformers/all-MiniLM-L12-v2" if second_built else ""),
    )


def _probe_upstreams() -> tuple[bool, bool]:
    """Quick health probes for :8002 (indexing) and :8004 (refinement).

    Returns ``(bm25_ok, refinement_ok)``. A 0.5 s timeout is used so the
    /health endpoint is always snappy.
    """
    import os

    import httpx

    bm25_url = os.environ.get("IR_INDEXING_URL", "http://127.0.0.1:8002") + "/health"
    refinement_url = os.environ.get("IR_REFINEMENT_URL", "http://127.0.0.1:8004") + "/health"

    def _ok(url: str) -> bool:
        try:
            r = httpx.get(url, timeout=0.5)
            return r.status_code == 200
        except Exception:
            return False

    return _ok(bm25_url), _ok(refinement_url)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def run() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8003)


if __name__ == "__main__":  # pragma: no cover
    run()

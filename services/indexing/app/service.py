"""Indexing service -- FastAPI on port 8002.

Phase 2 of the project. Exposes three retrieval primitives (Inverted
Index, TF-IDF, BM25) as HTTP endpoints, following the guide §2.4
contract:

    GET  /health
    GET  /index/{dataset_id}/stats
    GET  /index/{dataset_id}/exists
    POST /index/{dataset_id}/build
    POST /index/{dataset_id}/load
    POST /index/{dataset_id}/search   body: {query_tokens, model, k, k1?, b?}
    GET  /index/{dataset_id}/postings/{term}?cap=1000

Memory strategy
---------------
Three large objects per dataset:

  * TF-IDF sparse matrix (~100-200 MB)
  * BM25 token_ids + default BM25 instance (~1.5-2.5 GB)
  * InvertedIndex dict-of-dicts (~2-4 GB after the cap)

We only auto-load the **TF-IDF** and **BM25** artifacts because those
back the high-throughput ``/search`` endpoint. The InvertedIndex is
loaded *on demand* for the ``/postings`` debug endpoint and then
evicted (LRU-1).

Only one dataset is "hot" at a time. The LRU cache has size 1, so a
switching query (touche -> nq -> touche) loads the first dataset
twice. We trade that for peak RAM.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict

# Force UTF-8 on Windows before any logging/output.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.indexing.app import bm25 as bm25_mod
from services.indexing.app import inverted_index as inverted_index_mod
from services.indexing.app import tfidf as tfidf_mod
from services.indexing.app.config import index_dir
from services.indexing.app.corpus import load_tokenized_corpus
from shared.ir_common.schemas import (
    DATASET_IDS,
    BuildRequest,
    BuildResponse,
    HealthResponse,
    Posting,
    PostingsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

# Permissive CORS for local dev. Tightened in Phase 6.
app = FastAPI(
    title="IR Indexing Service",
    version="0.1.0",
    description=(
        "Classical indexing primitives (Inverted Index, TF-IDF, BM25) "
        "behind a FastAPI service. Phase 2 of the IR project."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────
# LRU caches
# ─────────────────────────────────────────────────────────────────────────

# Only one dataset hot at a time. Switching loads the new one and
# evicts the old.
_TFIDF_CACHE: OrderedDict[str, tfidf_mod.TfidfRetriever] = OrderedDict()
_BM25_CACHE: OrderedDict[str, bm25_mod.BM25Retriever] = OrderedDict()
_INVIDX_CACHE: OrderedDict[str, inverted_index_mod.InvertedIndex] = OrderedDict()

# Lock around hot-path mutations (load / evict) so concurrent /search
# calls don't race.
_LOCK = threading.Lock()

# Background build jobs: {job_id: {dataset_id, status, ...}}
_BUILD_JOBS: dict[str, dict] = {}
_BUILD_LOCK = threading.Lock()


def _is_known(ds: str) -> bool:
    return ds in DATASET_IDS


# ─────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────


def _load_tfidf(dataset_id: str) -> tfidf_mod.TfidfRetriever:
    """Load the TF-IDF retriever for ``dataset_id`` (LRU-1)."""
    with _LOCK:
        if dataset_id in _TFIDF_CACHE:
            _TFIDF_CACHE.move_to_end(dataset_id)
            return _TFIDF_CACHE[dataset_id]
        path = index_dir(dataset_id)
        if not (path / tfidf_mod.VECTORIZER_FILENAME).exists():
            raise FileNotFoundError(
                f"TF-IDF index for '{dataset_id}' not found at {path}. "
                "Run `make build-indexes` first."
            )
        r = tfidf_mod.TfidfRetriever.load(path)
        _TFIDF_CACHE.clear()
        _TFIDF_CACHE[dataset_id] = r
        return r


def _load_bm25(dataset_id: str) -> bm25_mod.BM25Retriever:
    with _LOCK:
        if dataset_id in _BM25_CACHE:
            _BM25_CACHE.move_to_end(dataset_id)
            return _BM25_CACHE[dataset_id]
        path = index_dir(dataset_id)
        if not (path / bm25_mod.DEFAULT_BM25_FILENAME).exists():
            raise FileNotFoundError(
                f"BM25 index for '{dataset_id}' not found at {path}. "
                "Run `make build-indexes` first."
            )
        r = bm25_mod.BM25Retriever.load(path)
        _BM25_CACHE.clear()
        _BM25_CACHE[dataset_id] = r
        return r


def _load_invidx(dataset_id: str) -> inverted_index_mod.InvertedIndex:
    """Load the InvertedIndex for the /postings debug endpoint (LRU-1)."""
    with _LOCK:
        if dataset_id in _INVIDX_CACHE:
            _INVIDX_CACHE.move_to_end(dataset_id)
            return _INVIDX_CACHE[dataset_id]
        path = index_dir(dataset_id)
        pkl = path / inverted_index_mod.INDEX_FILENAME
        if not pkl.exists():
            raise FileNotFoundError(
                f"InvertedIndex for '{dataset_id}' not found at {pkl}. "
                "Run `make build-indexes` first."
            )
        idx = inverted_index_mod.InvertedIndex.load(pkl)
        _INVIDX_CACHE.clear()
        _INVIDX_CACHE[dataset_id] = idx
        return idx


# ─────────────────────────────────────────────────────────────────────────
# Health + stats endpoints
# ─────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = next(iter(_TFIDF_CACHE.keys()), None)
    if loaded is None:
        loaded = next(iter(_BM25_CACHE.keys()), None)
    return HealthResponse(loaded_dataset=loaded)


@app.get("/index/{dataset_id}/exists")
def exists(dataset_id: str) -> dict[str, bool]:
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    d = index_dir(dataset_id)
    inv = (d / inverted_index_mod.INDEX_FILENAME).exists()
    tf = (d / tfidf_mod.MATRIX_FILENAME).exists()
    bm = (d / bm25_mod.DEFAULT_BM25_FILENAME).exists()
    return {"exists": inv and tf and bm}


@app.get("/index/{dataset_id}/stats", response_model=StatsResponse)
def stats(dataset_id: str) -> StatsResponse:
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    d = index_dir(dataset_id)
    inv_pkl = d / inverted_index_mod.INDEX_FILENAME
    if not inv_pkl.exists():
        return StatsResponse(dataset_id=dataset_id, exists=False)
    size_mb = 0.0
    for fname in (
        inverted_index_mod.INDEX_FILENAME,
        tfidf_mod.VECTORIZER_FILENAME,
        tfidf_mod.MATRIX_FILENAME,
        bm25_mod.DEFAULT_BM25_FILENAME,
        bm25_mod.TOKEN_IDS_FILENAME,
        bm25_mod.VOCAB_FILENAME,
        tfidf_mod.DOC_IDS_FILENAME,
        bm25_mod.DOC_IDS_FILENAME,
    ):
        p = d / fname
        if p.exists():
            size_mb += p.stat().st_size / (1024 * 1024)
    meta_path = d / "build_meta.json"
    build_at = ""
    build_seconds = 0.0
    vocab_size = 0
    total_docs = 0
    avg_doc_length = 0.0
    min_df = 0
    max_df_ratio = 0.0
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            build_at = str(m.get("built_at", ""))
            build_seconds = float(m.get("elapsed_seconds", 0.0))
            vocab_size = int(m.get("inverted_vocab_post_cap", 0))
            total_docs = int(m.get("total_docs", 0))
            avg_doc_length = float(m.get("avg_doc_length", 0.0))
            min_df = int(m.get("min_df", 0))
            max_df_ratio = float(m.get("max_df_ratio", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    if avg_doc_length == 0.0 or vocab_size == 0:
        inv = inverted_index_mod.InvertedIndex.load(inv_pkl)
        inv_stats = inv.stats()
        vocab_size = inv_stats["vocab_size"]
        total_docs = inv_stats["total_docs"]
        avg_doc_length = inv_stats["avg_doc_length"]
        min_df = inv_stats["min_df"]
        max_df_ratio = inv_stats["max_df_ratio"]
    loaded = dataset_id in _TFIDF_CACHE or dataset_id in _BM25_CACHE or dataset_id in _INVIDX_CACHE
    return StatsResponse(
        dataset_id=dataset_id,
        exists=True,
        loaded=loaded,
        vocab_size=vocab_size,
        total_docs=total_docs,
        avg_doc_length=avg_doc_length,
        build_seconds=build_seconds,
        build_at=build_at,
        size_mb=round(size_mb, 1),
        cap={
            "min_df": min_df,
            "max_df_ratio": max_df_ratio,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Build endpoint (async, in a background thread)
# ─────────────────────────────────────────────────────────────────────────


def _build_index_sync(dataset_id: str, req: BuildRequest, job_id: str) -> None:
    """Background build: tokenize from tokens.jsonl, build all 3 indexes, save."""
    with _BUILD_LOCK:
        _BUILD_JOBS[job_id]["status"] = "running"
    started = time.time()
    try:
        # 1. Materialize the corpus from tokens.jsonl.
        doc_ids, corpus = load_tokenized_corpus(dataset_id)

        # 2. InvertedIndex (cap via req).
        idx = inverted_index_mod.InvertedIndex(min_df=req.min_df, max_df_ratio=req.max_df_ratio)
        idx.build(zip(doc_ids, corpus, strict=True))
        idx.save(index_dir(dataset_id) / inverted_index_mod.INDEX_FILENAME)

        # 3. TF-IDF.
        tfidf = tfidf_mod.TfidfRetriever()
        tfidf.build(corpus, doc_ids)
        tfidf.save(index_dir(dataset_id))

        # 4. BM25.
        bm25 = bm25_mod.BM25Retriever()
        bm25.build(corpus, doc_ids, method=req.bm25_method, show_progress=False)
        bm25.save(index_dir(dataset_id))

        # 5. Write build_meta.json with stats + timestamp.
        elapsed = time.time() - started
        meta = {
            "dataset_id": dataset_id,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "total_docs": len(doc_ids),
            "inverted_vocab_post_cap": len(idx.inverted_index),
            "tfidf_vocab": len(tfidf.vectorizer.vocabulary_) if tfidf.vectorizer else 0,
            "tfidf_nnz": int(tfidf.matrix.nnz) if tfidf.matrix is not None else 0,
            "bm25_vocab": len(bm25.vocab),
            "min_df": req.min_df,
            "max_df_ratio": req.max_df_ratio,
            "bm25_method": req.bm25_method,
        }
        (index_dir(dataset_id) / "build_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # 6. Invalidate caches so the next /search will reload.
        with _LOCK:
            _TFIDF_CACHE.pop(dataset_id, None)
            _BM25_CACHE.pop(dataset_id, None)
            _INVIDX_CACHE.pop(dataset_id, None)

        with _BUILD_LOCK:
            _BUILD_JOBS[job_id].update(
                {
                    "status": "done",
                    "elapsed_seconds": round(elapsed, 2),
                    "vocab_size": meta["inverted_vocab_post_cap"],
                    "total_docs": meta["total_docs"],
                }
            )
    except Exception as exc:  # noqa: BLE001
        with _BUILD_LOCK:
            _BUILD_JOBS[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }


@app.post("/index/{dataset_id}/build", response_model=BuildResponse)
def build(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    req: BuildRequest | None = None,
) -> BuildResponse:
    """Build all indexes for ``dataset_id`` in the background.

    Returns immediately with 202-style ``started=true`` and a ``job_id``
    that the caller can correlate via the /stats endpoint (which reads
    ``build_meta.json`` once it exists).
    """
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    if req is None:
        from services.indexing.app.config import DEFAULT_MAX_DF_RATIO, DEFAULT_MIN_DF

        req = BuildRequest(min_df=DEFAULT_MIN_DF, max_df_ratio=DEFAULT_MAX_DF_RATIO)
    job_id = str(uuid.uuid4())[:8]
    with _BUILD_LOCK:
        _BUILD_JOBS[job_id] = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "status": "queued",
        }
    background_tasks.add_task(_build_index_sync, dataset_id, req, job_id)
    return BuildResponse(dataset_id=dataset_id, job_id=job_id)


# ─────────────────────────────────────────────────────────────────────────
# Load (warm cache) endpoint
# ─────────────────────────────────────────────────────────────────────────


@app.post("/index/{dataset_id}/load")
def load(dataset_id: str) -> dict:
    """Load all hot-path artifacts for ``dataset_id`` into memory."""
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    started = time.time()
    _load_tfidf(dataset_id)
    _load_bm25(dataset_id)
    return {
        "loaded": True,
        "dataset_id": dataset_id,
        "took_seconds": round(time.time() - started, 3),
    }


# ─────────────────────────────────────────────────────────────────────────
# Search endpoint
# ─────────────────────────────────────────────────────────────────────────


@app.post("/index/{dataset_id}/search", response_model=SearchResponse)
def search(dataset_id: str, req: SearchRequest) -> SearchResponse:
    """Search ``dataset_id`` with the requested ``model``.

    The body is a ``SearchRequest`` (see ``shared/ir_common.schemas``).
    The query must be pre-tokenized (output of the preprocessing
    service); the indexing service does not call ``preprocess()`` at
    query time. (The gateway in Phase 6 will call preprocessing first.)
    """
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")

    started = time.perf_counter()
    model = req.model

    if model == "inverted":
        # The InvertedIndex isn't a native ranked retriever; we sum
        # the per-term tf across query terms and sort descending. This
        # is a useful "did any term match at all?" ranking. For real
        # ranking, use bm25 or tfidf.
        try:
            inv = _load_invidx(dataset_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        scores: dict[str, float] = {}
        for tok in req.query_tokens:
            for p in inv.get_postings(tok):
                scores[p.doc_id] = scores.get(p.doc_id, 0.0) + p.tf
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: req.k]
        results = [
            SearchResult(rank=i + 1, doc_id=doc_id, score=float(s))
            for i, (doc_id, s) in enumerate(top)
            if s > 0
        ]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SearchResponse(
            dataset_id=dataset_id,
            model=model,
            k=req.k,
            latency_ms=elapsed_ms,
            results=results,
        )

    if model == "tfidf":
        try:
            tfidf = _load_tfidf(dataset_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        hits: list[tfidf_mod.TfidfHit] = tfidf.search(req.query_tokens, k=req.k)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SearchResponse(
            dataset_id=dataset_id,
            model=model,
            k=req.k,
            latency_ms=elapsed_ms,
            results=[SearchResult(rank=h.rank, doc_id=h.doc_id, score=h.score) for h in hits],
        )

    if model == "bm25":
        try:
            bm25 = _load_bm25(dataset_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        bm25_hits: list[bm25_mod.BM25Hit]
        bm25_hits, cached = bm25.search(
            req.query_tokens,
            k=req.k,
            k1=req.k1,
            b=req.b,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SearchResponse(
            dataset_id=dataset_id,
            model=model,
            k=req.k,
            latency_ms=elapsed_ms,
            results=[SearchResult(rank=h.rank, doc_id=h.doc_id, score=h.score) for h in bm25_hits],
            k1=req.k1,
            b=req.b,
            cached=cached,
        )

    # unreachable: SearchModel is a Literal
    raise HTTPException(status_code=400, detail=f"Unknown model: {model!r}")


# ─────────────────────────────────────────────────────────────────────────
# Postings (debug) endpoint
# ─────────────────────────────────────────────────────────────────────────


@app.get("/index/{dataset_id}/postings/{term}", response_model=PostingsResponse)
def postings(dataset_id: str, term: str, cap: int = 1000) -> PostingsResponse:
    """Return the postings list for ``term`` (debug endpoint)."""
    if not _is_known(dataset_id):
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id: {dataset_id!r}")
    cap = max(1, min(cap, 10000))
    try:
        inv = _load_invidx(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    plist = inv.get_postings(term)
    doc_freq = len(plist)
    truncated = doc_freq > cap
    page = plist[:cap]
    return PostingsResponse(
        dataset_id=dataset_id,
        term=term,
        doc_freq=doc_freq,
        postings=[Posting(doc_id=p.doc_id, tf=p.tf) for p in page],
        truncated=truncated,
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI runner
# ─────────────────────────────────────────────────────────────────────────


def run() -> None:  # pragma: no cover -- entry-point only
    """Run the service on 127.0.0.1:8002 (called by ``make dev-indexing``)."""
    import uvicorn

    uvicorn.run(
        "services.indexing.app.service:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()

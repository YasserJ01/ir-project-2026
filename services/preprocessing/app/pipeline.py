"""Preprocessing service for the IR system.

Phase 1 ships a minimal FastAPI wrapper around the shared
``preprocess()`` function. The gateway and other services will call
``POST /preprocess`` in Phase 6. For now the service is a standalone
HTTP endpoint useful for smoke-testing the pipeline and for ad-hoc
queries during development.

The service re-exports ``preprocess`` so callers can either:
  - import it directly (fastest, no network):  ``from services.preprocessing.app.pipeline import preprocess``
  - call the HTTP endpoint (service-style):   ``POST http://localhost:8001/preprocess``

Both paths use the **same** implementation. This is the single source
of truth guarantee from guide §1.5.
"""

from __future__ import annotations

import sys

# Force UTF-8 on Windows before any logging/output.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

import ir_datasets
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.ir_common.preprocess import (
    PIPELINE_STEPS,
    preprocess,
)

# Re-export for callers that want a service-style import.
__all__ = ["app", "preprocess", "PIPELINE_STEPS"]


class PreprocessRequest(BaseModel):
    """Body for ``POST /preprocess``."""

    text: str = Field(..., min_length=0, description="Raw input text to tokenize.")


class PreprocessResponse(BaseModel):
    """Body for ``POST /preprocess`` response."""

    tokens: list[str] = Field(..., description="Stemmed tokens, ready for indexing/search.")


app = FastAPI(
    title="IR Preprocessing Service",
    version="0.1.0",
    description=(
        "Tokenize raw text into stemmed tokens for the IR pipeline. "
        "Phase 1 of the IR project; consumed by the gateway in Phase 6."
    ),
)

# CORS tightened in Phase 6 to the local UI origins. The gateway
# (port 8000) and the React dev server (port 5173) are the only
# legitimate callers in production; same-host (:3000 / :5173) covers
# `docker compose up` + `npm run dev` scenarios.
_LOCAL_UI_ORIGINS = (
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_LOCAL_UI_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 if the process is up."""
    return {"status": "ok", "service": "preprocessing"}


@app.get("/pipeline")
def pipeline_steps() -> dict[str, list[str]]:
    """Expose the preprocessing pipeline steps (for the UI's debug panel)."""
    return {"steps": list(PIPELINE_STEPS)}


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_endpoint(req: PreprocessRequest) -> PreprocessResponse:
    """Run the preprocessing pipeline on a single string.

    Returns the stemmed token list. Empty / whitespace-only input returns ``[]``.
    """
    return PreprocessResponse(tokens=preprocess(req.text))


# ─────────────────────────────────────────────────────────────────────────
# Document-by-ID retrieval (Phase 7.5)
# ─────────────────────────────────────────────────────────────────────────

_DS_ID_TO_BEIR = {
    "touche2020": "beir/webis-touche2020",
    "nq": "beir/nq",
}
_doc_stores: dict[str, ir_datasets.DocStore] = {}


def _get_doc_store(dataset_id: str) -> ir_datasets.DocStore:
    """Lazy-load and cache the ``docs_store`` for a dataset.

    The ``ir_datasets`` cache (``~/.ir_datasets/``) is the single
    source of truth for document text. The raw :file:`docs.jsonl`
    in ``data/processed/`` exists too, but the pklz4 full-store
    gives O(1) random access by doc_id with no RAM overhead.
    """
    if dataset_id not in _doc_stores:
        beir_id = _DS_ID_TO_BEIR.get(dataset_id)
        if beir_id is None:
            raise HTTPException(400, f"Unknown dataset_id {dataset_id!r}")
        _doc_stores[dataset_id] = ir_datasets.load(beir_id).docs_store()
    return _doc_stores[dataset_id]


@app.get("/docs/{dataset_id}/{doc_id}")
def get_doc(dataset_id: str, doc_id: str) -> dict[str, str]:
    """Return the full text of a single document by its ID.

    ``dataset_id`` must be one of the canonical dataset IDs
    (``touche2020`` / ``nq``). ``doc_id`` is the opaque string
    returned by the search endpoints (e.g.
    ``bb913bfc-2019-04-18T17:06:16Z-00009-000``).

    Responses
    ---------
    * ``200`` — ``{"id": str, "text": str}``
    * ``400`` — unknown ``dataset_id``
    * ``404`` — ``doc_id`` not found in the dataset
    """
    if dataset_id not in _DS_ID_TO_BEIR:
        raise HTTPException(400, f"Unknown dataset_id {dataset_id!r}")
    try:
        store = _get_doc_store(dataset_id)
        doc = store.get(doc_id)
    except Exception as exc:
        raise HTTPException(404, f"doc_id {doc_id!r} not found: {exc}") from exc
    if doc is None:
        raise HTTPException(404, f"doc_id {doc_id!r} not found")
    return {"id": doc.doc_id, "text": doc.text}


# CLI runner: ``python -m services.preprocessing.app.pipeline`` -> uvicorn on :8001
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.preprocessing.app.pipeline:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )

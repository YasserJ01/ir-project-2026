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

from fastapi import FastAPI
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


# CLI runner: ``python -m services.preprocessing.app.pipeline`` -> uvicorn on :8001
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.preprocessing.app.pipeline:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )

"""FastAPI service for query refinement (Phase 4).

Runs on port 8004. The single public endpoint is ``POST /refine``
plus the standard ``GET /health`` + ``GET /`` for hand-testing.

The service is intentionally **stateless** across requests: SymSpell
+ WordNet are loaded once at import time (or first /refine, if
``EAGER_INIT=False``), and the user-log directory is the only
on-disk state.

A word on the 422 contract: ``RefineRequest`` allows ``extra="ignore"``
so a Phase 5 caller can pass a ``dataset_id`` field without us
rejecting it. That keeps the wire format forward-compatible as
later phases layer in.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.refinement.app.config import (
    EAGER_INIT,
    USER_LOG_DIR,
)
from services.refinement.app.pipeline import build_pipeline, measure_latency_ms
from shared.ir_common.schemas import (
    RefinementHealthResponse,
    RefineRequest,
    RefineResponse,
)

__all__ = ["app", "run"]

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Eagerly initialize the pipeline at startup if ``EAGER_INIT`` is true."""
    if EAGER_INIT:
        logger.info("Eagerly building the refinement pipeline...")
        # The factory is idempotent; we just touch it so any import /
        # download error surfaces *here* rather than on the first
        # /refine call.
        build_pipeline()
        logger.info("Refinement pipeline ready.")
    yield


app = FastAPI(
    title="Query Refinement Service",
    version="0.1.0",
    description=(
        "Phase 4. Takes a raw user query, runs it through grammar → spell "
        "→ synonyms → personalization, and returns a preprocessed token "
        "list with per-token weights."
    ),
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────
# Public endpoints
# ─────────────────────────────────────────────────────────────────────────


@app.get("/")
def root() -> dict:
    """Tiny landing page for hand-testing from a browser."""
    return {
        "service": "refinement",
        "version": "0.1.0",
        "endpoints": [
            "GET  /health",
            "POST /refine",
        ],
        "see": "docs/PHASE_4.md",
    }


@app.get("/health", response_model=RefinementHealthResponse)
def health() -> RefinementHealthResponse:
    """Return service liveness + which stage modules are loaded.

    We don't *call* SymSpell/WordNet/LanguageTool here -- a successful
    import is enough proof of liveness. The per-request latency field
    on ``/refine`` is the real performance signal.
    """
    # Probe the lazy importers by calling their no-op factories.
    from services.refinement.app.grammar import is_grammar_enabled
    from services.refinement.app.spell import build_spell_corrector
    from services.refinement.app.synonyms import build_synonym_expander

    spell_loaded = False
    try:
        build_spell_corrector()
        spell_loaded = True
    except FileNotFoundError:
        pass
    wordnet_loaded = False
    try:
        build_synonym_expander()
        wordnet_loaded = True
    except LookupError:
        pass
    return RefinementHealthResponse(
        spell_loaded=spell_loaded,
        wordnet_loaded=wordnet_loaded,
        grammar_loaded=is_grammar_enabled(),
        grammar_enabled=is_grammar_enabled(),
        user_log_dir=str(USER_LOG_DIR),
    )


@app.post("/refine", response_model=RefineResponse)
def refine(request: RefineRequest) -> RefineResponse:
    """Run the full refinement pipeline on ``request.query``."""
    start = time.perf_counter()
    pipeline = build_pipeline()
    result = pipeline.run(request)
    latency_ms = measure_latency_ms(start)
    logger.info(
        "refine user=%r q=%r latency_ms=%d tokens=%d",
        request.user_id,
        request.query[:60],
        latency_ms,
        len(result.tokens),
    )
    return RefineResponse(
        query=request.query,
        refined_query=result.refined_query,
        expanded_query=result.expanded_query,
        tokens=result.tokens,
        weighted_tokens=result.weighted_tokens,
        stages=result.stages,
        latency_ms=latency_ms,
        user_id=request.user_id,
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────────────────────


def run() -> None:
    """``python -m services.refinement.app.service`` -- uvicorn entrypoint."""
    import uvicorn

    uvicorn.run(
        "services.refinement.app.service:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    run()

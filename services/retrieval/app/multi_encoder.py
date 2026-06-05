"""Multi-encoder parallel rank for the retrieval service (:8003).

Phase 5 bonus (guide §5.3 + spec line 34). Fuses two SBERT encoders in
parallel by:
  1. Encoding the query with each of the two encoders.
  2. Searching each encoder's corresponding FAISS index.
  3. Fusing the two ranked lists with RRF / CombSUM / CombMNZ.

The two encoders are L6 and L12 (both 384-dim). We assume the FAISS
indexes share the same ``doc_ids.json`` (corpus order is identical
across builds -- the only thing that changes is the embedding vectors).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import numpy as np

from services.retrieval.app import config as _config
from services.retrieval.app.config import (
    DEFAULT_MODEL_NAME,
    SECOND_ENCODER_EMBEDDINGS_FILENAME,
    SECOND_ENCODER_INDEX_FILENAME,
    SECOND_ENCODER_NAME,
)
from services.retrieval.app.fusion import (
    FusedHit,
    RankedHit,
    fuse,
)
from shared.ir_common.schemas import (
    HybridSearchHit,
    HybridSearchResponse,
    MultiEncoderSearchRequest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MultiEncoderRunner",
    "MultiEncoderError",
    "MultiEncoderTimings",
    "DEFAULT_ENCODER_1",
    "DEFAULT_ENCODER_2",
]

# Public aliases so the service layer and the smoke test refer to the
# same names.
DEFAULT_ENCODER_1: str = DEFAULT_MODEL_NAME  # L6
DEFAULT_ENCODER_2: str = SECOND_ENCODER_NAME  # L12


class MultiEncoderError(Exception):
    """Raised when the multi-encoder path can't run.

    The HTTP layer translates this to a 503 (build pending) or 500
    (encoder failure).
    """

    def __init__(self, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


# ─────────────────────────────────────────────────────────────────────────
# Type alias for the encode+search function
# ─────────────────────────────────────────────────────────────────────────

# Multi-encoder's search path: (query_text, dataset_id, model_name, k) -> (scores, doc_ids)
# This is the same shape as the dense service's /retrieval/{ds}/search
# but takes an explicit model_name (so the same function can serve
# both L6 and L12).
MultiEncoderSearchFn = Callable[
    [str, str, str, int], Coroutine[Any, Any, tuple[list[float], list[str]]]
]


@dataclass
class MultiEncoderTimings:
    """Per-encoder wall-clock latencies in milliseconds."""

    timings: dict[str, int]

    def to_dict(self) -> dict[str, int]:
        return dict(self.timings)


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────


class MultiEncoderRunner:
    """Run two encoders in parallel + fuse the results.

    The :class:`MultiEncoderSearchFn` is injected (the production
    implementation is built in :mod:`service.py`).
    """

    def __init__(self, search_fn: MultiEncoderSearchFn) -> None:
        self._search = search_fn

    async def search(
        self,
        dataset_id: str,
        req: MultiEncoderSearchRequest,
    ) -> HybridSearchResponse:
        encoder_1 = req.encoder_1 or DEFAULT_ENCODER_1
        encoder_2 = req.encoder_2 or DEFAULT_ENCODER_2
        if encoder_1 == encoder_2:
            raise MultiEncoderError(
                "encoder_1 and encoder_2 must be different models; got " f"{encoder_1!r} for both",
                status_code=400,
            )
        if not _config.has_second_encoder_index(dataset_id):
            raise MultiEncoderError(
                f"Second-encoder index not built for {dataset_id!r}. "
                "Run `make download-second-model` then `make build-dense-2`.",
                status_code=503,
            )

        started = time.perf_counter()
        t1 = time.perf_counter()
        t2 = time.perf_counter()

        async def _do(name: str) -> tuple[list[float], list[str]]:
            return await self._search(req.query, dataset_id, name, req.k)

        # Run both encoders in parallel.
        (s1, ids1), (s2, ids2) = await _run_parallel(
            _do(encoder_1),
            _do(encoder_2),
        )
        ms1 = int((time.perf_counter() - t1) * 1000)
        ms2 = int((time.perf_counter() - t2) * 1000)
        # Use a short name for the timings dict.
        short_1 = _short_name(encoder_1)
        short_2 = _short_name(encoder_2)
        timings = MultiEncoderTimings(
            timings={short_1: ms1, short_2: ms2},
        )

        # Build RankedHit lists for the fusion functions.
        ranked_1 = [RankedHit(doc_id=d, score=float(s)) for s, d in zip(s1, ids1, strict=True)]
        ranked_2 = [RankedHit(doc_id=d, score=float(s)) for s, d in zip(s2, ids2, strict=True)]
        fused: list[FusedHit] = fuse({short_1: ranked_1, short_2: ranked_2}, method=req.fusion)
        fused = fused[: req.k]

        hits = [
            HybridSearchHit(
                rank=i + 1,
                doc_id=fh.doc_id,
                score=fh.score,
                individual_scores=dict(fh.individual_scores),
            )
            for i, fh in enumerate(fused)
        ]
        total_ms = int((time.perf_counter() - started) * 1000)
        return HybridSearchResponse(
            dataset_id=dataset_id,
            representation="embedding",  # closest sibling; not "multi_encoder" in the type
            fusion=req.fusion,
            k=req.k,
            latency_ms=total_ms,
            results=hits,
            per_retriever_latency_ms=timings.to_dict(),
            refined_query=None,
            refinement_fell_back=False,
            stages={
                short_1: f"top-{req.k} (encoder={encoder_1})",
                short_2: f"top-{req.k} (encoder={encoder_2})",
                "fuse": f"{req.fusion}",
            },
        )


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


async def _run_parallel(
    *coros: Coroutine[Any, Any, Any],
) -> list[Any]:
    """Run a sequence of coros in parallel and return their results.

    Equivalent to ``asyncio.gather(*coros)`` but spelled out so the
    intent is obvious in the orchestrator. Returns a list (not a
    tuple) to match ``asyncio.gather``'s return type.
    """
    return list(await asyncio.gather(*coros))


def _short_name(model_name: str) -> str:
    """Map a Hugging Face model name to a short key for individual_scores.

    The keys have to be short because they appear in the JSON response
    for every hit. The mapping is:

    * ``sentence-transformers/all-MiniLM-L6-v2``  -> ``l6``
    * ``sentence-transformers/all-MiniLM-L12-v2`` -> ``l12``
    * anything else                               -> the full model name

    The mapping is by *string equality* (not regex) so an unexpected
    model name goes through unchanged. This keeps the API stable for
    the common case while not silently miscategorising a custom model.
    """
    if model_name == DEFAULT_MODEL_NAME:
        return "l6"
    if model_name == SECOND_ENCODER_NAME:
        return "l12"
    return model_name


# ─────────────────────────────────────────────────────────────────────────
# Factory for the production search closure
# ─────────────────────────────────────────────────────────────────────────


def build_default_multi_encoder_search() -> MultiEncoderSearchFn:
    """Build the production :data:`MultiEncoderSearchFn` closure.

    Used by :mod:`service.py`. The closure wraps the live ``Embedder``
    and :class:`DenseIndex` -- one call to ``_load_faiss`` for the L6
    index, one for the L12 index.

    The L12 index is loaded from
    ``data/indexes/<dataset_id>/faiss_l12.index`` + ``embeddings_l12.npy``
    (see :func:`services.retrieval.app.config.second_encoder_index_path`).
    """
    from services.retrieval.app import service as service_mod

    async def _default_multi_encoder_search(
        query_text: str,
        dataset_id: str,
        model_name: str,
        k: int,
    ) -> tuple[list[float], list[str]]:
        emb = service_mod._embedder()
        t0 = time.perf_counter()
        # Encode the query with the requested model.
        q_vec: np.ndarray = emb.encode_query(query_text, model_name=model_name)
        _ = int((time.perf_counter() - t0) * 1000)  # encode_ms (unused)

        # Pick the right FAISS index based on the encoder name.
        if model_name == SECOND_ENCODER_NAME:
            idx = service_mod._load_faiss(
                dataset_id,
                index_filename=SECOND_ENCODER_INDEX_FILENAME,
                embeddings_filename=SECOND_ENCODER_EMBEDDINGS_FILENAME,
            )
        else:
            idx = service_mod._load_faiss(dataset_id)
        scores, ids = idx.search(q_vec, k)
        scores_list = [float(s) for s in scores]
        doc_ids_list = [idx.doc_ids[int(i)] for i in ids if int(i) >= 0]
        # Trim to the same length (FAISS pads with -1 for empty slots).
        n = min(len(scores_list), len(doc_ids_list))
        return scores_list[:n], doc_ids_list[:n]

    return _default_multi_encoder_search

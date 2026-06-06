"""Gateway-side schemas (Phase 6).

Most request/response models are re-exported from
``shared.ir_common.schemas`` so the gateway's wire format is identical
to the backend services' wire format. The gateway-specific models
(``GatewayHealthResponse``, ``GatewayErrorResponse``) live in the
shared module so other services can construct them too.

We also define :class:`GatewaySearchRequest` here — the body for
``POST /api/search``. The shared :class:`~ir_common.schemas.SearchRequest`
makes ``query`` and ``dataset_id`` optional (so backend services can
serve multiple call sites), but the gateway's contract is stricter:
both fields are **required**, so Pydantic returns 422 on a missing
field (rather than the gateway having to re-validate).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.ir_common.schemas import (
    DATASET_IDS,
    GatewayErrorResponse,
    GatewayHealthResponse,
    HybridSearchHit,
    HybridSearchRequest,
    HybridSearchResponse,
    LogClickRequest,
    MultiEncoderSearchRequest,
    RefinementHealthResponse,
    RefineRequest,
    RefineResponse,
)


class GatewaySearchRequest(BaseModel):
    """Body for ``POST /api/search``.

    ``query`` and ``dataset_id`` are required so Pydantic returns 422
    on missing/invalid fields. Backend services are more lenient and
    accept these fields as optional.
    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, max_length=2048)
    dataset_id: Literal["touche2020", "nq"]
    representation: Literal["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"] = (
        "bm25"
    )
    k: int = Field(default=10, ge=1, le=200)
    mode: Literal["basic", "with_features"] = "basic"
    fusion: Literal["rrf", "combsum", "combmnz"] = "rrf"
    user_id: str | None = Field(default=None, max_length=64)
    enable_grammar: bool = False


__all__ = [
    "DATASET_IDS",
    "GatewayErrorResponse",
    "GatewayHealthResponse",
    "GatewaySearchRequest",
    "HybridSearchHit",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "LogClickRequest",
    "MultiEncoderSearchRequest",
    "RefineRequest",
    "RefineResponse",
    "RefinementHealthResponse",
]

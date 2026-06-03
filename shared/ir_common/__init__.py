"""Shared package for cross-service types and utilities.

Phase 1 ships the preprocessing library. Phase 2 adds Pydantic schemas
(``SearchRequest``/``SearchResponse``/etc.) consumed by the indexing
service, the gateway (Phase 6), and the React UI (Phase 7).
"""

from shared.ir_common.preprocess import (
    PIPELINE_STEPS,
    drop_non_alpha,
    drop_short,
    preprocess,
    preprocess_batch,
    preprocess_cached,
    remove_stopwords,
    stem_tokens,
    strip_html,
    tokenize,
)
from shared.ir_common.schemas import (
    DATASET_IDS,
    BuildRequest,
    BuildResponse,
    HealthResponse,
    Posting,
    PostingsRequest,
    PostingsResponse,
    SearchModel,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

__all__ = [
    # Preprocessing (Phase 1)
    "PIPELINE_STEPS",
    "preprocess",
    "preprocess_batch",
    "preprocess_cached",
    "drop_non_alpha",
    "drop_short",
    "remove_stopwords",
    "stem_tokens",
    "strip_html",
    "tokenize",
    # Schemas (Phase 2)
    "DATASET_IDS",
    "BuildRequest",
    "BuildResponse",
    "HealthResponse",
    "Posting",
    "PostingsRequest",
    "PostingsResponse",
    "SearchModel",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "StatsResponse",
]

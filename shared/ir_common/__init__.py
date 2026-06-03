"""Shared package for cross-service types and utilities.

Phase 1 ships the preprocessing library. Later phases add pydantic
schemas (Phase 5 search requests/responses) and shared config.
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

__all__ = [
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
]

"""Indexing service.

Phase 2 of the IR project. Provides three retrieval primitives (Inverted
Index, TF-IDF, BM25) behind a FastAPI service on port 8002.

The single source of truth for tokenization remains
``shared.ir_common.preprocess`` (Phase 1). This package consumes the
``tokens.jsonl`` files produced by ``scripts/tokenize_corpus.py`` and
builds on-disk artifacts under ``data/indexes/<dataset_id>/``.

Public entry points:
  - ``app.service.app`` -- the FastAPI application
  - ``app.service.run`` -- CLI helper to run the service via uvicorn
"""

from __future__ import annotations

from shared.ir_common.schemas import DATASET_IDS

__all__ = ["DATASET_IDS"]

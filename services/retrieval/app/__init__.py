"""Dense-retrieval service.

Phase 3 of the IR project. Provides a sentence-transformer
encoder + FAISS ``IndexFlatIP`` behind a FastAPI service on port
8003.

The single source of truth for **tokenization** remains
``shared.ir_common.preprocess`` (Phase 1), but dense retrieval does
*not* use it -- the sentence-transformer has its own WordPiece BPE
tokenizer and expects natural text. The build script reads
``data/processed/{ds}/docs.jsonl`` directly.

Public entry points:
  - ``app.service.app`` -- the FastAPI application
  - ``app.service.run`` -- CLI helper to run the service via uvicorn
"""

from __future__ import annotations

from shared.ir_common.schemas import DATASET_IDS

__all__ = ["DATASET_IDS"]

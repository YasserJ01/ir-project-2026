from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("RAG_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_pipe: Any = None


def _load() -> None:
    global _pipe
    import torch
    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading %s on %s (dtype=%s)", MODEL_ID, device, dtype)
    _pipe = hf_pipeline(
        "text-generation",
        model=MODEL_ID,
        device=device,
        torch_dtype=dtype,
    )
    logger.info("Model loaded successfully")


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    if _pipe is None:
        _load()

    out = _pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()  # type: ignore[index]

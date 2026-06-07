from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_ID = os.environ.get("RAG_MODEL", HF_MODEL_ID)

LOCAL_MODEL_DIR = (
    Path(os.environ.get("REPO_ROOT", r"F:\IR project"))
    / "data" / "models" / "tinyllama"
)
_pipe: Any = None


def _load() -> None:
    global _pipe

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    model_dir = Path(LOCAL_MODEL_DIR)
    is_cuda = torch.cuda.is_available()
    dtype = torch.float16 if is_cuda else torch.float32

    if model_dir.exists():
        logger.info("Loading local model on %s (dtype=%s)", "cuda" if is_cuda else "cpu", dtype)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.cuda().half() if is_cuda else model.float()
        model.eval()

        _pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if is_cuda else -1,
        )
        logger.info("Model loaded on %s (%.1f GB VRAM)", "cuda" if is_cuda else "cpu",
                     torch.cuda.memory_allocated() / 1e9 if is_cuda else 0)
        return

    # Fallback: download from HuggingFace
    logger.info("Downloading %s from HuggingFace", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = model.cuda().half() if is_cuda else model.float()
    model.eval()

    _pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if is_cuda else -1,
    )
    logger.info("Model loaded from HuggingFace")


_INSTRUCTION_PHRASES = (
    'if the answer is not in the context',
    'cite sources as [doc_id]',
    'use only the context below',
)


def _is_instruction_echo(text: str) -> bool:
    """Return True if *text* looks like the model echoed the system prompt
    instead of answering the question."""
    lower = text.lower()
    return any(p in lower for p in _INSTRUCTION_PHRASES)


def generate(prompt: str, max_new_tokens: int = 128) -> str:
    if _pipe is None:
        _load()

    out = _pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    raw = out[0]["generated_text"].strip()  # type: ignore[index]
    if _is_instruction_echo(raw):
        logger.warning("Model echoed instruction; falling back to 'I don't know'")
        return "I don't know based on the given documents."
    return raw

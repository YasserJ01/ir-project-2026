from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

LOCAL_MODEL_DIR = (
    Path(os.environ.get("REPO_ROOT", r"F:\IR project"))
    / "data" / "models" / "tinyllama"
)
_llm: Any = None


def _load() -> None:
    global _llm

    from llama_cpp import Llama

    model_path = str(LOCAL_MODEL_DIR / GGUF_FILENAME)
    if not os.path.isfile(model_path):
        logger.error("GGUF model not found at %s. Run scripts/dev/download_tinyllama_gguf.py first.", model_path)
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    logger.info("Loading GGUF model from %s (GPU offload)", model_path)
    _llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    logger.info("GGUF model loaded (GPU offload: all layers)")


_INSTRUCTION_PHRASES = (
    'if the answer is not in the context',
    'cite sources as [',
    'use only the context below',
)


def _is_instruction_echo(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _INSTRUCTION_PHRASES)


def generate(prompt: str, max_new_tokens: int) -> str:
    if _llm is None:
        _load()

    out = _llm(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        echo=False,
    )
    raw = out["choices"][0]["text"].strip()
    if _is_instruction_echo(raw):
        logger.warning("Model echoed instruction; falling back to 'I don't know'")
        return "I don't know based on the given documents."
    return raw


def generate_stream(prompt: str, max_new_tokens: int):
    """Yield ``{"token": str, "done": bool}`` dicts from the GGUF model.

    Each call yields one token at a time. After the final token the
    last dict has ``done=True`` and ``answer`` set. If the model
    echoes the system instruction an ``override`` event is yielded
    instead of the final token bundle.
    """
    if _llm is None:
        _load()

    stream = _llm(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        stream=True,
        echo=False,
    )
    full_text = ""
    for output in stream:
        token = output["choices"][0]["text"]
        full_text += token
        yield {"token": token, "done": False}

    text = full_text.strip()
    if _is_instruction_echo(text):
        logger.warning("Streaming: model echoed instruction; overriding")
        yield {"override": True, "answer": "I don't know based on the given documents."}
    else:
        yield {"token": "", "done": True, "answer": text}

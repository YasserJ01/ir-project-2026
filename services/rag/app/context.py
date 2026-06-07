from __future__ import annotations

MAX_CONTEXT_TOKENS = 2000


def build_context(docs: list[dict[str, str]]) -> str:
    """Build a context string from a list of ``{"id": ..., "text": ...}`` docs.

    Each document is formatted as ``[doc_id=xxx] Text...``, separated by
    blank lines. The total length is capped at ``MAX_CONTEXT_TOKENS``
    words (a rough heuristic — we split on whitespace rather than
    loading a real tokenizer).
    """
    parts: list[str] = []
    remaining = MAX_CONTEXT_TOKENS

    for doc in docs:
        doc_id = doc.get("id", "?")
        text = doc.get("text", "")
        words = text.split()
        if not words:
            continue
        take = min(len(words), remaining)
        chunk = " ".join(words[:take])
        parts.append(f"[doc_id={doc_id}] {chunk}")
        remaining -= take
        if remaining <= 0:
            break

    return "\n\n".join(parts)

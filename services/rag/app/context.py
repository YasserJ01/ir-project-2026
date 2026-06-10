from __future__ import annotations

MAX_CONTEXT_TOKENS = 800


def build_context(docs: list[dict[str, str]]) -> str:
    """Build a context string from a list of ``{"id": ..., "text": ...}`` docs.

    Each document is formatted as ``[N] Text...`` (N = 1-based index), separated by
    blank lines. The total length is capped at ``MAX_CONTEXT_TOKENS``
    words (a rough heuristic — we split on whitespace rather than
    loading a real tokenizer).
    """
    parts: list[str] = []
    remaining = MAX_CONTEXT_TOKENS

    for idx, doc in enumerate(docs, start=1):
        doc_id = doc.get("id", "?")
        text = doc.get("text", "")
        words = text.split()
        if not words:
            continue
        take = min(len(words), remaining)
        chunk = " ".join(words[:take])
        parts.append(f"[{idx}] {chunk}")
        remaining -= take
        if remaining <= 0:
            break

    return "\n\n".join(parts)

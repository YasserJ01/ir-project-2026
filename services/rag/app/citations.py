from __future__ import annotations

import re

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, source_doc_ids: list[str]) -> dict[str, str]:
    """Parse ``[N]`` citation markers from the model output and map them to
    actual ``doc_id`` values from ``source_doc_ids`` (1-indexed).

    Returns a dict like ``{"1": "doc-abc", "2": "doc-xyz"}``. Malformed
    (out-of-range) citations are silently dropped.
    """
    citations: dict[str, str] = {}
    for match in _CITATION_PATTERN.finditer(answer):
        idx = int(match.group(1))
        if 1 <= idx <= len(source_doc_ids):
            citations[match.group(1)] = source_doc_ids[idx - 1]
    return citations

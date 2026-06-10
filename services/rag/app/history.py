from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTORY_TOKENS = 400
_MAX_TURNS = 3
_EOS = "</s>"

_Turn = dict[str, Any]


class ConversationStore:
    """In-memory conversation history store.

    Each conversation is a list of turns (``{"role": ..., "text": ...,
    "source_doc_ids": [...]}``). History is pruned by turn count and
    a rough word-count token budget.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[_Turn]] = {}

    def push(
        self,
        conversation_id: str,
        role: str,
        text: str,
        source_doc_ids: list[str] | None = None,
    ) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(
            {"role": role, "text": text, "source_doc_ids": source_doc_ids or []}
        )

    def format_history(self, conversation_id: str) -> str:
        """Return history as a TinyLlama-formatted string for prompt injection.

        Returns empty string if the conversation doesn't exist or has no
        turns. Prunes to the last ``_MAX_TURNS`` turns and the last
        ``_MAX_HISTORY_TOKENS`` rough-word tokens.
        """
        turns = self._store.get(conversation_id)
        if not turns:
            return ""

        # Keep only the last _MAX_TURNS turns (must be pairs: user + assistant)
        turns = turns[-_MAX_TURNS * 2 :]

        # Prune by token budget (rough word count)
        parts: list[str] = []
        remaining = _MAX_HISTORY_TOKENS
        for turn in reversed(turns):
            words = turn["text"].split()
            chunk = " ".join(words[:remaining])
            role = turn["role"]
            if role == "user":
                parts.append(f"<|user|>\nPrevious question: {chunk}{_EOS}\n")
            elif role == "assistant":
                parts.append(
                    f"<|assistant|>\nPrevious answer: {chunk}{_EOS}\n"
                )
            remaining -= len(words)
            if remaining <= 0:
                break

        parts.reverse()
        return "".join(parts)


# Module-level singleton so all request handlers share one store.
_store: ConversationStore | None = None


def get_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store

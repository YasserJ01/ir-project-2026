from __future__ import annotations

from services.rag.app.citations import extract_citations
from services.rag.app.history import ConversationStore


def test_extract_citations_basic() -> None:
    result = extract_citations("Paris is the capital [1].", ["doc-abc"])
    assert result == {"1": "doc-abc"}


def test_extract_citations_multiple() -> None:
    result = extract_citations(
        "France [1] is in Europe [2]. The capital is Paris [1].",
        ["doc-france", "doc-europe"],
    )
    assert result == {"1": "doc-france", "2": "doc-europe"}


def test_extract_citations_out_of_range() -> None:
    result = extract_citations("Out of range [5].", ["doc-a", "doc-b"])
    assert result == {}


def test_extract_citations_no_matches() -> None:
    result = extract_citations("No citations here.", ["doc-a"])
    assert result == {}


def test_extract_citations_mixed_valid_invalid() -> None:
    result = extract_citations(
        "Valid [1] and invalid [99].",
        ["doc-valid"],
    )
    assert result == {"1": "doc-valid"}


def test_extract_citations_adjacent() -> None:
    result = extract_citations("Both [1][2] here.", ["doc-a", "doc-b"])
    assert result == {"1": "doc-a", "2": "doc-b"}


def test_extract_citations_empty_answer() -> None:
    result = extract_citations("", ["doc-a"])
    assert result == {}


def test_extract_citations_empty_sources() -> None:
    result = extract_citations("Citation [1] here.", [])
    assert result == {}


def test_history_push_and_format() -> None:
    store = ConversationStore()
    conv_id = "test-1"
    store.push(conv_id, "user", "What is the capital of France?")
    store.push(conv_id, "assistant", "Paris is the capital [1].", ["doc-paris"])

    formatted = store.format_history(conv_id)
    assert "Previous question" in formatted
    assert "Previous answer" in formatted
    assert "Paris is the capital" in formatted


def test_history_unknown_returns_empty() -> None:
    store = ConversationStore()
    assert store.format_history("nonexistent") == ""


def test_history_empty_returns_empty() -> None:
    store = ConversationStore()
    store.push("empty-conv", "user", "hi")
    store.format_history("empty-conv")  # should not crash


def test_history_prune_to_max_turns() -> None:
    store = ConversationStore()
    conv_id = "prune-test"
    for i in range(5):
        store.push(conv_id, "user", f"Question {i}")
        store.push(conv_id, "assistant", f"Answer {i}")

    formatted = store.format_history(conv_id)
    assert "Question 0" not in formatted
    assert "Question 1" not in formatted
    assert "Question 3" in formatted
    assert "Question 4" in formatted
    assert formatted.count("Previous question") <= 3


def test_history_multiple_conversations_independent() -> None:
    store = ConversationStore()
    store.push("conv-a", "user", "Q from A")
    store.push("conv-b", "user", "Q from B")

    fa = store.format_history("conv-a")
    fb = store.format_history("conv-b")
    assert "Q from A" in fa
    assert "Q from B" not in fa
    assert "Q from B" in fb
    assert "Q from A" not in fb

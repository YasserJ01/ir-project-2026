from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from services.rag.app.service import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "rag"}


def test_answer_unknown_dataset_returns_422(client: TestClient) -> None:
    r = client.post(
        "/rag/answer",
        json={"dataset_id": "unknown", "query": "test", "k": 3},
    )
    assert r.status_code == 422


@patch("services.rag.app.service.search_retrieval")
@patch("services.rag.app.service.fetch_doc_text")
@patch("services.rag.app.service.generate")
def test_answer_returns_rag_response(
    mock_generate, mock_fetch_doc, mock_search, client: TestClient
) -> None:
    mock_search.return_value = [
        {"doc_id": "doc-1", "score": 1.0},
        {"doc_id": "doc-2", "score": 0.8},
    ]
    mock_fetch_doc.side_effect = [
        {"id": "doc-1", "text": "The capital of France is Paris."},
        {"id": "doc-2", "text": "France is a country in Europe."},
    ]
    mock_generate.return_value = (
        "Based on the provided context, the capital of France is Paris. [doc_id=doc-1]"
    )

    r = client.post(
        "/rag/answer",
        json={"dataset_id": "touche2020", "query": "What is the capital of France?", "k": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert "Paris" in body["answer"]
    assert "doc-1" in body["source_doc_ids"]
    assert "doc-2" in body["source_doc_ids"]
    assert body["latency_ms"] > 0
    assert isinstance(body["latency_ms"], float | int)


@patch("services.rag.app.service.search_retrieval")
def test_answer_empty_results_returns_dont_know(mock_search, client: TestClient) -> None:
    mock_search.return_value = []

    r = client.post(
        "/rag/answer",
        json={"dataset_id": "touche2020", "query": "Nothing in corpus", "k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "don't know" in body["answer"].lower()
    assert body["source_doc_ids"] == []


@patch("services.rag.app.service.search_retrieval")
def test_answer_retrieval_error_returns_502(mock_search, client: TestClient) -> None:
    from services.rag.app.rag_client import RagClientError

    mock_search.side_effect = RagClientError("retrieval down")

    r = client.post(
        "/rag/answer",
        json={"dataset_id": "touche2020", "query": "test", "k": 3},
    )
    assert r.status_code == 502


def test_is_instruction_echo_detects_instruction() -> None:
    from services.rag.app.generator import _is_instruction_echo

    assert _is_instruction_echo(
        'Based on the given documents, the answer is: '
        '- If the answer is not in the context, say "I don\'t know."'
    )
    assert _is_instruction_echo(
        'Cite sources as [doc_id].'
    )
    assert _is_instruction_echo(
        'Use ONLY the context below'
    )
    assert not _is_instruction_echo(
        'Climate change is caused by human activities. [doc_id=abc-123]'
    )
    assert not _is_instruction_echo(
        'Paris is the capital of France.'
    )


@patch("services.rag.app.generator._llm")
def test_generator_instruction_guard_catches_echo(mock_llm) -> None:
    from services.rag.app.generator import generate

    mock_llm.return_value = {
        "choices": [{
            "text": (
                'Based on the given documents, the answer is: \n'
                '- If the answer is not in the context, say "I don\'t know."\n'
                '- Cite sources as [doc_id].'
            ),
            "index": 0,
            "finish_reason": "stop",
        }]
    }
    result = generate("test prompt")
    assert "don't know" in result.lower()
    assert "If the answer is not in the context" not in result


@patch("services.rag.app.service.search_retrieval")
@patch("services.rag.app.service.fetch_doc_text")
@patch("services.rag.app.service.generate")
def test_answer_partial_missing_docs(
    mock_generate, mock_fetch_doc, mock_search, client: TestClient
) -> None:
    mock_search.return_value = [
        {"doc_id": "doc-1", "score": 1.0},
        {"doc_id": "doc-404", "score": 0.5},
    ]
    mock_fetch_doc.side_effect = [
        {"id": "doc-1", "text": "Some text."},
        {"id": "doc-404", "text": ""},  # missing doc returns empty
    ]
    mock_generate.return_value = "Answer with [doc_id=doc-1]."

    r = client.post(
        "/rag/answer",
        json={"dataset_id": "touche2020", "query": "test", "k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source_doc_ids"] == ["doc-1", "doc-404"]

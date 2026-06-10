from __future__ import annotations

import json
import logging
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from shared.ir_common.schemas import DATASET_IDS, RagRequest, RagResponse

from .citations import extract_citations
from .context import build_context
from .generator import generate, generate_stream
from .history import get_store
from .rag_client import RagClientError, fetch_doc_text, refine_query, search_retrieval

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise assistant. Use ONLY the context below.\n"
    'If the answer is not in the context, say "I don\'t know based on the given documents."\n'
    "Cite sources as [1], [2], etc. matching the document numbers below."
)

_EOS = "</s>"


def _build_prompt(query: str, context: str, history_str: str) -> str:
    """Build a TinyLlama chat prompt from history + context + query."""
    parts = [f"<|system|>\n{_SYSTEM_PROMPT}{_EOS}\n"]
    if history_str:
        parts.append(history_str)
    if context:
        parts.append(f"<|user|>\nContext:\n{context}\n\n")
    parts.append(f"Question: {query}{_EOS}\n<|assistant|>\n")
    return "".join(parts)


def _doc_ids_and_docs(results: list[dict], dataset_id: str) -> tuple[list[str], list[dict[str, str]]]:
    """Extract parallel ``doc_ids`` / ``docs`` lists from the retrieval results."""
    doc_ids: list[str] = []
    docs: list[dict[str, str]] = []
    for hit in results:
        doc_id = hit.get("doc_id", "")
        if not doc_id:
            continue
        doc_ids.append(doc_id)
        try:
            doc = fetch_doc_text(dataset_id, doc_id)
        except RagClientError:
            doc = {"id": doc_id, "text": ""}
        docs.append(doc)
    return doc_ids, docs


app = FastAPI(
    title="IR RAG Service",
    version="0.1.0",
    description="RAG service (Phase 8). Uses TinyLlama-1.1B for grounded answer generation.",
)

_LOCAL_UI_ORIGINS = (
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_LOCAL_UI_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "rag"}


@app.post("/rag/answer", response_model=RagResponse)
def answer(req: RagRequest) -> RagResponse:
    if req.dataset_id not in DATASET_IDS:
        raise HTTPException(400, f"Unknown dataset_id {req.dataset_id!r}")

    t0 = time.perf_counter()

    # 1. Expand query via refinement service (spell + synonyms)
    expanded_query = refine_query(req.query)

    # 2. Retrieve top-k docs from the retrieval service
    try:
        results = search_retrieval(req.dataset_id, expanded_query, k=req.k, representation=req.retriever)
    except RagClientError as exc:
        raise HTTPException(502, str(exc)) from exc

    if not results:
        refined = expanded_query if expanded_query != req.query else None
        return RagResponse(
            answer="I don't know based on the given documents.",
            source_doc_ids=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
            refined_query=refined,
            citations={},
        )

    # 3. Fetch full text for each result
    doc_ids, docs = _doc_ids_and_docs(results, req.dataset_id)

    # 4. Build context window
    context = build_context(docs)

    # 5. Inject conversation history (if any)
    history_str = ""
    if req.conversation_id:
        store = get_store()
        history_str = store.format_history(req.conversation_id)

    # 6. Format prompt with TinyLlama chat template
    prompt = _build_prompt(req.query, context, history_str)

    # 7. Generate answer
    try:
        answer_text = generate(prompt, max_new_tokens=req.max_tokens)
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}") from exc

    elapsed = (time.perf_counter() - t0) * 1000

    # 8. Extract citations
    answer_text = answer_text or "I don't know based on the given documents."
    citations = extract_citations(answer_text, doc_ids)

    # 9. Save conversation turn
    if req.conversation_id:
        store = get_store()
        store.push(req.conversation_id, "user", req.query, doc_ids)
        store.push(req.conversation_id, "assistant", answer_text, doc_ids)

    refined = expanded_query if expanded_query != req.query else None
    return RagResponse(
        answer=answer_text,
        source_doc_ids=doc_ids,
        latency_ms=round(elapsed, 1),
        refined_query=refined,
        citations=citations,
    )


@app.post("/rag/answer/stream")
async def answer_stream(req: RagRequest):
    """SSE streaming endpoint — yields tokens as the model generates them."""
    if req.dataset_id not in DATASET_IDS:
        raise HTTPException(400, f"Unknown dataset_id {req.dataset_id!r}")

    t0 = time.perf_counter()

    # 1. Expand query via refinement service (spell + synonyms)
    expanded_query = refine_query(req.query)

    # 2. Retrieve top-k docs from the retrieval service
    try:
        results = search_retrieval(req.dataset_id, expanded_query, k=req.k, representation=req.retriever)
    except RagClientError as exc:
        raise HTTPException(502, str(exc)) from exc

    doc_ids: list[str] = []
    docs: list[dict[str, str]] = []

    if results:
        doc_ids, docs = _doc_ids_and_docs(results, req.dataset_id)
        context = build_context(docs)
        history_str = ""
        if req.conversation_id:
            history_str = get_store().format_history(req.conversation_id)
        prompt = _build_prompt(req.query, context, history_str)
    else:
        prompt = ""

    refined = expanded_query if expanded_query != req.query else None

    async def event_stream():
        yield f"data: {json.dumps({'stage': 'retrieval', 'source_doc_ids': doc_ids, 'refined_query': refined})}\n\n"

        if not prompt:
            elapsed = (time.perf_counter() - t0) * 1000
            fallback = "I don't know based on the given documents."
            yield f"data: {json.dumps({'done': True, 'answer': fallback, 'source_doc_ids': [], 'latency_ms': round(elapsed, 1), 'refined_query': refined, 'citations': {}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        full_text = ""
        for chunk in generate_stream(prompt, max_new_tokens=req.max_tokens):
            if chunk.get("override"):
                yield f"data: {json.dumps({'done': True, 'answer': chunk['answer'], 'source_doc_ids': doc_ids, 'latency_ms': 0, 'refined_query': refined, 'citations': {}})}\n\n"
            elif chunk.get("done"):
                elapsed = (time.perf_counter() - t0) * 1000
                answer_text = chunk["answer"]
                citations = extract_citations(answer_text, doc_ids)
                if req.conversation_id:
                    store = get_store()
                    store.push(req.conversation_id, "user", req.query, doc_ids)
                    store.push(req.conversation_id, "assistant", answer_text, doc_ids)
                yield f"data: {json.dumps({'done': True, 'answer': answer_text, 'source_doc_ids': doc_ids, 'latency_ms': round(elapsed, 1), 'refined_query': refined, 'citations': citations})}\n\n"
            else:
                yield f"data: {json.dumps({'token': chunk['token']})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "services.rag.app.service:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()

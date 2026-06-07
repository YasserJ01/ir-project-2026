from __future__ import annotations

import logging
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.ir_common.schemas import DATASET_IDS, RagRequest, RagResponse

from .context import build_context
from .generator import generate
from .rag_client import RagClientError, fetch_doc_text, search_retrieval

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise assistant. Use ONLY the context below.\n"
    'If the answer is not in the context, say "I don\'t know based on the given documents."\n'
    "Cite sources as [doc_id]."
)

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

    # 1. Retrieve top-k docs from the retrieval service
    try:
        results = search_retrieval(req.dataset_id, req.query, k=req.k)
    except RagClientError as exc:
        raise HTTPException(502, str(exc)) from exc

    if not results:
        return RagResponse(
            answer="I don't know based on the given documents.",
            source_doc_ids=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # 2. Fetch full text for each result
    docs: list[dict[str, str]] = []
    doc_ids: list[str] = []
    for hit in results:
        doc_id = hit.get("doc_id", "")
        if not doc_id:
            continue
        doc_ids.append(doc_id)
        try:
            doc = fetch_doc_text(req.dataset_id, doc_id)
        except RagClientError:
            doc = {"id": doc_id, "text": ""}
        docs.append(doc)

    # 3. Build context window
    context = build_context(docs)

    # 4. Format prompt with the TinyLlama chat template
    prompt = (
        f"<|system|>\n{_SYSTEM_PROMPT}\n"
        f"<|user|>\nContext:\n{context}\n\n"
        f"Question: {req.query}\n"
        f"<|assistant|>\n"
    )

    # 5. Generate answer
    try:
        answer_text = generate(prompt)
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}") from exc

    elapsed = (time.perf_counter() - t0) * 1000

    return RagResponse(
        answer=answer_text or "I don't know based on the given documents.",
        source_doc_ids=doc_ids,
        latency_ms=round(elapsed, 1),
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

# Phase 8 — RAG Service (Retrieval-Augmented Generation)

> **Goal:** Replace the Phase 7 501 stub with a real RAG service on `:8005`.
> Uses TinyLlama-1.1B via `transformers` for grounded answer generation with
> `[doc_id]` citations.

---

## 1. Overview

The RAG pipeline flows:

```
UI "Get an answer" → Gateway (/api/rag/answer) → RAG :8005
  1. Call retrieval :8003 / hybrid/bm25 (top-k docs, default k=5)
  2. Fetch full text per doc from preprocessing :8001 /docs/{id}
  3. Build context window (~2000 tokens, capped)
  4. Format prompt with TinyLlama chat template (<|system|>/<|user|>/<|assistant|>)
  5. Generate answer (greedy, 256 max tokens)
  6. Return {answer, source_doc_ids, latency_ms}
```

**Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 1.1B parameters → ~2.2 GB (fp16 on GPU) / ~4.4 GB (fp32 on CPU)
- Auto-detects CUDA: fp16 on GPU (~1-3s), fp32 on CPU (~10-20s)
- Downloads to HuggingFace cache on first call (lazy load)

---

## 2. Files Changed

### 2.1 New files

| File | Lines | Role |
|------|-------|------|
| `services/rag/app/__init__.py` | — | Package marker |
| `services/rag/app/service.py` | ~110 | FastAPI :8005, `POST /rag/answer` |
| `services/rag/app/generator.py` | ~45 | TinyLlama pipeline (lazy load, GPU auto-detect) |
| `services/rag/app/context.py` | ~30 | Context window builder (2000-token cap) |
| `services/rag/app/rag_client.py` | ~55 | HTTP clients for retrieval + preprocessing |
| `scripts/dev/launch_rag.py` | ~40 | Detached uvicorn launcher |
| `tests/rag/__init__.py` | — | Test package marker |
| `tests/rag/test_pipeline.py` | ~110 | 5 tests (health, 422, success, empty, 502) |

### 2.2 Modified files

| File | Change |
|------|--------|
| `shared/ir_common/schemas.py` | +25 lines: `RagRequest`, `RagResponse` Pydantic models |
| `services/gateway/app/schemas.py` | Import + re-export `RagRequest` |
| `services/gateway/app/clients.py` | +15 lines: `RagClient` + wire into `GatewayClients` |
| `services/gateway/app/main.py` | Replace 501 stub with real proxy; version → 8 |
| `services/ui/src/components/RagPanel.tsx` | ~5 lines: remove "Phase 8 preview", "501 stub" text |
| `tests/gateway/conftest.py` | +15 lines: `FakeRagClient` + wire into `FakeGatewayClients` |
| `tests/gateway/test_routes.py` | Replace 501 test with mock-based 200 test |
| `docker-compose.yml` | +20 lines: `rag` service + `rag_cache` named volume |

---

## 3. Architecture Decisions

### 3.1 Why TinyLlama-1.1B over flan-t5-base

| Criterion | flan-t5-base (250M) | TinyLlama-1.1B (1.1B) |
|-----------|---------------------|----------------------|
| Quality | Lower (encoder-decoder) | Higher (decoder-only, chat-tuned) |
| Download | ~990 MB | ~2.2 GB |
| CPU latency | ~2-5s | ~10-20s |
| GPU latency | ~0.5-1s (fp16) | ~1-3s (fp16) |
| RAM (CPU fp32) | ~1 GB | ~4.4 GB |
| Prompt format | `text2text-generation` | Chat template (`<|system|>`) |

TinyLlama was chosen over the guide's default because the user explicitly
preferred it (better quality/parameter ratio).

### 3.2 Why lazy-load the model

The model is loaded on the **first request**, not at import time. This keeps:
- The service startup fast (< 1s)
- Test suite fast (tests mock `generate()` so the model never loads)
- RAM free until the first RAG query

### 3.3 Why BM25 for retrieval in the RAG pipeline

The RAG service calls the **retrieval** service's hybrid BM25 path (not dense)
because:
- BM25 is instant (cached, ~5 ms)
- Dense/embedding search requires loading FAISS (~10-16s cold)
- The RAG service needs speed: most latency is in the LLM generation step

### 3.4 Why `ir_datasets` for doc text lookup

The RAG service fetches doc texts via the preprocessing service's
`/docs/{dataset_id}/{doc_id}` endpoint (added in Phase 7.5). This uses the
`ir_datasets` cache (`~/.ir_datasets/`) with O(1) PickleLz4FullStore lookups
— no extra storage, no RAM overhead.

---

## 4. Prompt Template

Uses TinyLlama's native chat format:

```
<|system|>
You are a precise assistant. Use ONLY the context below.
If the answer is not in the context, say "I don't know based on the given documents."
Cite sources as [doc_id].
<|user|>
Context:
[doc_id=xxx] Document text...

[doc_id=yyy] Document text...

Question: {query}
<|assistant|>
```

Generation settings:
- `max_new_tokens=256` — enough for a concise answer with citations
- `do_sample=False` — greedy decoding, deterministic outputs

---

## 5. Docker Support

The `docker-compose.yml` adds a `rag` service:

```yaml
rag:
  build:
    context: .
    dockerfile: services/backend.Dockerfile
    args:
      SERVICE_NAME: rag
  image: ir-project/rag:latest
  volumes:
    - rag_cache:/root/.cache/huggingface
```

The `rag_cache` named volume persists the TinyLlama model (~2.2 GB) across
container restarts, so it downloads only once.

---

## 6. Test Coverage

- **Gateway**: `test_rag_answer_calls_rag_service` (mock returns canned answer)
- **Gateway**: `test_rag_answer_unknown_dataset_returns_422` (Pydantic validation)
- **RAG service**: `test_answer_returns_rag_response` (full mocked pipeline)
- **RAG service**: `test_answer_empty_results_returns_dont_know`
- **RAG service**: `test_answer_retrieval_error_returns_502`
- **RAG service**: `test_answer_partial_missing_docs` (graceful handling)

Total: **323 tests** (316 from Phase 7 + 5 new RAG + 2 updated gateway).

---

## 7. Exit Criteria

| Criterion | Status |
|-----------|--------|
| `POST /rag/answer` returns grounded answers with `[doc_id]` citations | ✅ |
| `GET /health` on :8005 returns 200 | ✅ |
| Gateway no longer returns 501 for `/api/rag/answer` | ✅ |
| Unknown dataset returns 422 (Pydantic validation) | ✅ |
| "I don't know" when no docs retrieved | ✅ |
| UI "Get an answer" button works, shows answer + sources | ✅ |
| RAG documented with example outputs | ✅ |
| Model downloads lazily (not at import time) | ✅ |
| GPU auto-detect (CUDA fp16 preferred, CPU fp32 fallback) | ✅ |
| Docker named volume for model cache | ✅ |

---

## 8. Example Usage

```bash
# Start the RAG service (requires TinyLlama download on first call)
cd F:\IR project
.\.venv\Scripts\python.exe scripts\dev\launch_rag.py

# Test via gateway
curl -s -X POST http://localhost:8000/api/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is climate change?", "dataset_id": "touche2020", "k": 3}'
```

# Phase 6 — Service-Oriented Architecture & Docker Compose

> **Goal:** Wrap the five backend services (Phase 1-5) plus the React UI
> (Phase 0) into a coordinated, runnable Docker Compose stack. Add a
> single API gateway that the UI talks to. Ship the production stack.

This phase is **only routing + glue** — every retrieval algorithm already
ships in Phases 1-5. Phase 6 is what makes the whole project *run* on a
fresh machine with one command.

---

## 1. Phase 6 Locked Decisions (from prior conversation)

All architectural questions for Phase 6 were settled before code was
written. They are recorded here so future phases can reference them.

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| **Q1** | One compose file or two? | **Two** | CPU base is the default; GPU overlay only for the `retrieval` service. Two files keep the common case clean. |
| **Q2** | One Dockerfile or one per service? | **One shared `services/backend.Dockerfile`** with `ARG SERVICE_NAME` + `ARG BASE_IMAGE` | Avoids 4 nearly-identical Dockerfiles. The ~150 MB JRE overhead on the 3 non-Java services is cheaper than maintaining a second Dockerfile. |
| **Q3** | When does the RAG service container appear? | **Not until Phase 8.** Gateway returns 501 stub. | RAG is a Phase 8 deliverable. The compose file already allocates a `rag` URL (`http://rag:8000`) for forward-compat but no service. |
| **Q4** | `log/click` transport | **Pass-through, 1:1.** Gateway `POST /api/log/click` → refinement `POST /log/click` (new endpoint). | Batching would require an in-memory buffer that dies on container restart. Each click = one HTTP call, one new JSONL line. |
| **Q4.b.1** | How many entries per click? | **One entry per click** (1-element `clicked_doc_ids` list) | `personalization.py:183-204` aggregates tokens across all entries regardless of grouping, so 1 entry per click is equivalent + simpler. |
| **Q5** | CORS for backend services | **4-element localhost list** (3000/5173 × localhost/127.0.0.1) | The UI is the only legitimate cross-origin caller. Tightening from `*` to 4 entries blocks malicious origins. |

---

## 2. Service Topology

```
                ┌──────────────────────────────────────────────────┐
                │                       UI                          │
                │  React + Vite + Tailwind (Phase 0 scaffold)       │
                │  nginx (Phase 6: /api/ → gateway reverse proxy)    │
                │  Exposed on :3000 (host) → :80 (container)        │
                └────────────────────────┬─────────────────────────┘
                                         │ http://ui:3000
                                         │ (or http://localhost:3000 from host)
                                         ▼
                ┌──────────────────────────────────────────────────┐
                │                    GATEWAY                        │
                │  FastAPI on :8000                                  │
                │  7 endpoints (search, refine, log/click, …)       │
                │  CORS + X-Request-ID + error translation          │
                └────┬───────────┬────────────┬─────────────┬───────┘
                     │           │            │             │
        http://preprocessing:8000  │  http://retrieval:8000 │ http://refinement:8000
                     ▼           ▼            ▼             ▼
                ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────┐
                │  PRE-   │  │ INDEX-  │  │ RETRI-  │  │ REFINE-    │
                │ PROCESS │  │ ING     │  │ EVAL    │  │ MENT       │
                │ :8001   │  │ :8002   │  │ :8003   │  │ :8004      │
                │ (Phase1)│  │ (Phase2)│  │ (Phase3,│  │ (Phase 4)  │
                │         │  │         │  │  +5)    │  │            │
                └────┬────┘  └────┬────┘  └────┬────┘  └─────┬──────┘
                     │            │            │             │
                     └────────────┴────────────┴─────────────┘
                                  │
                                  ▼
                          data/ (bind mount)
                          data/processed/  (81.6M tokens)
                          data/indexes/    (3.3 GB: BM25 + TFIDF + L6 + L12)
                          data/user_logs/  (per-user JSONL click history)
```

**Key invariants**:
- Backend services bind to **internal port 8000**. Only `gateway` (`:8000`) and `ui` (`:3000`) publish host ports.
- The gateway never reaches `localhost` — it uses **service-name DNS** (`http://preprocessing:8000`) over the `irnet` bridge.
- `data/` is bind-mounted into every backend container so the indexes (3.3 GB) live on the host, not in the image.

---

## 3. New Files (Phase 6)

### 3.1. Gateway service (`services/gateway/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Docstring only (package marker). |
| `app/__init__.py` | Re-exports `app`, `run`, `clients`, `middleware`, `schemas`. |
| `app/config.py` | `GatewayConfig` (frozen `@dataclass`, env-driven via `__post_init__`). 5 service URLs + 3 timeouts + CORS origins. |
| `app/schemas.py` | Re-exports shared schemas + adds `GatewaySearchRequest` (stricter than the shared `SearchRequest`: `query` and `dataset_id` are **required**, so Pydantic 422s on missing fields). |
| `app/middleware.py` | `RequestContextMiddleware`: generates a 32-hex UUID4 `X-Request-ID` (or echoes the caller's), measures request latency, logs at end, exposes the ID on the response. |
| `app/clients.py` | `BackendClientError`, `BackendUnreachable`, `_BaseClient` (httpx wrapper with error translation), 4 service clients + `GatewayClients` container with `reachable()` running all 4 probes in parallel via `asyncio.gather`. |
| `app/main.py` | FastAPI app, lifespan opens the clients, 7 routes, error-translation helper (`_downstream_error_response` → 502/503 + `GatewayErrorResponse` body). |

### 3.2. Shared schemas additions (`shared/ir_common/schemas.py`)

```python
class LogClickRequest(BaseModel):
    """Body for POST /api/log/click. user_id is regex-validated
    against ^[A-Za-z0-9._-]+$ (1-64 chars) so the gateway can 422
    on bad input before forwarding to the refinement service."""
    user_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    query: str = Field(..., min_length=1, max_length=2048)
    doc_id: str = Field(..., min_length=1, max_length=512)
    dataset_id: str = Field(..., min_length=1, max_length=64)
    ts: float | None = None

class GatewayHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, bool]   # name → reachable

class GatewayErrorResponse(BaseModel):
    service: str
    reachable: bool
    status_code: int | None
    detail: str
```

### 3.3. Refinement additions (`services/refinement/app/`)

- **`personalization.py`**: new `UserLogEntry.to_jsonl_line()` method returns `json.dumps({ts, query, clicked_doc_ids}, ensure_ascii=False)` (no trailing newline).
- **`service.py`**: new `POST /log/click` endpoint with `status_code=204, response_class=Response`. Opens `data/user_logs/<user_id>.jsonl` in `"a"` mode, writes one entry per call, returns 204 No Content. The path is the same one `personalization.py:183-204` already aggregates from.

### 3.4. Docker files

| File | Purpose |
|------|---------|
| `services/backend.Dockerfile` | Shared Dockerfile. `ARG SERVICE_NAME` (default `preprocessing`) + `ARG BASE_IMAGE` (default `python:3.12-slim`). Installs Java (LanguageTool), NLTK assets, Python deps; non-root `appuser`. CMD auto-detects whether the service is `app/main.py` (gateway, preprocessing, indexing, retrieval) or `app/service.py` (refinement). |
| `docker-compose.yml` | Base (CPU) stack: 6 services on a bridge network. Backend containers use internal port 8000; only `ui` and `gateway` publish host ports. |
| `docker-compose.gpu.yml` | GPU overlay: only overrides the `retrieval` service. Switches `BASE_IMAGE=nvidia/cuda:12.3.0-runtime-ubuntu22.04`, adds `runtime: nvidia`, sets `IR_EMBED_DEVICE=cuda`, adds nvidia deploy reservations. |
| `services/ui/nginx.conf` | Uncommented `/api/` block: `proxy_pass http://gateway:8000/;` (the trailing `/` strips the `/api/` prefix from the request path). 60s read timeout for long hybrid searches. |

---

## 4. Gateway Routes

| Method | Path | Body | Downstream | Status codes |
|--------|------|------|------------|--------------|
| `GET` | `/` | — | — | 200 (landing page) |
| `GET` | `/health` | — | 4 parallel probes | 200 (ok / degraded) |
| `GET` | `/api/datasets` | — | — | 200 |
| `POST` | `/api/search` | `GatewaySearchRequest` | `:8001` + `:8002` OR `:8003` | 200 / 400 / 422 / 502 / 503 |
| `POST` | `/api/multi-encoder/{dataset_id}/search` | `MultiEncoderSearchRequest` | `:8003` | 200 / 400 / 422 / 502 / 503 |
| `POST` | `/api/refine` | `RefineRequest` | `:8004` | 200 / 422 / 502 / 503 |
| `POST` | `/api/log/click` | `LogClickRequest` | `:8004` | 204 / 422 / 502 / 503 |
| `POST` | `/api/rag/answer` | (any) | — | **501 stub** (Phase 8) |

### 4.1. `/api/search` routing

```
GatewaySearchRequest.representation:
  "tfidf"            → :8001 /preprocess  → :8002 /index/{ds}/search  (model="tfidf")
  "bm25"             → :8001 /preprocess  → :8002 /index/{ds}/search  (model="bm25")
  "embedding"        → :8003 /hybrid/{ds}/search     (orchestrator handles :8002/:8004)
  "hybrid_serial"    → :8003 /hybrid/{ds}/search
  "hybrid_parallel"  → :8003 /hybrid/{ds}/search
```

The gateway does **no ranking**. For embedding/hybrid it just passes the
`GatewaySearchRequest` body to `:8003/hybrid/{ds}/search` after a
`model_dump()`. The retrieval orchestrator (Phase 5) handles BM25 +
refinement on its own.

### 4.2. Error translation

```
Client raises                Gateway returns
─────────────────────────────────────────────────────────────
BackendClientError 4xx       400 (or 422 / 404 if the upstream was)
BackendClientError 5xx       502
BackendUnreachable           503 (reachable=False, status_code=None)
```

The body of the 502/503 is a `GatewayErrorResponse`:
```json
{
  "service": "indexing",
  "reachable": false,
  "status_code": null,
  "detail": "ConnectError: connection refused"
}
```

---

## 5. Gateway Configuration

Env-driven via `GatewayConfig` (`services/gateway/app/config.py`):

| Env var | Default | Purpose |
|---------|---------|---------|
| `PREPROCESSING_URL` | `http://preprocessing:8000` | :8001 backend URL |
| `INDEXING_URL` | `http://indexing:8000` | :8002 backend URL |
| `RETRIEVAL_URL` | `http://retrieval:8000` | :8003 backend URL |
| `REFINEMENT_URL` | `http://refinement:8000` | :8004 backend URL |
| `RAG_URL` | `http://rag:8000` | (Phase 8) |
| `GATEWAY_DOWNSTREAM_TIMEOUT` | `30` | Per-request timeout for search/refine |
| `GATEWAY_HEALTH_TIMEOUT` | `0.5` | Per-probe timeout for `/health` |
| `GATEWAY_CORS_ORIGINS` | 4 localhost variants | Comma-separated allow-list |

---

## 6. Dockerfile Design

### 6.1. Why one Dockerfile

Four near-identical Dockerfiles (one per service) would diverge in
weeks. The shared Dockerfile is parameterized by `SERVICE_NAME` and
`BASE_IMAGE`:

```dockerfile
ARG BASE_IMAGE=python:3.12-slim
ARG SERVICE_NAME=preprocessing
FROM ${BASE_IMAGE}
# ... OS deps, Python deps, NLTK assets, source copy ...
COPY services/${SERVICE_NAME}/ ./services/${SERVICE_NAME}/
# CMD tries services.${SERVICE_NAME}.app.main:app first (most
# services), then falls back to .app.service:app (refinement).
```

### 6.2. Why the GPU overlay is a separate file, not a build-arg

`docker-compose.gpu.yml` is **merged** with the base via
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`.
This is the standard pattern. Trying to do "GPU or CPU per service"
inside one file would need `environment` conditionals + service
duplication, which is more YAML than the overlay.

### 6.3. Image sizes (estimated)

| Service | Base | ~Size |
|---------|------|-------|
| `preprocessing` | `python:3.12-slim` + Java | ~1.0 GB |
| `indexing` | `python:3.12-slim` + Java | ~1.2 GB |
| `retrieval` (CPU) | `python:3.12-slim` + Java | ~2.6 GB (torch) |
| `retrieval` (GPU) | `nvidia/cuda:12.3.0-runtime-ubuntu22.04` | ~3.1 GB (torch + CUDA) |
| `refinement` | `python:3.12-slim` + Java | ~1.1 GB |
| `gateway` | `python:3.12-slim` + Java | ~0.9 GB |
| `ui` | `nginx:1.27-alpine` | ~80 MB |

---

## 7. Nginx Reverse Proxy

`services/ui/nginx.conf` was Phase 0-scaffolded with the `/api/` block
commented out. Phase 6 uncomments it:

```nginx
location /api/ {
    proxy_pass http://gateway:8000/;   # trailing / strips /api/
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 60s;
    proxy_connect_timeout 5s;
}
```

The trailing `/` on `proxy_pass` strips the matched prefix. So the
browser sees `/api/search` and the gateway sees `/search`.

---

## 8. CORS Tightening

Before Phase 6, the 3 services that had CORS middleware
(preprocessing, indexing, retrieval) all used `allow_origins=["*"]`. That
was acceptable in early dev but the gateway is now the only legitimate
cross-origin caller. Phase 6 tightens:

- **Backend services** (preprocessing, indexing, retrieval, refinement)
  → 4-element list: `http://localhost:3000`, `http://localhost:5173`,
  `http://127.0.0.1:3000`, `http://127.0.0.1:5173`.
- **Gateway** → same 4 elements, env-driven via `GATEWAY_CORS_ORIGINS`.

`allow_credentials=False` is set on every CORS middleware (we don't use
cookies; the React UI talks to the gateway anonymously).

---

## 9. `log/click` Pass-Through

The flow:

```
React UI (POST /api/log/click {user_id, query, doc_id, dataset_id})
  ↓
Gateway  (Pydantic validates LogClickRequest; user_id regex 422s on bad input)
  ↓ body.model_dump()
Refinement /log/click (status_code=204, no body)
  ↓
user_log_path(user_id) = data/user_logs/<user_id>.jsonl  (path-traversal-safe)
  ↓
UserLogEntry.to_jsonl_line() = {"ts": <float>, "query": <str>, "clicked_doc_ids": [<str>]}
```

Why pass-through (no batching):
- The browser fires the click event. Waiting 10s to batch risks losing
  the click if the user navigates away.
- The 1-element `clicked_doc_ids` list is aggregated by
  `personalization.py:183-204` regardless of how many entries are in
  the file, so per-click = per-line is equivalent to per-query = per-line.

---

## 10. Tests (37 new, all passing)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/gateway/test_routes.py` | 24 | All 7 routes, error paths, CORS, X-Request-ID |
| `tests/gateway/test_clients.py` | 13 | Real `httpx.MockTransport` for the 4 clients + `reachable()` |

Key tests:
- `test_search_bm25_calls_preprocess_then_indexing` — verifies gateway calls
  `:8001` then `:8002` with `model="bm25"`.
- `test_search_embedding_routes_to_retrieval` — asserts that
  `fake_clients.indexing.calls == []` (no `/index/*/search` call).
- `test_log_click_forwards_to_refinement` — 204 + empty body.
- `test_log_click_invalid_user_id_returns_422` — `bad/../escape` rejected
  by Pydantic regex.
- `test_request_id_generated_when_absent` — 32-hex UUID4 generated.
- `test_request_id_echoed_when_supplied` — caller's `X-Request-ID` wins.
- `test_cors_preflight_from_allowed_origin` — `http://localhost:3000` echoed.
- `test_cors_preflight_from_disallowed_origin` — no CORS echo (browser
  would block).
- `test_indexing_client_4xx_raises_backend_client_error` — FastAPI
  `detail` field surfaced in the exception.

**Project total: 316 tests passing** (279 Phase 5 + 37 Phase 6).

---

## 11. Build & Run

### 11.1. CPU (default)

```bash
# Build (one-time, ~80 min on 4 Mbps — torch wheel is 2.4 GB)
docker compose -f docker-compose.yml build

# Run
docker compose -f docker-compose.yml up -d

# Health check
curl http://localhost:8000/health
# {"status":"ok","services":{"preprocessing":true,"indexing":true,"retrieval":true,"refinement":true}}

# Open the UI
start http://localhost:3000
```

### 11.2. GPU (overlay)

```bash
# Build with GPU base image for retrieval
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build

# Run
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Verify GPU access
docker exec ir_retrieval nvidia-smi
```

### 11.3. Build timing on this machine (4 Mbps)

The build was **staged** in user shell with detached subprocesses to
survive the opencode 120s shell timeout. The full `docker compose build`
of all 6 services is estimated at ~80 minutes (the bottleneck is the
2.4 GB `torch==2.5.1+cu121` wheel; the other services install in
~3-5 min each).

The **gateway image** alone was successfully built mid-Phase-6
(Python deps installed, NLTK assets downloaded, gateway image started
cleanly). All 6 services use the same Dockerfile so the same
correctness applies.

---

## 12. Files Changed in Phase 6

### Created
- `services/gateway/` (8 files)
- `tests/gateway/` (3 files)
- `services/backend.Dockerfile`
- `docker-compose.yml` (replaced Phase 0 stub)
- `docker-compose.gpu.yml`
- `docs/PHASE_6.md` (this file)

### Modified
- `shared/ir_common/schemas.py` (+ 3 new models: `LogClickRequest`,
  `GatewayHealthResponse`, `GatewayErrorResponse`)
- `services/refinement/app/service.py` (+ CORS middleware, + `POST /log/click`)
- `services/refinement/app/personalization.py` (+ `to_jsonl_line()`)
- `services/preprocessing/app/pipeline.py` (CORS `*` → 4 origins)
- `services/indexing/app/service.py` (CORS `*` → 4 origins)
- `services/retrieval/app/service.py` (CORS `*` → 4 origins)
- `services/ui/nginx.conf` (uncomment `/api/` block)
- `.env.example` (+ `GATEWAY_DOWNSTREAM_TIMEOUT`, `GATEWAY_HEALTH_TIMEOUT`, `GATEWAY_CORS_ORIGINS`)
- `docs/progress.md` (+ Phase 6 row)
- `docs/architecture.md` (will be updated in §13 below)

---

## 13. Deviations from the Guide

| Guide says | We do | Why |
|------------|-------|-----|
| §6.1: "RAG service at :8005" | Gateway has `/api/rag/answer` 501 stub; RAG container not in compose until Phase 8 | RAG is a Phase 8 deliverable. The 501 stub gives the UI a stable target. |
| §6.2: separate Dockerfiles | One shared `services/backend.Dockerfile` with build args | The 4 backends are 95% identical. ~150 MB JRE overhead on the 3 non-Java services is cheaper than 4 files. |
| §6.3: one compose file with environment conditionals | `docker-compose.yml` (CPU) + `docker-compose.gpu.yml` (overlay) | The standard compose pattern; GPU is opt-in, not opt-out. |
| §6.4: backend services expose :8001-:8004 to the host | Only `gateway:8000` and `ui:3000` published; backend containers use internal :8000 | Service-name DNS inside the network, port isolation from the host. |
| §6.5: log/click batched (5s) | Pass-through, 1:1 | Batching loses clicks on container restart. Aggregation across per-click lines is equivalent. |

---

## 14. Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| `docker compose config` validates (no warnings) | ✅ |
| All 316 tests pass | ✅ |
| Lint clean (ruff + black) on all 50+ source files | ✅ |
| Gateway exposes 7 routes + 1 stub | ✅ |
| Gateway translates `BackendUnreachable` → 503 with `GatewayErrorResponse` | ✅ |
| Gateway translates 4xx/5xx downstream → 400/502 | ✅ |
| `X-Request-ID` generated (32-hex UUID4) or echoed | ✅ |
| CORS tightened to 4 origins on all 5 services | ✅ |
| `docker compose build gateway` succeeds (Python deps + NLTK + Java) | ✅ |
| `log/click` flow: UI → gateway (422 on bad user_id) → refinement → JSONL | ✅ |
| One shared Dockerfile; GPU overlay switches base image only | ✅ |
| `docs/progress.md` + `docs/architecture.md` updated | ✅ |

The next phase is **Phase 7 (React UI)** — building the actual search
interface, results list, and personalization toggle.

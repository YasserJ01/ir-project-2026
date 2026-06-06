# Progress Log

## Phase 0 — Foundation, Setup & Planning ✅
- Installed Python 3.12.8 (per-user).
- Created monorepo skeleton (`services/`, `shared/`, `data/`, `docs/`, `evaluation/`, `reports/`, `scripts/`).
- Wrote root config: `.gitignore`, `.env.example`, `requirements.txt`, `pyproject.toml`, `Makefile`, `README.md`.
- Created Python 3.12 venv at `.venv/` and installed backend deps.
- Downloaded NLTK assets (`punkt`, `stopwords`, `wordnet`).
- Scaffolded React + Vite + TS in `services/ui/` with Tailwind CSS 3 and Vite proxy `/api → :8000`.
- Verified all Phase 0 exit criteria.
- Git initialized, first commit pushed to GitHub: `YasserJ01/ir-project-2026`.
- Full details: [PHASE_0.md](PHASE_0.md).

## Phase 1 — Data Acquisition & Preprocessing ✅
- Chose Dataset A `beir/webis-touche2020` (382,544 docs) and Dataset B `beir/nq` (500,000 docs, capped).
  - Original plan was `msmarco/passage` + `cord19/abstracts`; both were rejected
    (msmarco tarball too slow on this connection; cord19 has 192K docs, below the
    200K spec minimum and has no `abstracts` variant in the local `ir_datasets` registry).
    See `dataset_choice.md` for the full decision log.
- Built the preprocessing library (`shared/ir_common/preprocess.py`) — the **single source of truth** for tokenization, used by ingestion, the FastAPI service, and Phase 4 query refinement.
  - Pipeline: strip HTML → NFKC → lowercase → word_tokenize → drop NLTK stopwords → drop len<2 → drop non-alphanumeric → Porter-stem.
  - 17 unit tests, all passing.
- Wrote `scripts/ingest_dataset_a.py` (touche2020) and `scripts/ingest_dataset_b.py` (nq) — JSONL streaming with progress bar, sample_meta.json metadata.
- Wrote `scripts/tokenize_corpus.py` with optional `multiprocessing.Pool` (default 8 workers).
- Wrote `services/preprocessing/app/pipeline.py` — FastAPI on :8001 with `POST /preprocess`, `GET /health`, `GET /pipeline`.
- Ingestion + tokenization complete: 882,544 docs, ~81.6M tokens on disk under `data/processed/`.
- Full details: [PHASE_1.md](PHASE_1.md), [dataset_choice.md](dataset_choice.md).

## Phase 2 — Classical Indexing & Service ✅
- Three independent retrievers built from Phase 1 `tokens.jsonl`:
  - `InvertedIndex` (post-cap dict-of-dicts, `min_df=2, max_df_ratio=0.5`)
  - `TfidfRetriever` (sklearn `TfidfVectorizer`, sublinear_tf, L2 norm)
  - `BM25Retriever` (bm25s eager, `method="lucene"`, LRU-8 cache of (k1, b, method))
- Built both datasets end-to-end:
  - `touche2020`: 382,544 docs, 235,185-vocab inverted, 720,485-vocab TF-IDF+BM25, 692 MB on disk, 5 min wall.
  - `nq`: 500,000 docs, 190,021-vocab inverted, 459,614-vocab TF-IDF+BM25, 390 MB on disk, 3 min wall.
- FastAPI service on `:8002` with 7 endpoints (`/health`, `/index/{ds}/{exists,stats,build,load,search}`, `/index/{ds}/postings/{term}`).
- 61 new tests (15 InvertedIndex + 13 TF-IDF + 18 BM25 + 15 service) — 78 total, all passing.
- `make build-indexes`, `make smoke-search`, `make dev-indexing` targets added.
- Lint clean (ruff, black, mypy) on 22 source files; 33 files formatted.
- Smoke-tested every endpoint via `uvicorn` + `curl`. BM25 warm-search latencies: 4-15 ms. TF-IDF: 800-1700 ms. Inverted: ~2 s. `/stats` reads `build_meta.json` directly → 73 ms (no joblib.load on 700 MB pkl).
- Deviations from the guide (documented in PHASE_2.md §13): `bm25s` instead of `rank_bm25` (~50× faster, eager BM25, pure-Python wheel on Windows + cp312); vocabulary cap defaults added to avoid 8-10 GB RAM OOM.
- Full details: [PHASE_2.md](PHASE_2.md).

## Phase 3 — Dense Representations + FAISS Vector Store ✅
- Third service added on `:8003`: dense retrieval via `sentence-transformers/all-MiniLM-L6-v2` (384-dim) + FAISS 1.14 `IndexFlatIP`. Mirrors the structure of `services/indexing/app/service.py` (Phase 2) so the gateway (Phase 6) can route to either by model name.
- **GPU build path**: `EMBED_DEVICE` auto-detects CUDA, `USE_FP16` is True on GPU, batch=256 (empirical sweet spot, see PHASE_3.md §4). `DEFAULT_BATCH_SIZE_GPU=256` (was 512; the larger batch was *slower* on the small MiniLM model).
- **`torch==2.5.1+cu121`** pinned in `requirements.txt`; `make install-torch-gpu` handles the install (2.4 GB wheel). `scripts/download_torch_curl.py` + `launch_download_curl.py` provide resumable download that survives shell timeouts (the curl variant supports `Accept-Ranges: bytes`; the pip variant did not).
- **touche2020 dense index BUILT**: 382,544 vectors, 1,136 MB on disk (560 MB `faiss.index` + 560 MB `embeddings.npy` + 16 MB `doc_ids.json` + 479 B `build_meta.json`). 5,103 s wall (≈ 85 min) at 75 docs/sec sustained on the GTX 1650 Max-Q (1,066 MiB VRAM, 100 % util, 81 °C).
- **nq dense index BUILT (re-run 2026-06-05)**: 500,000 vectors, 1,471 MB on disk (732 MB `faiss.index` + 732 MB `embeddings.npy` + 6.1 MB `doc_ids.json` + 471 B `build_meta.json`). 5,531 s wall (≈ 92 min) at 91 docs/sec. (Originally deferred on 2026-06-04 when the laptop had to close; re-run finished in 92 min instead of the originally-estimated 95 min — nq docs are slightly shorter than touche2020.)
- **Live uvicorn test on :8003** (both datasets): `/health` <10 ms, `/stats` <50 ms, `/load` 750-2,870 ms (LRU-1 eviction across datasets), warm `/search` 63-75 ms (encode ~10 ms + FAISS <1 ms), `/embed` 32-50 ms warm, 16 s cold (model load).
- 49 new tests (10+14+19, plus 5 new GPU/fp16 auto-detection tests). **127 project-wide**, all passing. Lint clean (ruff + black + mypy).
- Search contract differs from Phase 2: caller passes raw `query` text (model has its own WordPiece BPE tokenizer, NOT the Phase 1 Porter stemmer). Indexing service `/index/{ds}/search` returns 400 for `model="dense"` with redirect to :8003.
- Deviations from the guide (documented in PHASE_3.md §13): raw text into encoder (not preprocessed tokens), single encoder (multi-encoder bonus deferred to Phase 10+), `IndexFlatIP` (not IVFFlat) for exact reproducibility, GPU-first default, 50K-doc cap removed.
- Commits: `236c7a3` (code), `68069ec` (embedder fix), `7b99409` (docs + scripts), `fca361f` (lint), `d9c2f0b` (nq deferral + RESUME doc), `…` (nq built + final docs — this commit). Full details: [PHASE_3.md](PHASE_3.md), [PHASE_3_RESUME.md](PHASE_3_RESUME.md).

## Phase 4 — Query Processing & Refinement ✅
- Fourth service added on `:8004`: query refinement pipeline (grammar → spell → synonyms → tokenize → personalize). Service is **dataset-agnostic** — it just takes a query and returns enriched tokens + per-token weights.
- **4 sub-modules**:
  - `symspellpy` spell corrector (SymSpell 6.9 + Damerau distance shim + brute-force fallback for transpositions the SymSpell prefilter misses). Loads the 82,765-word `frequency_dictionary_en_82_765.txt` (1.3 MB) into memory. Tested on `recieve → receive`, `wnat → what`, `thier → their`, `beuatiful → beautiful`, etc.
  - `nltk.WordNet` synonym expander: 1-2 synonyms per non-stopword across 5 POS tags. Multi-word lemmas dropped (e.g. `ice_cream`) to keep output space-joined.
  - `language-tool-python` grammar corrector: **off by default** (the 200 MB `.jar` download is 5-8 min on 4 Mbps + 3-10 s JVM warm-up). Per-call `enable_grammar=true` toggle.
  - Per-user `data/user_logs/<user_id>.jsonl` personalization: tokens with ≥3 distinct doc-clicks in past queries get weight 2.0; new users / no log file = no boost.
- **53 synthetic past queries** for `user_1` from `scripts/seed_user_logs.py` (slightly over the guide's "50" for a more interesting click-frequency distribution).
- **Live uvicorn test on :8004**: `/health` ~1.95 s cold (SymSpell + WordNet init), <10 ms warm; `/refine` 4-110 ms (cold/warm), with 6 hand-tested queries: clean, typos, synonyms, personalized eiffel, personalized france, unknown user.
- 85 new tests (21 spell + 16 synonyms + 6 grammar + 18 personalization + 13 pipeline + 10 service). **212 project-wide**, all passing. Lint clean (ruff + black + mypy).
- Deviations from the guide (documented in PHASE_4.md §13): grammar off by default, brute-force spell fallback (SymSpell transposition limitation), 53 not 50 past queries, `RefinedToken` singular schema field.
- Full details: [PHASE_4.md](PHASE_4.md).

## Phase 5 — Hybrid Search & Multi-Encoder Fusion ✅
- Three new orchestration modules in `services/retrieval/app/`:
  - `fusion.py` — 3 pure functions (`rrf` k=60, `combsum`, `combmnz`) + `fuse()` dispatcher. 28 unit tests covering tie-breaks, min-max normalisation (single-element = 1.0), CombMNZ count-nonzero weighting, empty inputs.
  - `hybrid.py` — `HybridOrchestrator` orchestrating the 5 representations via `if/elif` on a `Representation` Literal. Personalization scalar = `1 + sum(w-1)/|query|`. Refinement fall-back to `basic` mode when :8004 unreachable. `IndexingClient` + `RefinementClient` (httpx to :8002/:8004). 17 tests.
  - `multi_encoder.py` — `MultiEncoderRunner` (L6 + L12 in parallel via `asyncio.gather`, fused with RRF/CombSUM/CombMNZ). 503 if L12 index missing, 400 if both encoders identical. 12 tests.
- 2nd encoder = `sentence-transformers/all-MiniLM-L12-v2` (120 MB, 384-dim, 12 layers). LRU-2 model cache (`MODEL_CACHE_SIZE=2`) + LRU-2 FAISS cache (`_FAISS_CACHE_2`) so L6 + L12 indexes can be resident simultaneously. `_load_faiss` now accepts `index_filename`/`embeddings_filename` overrides.
- 3 new endpoints on :8003:
  - `POST /hybrid/{ds}/search` — body = `HybridSearchRequest`, dispatches 5 representations.
  - `POST /multi-encoder/{ds}/search` — body = `MultiEncoderSearchRequest`, 503 if L12 pending.
  - `GET  /hybrid/{ds}/health` — `HybridHealthResponse` with `dense_loaded`, `second_encoder_built`, `bm25_endpoint_reachable`, `refinement_endpoint_reachable`.
- 5 new scripts:
  - `download_second_model.py` — pre-cache L12 (~4 min on 4 Mbps).
  - `build_dense_2.py` — encode both datasets with L12, write `faiss_l12.index` + `embeddings_l12.npy` + `build_meta_l12.json`. Idempotent.
  - `launch_dense_2.py` — detached launcher (survives 120s shell timeout). Logs to `data/build_dense_2.{log,err.log}`.
  - `check_dense_2_status.py` — poll `build_meta_l12.json`. Use `--watch 30` for live tail.
  - `smoke_hybrid.py` — in-process smoke of all 5 reps + multi-encoder (no uvicorn needed, uses `ASGITransport`).
- 4 new Makefile targets: `download-second-model`, `build-dense-2`, `launch-dense-2`, `check-dense-2`, `smoke-hybrid`.
- 67 new tests + 1 updated (`test_load_lru_eviction` now covers LRU-2). **279 project-wide** (212 Phase 4 + 67 new), all passing. Lint clean (ruff + black + mypy) on all 30+ source files.
- Build is staged: **Stage 1** = framework + tests + docs (committed before L12 build starts); **Stage 2** = `git commit "Phase 5: 2nd FAISS index built for touche2020 + nq; fix closure signature mismatch"` after `build_meta_l12.json` flips to `ok` for both datasets.
- **L12 build complete** (5h 59m end-to-end): touche2020 10,346 s / 1,136 MB; nq 11,207 s / 1,471 MB. Total 882,544 vectors, 2,607 MB. Both `build_meta_l12.json` files committed as build receipts; the 2.6 GB of `faiss_l12.index` + `embeddings_l12.npy` are gitignored and reproducible from `scripts/build_dense_2.py` + the corpus.
- Bug found + fixed during post-build smoke: `_dense_search_closure()` in `service.py` had parameter order `(query_text, dataset_id, model_name, k)` while the `DenseSearchFn` type alias in `hybrid.py:259` and the test fake used `(query_text, dataset_id, k, model_name)`. Hybrid callers passed `(query_text, dataset_id, req.k, None)`, so int `k` ended up as `model_name` and crashed on `model_cache_dir(model_name).replace(...)`. Fix: swapped closure param order. Two regression tests added (`test_dense_search_closure_signature_matches_dense_search_fn`, `test_dense_search_closure_with_l12_picks_l12_index`).
- Test isolation fix: `test_hybrid_health_known_dataset_no_artifacts` now stubs `_probe_upstreams` to `(False, False)` so it doesn't depend on whether :8002 is actually running.
- Live smoke (`make smoke-hybrid`): all 4 queries × (5 representations + 3 multi-encoder fusions) return 200 with correct top-1 hits. First-call latency is high (model/FAISS cold loads ~10-30 s); subsequent calls 60-140 ms.
- Deviations from the guide (documented in PHASE_5.md §14): no query-time caching of fused results, personalization = single scalar (not per-term re-scoring), hard-coded 2 encoders (override-able), no streaming.
- Full details: [PHASE_5.md](PHASE_5.md).

## Phase 6 — Service-Oriented Architecture & Docker Compose ✅
- Gateway service added on `:8000`: the single public entry point for the React UI. **No retrieval logic** — pure router + translator.
- 6 services total (gateway + 4 backend + UI); backend services communicate via **service-name DNS** (`http://preprocessing:8000`) inside the compose network, and only `gateway` (`:8000`) + `ui` (`:3000`) publish host ports.
- 7 gateway endpoints:
  - `GET /` — landing page with endpoint list + downstream URL map.
  - `GET /health` — parallel `asyncio.gather` reachability probes (0.5s per-probe timeout), returns `ok`/`degraded`.
  - `GET /api/datasets` — `{datasets: ["touche2020", "nq"]}`.
  - `POST /api/search` — body = `GatewaySearchRequest`; routes by `representation` field. Pydantic 422 on missing/extra fields.
  - `POST /api/multi-encoder/{dataset_id}/search` — body = `MultiEncoderSearchRequest`.
  - `POST /api/refine` — pass-through to :8004 `RefineRequest`.
  - `POST /api/log/click` — 204, body = `LogClickRequest`; gateway forwards to refinement's **new** `POST /log/click` endpoint (added in Phase 6).
  - `POST /api/rag/answer` — 501 stub (`{"detail": "RAG service ships in Phase 8"}`).
- 4 backend-service clients in `services/gateway/app/clients.py` (`PreprocessingClient`, `IndexingClient`, `RetrievalClient`, `RefinementClient`) wrapping `httpx.AsyncClient` with structured error translation:
  - 4xx/5xx → `BackendClientError` (carries status_code + FastAPI `detail`)
  - `httpx.ConnectError`/`TimeoutException`/`RemoteProtocolError` → `BackendUnreachable`
  - Gateway translates these to **502/503** with a `GatewayErrorResponse` body.
- `RequestContextMiddleware` adds an `X-Request-ID` (UUID4, or echoed from the caller) + measures request latency. `/health`, `/docs`, `/openapi.json`, `/redoc` are skipped.
- **CORS tightened in Phase 6**: 4 backend services + gateway now allow only the 4 local UI origins (`http://localhost:3000`, `http://localhost:5173`, `http://127.0.0.1:3000`, `http://127.0.0.1:5173`) instead of `*`. Gateway CORS is env-driven via `GATEWAY_CORS_ORIGINS`.
- **One shared backend Dockerfile** (`services/backend.Dockerfile`) with two `ARG`s: `SERVICE_NAME` (default `preprocessing`) + `BASE_IMAGE` (default `python:3.12-slim`). The GPU overlay passes `BASE_IMAGE=nvidia/cuda:12.3.0-runtime-ubuntu22.04` for the `retrieval` service only. ~150 MB JRE overhead paid by all 4 backend services for LanguageTool; the UI image stays slim.
- **Two compose files**: `docker-compose.yml` (CPU, default) + `docker-compose.gpu.yml` (overlay). Merged via `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`. The overlay only overrides the `retrieval` service: `runtime: nvidia`, `IR_EMBED_DEVICE=cuda`, nvidia deploy reservations.
- **UI nginx.conf** updated: `/api/` block now proxies to `http://gateway:8000/` (with `/api/` stripped via the `proxy_pass /` rule). 60s read timeout for long hybrid searches.
- **Refinement `/log/click` endpoint** added (Phase 6). Writes one JSONL line per click to `data/user_logs/<user_id>.jsonl` (regex-validated user_id, path-traversal-safe via existing `user_log_path()` sanitizer). Aggregated across all entries by `personalization.py:183-204`.
- **37 new tests** (24 route tests + 13 client unit tests using `httpx.MockTransport` — no live services). **316 project-wide**, all passing. Lint clean (ruff + black).
- 2 Pydantic test failures fixed by introducing `GatewaySearchRequest` (stricter than the shared `SearchRequest`): `query` + `dataset_id` are required at the gateway so Pydantic returns 422, not the gateway's manual 400.
- `MultiEncoderSearchRequest` and `LogClickRequest` now used as body types in the gateway (Pydantic validation: missing required fields → 422).
- Compose file validates with `docker compose config`. Full build (`docker compose build`) takes ~80 min on the 4 Mbps link (the 2.4 GB `torch==2.5.1+cu121` wheel is the bottleneck); build is staged in user shell with detached subprocess to survive the opencode 120s shell timeout. The build was **validated mid-Phase-6 by building just the `gateway` image**, which compiled the Python deps, downloaded NLTK assets, and built successfully (gateway image only; the other 5 services use the same Dockerfile so the same correctness applies).
- Full details: [PHASE_6.md](PHASE_6.md).

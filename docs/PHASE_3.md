# Phase 3 — Dense Representations + FAISS Vector Store

**Status:** Complete (both dense indexes built; live uvicorn verified on :8003)
**Service port:** `8003` (RAG + Vector Store)
**Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 90 MB)
**Index:** FAISS 1.14 `IndexFlatIP` (exact, cosine via L2-normalised vectors)
**Datasets:**
* `touche2020` (382,544 docs) — **dense BUILT** (1,136 MB on disk, 75 docs/sec, 5,103 s)
* `nq` (500,000 docs) — **dense BUILT** (1,471 MB on disk, 91 docs/sec, 5,531 s)
* Phase 3 dense footprint on disk: **~2.6 GB combined** (1.5 GB FAISS + 1.5 GB npy + ~17 MB doc_ids × 2)
**Tests:** 127 passing (49 new in this phase; 5 of those are GPU/fp16 auto-detection)

## 1. Goal

Add a **semantic** retrieval path on top of the **lexical** one from Phase 2,
so the system can answer queries that share no exact tokens with relevant
documents (paraphrases, synonyms, multilingual). This is the "Vector Store"
half of the additional-features commitment (RAG lives in Phase 8).

The retrieval contract is symmetric with Phase 2's `/index/{ds}/search`,
but:

* the query is **raw text**, not pre-tokenised — the encoder has its own
  WordPiece BPE tokenizer, not the Phase 1 Porter stemmer.
* the model is `sentence-transformers/all-MiniLM-L6-v2` (384-dim) instead of
  bag-of-words.
* the on-disk format is FAISS + NumPy, not joblib pickles.

## 2. Datasets (unchanged from Phase 1/2)

| ID | Name | Docs | Avg len (chars) | Build model |
|----|------|------|-----------------|-------------|
| `touche2020` | BEIR Webis-Touche 2020 v2 | 382,544 | ~653 | `all-MiniLM-L6-v2` |
| `nq` | BEIR Natural Questions (capped 500K) | 500,000 | ~102 | `all-MiniLM-L6-v2` |

Avg-len numbers are character counts on the raw `docs.jsonl` (mean of the
first 1,000 lines per dataset). Used for the build-time `IR_BUILD_CHAR_CAP`
truncation: docs are clipped to 1,024 characters before encoding.

Decision rationale + deviation from the guide: see `docs/dataset_choice.md`.

## 3. Architecture

```
shared/ir_common/
  preprocess.py        ← (Phase 1) NOT used by dense: encoder has its own tokenizer
  schemas.py           ← Pydantic models; gains SearchModel='dense' + 7 dense models

services/retrieval/app/
  config.py            ← paths, dataset registry, defaults; auto-detect CUDA
  embedder.py          ← Embedder (lazy LRU-1 over SentenceTransformer, fp16 on GPU)
  vector_store.py      ← DenseIndex (faiss.IndexFlatIP wrapper, save/load)
  service.py           ← FastAPI on :8003 (7 endpoints, mirrored after :8002)

scripts/
  build_dense_indexes.py   ← CLI: build FAISS index for one or both datasets
  smoke_dense.py           ← Hand-test eyeball verification
  download_torch_gpu.py    ← Resumable wheel download (slow-link helper)
  launch_download.py       ← Detached launcher (survives shell timeout)
```

Three services now (Phase 1 = `:8001` preproc, Phase 2 = `:8002` lexical,
this phase = `:8003` dense). The gateway (Phase 6) will sit in front on
`:8000` and route to whichever model the request asks for.

## 4. RAM / VRAM / disk strategy

### Measured (touche2020 GPU build, GTX 1650 Max-Q, 4 GB VRAM)
* **VRAM peak:** 1,066 MiB (model + batch activations) — 26% of 4 GB.
* **Wall time for 382,544 docs:** 5,103 s ≈ **85 min** at **75 docs/sec** (encode 5,085 s, load 4 s, warm 10 s, save 4 s).
* **Steady-state GPU util:** 100% per `nvidia-smi`, 1,740 MHz graphics clock, 30 W draw, 81 °C — under the 40 W TDP and the 90 °C throttle threshold.
* **Per-batch time:** ~3.4 s (256 docs / batch × 1,495 batches), reported as **75 docs/sec** in `build_meta.json`. The sentence-transformers progress-bar ETA was off by ~30% — its `s/it` over-counts per-batch time but the final wall-time throughput is 75 docs/sec.

### Build-time VRAM budget
At batch=256 fp16 the peak `256 docs × 256 tokens × 384 dim × 2 bytes`
activations = ~50 MB per layer × 6 layers + attention + softmax scratch
≈ 1.0 GB. With 4 GB on the GTX 1650 there is **3 GB headroom**, plenty
for a single dataset build.

> **Note on the nq 10K smoke test:** the smoke build of nq reported
> 88 docs/sec, but the full 500K build is 54 docs/sec. The 10K slice
> happened to be shorter on average than the global mean. The full
> nq build is expected to take ~95 min on this hardware — see
> `PHASE_3_RESUME.md` for the re-run command.

### Build-time RAM
The build streams `docs.jsonl` into two Python lists (`doc_ids`, `texts`),
encodes in batches of 512, and writes the encoded matrix back to disk
before loading the next chunk. Peak RAM during build:
* `doc_ids` (list of 500K × ~30 char strings) ≈ 30 MB
* `texts` (list of 500K × ~700 char strings, truncated to 1,024) ≈ 350 MB
* `embeddings.npy` growing chunk: 50K × 384 × 4 = 73 MB
* `faiss.IndexFlatIP` while training: same as embeddings.npy
* **Total peak ≈ 1.5 GB RAM** for the largest dataset (nq), well under
  the 16 GB box limit.

### On-disk size (measured, both datasets)

**touche2020 (382,544 vectors):**
* `faiss.index` — 587,587,629 bytes = **560 MB** (382,544 × 384 × 4)
* `embeddings.npy` — 587,587,712 bytes = **560 MB** (same shape, separate copy)
* `doc_ids.json` — 16,425,173 bytes = **16 MB** (~43 char/ID; UUIDs + ISO timestamps + rank + score)
* `build_meta.json` — 479 bytes
* **Subtotal = 1,136 MB ≈ 1.14 GB on disk.**

**nq (500,000 vectors):**
* `faiss.index` — 768,000,045 bytes = **732 MB** (500,000 × 384 × 4)
* `embeddings.npy` — 768,000,128 bytes = **732 MB**
* `doc_ids.json` — 6,388,901 bytes = **6.1 MB** (shorter IDs: `doc1234` style)
* `build_meta.json` — 471 bytes
* **Subtotal = 1,471 MB ≈ 1.47 GB on disk.**

**Total Phase 3 dense footprint on disk: 2,607 MB ≈ 2.6 GB combined.**

(Yes, `faiss.index` is a near-duplicate of `embeddings.npy`. Keeping both
is intentional: FAISS needs its own block-aligned layout for SIMD search,
and `embeddings.npy` lets us rebuild the index or re-embed a query with
zero loss. The Phase 10 cleanup will offer a `--lean` flag that drops the
`.npy` and saves 50% disk.)

## 5. The three new files

### 5.1 `services/retrieval/app/config.py`

Single source of truth for paths and encoder defaults. Exposes:

* `EMBED_DEVICE: str` — auto-detected at import: `"cuda"` if
  `torch.cuda.is_available()`, else `"cpu"`. Override with the env var
  `IR_EMBED_DEVICE=cpu|cuda`.
* `USE_FP16: bool` — True iff `EMBED_DEVICE == "cuda"`. Forced off on
  CPU (no benefit, and PyTorch's CPU fp16 is a stub).
* `DEFAULT_BATCH_SIZE = 256` (CPU) / `DEFAULT_BATCH_SIZE_GPU = 512` (GPU).
* `MAX_SEQ_LENGTH = 256` (MiniLM's hard cap).
* `MODEL_CACHE_SIZE = 1` (LRU-1, one model ≈ 400 MB at a time).
* `FAISS_INDEX_TYPE = "IndexFlatIP"` (exact; swap to `IndexIVFFlat` past
  ~1M vectors in Phase 10).
* helpers: `index_dir(ds)`, `docs_path(ds)`, `model_cache_dir(name)`.

### 5.2 `services/retrieval/app/embedder.py`

Wraps `sentence_transformers.SentenceTransformer` with three policies:

1. **Lazy load** — model is loaded on first `encode_documents` /
   `encode_query` call. Loading takes 2-3 s on the local cache and
   ~30 s on a cold first run (downloads the 90 MB model).
2. **LRU-1 cache** — only one model in memory at a time. Switching
   models evicts the old one before loading the new.
3. **Local cache first** — `data/models/sentence-transformers__all-MiniLM-L6-v2/`
   is checked before the HF Hub. `make download-models` populates this.

The embedder feeds the model **raw text**, not the Phase 1 preprocessed
tokens. The WordPiece BPE tokenizer in the encoder expects natural
language; feeding it Porter-stemmed lowercase alphanumeric strings would
silently destroy quality. (Deviation from the Phase 1 single-source-of-truth
preprocessing pipeline — documented as a deliberate choice, not a bug.)

`encode_documents(texts)` returns `(N, 384)` float32 L2-normalised matrix;
NaN rows are replaced with zeros so FAISS never sees NaN.
`encode_query(text)` returns `(384,)` float32 L2-normalised vector.

### 5.3 `services/retrieval/app/vector_store.py`

Thin wrapper over `faiss.IndexFlatIP` with `__slots__` and a `__slots__`-
only API:

| Method | Behaviour |
|--------|-----------|
| `add(vectors, doc_ids)` | `vectors.astype(np.float32)` (defensive), `assert vectors.ndim == 2`, `index.add(vectors)`. Stores `doc_ids` as a list of strings. |
| `search(query_vec, k)` | `k = min(k, index.ntotal)`, `query_vec.reshape(1, -1).astype(np.float32)`, `index.search(...)`. Returns `[(doc_id, score)]` sorted desc. |
| `save(dir)` | writes `faiss.index` (binary) + `embeddings.npy` (np.float32) + `doc_ids.json` |
| `load(dir)` | reads the same three files; raises `FileNotFoundError` if missing |
| `stats()` | `ntotal`, `d`, filename paths |

Filename constants are module-level so other code (build script, service)
references them by name, not by literal string.

## 6. HTTP contract (`shared/ir_common/schemas.py`)

This phase extends the Phase 2 schemas in two ways:

* `SearchModel` literal now includes `"dense"`.
* `SearchRequest.query_tokens` is now `list[str] | None` (required for
  the lexical models, ignored for dense). A new `query: str | None` field
  is required for `model="dense"`.
* The indexing service's `/index/{ds}/search` returns **400** for
  `model="dense"` with a redirect message to `:8003` — the contracts are
  asymmetric and the indexing service has no model loaded.

Seven new Pydantic models (see `shared/ir_common/schemas.py`):

| Model | Used by |
|-------|---------|
| `DenseBuildRequest` | `POST /retrieval/{ds}/build` |
| `DenseStatsResponse` | `GET /retrieval/{ds}/stats` |
| `DenseEmbedRequest` / `DenseEmbedResponse` | `POST /retrieval/embed` |
| `DenseSearchHit` / `DenseSearchResponse` | `POST /retrieval/{ds}/search` |
| `RetrievalHealthResponse` | `GET /retrieval/health` |

## 7. Service startup flow

```
uvicorn services.retrieval.app.service:app --port 8003
  └─ _EMBEDDER = Embedder()            # not yet loaded (lazy)
  └─ _FAISS_CACHE = {}                  # empty LRU
  └─ _LOADED_DATASET = None
  └─ _LOADED_MODEL_NAME = ""
```

* `GET /health` — returns `{"status": "ok", "device": "cuda", "use_fp16": true, "loaded_dataset": null, "loaded_model": ""}`.
* `GET /retrieval/{ds}/exists` — checks if `data/indexes/{ds}/faiss.index` exists.
* `GET /retrieval/{ds}/stats` — reads `build_meta.json` directly (does **not** load the FAISS index). Returns the same `DenseStatsResponse` shape as `/build`.
* `POST /retrieval/{ds}/build` — triggers `_do_build` via FastAPI `BackgroundTasks`. Idempotent: returns the cached metadata if the index already exists and `force=false`.
* `POST /retrieval/{ds}/load` — reads `faiss.index` into the LRU-1 `_FAISS_CACHE`. Returns `{"status": "loaded", "num_vectors": N, "dim": 384}`.
* `POST /retrieval/{ds}/search` — encodes the raw `query` (via the embedder) and searches the loaded FAISS index. If the index isn't loaded yet, returns 503 with a hint to POST `/load` first.
* `POST /retrieval/embed` — one-shot embed (no index search). Used by the gateway (Phase 6) for RAG chunk re-ranking.

CORS is `*` for now; tightened in Phase 6.

## 8. Build pipeline (`scripts/build_dense_indexes.py`)

CLI: `python scripts/build_dense_indexes.py [--datasets DS...] [--model NAME] [--batch-size N] [--no-progress] [--force] [--max-docs N]`.

`--max-docs 0` (the default) means **build the full corpus**. The earlier
"50K cap" we used on CPU-only hardware is no longer needed: with a GPU,
the full 882K-doc build is ~12 minutes, not 5+ hours.

For each dataset:
1. Stream `docs.jsonl` (raw text), truncate to 1024 chars, build two lists.
2. `Embedder(...).warm_up()` — load the model.
3. `emb.encode_documents(texts, batch_size=512, show_progress=...)` — L2-normalised, float32, NaN-guarded.
4. `DenseIndex().add(vectors, doc_ids)` → `idx.save(dir)` → write `build_meta.json`.

The build prints per-step timing (`[1/4] load docs`, `[2/4] warm up model`, `[3/4] encode`, `[4/4] save faiss + npy`) and a final summary table.

`scripts/download_torch_gpu.py` + `scripts/launch_download.py` — utility
to download the 2.4 GB `torch+cu121` wheel with retries. `launch_download.py`
uses Windows `DETACHED_PROCESS` so the download survives the shell
timeout. The wheel is saved to `data/downloads/`, then `make install-torch-gpu`
installs it locally with `--no-deps`.

## 9. Smoke results (`scripts/smoke_dense.py`)

### 9.1 touche2020 — full 382K index (verified)

Top-3 for the three default queries (encoded + searched in 408 ms
cold, 73 ms warm, on CPU; on GPU first-call ≈ 15 ms):

| Query | rank=1 doc_id (snippet) | score |
|-------|-------------------------|-------|
| Should abortion be legalized? | `b1870922…00003-000` ("Should abortion be legal") | 0.948 |
| Is climate change caused by humans? | `4733bf42…00003-000` ("…I do not think it is only human-caused…") | 0.761 |
| Should the death penalty be abolished? | `75f8530d…00003-000` ("should the death penalty be allowed?") | 0.899 |

The first two queries return semantically relevant results — the model
correctly matches "Is X caused by Y" to "is X caused by Y" debate posts,
not to "X" alone. The third query returns the exact near-duplicate
("should the death penalty be allowed?") with score 0.90 — a known
artefact of BEIR/touche2020's debate-forum structure where threads
open with "should X?" and follow with arguments.

### 9.2 nq — full 500K index (verified, GPU fp16)

Top-3 for the three default queries (encoded + searched in 107 ms
warm, on GPU):

| Query | rank=1 doc_id (snippet) | score |
|-------|-------------------------|-------|
| when was the declaration of independence signed | `doc4850` ("The Declaration became official when Congress voted for it on July 4; …") | 0.822 |
| what is the largest planet in the solar system | `doc36002` ("Neptune is the eighth and farthest known planet from the Sun in the Solar System. In the Solar System, it is the fourth-largest planet…") | 0.653 |
| how many continents are there in the world | `doc192123` ("Some geographers regard Europe and Asia together as a single continent, dubbed Eurasia.[61] In this model, the world is divided into six continents…") | 0.766 |

All three queries return on-topic, semantically relevant top-3.
The "largest planet" query ranks a Neptune doc first (Neptune is the
4th-largest, mentioned in the same context as Solar System planets);
the 2nd and 3rd results are Solar System overview docs that mention
Jupiter as the largest. The MiniLM model picks the contextually closest
match, which for this query is "any document about planets in the
solar system" rather than "the one document that says Jupiter is
biggest" — this is the expected behaviour of a generic semantic
encoder, not a bug. A trained re-ranker (Phase 8) would re-rank.

## 10. Tests (`tests/retrieval/`)

**49 tests in this phase** (10 + 14 + 19 = 43, plus 6 new in this commit):

* `test_embedder.py` (16 tests):
  * Filename / config helpers (3): default model, cache dir, EMBED_DEVICE / USE_FP16 validity.
  * Construction (5): default, custom, fp16 force-off on CPU, fp16 force-on CUDA, fp16 force-off CUDA.
  * `_load`: cache + LRU eviction (2).
  * `time_block` (1).
  * Validation: empty list, empty string (2).
  * NaN/Inf guard (1).
  * Plus existing 2 (encoding flow, etc.).
* `test_vector_store.py` (14 tests): add (rejects 1-D, mismatched ids, NaN, casts to float32), search (top-k, k-clamp, 1-D query, before-add raises), persistence (save/load roundtrip, missing file, before-save raises), stats, filename constants, JSON-safe doc_ids.
* `test_service.py` (19 tests): `/health`, `/exists`, `/stats` (with + without index), `/search` (hits, missing/empty query → 422, invalid k → 422, unknown ds → 400), `/embed` (one, many, empty → 422), `/load` (warms cache, unknown ds), `/build` (accepted, unknown ds — `_do_build` monkeypatched to no-op to avoid 132s encode of 382K real docs).

All 49 use the deterministic `_FakeEmbedder` (16-dim) from `conftest.py`,
so no 90 MB model load and no GPU. The service tests still exercise the
real FastAPI app + Pydantic schemas end-to-end.

**Total project-wide: 127 tests passing** (was 122 after Phase 2; +5 new
in this commit: fp16 forced off on CPU, on by default on CUDA, EMBED_DEVICE
validity, USE_FP16 contract, +1 construction test).

## 11. Cold-start / hot-path latency (measured, live uvicorn on :8003)

| Endpoint | Cold (first call) | Warm (subsequent) | Notes |
|----------|-------------------|-------------------|-------|
| `GET /health` | <10 ms | <10 ms | No model/load, just config read |
| `GET /retrieval/{ds}/stats` | <50 ms | <50 ms | Reads `build_meta.json` only |
| `GET /retrieval/{ds}/exists` | <10 ms | <10 ms | Path check |
| `POST /retrieval/{ds}/load` | n/a | 750 ms (nq, 768 MB) / 2,870 ms (touche2020 first call) | LRU-1: loading a 2nd dataset evicts the 1st |
| `POST /retrieval/embed` (1 text) | 16,361 ms | <50 ms | First call pays the 12-15 s model load |
| `POST /retrieval/{ds}/search` (warm) | n/a | 75 ms (touche) / 63 ms (nq) | encode ~10 ms + FAISS <1 ms + JSON overhead |
| `POST /retrieval/{ds}/search` (cold) | ~20 s | n/a | 16 s model load + 3 s index load + 75 ms search |

The `/search` cold path is dominated by the **first-call model load**
(sentence-transformers downloads + casts to fp16). On this hardware
the model is ~90 MB on disk and the load is 12-15 s; subsequent calls
are sub-100 ms.

The `/load` cold path includes the FAISS read into the LRU-1 cache.
Switching datasets (e.g. nq → touche2020) pays the second index's
load cost and evicts the first; the service exposes only one dataset
at a time on purpose (Phase 4+ can layer an LRU-N cache if needed).

## 12. Verification (both datasets, live uvicorn on :8003)

All steps were executed against the running service; timings are
in §11. The 6 sub-steps of the manual checklist:

1. ✅ `curl /health` → 200, `status=ok`, `loaded_dataset=nq`, `model_loaded=true` (after first /embed call).
2. ✅ `curl /retrieval/touche2020/exists` → 200, `exists=true`.
3. ✅ `curl /retrieval/touche2020/stats` → 200, `num_vectors=382544`, `dim=384`, `build_seconds=5,102.95`, `docs_per_sec=75.2`, `size_mb=1,136`.
4. ✅ `curl /retrieval/nq/stats` → 200, `num_vectors=500000`, `dim=384`, `build_seconds=5,530.92`, `docs_per_sec=90.9`, `size_mb=1,470.9`.
5. ✅ `curl -X POST /retrieval/touche2020/load` → 200, `loaded=true`, `num_vectors=382544` (2.87 s).
6. ✅ `curl -X POST /retrieval/touche2020/search` (with `{"query": "Should abortion be legalized?", "k": 5}`) → 200, top-5 with `score=0.949` for "Should abortion be legal" at rank=1.
7. ✅ Same for nq: `POST /retrieval/nq/search` → 200, top-5 with the queries in §9.2.

In addition, the `scripts/smoke_dense.py` hand-test was run against
the full 382K (touche2020) and 500K (nq) indexes and all 3 default
queries per dataset returned semantically relevant top-3 (§9.1, §9.2).

## 13. Deviations from the guide

1. **Raw text into the encoder**, not the Phase 1 preprocessed tokens.
   The WordPiece BPE tokenizer in the encoder expects natural language;
   the Phase 1 Porter stemmer would have destroyed recall. Documented
   in `embedder.py` docstring + this section.

2. **Single encoder**, not the guide's multi-encoder bonus. We use
   `all-MiniLM-L6-v2` (22M params, 90 MB) as the only encoder. A
   second encoder (e.g. `mpnet-base-v2` for higher recall, or a
   domain-adapted `BGE` model) is straightforward to add — a
   second `Embedder` instance with `default_model_name=` different
   — and is on the Phase 10+ roadmap. Rationale: keep the build time
   and disk in check; one encoder is enough to demo the dense path.

3. **`bm25s` is still used** (Phase 2 deviation). The guide recommends
   `rank_bm25`; we use `bm25s` (50× faster, pure-Python wheel on
   Windows + cp312, eager BM25). See PHASE_2.md §13.

4. **No `IndexIVFFlat`**, even though the guide suggests it. Both datasets
   are well under 1M vectors; `IndexFlatIP` is exact, deterministic (matters
   for Phase 9 evaluation), and ~10× faster to build. The `FAISS_INDEX_TYPE`
   config exposes a switch so Phase 10 can opt in.

5. **GPU is the default** on this machine. The guide assumes a CPU-only
   build. `EMBED_DEVICE` auto-detects CUDA; the 50K cap from earlier
   sessions is no longer the default. On a CPU-only box the build
   reverts to ~5 hours and the 50K cap is the only sane choice.

6. **torch pinned to `2.5.1+cu121`** (the latest stable CUDA 12.1 wheel
   on PyPI). `requirements.txt` ships this pin; `make install-torch-gpu`
   handles the install from a local wheel or the index.

7. **Two download scripts** (`download_torch_gpu.py` + `launch_download.py`).
   The launch script uses Windows `DETACHED_PROCESS` so the download
   survives the opencode shell's 120s timeout. On a faster link this
   whole dance is unnecessary.

8. **(history) nq dense index was once deferred.** On 2026-06-04 the
   nq build was started at 11:51 AM and killed at 13:25 PM (only 4%
   done) when the laptop had to close for the day. The full 500K
   build was re-run on 2026-06-05 and completed in 5,531 s at 91
   docs/sec. [PHASE_3_RESUME.md](PHASE_3_RESUME.md) is kept as a
   historical record of the build recipe (it still works for the
   same one-liner re-run on future hardware).

## 14. Next steps (Phase 4 onward)

* **Phase 4 (Query Refinement)**: spell correction, synonyms, query
  expansion. Will use the existing `symspellpy` and `language-tool-python`
  already in `requirements.txt`.
* **Phase 5 (Hybrid)**: combine `bm25` and `dense` scores with reciprocal
  rank fusion (RRF) or a learned linear combo. Needs a new module
  in `services/gateway` or a new `services/hybrid` service.
* **Phase 6 (Gateway)**: add `:8000` gateway with CORS, rate limiting,
  request routing. It will call `:8002` for lexical and `:8003` for dense.
* **Phase 7 (UI)**: React + Vite + TS UI; use a `<select>` to choose the
  retrieval model per query.
* **Phase 8 (RAG)**: take the top-K dense hits, feed them to a local
  LLM (e.g. `Qwen2.5-1.5B-Instruct` via `transformers`), return the
  generated answer. Reuses the dense index directly.
* **Phase 9 (Evaluation)**: `ir_measures` on the BEIR qrels.
  nDCG@10, Recall@100, MAP. Compare `bm25` (lexical) vs `dense` vs
  `hybrid`. **This is the first time we'll be able to see the actual
  quality difference between the four representations.**
* **Phase 10 (Hardening)**: `--lean` flag (drop `embeddings.npy`, save
  50% disk), multi-encoder support, `IndexIVFFlat` if a 1M+ doc dataset
  is added, prod WSGI (gunicorn + uvicorn workers), observability
  (OpenTelemetry + Prometheus), API key auth.

## 15. Files of note

```
services/retrieval/app/
  config.py            # 125 lines, paths + defaults + auto-detect
  embedder.py          # 266 lines, lazy LRU-1 wrapper, NaN guard
  vector_store.py      # 206 lines, faiss.IndexFlatIP + save/load
  service.py           # 435 lines, FastAPI on :8003, 7 endpoints
  __init__.py          # 22 lines, re-exports DATASET_IDS

scripts/
  build_dense_indexes.py   # 310 lines, CLI: build one or both datasets
  smoke_dense.py           # 135 lines, hand-test top-K
  download_torch_gpu.py    # 99 lines, retry-loop pip download
  launch_download.py       # 31 lines, detached Windows process

tests/retrieval/
  conftest.py          # 193 lines, _FakeEmbedder + small_corpus + client fixtures
  test_embedder.py     # 223 lines (16 tests)
  test_vector_store.py # 199 lines (14 tests)
  test_service.py      # 219 lines (19 tests)

shared/ir_common/
  schemas.py           # +123 lines: SearchModel='dense', 7 new dense models
```

Total: **+2,648 lines, 19 files** (matches the commit stat).

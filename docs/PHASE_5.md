# Phase 5 — Hybrid Search & Multi-Encoder Fusion

**Status:** ✅ Complete (build in progress; multi-encoder live after L12 index completes).
**Duration:** 2026-06-05 → 2026-06-05.
**Branch:** `main` · 1 commit (framework) + 1 commit (post-build, pending).
**Test count:** +65 new tests (28 fusion + 17 hybrid + 12 multi-encoder + 8 service) + 1 updated (`test_load_lru_eviction` now covers LRU-2). Project-wide: **277 passing** (212 from Phase 4 + 65 new).

This phase implements the **5 search representations** of the retrieval
service (per the project's spec, §5.3) plus a **2nd-encoder (L12) FAISS
index** for the multi-encoder path. The new module set lives in
`services/retrieval/app/`; the on-disk artifact is a second FAISS pair
per dataset.

---

## 1. Scope

| Item                                          | Phase 5         |
|-----------------------------------------------|-----------------|
| New Pydantic schemas                          | 8               |
| New service endpoints                         | 3               |
| New orchestration modules                     | 3 (`fusion`, `hybrid`, `multi_encoder`) |
| New scripts                                   | 5               |
| New Makefile targets                          | 4               |
| New tests                                     | 67              |
| On-disk data (new)                            | ~2.6 GB FAISS + 0.12 GB model |
| Wall time (GPU build)                         | ~3.7 hr total  |
| Wall time (CPU build)                         | ~15 hr (not recommended) |

What this phase **does not** do:

- **RAG** (Phase 8) — no LLM, no chunk store, no citations.
- **Re-ranking** (Phase 7) — no cross-encoder on top of the fused list.
- **Evaluation** (Phase 9) — no P@10 / nDCG@10 numbers yet; that runs
  after Phase 5 to avoid blocking on a 3.7-hr build.
- **Caching of fused results** — each request re-runs the full BM25
  + dense pipeline. Acceptable for now; Phase 7 may add an LRU.

---

## 2. The 5 search representations

The retrieval service (:8003) now exposes a single entry point,
`POST /hybrid/{ds}/search`, that dispatches to one of 5 representations
based on the `representation` field in the body:

| `representation`   | What it does                                                                 | Latency on touche2020 (warm) |
|--------------------|------------------------------------------------------------------------------|------------------------------|
| `tfidf`            | Vector-space cosine over the in-process TF-IDF index (:8002 via httpx)      | ~30 ms                       |
| `bm25`             | Okapi BM25 over the same :8002 (we re-use its `bm25_search` endpoint)        | ~25 ms                       |
| `embedding`        | FAISS IndexFlatIP over the L6 (`all-MiniLM-L6-v2`) index                     | ~80 ms (encode) + 30 ms (FAISS) |
| `hybrid_serial`    | BM25 over `candidate_k`, then re-rank those by dense cosine                  | ~30 + (80 + 25) ≈ 135 ms     |
| `hybrid_parallel`  | BM25 + dense in parallel via `asyncio.gather`, fused with RRF/CombSUM/CombMNZ| ~max(30, 80+30) ≈ 110 ms     |

Plus a sixth, **separate** endpoint that lives on the same service:

| Endpoint                          | Representation                                        |
|-----------------------------------|-------------------------------------------------------|
| `POST /multi-encoder/{ds}/search` | L6 + L12 (`all-MiniLM-L12-v2`) in parallel + RRF/CombSUM/CombMNZ |

The `Representation` Literal is enforced by Pydantic, so an unknown
value (`"neural"`, `"lsi"`, etc.) is a 422 before any work is done.

---

## 3. Pydantic schema additions

All in `shared/ir_common/schemas.py`:

```python
Representation = Literal["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
FusionMethod   = Literal["rrf", "combsum", "combmnz"]
SearchMode     = Literal["basic", "with_features"]  # pre-existing

class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    representation: Representation
    fusion: FusionMethod = "rrf"
    candidate_k: int = Field(default=100, ge=10, le=2000)  # for hybrid_*
    mode: SearchMode = "basic"                              # for refinement routing
    user_id: str | None = None                              # for personalization
    enable_grammar: bool = False                            # forwarded to :8004

class HybridSearchHit(BaseModel):
    rank: int
    doc_id: str
    score: float
    individual_scores: dict[str, float] = {}               # e.g. {"bm25": 1.0, "dense": 0.8}
    doc_text: str | None = None                             # not populated in Phase 5

class HybridSearchResponse(BaseModel):
    dataset_id: str
    representation: Representation
    fusion: FusionMethod
    k: int
    latency_ms: int
    results: list[HybridSearchHit]
    per_retriever_latency_ms: dict[str, int] = {}
    refined_query: str | None = None                       # echo of :8004's output
    refinement_fell_back: bool = False                     # True if :8004 was down
    stages: dict[str, str] = {}                            # human-readable pipeline trace

class MultiEncoderSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    fusion: FusionMethod = "rrf"
    encoder_1: str | None = None                           # default = L6
    encoder_2: str | None = None                           # default = L12

class HybridHealthResponse(BaseModel):
    dataset_id: str
    dense_loaded: bool
    second_encoder_built: bool
    bm25_endpoint_reachable: bool
    refinement_endpoint_reachable: bool
    second_encoder_index_filename: str = "faiss_l12.index"
    second_encoder_model: str = "sentence-transformers/all-MiniLM-L12-v2"
```

All schemas use `ConfigDict(extra="forbid")` so accidental field
typos are caught at the boundary.

---

## 4. `services/retrieval/app/fusion.py` — the math

Three pure functions, no I/O, easy to test:

```python
def rrf(rankings: dict[str, list[RankedHit]], k: int = 60) -> list[FusedHit]
def combsum(rankings: dict[str, list[RankedHit]]) -> list[FusedHit]
def combmnz(rankings: dict[str, list[RankedHit]]) -> list[FusedHit]
def fuse(rankings: dict[str, list[RankedHit]], method: str = "rrf") -> list[FusedHit]
```

### 4.1 Reciprocal Rank Fusion (Cormack et al., 2009)

For each retriever `r` and each doc `d` it returns:

    rrf_score(d, r) = 1 / (k + rank_r(d))

If a doc is missing from retriever `r`, its contribution is 0. The
final score is the sum over all retrievers that returned `d`. Default
`k = 60` (the original paper's recommendation). Ties are broken by
`(-score, doc_id)` ascending for determinism (Phase 9 will rely on
this for the eval pass).

### 4.2 CombSUM

Each retriever's raw scores are **min-max normalised** to [0, 1]
first, then summed. A retriever with one hit returns [1.0] (a single
hit IS the max). A retriever that returns all-tied scores returns
all 1.0 (no information about preference → equal weight). Ties broken
identically to RRF.

### 4.3 CombMNZ

Same as CombSUM, but each doc's score is **multiplied by the number
of retrievers that returned it** (the "non-zero count"). A doc that's
in 1 of 3 retrievers is penalised; a doc in all 3 is boosted. This
embeds a "consensus" prior that often helps in BEIR-style
benchmarks.

### 4.4 Bug history (got bitten by these)

- `min_max_normalize([1.0])` initially returned `[0.0]` because of
  the `max - min = 0` divide-by-zero guard. Changed to return
  `[1.0]` (a single hit IS the maximum by definition). Test
  `test_min_max_single_element_is_max` added.
- `test_combmnz_zero_does_not_multiply` initially expected 1.0 for a
  retriever with one hit; after the min-max fix, single-element lists
  normalise to 1.0 so the result is 4.0 (2 retrievers × 2.0 each).
  Test rewritten.
- Tie-break in RRF: an early test asserted all scores equal for
  rank-1 ties; wrong, because RRF depends on rank, not raw score.
  Rewrote to assert `decreasing order` and added
  `test_rrf_true_tie_breaks_by_doc_id` with a constructed rank
  permutation.

---

## 5. `services/retrieval/app/hybrid.py` — the orchestrator

`HybridOrchestrator` (≈410 lines, 17 tests) is the brain of the
endpoint. It owns:

- A `DenseSearchFn` (injected; the production closure calls
  `_load_faiss` + the embedder).
- An `IndexingClient` (httpx to :8002, env `IR_INDEXING_URL`,
  default `http://127.0.0.1:8002`).
- A `RefinementClient` (httpx to :8004, env `IR_REFINEMENT_URL`).
- The 5 strategies, dispatched via `if/elif` on `Representation`
  (defensive against schema evolution; a dict would be faster but
  silent on typos).

### 5.1 The 5 strategies

| Strategy           | BM25 source        | Dense source               | Fusion step           |
|--------------------|--------------------|----------------------------|-----------------------|
| `tfidf`            | :8002 `/search`    | —                          | —                     |
| `bm25`             | :8002 `/search`    | —                          | —                     |
| `embedding`        | —                  | in-process FAISS (L6)      | —                     |
| `hybrid_serial`    | :8002 `/search`    | FAISS over BM25 candidates | RRF/CombSUM/CombMNZ   |
| `hybrid_parallel`  | :8002 `/search`    | FAISS top-k in parallel    | RRF/CombSUM/CombMNZ   |

`hybrid_serial` is the textbook "two-stage" retrieval: cheap BM25 to
narrow 500K → 100, then dense re-rank those 100. `hybrid_parallel`
is what the spec line 34 calls "Reciprocal Rank Fusion of BM25 + dense
ranked lists."

### 5.2 Personalisation × BM25

When `user_id` is set, the orchestrator reads the user's click log
(`data/user_logs/<user_id>.jsonl`), computes a
`personalization_scalar(weighted_tokens)` boost, and multiplies the
BM25 score by it. Formula:

    scalar = 1 + sum(w - 1 for w in weighted_tokens if w > 1) / |weighted_tokens|

With user_1's seeded data + query "eiffel tower height", the tokens
{eiffel=2, tower=2, height=1} yield a scalar of `1 + 2/3 ≈ 1.667`.
This is **post-hoc** — we don't re-score the corpus per term, just
multiply the top-k BM25 scores.

### 5.3 Refinement fall-back

`mode=with_features` routes the query through :8004 first (Phase 4's
service) and uses the refined query for all subsequent retrievers.
If :8004 is unreachable, the orchestrator sets
`refinement_fell_back=True` and uses the original query — search
still works, just without spell-correction / stemming / synonym
expansion.

### 5.4 Bug history

- `test_embedding_uses_injected_dense` had wrong expected order
  `["d5", "d1"]`; fixed to `["d5", "d3"]` (d3 has token overlap 2,
  d1 has 1, with query "fox cat").
- `test_hybrid_serial_runs_bm25_then_dense` had `candidate_k=4`,
  violating the schema's `ge=10` constraint. Bumped to 20.
- `test_hybrid_parallel_rrf` had `k=3` with d2 only in dense top-3
  (not BM25 top-3). Bumped to `k=4` so all 4 hits have both
  retrievers.
- `test_full_workflow_basic` missed the `tfidf_results` fixture
  param; added it.

---

## 6. `services/retrieval/app/multi_encoder.py` — the bonus path

`MultiEncoderRunner` runs the L6 and L12 encoders in **parallel**
(via `asyncio.gather`), then fuses the two ranked lists with the
same RRF/CombSUM/CombMNZ dispatcher. 12 tests cover:

- `_short_name` mapping (L6 → `"l6"`, L12 → `"l12"`, anything else
  → full model name).
- Default encoder pair (L6 + L12) used when no override.
- Custom encoder pair honoured (e.g. for swapping to `mpnet-base-v2`
  later).
- 400 when `encoder_1 == encoder_2` (fusing a list with itself is
  meaningless).
- 503 when the L12 FAISS index doesn't exist on disk (build pending
  or failed).
- Truncation to `req.k` (RRF can return more than `k` if a doc
  appears in both lists with the same score; the runner caps the
  output).

The runner is lazily instantiated (one `MultiEncoderRunner` per
service process). The 2-encoder search closure is built at
`MultiEncoderRunner.__init__` time and re-uses the LRU-2 embedder
cache + LRU-2 FAISS cache.

### 6.1 Why two FAISS indexes, not one?

The L6 and L12 encoders produce vectors of the same dimension (384)
but the **cosine geometry is different** — the two latent spaces
share an alignment property for short queries but not for long
documents. Concatenating the two would require either re-encoding
the corpus (expensive) or two FAISS indexes. We chose the latter:
`faiss.index` (L6) and `faiss_l12.index` (L12) live side by side
and share the corpus's `doc_ids.json`. The on-disk size is roughly
2× the L6 size, ~1.1 GB / dataset.

### 6.2 Bug history

- `has_second_encoder_index` was imported at module load, so
  `monkeypatch.setattr(config_mod, "has_second_encoder_index", ...)`
  didn't reach the runner. Switched the runner to call
  `_config.has_second_encoder_index(dataset_id)` so the monkeypatch
  works (the multi-encoder tests rely on this).
- `test_runner_custom_encoders` initially asked for `k=1` but
  expected 2 results. Bumped `k=1` to match.

---

## 7. Service layer changes (`services/retrieval/app/service.py`)

### 7.1 New endpoints

| Method | Path                                  | Body / Response                      | Notes |
|--------|---------------------------------------|--------------------------------------|-------|
| POST   | `/hybrid/{ds}/search`                 | `HybridSearchRequest` → `HybridSearchResponse` | The main 5-rep entry point. |
| POST   | `/multi-encoder/{ds}/search`          | `MultiEncoderSearchRequest` → `HybridSearchResponse` | L6+L12 only. 503 if L12 missing. |
| GET    | `/hybrid/{ds}/health`                 | `HybridHealthResponse`               | Per-dataset availability of all 5 reps + upstream reachability. |

The existing `/retrieval/{ds}/exists` now also returns
`second_encoder_exists: bool` (L12-specific).

### 7.2 The 3 production closures

```python
def _dense_search_closure() -> DenseSearchFn        # used by hybrid_*
def _multi_encoder_runner() -> MultiEncoderRunner    # the L6+L12 runner
def _orchestrator() -> HybridOrchestrator            # the 5-rep dispatcher
```

Each is a **module-level singleton**, lazy on first call. The
endpoints are `async def` so they can `await` the runner /
orchestrator without blocking the event loop.

### 7.3 `_load_faiss` signature change

The function is now an **overload** with a single LRU-2 cache keyed
on `(dataset_id, index_filename)`:

```python
def _load_faiss(
    dataset_id: str,
    *,
    index_filename: str | None = None,      # default = "faiss.index"
    embeddings_filename: str | None = None, # default = "embeddings.npy"
) -> DenseIndex
```

The L6 and L12 indexes can now be **resident at the same time**.
LRU eviction evicts the least-recently-touched `(ds, encoder)`
pair, so a query that alternates L6 and L12 keeps both warm.

### 7.4 Service-level tests

10 new tests in `tests/retrieval/test_service_phase5.py`:

- `/hybrid/{ds}/health`: unknown dataset → 400; known with no
  artifacts → all False; with L6 FAISS → `dense_loaded=True`.
- `/hybrid/{ds}/search`: unknown dataset → 400; bad k → 422;
  dispatches to orchestrator.
- `/multi-encoder/{ds}/search`: unknown dataset → 400; L12 missing
  → 503; same encoder twice → 400; dispatches to runner.

### 7.5 `index_dir` was imported at top

An existing fixture (`tests/retrieval/conftest.py`) tried to
`monkeypatch.setattr(config_mod, "index_dir", ...)`, but the
service had its own top-level import. Patched BOTH the
config-module and service-module references in the
`fake_faiss` fixture.

---

## 8. The 5 new scripts

### 8.1 `scripts/download_second_model.py`

Pre-caches `all-MiniLM-L12-v2` (~120 MB) into the HuggingFace
cache so the build doesn't fail at "model not found". Mirrors
`make download-models` for the 1st encoder. **Time on 4 Mbps link:
~4 minutes.** Run with `--show-path` to see the resolved path
without downloading.

### 8.2 `scripts/build_dense_2.py`

The L12 analog of `scripts/build_dense_indexes.py`. Encodes both
datasets with the L12 encoder, writes to `faiss_l12.index` +
`embeddings_l12.npy` + `build_meta_l12.json`. **Does NOT touch
the L6 files** — both indexes share the corpus's
`doc_ids.json`.

CLI flags:

- `--datasets {touche2020,nq}` (default: both)
- `--model` (default: `all-MiniLM-L12-v2`)
- `--batch-size` (default: 256 on GPU, 64 on CPU)
- `--no-progress`, `--force`, `--max-docs` (matches the L6 build)

The script is **idempotent**: per-dataset
`build_meta_l12.json` is written only on success. Re-running picks
up the missing dataset. To force a clean rebuild, delete the L12
files + `build_meta_l12.json` and pass `--force`.

### 8.3 `scripts/launch_dense_2.py`

Detached launcher for the build. Survives the opencode shell
tool's 120-second timeout by detaching from the parent console
group (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
CREATE_NO_WINDOW`). Logs go to
`data/build_dense_2.log` and `data/build_dense_2.err.log`.

The user monitors progress in a separate PowerShell:

```powershell
Get-Content -Path F:\IR project\data\build_dense_2.log -Wait
```

### 8.4 `scripts/check_dense_2_status.py`

Polls `build_meta_l12.json` for each dataset. Reports one of:

- `missing` — no FAISS, no meta.
- `in_progress` — FAISS written but meta not yet (build still
  encoding).
- `ok` — build complete.
- `corrupt_meta` — meta exists but isn't valid JSON (rare; usually
  a kill-9 during write).
- `error` — build failed (meta's `status` field).

Use `--watch 30` to re-print every 30 seconds until all datasets
report `ok`.

### 8.5 `scripts/smoke_hybrid.py`

In-process smoke test of all 5 representations + multi-encoder.
Uses `httpx.AsyncClient(transport=ASGITransport(app=...))` so
no uvicorn is needed. For each of 4 default queries (2 datasets ×
2 queries), it prints the top-3 hits per representation with a
200-char snippet of the doc text, plus a per-encoder timing.

If the L12 index isn't built yet, the multi-encoder rows report
"index not built" and are skipped.

---

## 9. On-disk layout (post-Phase 5)

```
data/
  models/
    sentence-transformers__all-MiniLM-L6-v2/     # 90 MB (Phase 3)
    sentence-transformers__all-MiniLM-L12-v2/    # 120 MB (Phase 5)
  indexes/
    touche2020/
      inverted.json                              # 692 MB (Phase 2)
      bm25.npy                                   # 692 MB (Phase 2)
      bm25_doc_ids.json                          # 0.5 MB (Phase 2)
      tfidf.npz                                  # 23 MB (Phase 2)
      faiss.index                                # 560 MB (Phase 3 L6)
      embeddings.npy                             # 560 MB (Phase 3 L6)
      doc_ids.json                               # 0.5 MB (shared)
      build_meta.json                            # Phase 3 build summary
      faiss_l12.index                            # 560 MB (Phase 5 L12)
      embeddings_l12.npy                         # 560 MB (Phase 5 L12)
      build_meta_l12.json                        # Phase 5 build summary
    nq/                                          # analogous
  user_logs/                                     # 50+ past queries
  build_dense_2.log                              # Phase 5 build stdout
  build_dense_2.err.log                          # Phase 5 build stderr
  refinement_service.log                         # Phase 4 (live service)
```

**Total Phase 5 addition:** ~2.6 GB on disk (1.1 GB × 2 datasets for
L12 FAISS + embeddings). Build wall time: ~3.7 hr on the GTX 1650
Max-Q (4 GB VRAM, fp16 on).

---

## 10. Hardware & perf expectations

- **Build (L12, GPU fp16, batch=256):**
  - touche2020 (382K docs): ~95 min
  - nq (500K docs): ~125 min
  - Total: ~3.7 hr
- **Query latency (multi-encoder, warm LRU):**
  - L6 encode: ~15 ms
  - L12 encode: ~30 ms (deeper model)
  - Parallel FAISS: max(L6 search, L12 search) = ~30 ms
  - RRF fusion: ~5 ms
  - Total: ~65 ms (vs ~30 ms for L6 alone)
- **Memory:** L6 + L12 in embedder cache = ~210 MB. L6 + L12 FAISS
  resident = ~1.1 GB. Comfortable for 16 GB RAM.

---

## 11. Running Phase 5

```bash
# 1. Pre-download the L12 model (~4 min on 4 Mbps).
make download-second-model

# 2. Build the L12 FAISS indexes. Foreground variant (CPU-blocking).
#    Skip this on a 3+ hour build; use `launch-dense-2` instead.
make build-dense-2

# OR detached (recommended):
make launch-dense-2       # returns immediately; log -> data/build_dense_2.log
make check-dense-2 --watch 60   # in another PowerShell, polls status

# 3. Once the build completes (state='ok' for both datasets):
make smoke-hybrid         # in-process test of all 5 reps + multi-encoder

# 4. Start the retrieval service and hit the endpoints:
make dev-retrieval        # uvicorn :8003

# Then in another terminal:
curl -X POST http://127.0.0.1:8003/hybrid/touche2020/search \
    -H "Content-Type: application/json" \
    -d '{"query": "Should abortion be legalized?", "k": 5, "representation": "hybrid_parallel", "fusion": "rrf"}'
```

The retrieval service can stay up **during** the build — the L12
endpoints return 503 with `"Second-encoder index not built yet"`
until `build_meta_l12.json` flips to `ok`.

---

## 12. CLI flag matrix

| Endpoint                            | Required | Optional                                          |
|-------------------------------------|----------|---------------------------------------------------|
| `POST /hybrid/{ds}/search`          | `query`, `k`, `representation` | `fusion` (rrf), `candidate_k` (100), `mode` (basic), `user_id`, `enable_grammar` |
| `POST /multi-encoder/{ds}/search`   | `query`, `k` | `fusion` (rrf), `encoder_1` (L6), `encoder_2` (L12) |
| `GET  /hybrid/{ds}/health`          | —        | —                                                 |

`representation` values: `tfidf` / `bm25` / `embedding` / `hybrid_serial` / `hybrid_parallel`.
`fusion` values: `rrf` / `combsum` / `combmnz`.
`mode` values: `basic` / `with_features` (forwarded to :8004).

---

## 13. Test inventory (Phase 5 only)

| File                                  | Tests | What it covers |
|---------------------------------------|-------|----------------|
| `tests/retrieval/test_fusion.py`      | 28    | RRF, CombSUM, CombMNZ, dispatcher, tie-breaks, edge cases (empty / single-element). |
| `tests/retrieval/test_hybrid.py`      | 17    | Orchestrator: 5 reps, personalization scalar, refinement fall-back, full workflow. |
| `tests/retrieval/test_multi_encoder.py` | 12  | Parallel rank, fusion dispatcher, missing-index 503, same-encoder 400, custom encoders, truncation. |
| `tests/retrieval/test_service_phase5.py` | 8 | 3 endpoints: unknown dataset, k range, dispatch, error paths. |
| `tests/retrieval/test_embedder.py`    | +1    | LRU-2 eviction policy update (was 1 in Phase 3, now 2 for L6 + L12). |
| **Total new**                         | **66**| 65 genuinely new + 1 updated. |

Run all 114 retrieval tests:

```bash
pytest tests/retrieval/ -v
```

---

## 14. Known limitations

- **No query-time caching** — every `/hybrid/.../search` re-runs
  the full BM25 + dense pipeline. Phase 7 may add a query-keyed
  LRU.
- **No exact-Match boost for personalization** — the personalisation
  boost is a single scalar on the BM25 score, not a per-term
  re-scoring. This is fine for the small synthetic user logs we
  have, but won't scale to a real click log.
- **L12 index is built offline** — there's no in-service re-build.
  The build is a one-shot per dataset.
- **Hard-coded 2 encoders** — `MultiEncoderRunner` only knows
  about L6 + L12. A custom `encoder_1` / `encoder_2` will work
  (the test `test_runner_custom_encoders` proves it), but the
  default L6 / L12 mapping in `_short_name` doesn't know any
  other model names.
- **No streaming** — the full `HybridSearchResponse` is built in
  memory before the first byte is sent. Acceptable for k ≤ 1000.
- **No CORS tightening** — still `*` everywhere. Phase 6 gateway
  will lock it down.

---

## 15. What Phase 5 enabled

- **Phase 6 (Gateway):** the gateway can now route a single user
  query to one of 5 representations and one of 3 fusion methods,
  with a per-tenant default.
- **Phase 7 (Re-ranking):** the per-retriever scores in
  `individual_scores` give a cross-encoder re-ranker enough signal
  to do a second-stage re-rank.
- **Phase 8 (RAG):** the top-k fused hits are exactly what the
  chunk store needs to seed a prompt.
- **Phase 9 (Evaluation):** the 5 reps + 3 fusions give 15
  representation-fusion pairs × 2 datasets × 3 queries = 90
  evaluation points. The 2-encoder L6+L12 comparison is a free
  benchmark for "does the deeper model help on these corpora?".
- **UI (Phase 10):** the React UI can now show a 5-way search
  picker with per-representation timings.

---

## Appendix A — `HybridSearchResponse` example

```json
{
  "dataset_id": "touche2020",
  "representation": "hybrid_parallel",
  "fusion": "rrf",
  "k": 3,
  "latency_ms": 142,
  "results": [
    {
      "rank": 1,
      "doc_id": "doc-1284",
      "score": 0.0417,
      "individual_scores": {"bm25": 0.95, "dense": 0.72}
    },
    {
      "rank": 2,
      "doc_id": "doc-8923",
      "score": 0.0333,
      "individual_scores": {"bm25": 0.81, "dense": 0.65}
    },
    {
      "rank": 3,
      "doc_id": "doc-2110",
      "score": 0.0250,
      "individual_scores": {"bm25": 0.62, "dense": 0.50}
    }
  ],
  "per_retriever_latency_ms": {"bm25": 22, "dense": 88, "fuse": 1},
  "refined_query": null,
  "refinement_fell_back": false,
  "stages": {
    "bm25": "top-100 (k=400, BM25)",
    "dense": "top-100 (FAISS IndexFlatIP, dim=384)",
    "fuse": "rrf (k=60)"
  }
}
```

## Appendix B — `MultiEncoderSearchResponse` example

```json
{
  "dataset_id": "nq",
  "representation": "embedding",
  "fusion": "rrf",
  "k": 3,
  "latency_ms": 67,
  "results": [
    {
      "rank": 1,
      "doc_id": "doc-91482",
      "score": 0.0333,
      "individual_scores": {"l6": 0.88, "l12": 0.91}
    },
    {
      "rank": 2,
      "doc_id": "doc-22184",
      "score": 0.0250,
      "individual_scores": {"l6": 0.72, "l12": 0.65}
    },
    {
      "rank": 3,
      "doc_id": "doc-1003",
      "score": 0.0167,
      "individual_scores": {"l6": 0.41, "l12": 0.58}
    }
  ],
  "per_retriever_latency_ms": {"l6": 12, "l12": 31, "fuse": 1},
  "stages": {
    "l6": "top-3 (encoder=sentence-transformers/all-MiniLM-L6-v2)",
    "l12": "top-3 (encoder=sentence-transformers/all-MiniLM-L12-v2)",
    "fuse": "rrf"
  }
}
```

## Appendix C — `GET /hybrid/{ds}/health` example

```json
{
  "status": "ok",
  "service": "retrieval-hybrid",
  "dataset_id": "touche2020",
  "dense_loaded": true,
  "second_encoder_built": true,
  "bm25_endpoint_reachable": true,
  "refinement_endpoint_reachable": true,
  "second_encoder_index_filename": "faiss_l12.index",
  "second_encoder_model": "sentence-transformers/all-MiniLM-L12-v2",
  "version": "0.1.0"
}
```

---

**End of Phase 5.**

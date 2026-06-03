# Phase 2 — Classical Indexing + Service

**Status:** Complete (committed `022bd85`)
**Service port:** `8002`
**Build time:** 8 min total (5 min touche2020 + 3 min nq)
**On-disk size:** 1.08 GB across both datasets
**Tests:** 78 passing (61 new in this phase)

## 1. Goal

Build three classical inverted-style retrievers and expose them over a
FastAPI service so the upcoming gateway (Phase 6) and React UI (Phase 7)
can issue searches without touching disk.

The three retrievers are independent primitives — a single `InvertedIndex`,
a `TfidfRetriever` over the same corpus, and a `BM25Retriever` wrapping
`bm25s` — all built from the Phase-1 preprocessed `tokens.jsonl`.

## 2. Datasets

| ID | Name | Docs | Tokens | Avg len | Qrels |
|----|------|------|--------|---------|-------|
| `touche2020` | BEIR Webis-Touche 2020 v2 | 382,544 | 57,069,964 | 149.19 | 2,962 |
| `nq` | BEIR Natural Questions (capped 500K) | 500,000 | 24,540,420 | 49.08 | 4,201 |

Decision rationale + deviation from the guide: see `docs/dataset_choice.md`.

## 3. Architecture

```
shared/ir_common/
  preprocess.py        ← single source of truth (Phase 1)
  schemas.py           ← Pydantic models for the HTTP contract

services/indexing/app/
  config.py            ← paths, dataset registry, defaults
  corpus.py            ← streaming reader over tokens.jsonl
  inverted_index.py    ← InvertedIndex (post-cap, joblib)
  tfidf.py             ← TfidfRetriever (sklearn)
  bm25.py              ← BM25Retriever (bm25s eager)
  service.py           ← FastAPI on :8002

scripts/
  build_indexes.py     ← CLI: build all three for a dataset
  smoke_search.py      ← Hand-test eyeball verification
```

The three retrievers are **independent**: each has its own on-disk
artefacts and its own `build` / `save` / `load` / `search` API. The
service can load any combination on demand.

## 4. RAM strategy

The naive `dict[str, list[tuple[int, int]]]` inverted index over
720K unique terms × 382K docs is **8-10 GB** in RAM. We have 15.8 GB
total with ~3 GB free; the naive build OOMs. We therefore cap the
vocabulary in two passes during `InvertedIndex.build`:

1. **First pass** — compute document frequency per term, drop
   singletons (`min_df=2`) and stop-words / boilerplate
   (`max_df_ratio=0.5`).
2. **Second pass** — emit only post-cap postings.

After the cap, the inverted index is **4-5 GB RAM** and **692 MB
on disk** for the 382K-doc corpus. Both flags are CLI-tunable on
`build_indexes.py`.

For the live service, the index is loaded on demand and held in an
LRU-1 cache (`_INVIDX_CACHE`, `_TFIDF_CACHE`, `_BM25_CACHE` in
`service.py`). Only one dataset is "hot" at a time; switching
datasets evicts the old one and the OS reclaims the pages.

## 5. The three retrievers

### 5.1 InvertedIndex (`services/indexing/app/inverted_index.py`)

Dictionary-of-dictionaries: `self.postings: dict[str, dict[int, int]]`
mapping term → `{doc_id: term_frequency}`. Also stores `self.doc_length:
dict[int, int]` and `self.doc_count: int`. `__slots__` keeps the
per-instance dict small.

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `build(corpus, doc_ids)` | O(total_tokens) | Two passes (df, then emit) |
| `save(path)` / `load(path)` | O(disk) | joblib, compress=3, ~5x ratio |
| `get_postings(term, cap)` | O(1) amortised | Returns sorted top-cap by tf |
| `tf(term, doc_id)` | O(1) | |
| `length(doc_id)` | O(1) | |
| `vocab` | O(1) | len() = distinct term count |
| `stats()` | O(1) | Reads from in-memory dicts |

The "inverted" model in `/search` sums `tf(term, doc_id)` across the
query tokens and sorts descending — a "did any term match?" ranking,
not a true relevance score. Useful as a sanity check; not the default
in production.

### 5.2 TfidfRetriever (`services/indexing/app/tfidf.py`)

Thin wrapper over `sklearn.feature_extraction.text.TfidfVectorizer` with
**identity preprocessor and tokenizer** (we feed pre-tokenised docs as
space-joined strings; the vectorizer tokenises on whitespace, no
lowercasing, no stop-word removal, no accent stripping). The query path
is identical: the gateway pre-tokenises with `preprocess()` and
joins with spaces.

| Param | Value | Why |
|-------|-------|-----|
| `preprocessor` | `_identity` | Don't re-tokenize |
| `tokenizer` | `_identity` | Don't re-tokenize |
| `lowercase` | `False` | Phase 1 already lower-cased |
| `token_pattern` | `None` | Identity tokeniser needs no regex |
| `sublinear_tf` | `True` | Apply `log(1 + tf)` |
| `norm` | `"l2"` | Cosine similarity = dot product |

Artefacts: `tfidf_vectorizer.pkl`, `tfidf_matrix.npz` (CSR sparse,
~350 MB for 382K docs × 720K vocab), `doc_ids.json`.

Search: vectorise the query, compute `cosine_similarity(q_vec, matrix)`.
argpartition the row of scores for O(N) top-k rather than O(N log N)
argsort. Latency: 800-1700 ms on the 382K-doc corpus (dominated by
the dense cos-sim row); 5-15 ms on 500K-doc nq because the BM25 takes
precedence as the default in the service.

### 5.3 BM25Retriever (`services/indexing/app/bm25.py`)

Wraps `bm25s.BM25(method="lucene", k1=1.5, b=0.75)`. Two important
properties of `bm25s` that drove the design:

1. **Eager BM25** — at `index()` time, `bm25s` precomputes the
   per-(doc, term) BM25 score into a sparse CSC matrix stored on
   `bm.scores`. `get_scores()` then sums those precomputed values for
   the query tokens. This is why bm25s is ~50× faster than
   `rank_bm25` on a 500K-doc corpus.
2. **k1/b is baked in at index time.** Mutating `bm.k1 = 0.5; bm.b = 0.3`
   after the fact has no effect on scores — the precomputed matrix
   used the old parameters. To change k1/b we must build a fresh
   `bm25s.BM25` and re-run `index()`. That pass is O(corpus); on
   500K docs it costs ~30 seconds.

This is why we keep an LRU-8 cache of `(k1, b, method) →
bm25s.BM25`: the first call with a new (k1, b) takes ~30 s; subsequent
calls with the same (k1, b) are O(1) cache hits. The cache key
includes `method` because the same bm25s object cannot serve
`lucene` and `atire` scoring formulas.

Pre-tokenisation: we feed `bm25s.tokenization.Tokenized(ids=…,
vocab=…)` constructed from **our** `preprocess()` output, so the
single source of truth guarantee is preserved (see
`tests/preprocessing/test_preprocess.py::test_preprocess_is_the_canonical_function`
from Phase 1).

Artefacts: `bm25.pkl` (the default `bm25s.BM25` object, ~150 MB for
touche2020), `bm25_token_ids.pkl` (the corpus in token-ID form for
re-tuning), `bm25_vocab.json`, `doc_ids.json`.

## 6. HTTP contract (`shared/ir_common/schemas.py`)

Pydantic v2 models. The endpoints are:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + currently-loaded dataset |
| `GET` | `/index/{ds}/exists` | Quick existence check on disk |
| `GET` | `/index/{ds}/stats` | Vocab / doc count / build time / size |
| `POST` | `/index/{ds}/build` | Async via `BackgroundTasks` |
| `POST` | `/index/{ds}/load` | Warm the in-process LRU caches |
| `POST` | `/index/{ds}/search` | Ranked retrieval |
| `GET` | `/index/{ds}/postings/{term}` | Inspect a term's postings |

`SearchRequest` carries `query_tokens: list[str]` — **already
pre-tokenised** by the caller (typically the gateway in Phase 6).
The service does not re-tokenise at query time. The same canonical
`preprocess()` from Phase 1 is the input contract everywhere.

`extra="forbid"` on most models; `extra="ignore"` on `SearchRequest`
because Phase 5 callers may send `fusion` or other fields the service
doesn't understand.

`dataset_id` is a path parameter, validated against `DATASET_IDS =
("touche2020", "nq")`. Unknown IDs → 400.

CORS is permissive (`*`) for now; tightened in Phase 6.

## 7. Service startup flow

```python
# uvicorn services.indexing.app.service:app --host 127.0.0.1 --port 8002
```

Startup is instant — no I/O at import time. The first `/search` or
`/load` triggers a joblib.load of the relevant artefact and warms
the LRU cache. Subsequent searches with the same `model` parameter
are O(1) cache hits. The response includes `"cached": true|false`
for diagnostics.

## 8. Build pipeline (`scripts/build_indexes.py`)

```
python scripts/build_indexes.py --datasets touche2020 --min-df 2 --max-df-ratio 0.5
```

CLI flags:

- `--datasets {touche2020,nq}` (default: both)
- `--min-df` (default 2)
- `--max-df-ratio` (default 0.5)
- `--bm25-method {lucene,atire,robertson,bm25l,bm25plus}` (default `lucene`)
- `--no-progress` (silence tqdm)

Per dataset the script:
1. Streams `tokens.jsonl` to compute the vocabulary and document
   frequencies (first pass).
2. Applies the cap (`min_df`, `max_df_ratio`).
3. Re-streams `tokens.jsonl` to build the post-cap inverted index
   (second pass).
4. Builds the TF-IDF matrix from the post-cap vocabulary.
5. Builds the BM25 index from the post-cap vocabulary + token-IDs.
6. Persists `build_meta.json` with the build time, vocabulary sizes,
   nnz count, cap parameters, and on-disk size.

**Wall times (this machine, 12 cores, 16 GB RAM):**

| Dataset | Inverted | TF-IDF | BM25 | Total | On-disk |
|---------|----------|--------|------|-------|---------|
| `touche2020` | 16.9 s | 9.6 s | 24.8 s | 299.4 s (5 min) | 692 MB |
| `nq` | 5.5 s | 4.9 s | 14.5 s | 166.5 s (3 min) | 390 MB |

(The "Total" is the wall-clock including the two stream-passes and
joblib.dump; the per-step breakdown is the in-memory work.)

## 9. Smoke results (`scripts/smoke_search.py`)

Hand-picked queries against both datasets and both retrievers. Each
result includes a snippet from `docs.jsonl` for eyeball verification.

**Sample BM25 results on `touche2020`:**

| Query | Top-1 doc snippet |
|-------|-------------------|
| "Should abortion be legalized?" | "Making abortion legal…" (rank 1) |
| "Is climate change caused by humans?" | "Humans do not cause CLIMATE CHANGE…" (rank 1) |
| "Should the death penalty be abolished?" | "The death penalty should be abolished…" (rank 1) |

**Sample BM25 results on `nq`:**

| Query | Top-1 doc snippet |
|-------|-------------------|
| "when was the declaration of independence signed" | "Fifty-six delegates eventually signed the Declaration of Independence…" |
| "what is the largest planet in the solar system" | "…Jupiter is the largest of the four giant planets in the Solar System…" |
| "how many continents are there in the world" | "While continent was used on the one hand for relatively small areas…" |

All top-3 results for every query are domain-relevant. Latencies:

| Model | touche2020 (382K docs) | nq (500K docs) |
|-------|------------------------|----------------|
| BM25 (warm) | 5-15 ms | 5-10 ms |
| TF-IDF (warm) | 1.6-1.7 s | 0.8 s |

## 10. Tests (`tests/indexing/`)

78 tests total; 61 new in this phase.

| File | Tests | Focus |
|------|-------|-------|
| `test_inverted_index.py` | 15 | build/save/load round-trip, postings, tf, length, cap behaviour, doc_count, stats |
| `test_tfidf.py` | 13 | build/save/load, search with empty query, top-k correctness, k > corpus, OOV tokens, sublinear_tf effect |
| `test_bm25.py` | 18 | build with custom k1/b, method variants, LRU cache hit/miss, LRU eviction, save/load, k1=0 invariant, search with empty query |
| `test_service.py` | 15 | health, exists, stats, search across all three models, /postings, unknown dataset 400, bad query 422 |

All tests use a tiny 5-document in-memory corpus via the
`fake_index_dir` fixture in `conftest.py` (no disk I/O). The
service tests use FastAPI's `TestClient` so they exercise the real
HTTP surface without spawning uvicorn.

Run with `make test`. Lint with `make lint` (ruff + black).
Type-check with `make type` (mypy, 22 source files).

## 11. Cold-start / hot-path latency

Measured via `curl` against `uvicorn` on `127.0.0.1:8002`:

| Endpoint | Cold (no in-process cache) | Hot (warm cache) |
|----------|----------------------------|-------------------|
| `GET /health` | 1 ms | 1 ms |
| `GET /index/touche2020/exists` | 2 ms | 2 ms |
| `GET /index/touche2020/stats` | **73 ms** *(reads `build_meta.json` only — no joblib.load)* | 73 ms |
| `POST /index/touche2020/search` (BM25 default) | 5-35 s *(joblib.load 150 MB bm25.pkl)* | **4-6 ms** |
| `POST /index/touche2020/search` (TF-IDF) | 5-10 s *(joblib.load + 8 MB pkl + 350 MB npz)* | 800-1700 ms |
| `POST /index/touche2020/search` (inverted) | 25-30 s *(joblib.load 100 MB pkl)* | ~2 s |
| `GET /index/touche2020/postings/abort` | similar to inverted search | <100 ms |

The cold-load times are one-time per worker per dataset; with the
LRU-1 cache they happen at most once per dataset per process. A
production deployment with two uvicorn workers would have two
cold-loads.

The `POST /search` with `model="bm25"` and a **non-default** `(k1, b)`
incurs a **~30 s rebuild** on the 500K-doc corpus; subsequent calls
with the same `(k1, b)` are 4-6 ms cache hits. This is the LRU-8
cache in action.

## 12. Verification

End-to-end smoke test on `uvicorn`:

```
$ curl -s http://127.0.0.1:8002/health
{"status":"ok","service":"indexing","loaded_dataset":null,"version":"0.1.0"}

$ curl -s http://127.0.0.1:8002/index/touche2020/stats
{"dataset_id":"touche2020","exists":true,"loaded":false,
 "vocab_size":235185,"total_docs":382544,"avg_doc_length":149.19,
 "build_seconds":299.42,"build_at":"2026-06-03T22:39:25",
 "size_mb":707.9,"cap":{"min_df":2,"max_df_ratio":0.5}}

$ curl -s -X POST -H 'Content-Type: application/json' \
       -d '{"query_tokens":["abort","legal"],"k":3,"model":"bm25"}' \
       http://127.0.0.1:8002/index/touche2020/search
{"dataset_id":"touche2020","model":"bm25","k":3,"latency_ms":6,
 "results":[{"rank":1,"doc_id":"1eb8b86d-2019-04-18T16:50:10Z-00002-000",
             "score":5.5258},
            ...],
 "k1":1.5,"b":0.75,"cached":true}
```

## 13. Deviations from the guide

| Guide | This project | Why |
|-------|--------------|-----|
| `rank_bm25>=0.2` | `bm25s>=0.3` | bm25s is ~50× faster on 500K-doc corpora (eager BM25 with NumPy-vectorised scoring); pure-Python wheel works on Windows + cp312 out of the box. Same API surface via the `BM25Retriever` wrapper. `rank_bm25` kept in `requirements.txt` commented as a reference. |
| `inverted_index.cap` defaults unspecified | `min_df=2, max_df_ratio=0.5` | Uncapped dict-of-dicts OOMs on 16 GB RAM. Both flags are CLI-tunable. |
| No service layer for the indexes | FastAPI on `:8002` | Gateway (Phase 6) is a separate process; indexes must be addressable over HTTP. |
| Search takes raw query strings | Search takes pre-tokenised `query_tokens` | The single-source-of-truth contract from Phase 1: `preprocess()` is the canonical tokeniser; the service does not re-tokenise. The gateway in Phase 6 will orchestrate the two-step pipeline. |

## 14. Next steps (Phase 3 onward)

- **Phase 3 — Vector store + embeddings.** Reuse the same
  pre-tokenised `tokens.jsonl` to encode each doc with
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim), build a FAISS
  `IndexFlatIP`, expose it on `:8003`. Mirror the Phase 2 service
  contract.
- **Phase 6 — Gateway.** A `services/gateway` on `:8000` that
  receives a raw query string, calls `preprocessing:8001` for
  `preprocess()`, then fans out to one or more retrieval services
  and fuses the result lists. The `/search` payload here already
  takes `query_tokens`, so the gateway is straightforward.
- **Phase 7 — UI.** The React UI calls the gateway; the gateway
  calls this service. End-to-end search bar in the browser.
- **Phase 9 — Evaluation.** Compute MAP@10, nDCG@10, MRR for both
  retrievers on both datasets, using the qrels from BEIR.

## 15. Files of note

- `services/indexing/app/service.py` — the FastAPI surface
- `services/indexing/app/bm25.py` — the BM25Retriever (LRU-8 cache)
- `services/indexing/app/inverted_index.py` — the InvertedIndex (cap logic)
- `shared/ir_common/schemas.py` — the HTTP contract
- `scripts/build_indexes.py` — the build CLI
- `data/indexes/{touche2020,nq}/build_meta.json` — the build artefact
  summary; read by `/stats` for sub-second responses without
  joblib.load.

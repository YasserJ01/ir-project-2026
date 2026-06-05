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
- 65 new tests + 1 updated (`test_load_lru_eviction` now covers LRU-2). **277 project-wide** (212 Phase 4 + 65 new), all passing. Lint clean (ruff + black + mypy) on all 30+ source files.
- Build is staged: **Stage 1** = framework + tests + docs (committed before L12 build starts); **Stage 2** = `git commit --allow-empty "Phase 5: 2nd FAISS index built; multi-encoder live"` after `build_meta_l12.json` flips to `ok` for both datasets.
- Expected L12 build time on GTX 1650 Max-Q: ~95 min touche2020 + ~125 min nq = ~3.7 hr total (FP16, batch=256).
- Deviations from the guide (documented in PHASE_5.md §14): no query-time caching of fused results, personalization = single scalar (not per-term re-scoring), hard-coded 2 encoders (override-able), no streaming.
- Full details: [PHASE_5.md](PHASE_5.md).

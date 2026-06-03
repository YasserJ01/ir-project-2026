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

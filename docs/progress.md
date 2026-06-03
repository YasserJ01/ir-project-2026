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

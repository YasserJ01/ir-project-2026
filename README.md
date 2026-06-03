# Information Retrieval System — Project 2026

> A production-grade, service-oriented Information Retrieval (IR) search engine.
> Two corpora (≥ 200K docs each), four representations (TF-IDF, BM25, Embeddings, Hybrid),
> query refinement, RAG, and a Vector Store — all behind a clean React UI.

## Status

| Phase | Status | Doc |
|-------|--------|-----|
| 0 — Foundation, Setup & Planning | ✅ done | [docs/PHASE_0.md](docs/PHASE_0.md) |
| 1 — Data Acquisition & Preprocessing | ✅ done | [docs/PHASE_1.md](docs/PHASE_1.md) |
| 2 — Indexing | ⏳ upcoming | — |
| 3 — Dense Representations + FAISS | ⏳ upcoming | — |
| 4 — Query Processing & Refinement | ⏳ upcoming | — |
| 5 — Query Matching, Ranking & Hybrid | ⏳ upcoming | — |
| 6 — Service-Oriented Architecture (SOA) | ⏳ upcoming | — |
| 7 — User Interface (React + Vite + TS) | ⏳ upcoming | — |
| 8 — Additional Features (Vector Store + RAG) | ⏳ upcoming | — |
| 9 — System Evaluation | ⏳ upcoming | — |
| 10 — Hardening, Documentation & Submission | ⏳ upcoming | — |

## Stack

- **Backend:** Python 3.12, FastAPI, NLTK, sentence-transformers, FAISS, rank_bm25, scikit-learn, pytrec_eval.
- **Frontend:** React 18, Vite 5, TypeScript 5, Tailwind CSS 3, TanStack Query, Zustand.
- **Orchestration:** Docker Compose (later phases).

## Quick Start (Windows / PowerShell)

```powershell
# 1. Create venv + install deps
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab

# 2. Frontend deps
cd services\ui
npm install
npm run build      # verify it builds
cd ..\..

# 3. Verify
python -c "import fastapi, faiss, sentence_transformers, ir_datasets; print('PY OK')"

# 4. Ingest the two datasets (Phase 1 — see docs/PHASE_1.md for why these picks)
make ingest-a      # beir/webis-touche2020  -> 382,544 docs
make ingest-b      # beir/nq                 -> 500,000 docs (capped)
make tokenize      # docs.jsonl -> tokens.jsonl (~10 min with 8 workers)

# 5. (Optional) Run the preprocessing service
make dev-preproc   # -> http://127.0.0.1:8001
#   POST /preprocess {"text": "Hello, World!"}  -> {"tokens": ["hello", "world"]}

# 6. (Optional) Run the UI in production mode via Docker
docker compose up -d --build
# → http://localhost:3000
# See docs/DOCKER.md for dev vs prod conventions.
```

## Architecture (high level)

```
React UI (:5173)  ─▶  FastAPI Gateway (:8000)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Preprocessing   Indexing     Retrieval
          (:8001)       (:8002)       (:8003)
                              │             │
                              └──────┬──────┘
                                     ▼
                              Refinement (:8004)
                                     │
                                     ▼
                                 RAG (:8005)
```

See [docs/architecture.md](docs/architecture.md) for the full diagram (filled in Phase 6).

## Documentation

- [SOLO_DEVELOPER_GUIDE.md](SOLO_DEVELOPER_GUIDE.md) — the master plan (phases 0 → 10).
- [TEAM_DOCUMENTATION.md](TEAM_DOCUMENTATION.md) — the team variant (for reference).
- [docs/PHASE_0.md](docs/PHASE_0.md) — what was done in Phase 0.
- [docs/PHASE_1.md](docs/PHASE_1.md) — what was done in Phase 1 (data + preprocessing).
- [docs/architecture.md](docs/architecture.md) — system architecture.
- [docs/dataset_choice.md](docs/dataset_choice.md) — chosen datasets (filled in Phase 1).
- [docs/progress.md](docs/progress.md) — running progress log.

## Repository Layout

```
ir-project-2026/
├── README.md
├── docker-compose.yml         # (Phase 6)
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .env.example
├── .gitignore
├── data/                      # gitignored — populated at runtime
├── docs/
├── evaluation/
├── reports/
├── scripts/
├── services/
│   ├── gateway/               # FastAPI Gateway
│   ├── preprocessing/
│   ├── indexing/
│   ├── retrieval/
│   ├── refinement/
│   ├── rag/
│   └── ui/                    # React + Vite + TS
└── shared/
    └── ir_common/             # shared config, schemas, http client
```

## License

MIT

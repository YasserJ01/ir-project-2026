# Information Retrieval System — Project 2026

> A production-grade, service-oriented Information Retrieval (IR) search engine.
> Two corpora (≥ 200K docs each), five search representations (TF-IDF, BM25, Embedding, Hybrid-Serial, Hybrid-Parallel) plus a 2-encoder fusion (L6 + L12),
> query refinement, RAG, and a Vector Store — all behind a clean React UI.

## Status

| Phase | Status | Doc |
|-------|--------|-----|
| 0 — Foundation, Setup & Planning | ✅ done | [docs/PHASE_0.md](docs/PHASE_0.md) |
| 1 — Data Acquisition & Preprocessing | ✅ done | [docs/PHASE_1.md](docs/PHASE_1.md) |
| 2 — Indexing | ✅ done | [docs/PHASE_2.md](docs/PHASE_2.md) |
| 3 — Dense Representations + FAISS | ✅ done | [docs/PHASE_3.md](docs/PHASE_3.md), [docs/PHASE_3_RESUME.md](docs/PHASE_3_RESUME.md) |
| 4 — Query Processing & Refinement | ✅ done | [docs/PHASE_4.md](docs/PHASE_4.md) |
| 5 — Query Matching, Ranking & Hybrid | ✅ done | [docs/PHASE_5.md](docs/PHASE_5.md) |
| 6 — Service-Oriented Architecture (SOA) | ⏳ upcoming | — |
| 7 — User Interface (React + Vite + TS) | ⏳ upcoming | — |
| 8 — Additional Features (Vector Store + RAG) | ⏳ upcoming | — |
| 9 — System Evaluation | ⏳ upcoming | — |
| 10 — Hardening, Documentation & Submission | ⏳ upcoming | — |

## Stack

- **Backend:** Python 3.12, FastAPI, NLTK, sentence-transformers, FAISS, bm25s, scikit-learn, ir_measures.
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

# 5. Build the classical indexes (Phase 2 — ~8 min, 1.1 GB on disk)
make build-indexes   # inverted + TF-IDF + BM25 for both datasets
make smoke-search   # hand-test the search results

# 6. (Phase 3) Install the GPU torch wheel (only if you have a CUDA-capable GPU)
#    On 4 Mbps links the 2.4 GB wheel takes ~50 min; populate data/downloads/
#    first with: python scripts/download_torch_gpu.py
make install-torch-gpu   # uninstalls CPU torch, installs torch==2.5.1+cu121

# 7. Build the dense FAISS indexes (Phase 3 — ~12 min on GPU, ~5+ hr on CPU)
make build-dense         # sentence-transformers MiniLM-L6-v2 + IndexFlatIP
make smoke-dense         # hand-test the dense search results

# 8. (Optional) Run the preprocessing service
make dev-preproc   # -> http://127.0.0.1:8001
#   POST /preprocess {"text": "Hello, World!"}  -> {"tokens": ["hello", "world"]}

# 9. (Optional) Run the indexing service (lexical)
make dev-indexing   # -> http://127.0.0.1:8002
#   GET  /health
#   GET  /index/{touche2020|nq}/stats
#   POST /index/{touche2020|nq}/search
#        body: {"query_tokens": ["abort", "legal"], "k": 10, "model": "bm25"}

# 10. (Optional) Run the dense-retrieval service
make dev-retrieval   # -> http://127.0.0.1:8003
#   GET  /retrieval/health
#   GET  /retrieval/{touche2020|nq}/stats
#   POST /retrieval/{touche2020|nq}/search
#        body: {"query": "When was X founded?", "k": 10}

# 11. (Optional) Run the query-refinement service (Phase 4)
make download-symspell-dict   # one-time: 1.3 MB SymSpell dict
make seed-user-logs           # 53 synthetic past queries for "user_1"
make dev-refinement           # -> http://127.0.0.1:8004
#   GET  /health
#   POST /refine
#        body: {"query": "recieve teh helo", "user_id": "user_1"}
#        body: {"query": "fast car", "enable_synonyms": true, "synonym_count": 2}
make smoke-refine             # hand-test the 6 default queries

# 12. (Optional) Run the Phase 5 hybrid / multi-encoder path
make download-second-model    # pre-cache all-MiniLM-L12-v2 (~4 min on 4 Mbps)
make launch-dense-2           # detached L12 FAISS build (~3.7 hr on GPU)
make check-dense-2 --watch 60 # poll build_meta_l12.json until both 'ok'
make dev-retrieval            # -> http://127.0.0.1:8003
#   POST /hybrid/{ds}/search
#        body: {"query": "...", "k": 10, "representation": "hybrid_parallel",
#               "fusion": "rrf", "candidate_k": 100, "mode": "with_features"}
#   POST /multi-encoder/{ds}/search
#        body: {"query": "...", "k": 10, "fusion": "rrf"}
#   GET  /hybrid/{ds}/health
make smoke-hybrid             # hand-test all 5 reps + multi-encoder

# 13. (Optional) Run the UI in production mode via Docker
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
- [docs/PHASE_2.md](docs/PHASE_2.md) — what was done in Phase 2 (inverted + TF-IDF + BM25 + service on :8002).
- [docs/PHASE_3.md](docs/PHASE_3.md) — what was done in Phase 3 (dense embeddings + FAISS + service on :8003).
- [docs/PHASE_4.md](docs/PHASE_4.md) — what was done in Phase 4 (query refinement: spell + synonyms + grammar + personalization + service on :8004).
- [docs/PHASE_5.md](docs/PHASE_5.md) — what was done in Phase 5 (5-representation hybrid search + L6+L12 multi-encoder fusion, all on :8003).
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

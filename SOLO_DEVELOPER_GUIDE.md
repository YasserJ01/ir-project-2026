# Information Retrieval System — Project 2026
## Solo Developer Phase-Based Implementation Guide (Windows + React)

> **Audience:** A single developer who will carry this project end-to-end.
> **OS:** **Windows 10/11** (PowerShell 5.1+).
> **Course:** Information Retrieval Systems 2026 — Practical Project
> **Deadline:** July 3rd, 2026
> **Language Constraint:** Python only (backend) + **React** (frontend).
> **Additional Features Committed:** 2 (RAG + Vector Stores)
> **Estimated effort:** ~6 weeks of focused work (~25–30 hours/week)

This document is your **single source of truth**. Work strictly phase by phase. Do **not** start a phase until the previous one's exit criteria are met.

> All shell snippets in this guide use **PowerShell** syntax. If you are on macOS/Linux, mentally translate (`\.venv\Scripts\activate` → `source .venv/bin/activate`, `;` instead of `&&`, etc.).

---

## Table of Contents

- [Phase 0 — Foundation, Setup & Planning](#phase-0--foundation-setup--planning)
- [Phase 1 — Data Acquisition & Preprocessing](#phase-1--data-acquisition--preprocessing)
- [Phase 2 — Indexing (Inverted Index, TF-IDF, BM25)](#phase-2--indexing-inverted-index-tf-idf-bm25)
- [Phase 3 — Dense Representations (Embeddings + FAISS)](#phase-3--dense-representations-embeddings--faiss)
- [Phase 4 — Query Processing & Refinement](#phase-4--query-processing--refinement)
- [Phase 5 — Query Matching, Ranking & Hybrid Retrieval](#phase-5--query-matching-ranking--hybrid-retrieval)
- [Phase 6 — Service-Oriented Architecture (SOA)](#phase-6--service-oriented-architecture-soa)
- [Phase 7 — User Interface (React + Vite + TypeScript)](#phase-7--user-interface-web)
- [Phase 8 — Additional Features (Vector Store + RAG)](#phase-8--additional-features-vector-store--rag)
- [Phase 9 — System Evaluation](#phase-9--system-evaluation)
- [Phase 10 — Hardening, Documentation & Submission](#phase-10--hardening-documentation--submission)
- [Appendix A — Daily/Weekly Checklist](#appendix-a--dailyweekly-checklist)
- [Appendix B — Architecture Sketch](#appendix-b--architecture-sketch)

---

## Phase 0 — Foundation, Setup & Planning

**Goal:** A clean, reproducible Python + Node.js environment and a project skeleton that will support six services, two datasets, and a React UI.

### 0.1 Tooling (Windows-first)

| Tool | Version | Why | Windows installer |
|------|---------|-----|-------------------|
| **Python** | 3.11+ | Mandated by spec; runs all backend services | [python.org](https://www.python.org/downloads/windows/) (check "Add to PATH") |
| **Node.js** | **20 LTS** | Runs the React dev server + build | [nodejs.org](https://nodejs.org/) (LTS) — `npm` is bundled |
| **Git** | latest | Version control | [git-scm.com](https://git-scm.com/download/win) |
| **Docker Desktop** | latest | One-command system demo at defense | [docker.com](https://www.docker.com/products/docker-desktop/) (enable WSL2) |
| **VS Code** | latest | IDE | [code.visualstudio.com](https://code.visualstudio.com/) |
| **PowerShell 7** (optional) | latest | Better scripting than Windows PowerShell 5.1 | `winget install Microsoft.PowerShell` |

> **Tip:** On Windows, after installing Python, open a **new** PowerShell so `$env:PATH` is refreshed. Verify with `python --version` and `node --version`.

### 0.2 Steps
1. Create a GitHub repository: `ir-project-2026` (public).
2. Clone locally (e.g. into `C:\dev\ir-project-2026`).
3. Create the directory structure (use the template below).
4. Create `pyproject.toml` and `requirements.txt` for Python; create `services/ui/package.json` for React (template in §0.7).
5. Add a `.gitignore` (Python + Node + data + `.env` + React build output).
6. Add a `.env.example`.
7. Add a `README.md` skeleton.
8. Initial commit: `chore: bootstrap project structure`.

### 0.3 Initial Folder Layout
```
ir-project-2026/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile                       # convenience: make up, make eval, etc.
├── docs/
│   ├── architecture.md
│   └── diagrams/
├── data/                          # gitignored
├── services/
│   ├── gateway/
│   ├── preprocessing/
│   ├── indexing/
│   ├── retrieval/
│   ├── refinement/
│   ├── rag/
│   └── ui/                        # React + Vite + TypeScript (Phase 7)
│       ├── package.json
│       ├── vite.config.ts
│       ├── tsconfig.json
│       ├── index.html
│       ├── Dockerfile
│       ├── nginx.conf             # used by the prod Dockerfile
│       ├── public/
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── api/
│           │   └── client.ts
│           ├── components/
│           ├── hooks/
│           ├── pages/
│           ├── store/             # zustand / react-query
│           ├── types/
│           └── styles/
├── shared/
│   └── ir_common/
├── scripts/
│   ├── ingest_dataset_a.py
│   ├── ingest_dataset_b.py
│   ├── build_indexes.py
│   └── run_evaluation.py
├── evaluation/
│   ├── queries/
│   ├── results/
│   └── reports/
└── reports/
    └── final_report_ar.md
```

### 0.4 `requirements.txt` (Python backend, baseline)
```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
httpx
requests
ir_datasets
nltk
spacy
gensim
sentence-transformers
torch
faiss-cpu
rank_bm25
scikit-learn
numpy
pandas
pytrec_eval
symspellpy
language-tool-python
langchain
transformers
accelerate
pytest
pytest-asyncio
ruff
black
mypy
joblib
tqdm
```

### 0.5 First-Run Verification (PowerShell)
```powershell
# Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
python -c "import fastapi, faiss, sentence_transformers, ir_datasets, pytrec_eval; print('PY OK')"

# Frontend
node --version
npm --version
cd services\ui
npm install
npm run build
cd ..\..
```

> **PowerShell execution policy** — if `Activate.ps1` is blocked, run once (as admin):
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 0.6 Exit Criteria
- [ ] Repo created, README skeleton, all folders present.
- [ ] `python -c "..."` prints `PY OK`.
- [ ] `node --version` prints `v20.x` or newer.
- [ ] `npm run build` inside `services/ui/` produces a `dist/` folder.
- [ ] Docker is installed and `docker --version` works.

### 0.7 React `package.json` starter (created in Phase 7)
You don't need this now — it's listed here so you know it's coming. Real values get finalized in Phase 7.
```json
{
  "name": "ir-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "eslint .",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "@tanstack/react-query": "^5.51.0",
    "axios": "^1.7.0",
    "zustand": "^4.5.0",
    "clsx": "^2.1.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^9.0.0"
  }
}
```

---

## Phase 1 — Data Acquisition & Preprocessing

**Goal:** Two clean, preprocessed corpora saved to disk; same pipeline reusable at query time.

### 1.1 Pick Datasets
1. Browse `https://ir-datasets.com`. **Hard rules**: ≥ 200K docs each, has qrels + test queries, **NOT** `antique`.
2. Recommended pair: **`msmarco/passage`** + **`cord19/abstracts`**. Confirm with your lab instructor (Eng. Marwa / Eng. Salyma).
3. Record your choice in `docs/dataset_choice.md` (size, qrels count, why).

### 1.2 Ingestion Script
File: `scripts/ingest_dataset_a.py` (and `b`).

For each dataset:
```python
import ir_datasets
import json
from pathlib import Path

ds = ir_datasets.load("msmarco/passage")
out = Path("data/processed/msmarco_passage/docs.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for i, doc in enumerate(ds.docs_iter()):
        f.write(json.dumps({"id": doc.doc_id, "text": doc.text}) + "\n")
print(f"Wrote {i+1} docs.")
```

Run both. Validate counts: must be ≥ 200K each.

### 1.3 Preprocessing Module
File: `services/preprocessing/app/pipeline.py`.

Pipeline:
1. Strip HTML.
2. NFKC normalize.
3. Lowercase.
4. Tokenize (`nltk.word_tokenize`).
5. Remove stopwords (NLTK English).
6. Remove tokens `len < 2`.
7. Stem **or** lemmatize (decide one, justify in report).
   - Default: **Porter Stemmer** (simple, fast). Alternative: Snowball / spaCy lemmatizer.

Expose both **library function** (`preprocess(text) -> List[str]`) and **HTTP endpoint** (FastAPI, port 8001).

### 1.4 Persist Tokens
For each dataset, write `data/processed/<dataset_id>/tokens.jsonl` with `{"id":..., "tokens":[...]}`. This is what BM25/TF-IDF will consume.

### 1.5 Exit Criteria
- [ ] Both datasets downloaded and preprocessed.
- [ ] Token count: > 200K docs each.
- [ ] `preprocess()` is the **single source of truth** — used by ingestion, indexing, and the query refinement service.

---

## Phase 2 — Indexing (Inverted Index, TF-IDF, BM25)

**Goal:** Three retrieval primitives (Inverted Index query helper, TF-IDF, BM25) saved to disk and queryable in < 1 second for top-100.

### 2.1 Inverted Index
File: `services/indexing/app/inverted_index.py`.

Data structures:
- `inverted_index: dict[term, dict[doc_id, tf]]`
- `doc_lengths: dict[doc_id, int]`
- `doc_freq: dict[term, int]`
- `avg_doc_length: float`
- `total_docs: int`

API:
- `build(tokens_iter) -> None`
- `save(path) -> None` / `load(path) -> InvertedIndex`
- `get_postings(term) -> list[(doc_id, tf)]`

Use `joblib` for fast persistence.

### 2.2 TF-IDF Retriever
File: `services/indexing/app/tfidf.py`.
- `TfidfVectorizer` over the preprocessed corpus.
- Persist vectorizer + sparse matrix.
- At query time: vectorize query → cosine similarity top-k.

### 2.3 BM25 Retriever
File: `services/indexing/app/bm25.py`.
- `rank_bm25.BM25Okapi(corpus_tokens)`.
- `score(query_tokens, k1, b) -> list[(doc_id, score)]`.
- `k1` and `b` are **runtime parameters** (default 1.5 / 0.75).

### 2.4 FastAPI Endpoints (port 8002)
```
GET  /index/{dataset_id}/stats
POST /index/{dataset_id}/build
POST /index/{dataset_id}/search      # body: {query_tokens, model, k, k1?, b?}
```

### 2.5 Exit Criteria
- [ ] Indexes built and saved for both datasets.
- [ ] `/search` endpoint returns ranked docs in < 1s for top-10.
- [ ] Sanity test: a known query returns relevant docs (eyeball 3 results).

---

## Phase 3 — Dense Representations (Embeddings + FAISS)

**Goal:** Encode all documents with a sentence-transformer model and store them in a FAISS index.

### 3.1 Pick Embedding Model
- Default: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast on CPU).
- For better quality on MS MARCO: `sentence-transformers/msmarco-MiniLM-L6-cos-v5` (already tuned for passage retrieval).
- (Optional, for the bonus "multiple embeddings" — pick a second model like `all-mpnet-base-v2`.)

### 3.2 Encode Documents
File: `services/retrieval/app/embedder.py`.
- Batch encode at `batch_size=256`, use GPU if available.
- L2-normalize vectors → cosine similarity becomes inner product.
- Save embeddings as `.npy` (in case you want to rebuild FAISS).

### 3.3 FAISS Wrapper
File: `services/retrieval/app/vector_store.py`.
- For ≤ 1M vectors: `IndexFlatIP` (exact).
- For > 1M: `IndexIVFFlat(nlist=4096, nprobe=16)` or `IndexHNSWFlat(M=32)`.
- Persist with `faiss.write_index`.

API:
- `add(vectors, ids) -> None`
- `search(query_vec, k) -> (scores, ids)`

### 3.4 FastAPI Endpoints (port 8003)
```
POST /retrieval/embed    # body: {texts[]} -> vectors
POST /retrieval/search   # body: {query, dataset_id, k} -> scored docs
```

### 3.5 Exit Criteria
- [ ] FAISS index built for both datasets.
- [ ] Dense retrieval returns sensible results on a hand-tested query.
- [ ] Per-query latency < 300ms for top-10.

---

## Phase 4 — Query Processing & Refinement

**Goal:** A user query gets cleaned, expanded, personalized, and converted to tokens before reaching retrieval.

### 4.1 Query Processing
- Reuse `preprocessing` pipeline (do not duplicate logic).
- Endpoint on `preprocessing`: `POST /preprocess`.

### 4.2 Refinement Service (port 8004)
File: `services/refinement/app/`.

Four sub-modules:
1. **Spell correction** (`symspellpy`):
   - Download `frequency_dictionary_en_82_765.txt` once.
   - `correct(query) -> str`.
2. **Synonym expansion** (NLTK WordNet):
   - For each non-stopword token, add 1–2 synonyms.
   - `expand(query) -> str` (space-joined).
3. **Grammar correction** (`language-tool-python`):
   - `correct_grammar(query) -> str`.
   - On first run it downloads a `.jar` — keep it offline afterwards.
4. **Personalization** (`personalization.py`):
   - Maintain `data/user_logs/<user_id>.jsonl` of past queries and clicked doc_ids.
   - For each token, if the user has clicked 3+ docs containing a related term in the past, **boost** that term's weight (simple +1 multiplier).
   - Start by simulating "user 1" with 50 hand-crafted past queries (we'll iterate).

Endpoint: `POST /refine` → `{query, user_id, enable_spell, enable_synonyms, enable_grammar, enable_personalization}` → `{expanded_query, tokens, weights}`.

### 4.3 Pipeline Order
1. Grammar correction → 2. Spell correction → 3. Synonym expansion → 4. Personalization weight map → 5. Tokenize via shared preprocessing.

### 4.4 Exit Criteria
- [ ] `/refine` returns enriched queries.
- [ ] Unit tests for each module.
- [ ] Personalization toggle works (off → no change, on → weight map populated).

---

## Phase 5 — Query Matching, Ranking & Hybrid Retrieval

**Goal:** A unified `/search` endpoint that can serve any of the 4 representations, with hybrid serial and parallel modes.

### 5.1 Single-Model Retrieval
- TF-IDF: cosine similarity (scikit-learn).
- BM25: built-in scoring.
- Embedding: cosine over FAISS vectors.

### 5.2 Hybrid — Serial
A pipeline: pass the candidate set from retriever A through retriever B for re-ranking.
- Default: **BM25 → Embedding re-rank** (BM25 narrows to top-1000, embedding re-ranks to top-10).
- Justify in the report (good balance of speed + quality).

### 5.3 Hybrid — Parallel
Run two or more retrievers independently, then fuse:
- **RRF** (Reciprocal Rank Fusion): `score(d) = Σ 1/(k + rank_i(d))`, default k=60.
- **CombSUM**: sum of min-max normalized scores.
- **CombMNZ**: sum × count of non-zero contributions.
- **Multiple embeddings** (bonus per spec): combine two SBERT models in parallel hybrid.

File: `services/retrieval/app/hybrid.py`, `services/retrieval/app/fusion.py`.

### 5.4 FastAPI Endpoint
```
POST /search
{
  "dataset_id": "msmarco_passage",
  "query": "what is the capital of france",
  "mode": "basic" | "with_features",
  "representation": "tfidf" | "bm25" | "embedding" | "hybrid_serial" | "hybrid_parallel",
  "fusion": "rrf" | "combsum" | "combmnz",  # only for hybrid_parallel
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "k": 10
}
```

### 5.5 Exit Criteria
- [ ] All 4 representations return results.
- [ ] Serial and parallel hybrid both functional.
- [ ] At least 2 fusion methods implemented.
- [ ] At least one configuration uses **multiple embedding models** in parallel (spec line 34 bonus).

---

## Phase 6 — Service-Oriented Architecture (SOA)

**Goal:** Six independently runnable services, one API Gateway, Docker Compose.

### 6.1 The Services

| Service | Port (dev) | Purpose |
|---------|------------|---------|
| gateway | 8000 | Public entry, routing, **CORS allow-origin for `http://localhost:5173`** |
| preprocessing | 8001 | Text preprocessing |
| indexing | 8002 | Inverted index, TF-IDF, BM25 |
| retrieval | 8003 | Embeddings, FAISS, hybrid |
| refinement | 8004 | Query refinement |
| rag | 8005 | RAG answer generation |
| ui | 5173 | **React frontend (Vite dev server)** — port 3000 inside Docker via nginx |

### 6.2 The API Gateway
File: `services/gateway/app/main.py`.
- Receives `/search`, `/refine`, `/rag/answer`.
- Internally calls the relevant services via `httpx`.
- Adds request_id, logs latency.
- Optionally aggregates results from multiple representations.
- **CORS**: install `fastapi.middleware.cors.CORSMiddleware` and allow `http://localhost:5173` (Vite dev) and the production origin. Without this the React app will be blocked by the browser.
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:5173", "http://localhost"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

### 6.3 Docker Compose
- One `Dockerfile` per service.
  - Backend services → `python:3.11-slim` + `pip install -r requirements.txt`.
  - **UI service** → multi-stage: `node:20-alpine` to `npm run build`, then `nginx:alpine` serving the static `dist/` (see Phase 7 §7.6).
- Shared volume for `data/`.
- `depends_on` with `healthcheck` (curl `/health`).
- `environment` block for service URLs.
- In dev, you usually run the UI outside Docker (`npm run dev`) and only containerize the backend. Keep this as a documented option.

### 6.4 Independence Tests
- `docker compose up` brings everything up.
- `docker compose up indexing` alone still works.
- Each service has a unit test that runs its app in-process.

### 6.5 Exit Criteria
- [ ] `docker compose up` starts all services.
- [ ] Gateway `/docs` (Swagger) is reachable.
- [ ] CORS preflight from `http://localhost:5173` succeeds.
- [ ] Each service has a `/health` route.
- [ ] Architecture diagram in `docs/diagrams/architecture.png` and `architecture.md` (use Mermaid + draw.io).

---

## Phase 7 — User Interface (React + Vite + TypeScript)

**Goal:** A modern, single-page React application that exposes every control the spec demands and talks to the Python gateway over REST.

### 7.1 Stack (and rationale)

| Choice | Why |
|--------|-----|
| **React 18** | Industry-standard UI library; mandated in the brief. |
| **Vite 5** | Fastest dev server on Windows; HMR is instant. |
| **TypeScript 5** | Strong types — the spec demands "clean, organized, maintainable" code. |
| **Tailwind CSS 3** | Utility-first styling, no external CSS files to fight. |
| **TanStack Query (React Query)** | Server-state cache, retries, background refresh — perfect for IR search calls. |
| **Zustand** | Tiny global store (selected dataset, user_id, prefs). |
| **Axios** | HTTP client with interceptors (auth/logging later). |
| **React Router 6** | One route is enough today, leaves room to grow. |
| **ESLint + Prettier** | Enforced code style. |
| **Vitest** | Unit tests for components. |

### 7.2 Bootstrap the React App (one-time, PowerShell)

```powershell
cd services
mkdir ui
cd ui

# Scaffold (non-interactive). The trailing `.` creates the project in the current folder.
npm create vite@latest . -- --template react-ts

# Answer the prompts:
#   ? Current directory is not empty. Please choose how to proceed:  Ignore files and continue
#   (or manually confirm any overwrites)

# Install deps
npm install

# Add the rest
npm install react-router-dom @tanstack/react-query axios zustand clsx
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p
```

Verify:
```powershell
npm run dev
# Vite prints: Local: http://localhost:5173/
# Open it — you should see the default React+Vite page.
```

### 7.3 Folder Layout (inside `services/ui/`)

```
services/ui/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── Dockerfile               # multi-stage: build with node, serve with nginx
├── nginx.conf               # SPA fallback + reverse proxy /api -> gateway:8000
├── public/
│   └── favicon.svg
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css            # Tailwind directives
    ├── api/
    │   └── client.ts        # axios instance with baseURL
    ├── components/
    │   ├── DatasetSelector.tsx
    │   ├── RepresentationPicker.tsx
    │   ├── Bm25Sliders.tsx
    │   ├── HybridConfigPicker.tsx
    │   ├── SearchBar.tsx
    │   ├── ResultsList.tsx
    │   ├── ResultCard.tsx
    │   ├── RagPanel.tsx
    │   ├── ModeToggle.tsx
    │   └── LatencyBadge.tsx
    ├── hooks/
    │   ├── useSearch.ts          # wraps React Query
    │   ├── useDatasets.ts
    │   └── useUserLog.ts         # POST click to gateway /log
    ├── pages/
    │   └── HomePage.tsx
    ├── store/
    │   └── useUiStore.ts         # zustand: dataset, mode, userId
    ├── types/
    │   └── api.ts                # mirrors shared/ir_common/schemas.py
    └── utils/
        └── highlight.ts
```

### 7.4 Vite Config (proxy `/api` → gateway)

`vite.config.ts`:
```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",   // gateway
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
```

This way the React app always calls `/api/...` — the same URL works in dev (Vite proxy) and in production (nginx proxy, see §7.6).

### 7.5 API Client (axios)

`src/api/client.ts`:
```ts
import axios from "axios";
import type { SearchRequest, SearchResponse, RagResponse, RefineRequest, RefineResponse } from "../types/api";

export const api = axios.create({
  baseURL: "/api",
  timeout: 30_000,
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    console.error("[api]", err?.response?.status, err?.config?.url, err?.message);
    return Promise.reject(err);
  }
);

export const search = (req: SearchRequest) =>
  api.post<SearchResponse>("/search", req).then((r) => r.data);

export const refine = (req: RefineRequest) =>
  api.post<RefineResponse>("/refine", req).then((r) => r.data);

export const ragAnswer = (params: { dataset_id: string; query: string; k?: number }) =>
  api.post<RagResponse>("/rag/answer", params).then((r) => r.data);

export const listDatasets = () =>
  api.get<string[]>("/datasets").then((r) => r.data);

export const logClick = (payload: { user_id: string; query: string; doc_id: string; dataset_id: string }) =>
  api.post("/log/click", payload).then((r) => r.data);
```

### 7.6 Production Build & Docker (nginx)

`services/ui/Dockerfile` (multi-stage):
```dockerfile
# --- build stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build          # produces /app/dist

# --- runtime stage ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

`services/ui/nginx.conf`:
```nginx
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;

  # SPA fallback
  location / {
    try_files $uri $uri/ /index.html;
  }

  # Proxy API calls to the Python gateway
  location /api/ {
    proxy_pass http://gateway:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 60s;
  }
}
```

In `docker-compose.yml`:
```yaml
  ui:
    build: ./services/ui
    container_name: ir_ui
    ports:
      - "3000:80"
    depends_on:
      - gateway
```

Then open `http://localhost:3000/`.

### 7.7 Required Controls (Component Map)

| Spec requirement | React component | Notes |
|------------------|-----------------|-------|
| Dataset selector before any search | `DatasetSelector` | Dropdown bound to `useUiStore.dataset`; persists in `localStorage` |
| Mode: Basic / With Features | `ModeToggle` | Radio group; default `basic` |
| Representation: TF-IDF / BM25 / Embedding / Hybrid Serial / Hybrid Parallel | `RepresentationPicker` | Conditional renders for sliders/fusion picker |
| BM25 sliders `k1`, `b` | `Bm25Sliders` | Range inputs, debounced (300ms) before re-fetching |
| Hybrid config (RRF / CombSUM / CombMNZ) | `HybridConfigPicker` | Shown only when `representation` is `hybrid_parallel` |
| Search input + button | `SearchBar` | Enter key triggers search; button shows loading spinner |
| Results list | `ResultsList` + `ResultCard` | Rank, snippet (~280 chars), score, "View" button |
| RAG toggle | `RagPanel` | Calls `/rag/answer` after a search; shows answer + source doc_ids |
| Click-to-log | `useUserLog` hook in `ResultCard.onClick` | POSTs to gateway `/log/click` |

### 7.8 HomePage Skeleton (sketch)

```tsx
// src/pages/HomePage.tsx
import { useState } from "react";
import { useSearch } from "../hooks/useSearch";
import { useUiStore } from "../store/useUiStore";
import DatasetSelector from "../components/DatasetSelector";
import ModeToggle from "../components/ModeToggle";
import RepresentationPicker from "../components/RepresentationPicker";
import Bm25Sliders from "../components/Bm25Sliders";
import HybridConfigPicker from "../components/HybridConfigPicker";
import SearchBar from "../components/SearchBar";
import ResultsList from "../components/ResultsList";
import RagPanel from "../components/RagPanel";
import LatencyBadge from "../components/LatencyBadge";

export default function HomePage() {
  const { dataset, mode, representation, bm25, fusion } = useUiStore();
  const [query, setQuery] = useState("");
  const { data, isFetching, refetch } = useSearch({ dataset, mode, representation, bm25, fusion, query });

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white p-4 shadow-sm">
        <h1 className="text-2xl font-bold">IR Search Engine — 2026</h1>
      </header>

      <section className="mx-auto max-w-5xl space-y-4 p-6">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <DatasetSelector />
          <ModeToggle />
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <RepresentationPicker />
          <HybridConfigPicker />
        </div>
        <Bm25Sliders />  {/* auto-hidden when not relevant via parent conditional */}
        <SearchBar value={query} onChange={setQuery} onSubmit={() => refetch()} loading={isFetching} />
        <LatencyBadge ms={data?.latency_ms} />
        <ResultsList results={data?.results ?? []} />
        <RagPanel query={query} dataset={dataset} />
      </section>
    </main>
  );
}
```

### 7.9 State Management (Zustand)

`src/store/useUiStore.ts`:
```ts
import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Representation = "tfidf" | "bm25" | "embedding" | "hybrid_serial" | "hybrid_parallel";
export type Mode = "basic" | "with_features";
export type Fusion = "rrf" | "combsum" | "combmnz";

interface UiState {
  dataset: string;
  mode: Mode;
  representation: Representation;
  fusion: Fusion;
  bm25: { k1: number; b: number };
  userId: string;
  setDataset: (d: string) => void;
  setMode: (m: Mode) => void;
  setRepresentation: (r: Representation) => void;
  setFusion: (f: Fusion) => void;
  setBm25: (b: { k1: number; b: number }) => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      dataset: "msmarco_passage",
      mode: "basic",
      representation: "bm25",
      fusion: "rrf",
      bm25: { k1: 1.5, b: 0.75 },
      userId: crypto.randomUUID(),
      setDataset: (dataset) => set({ dataset }),
      setMode: (mode) => set({ mode }),
      setRepresentation: (representation) => set({ representation }),
      setFusion: (fusion) => set({ fusion }),
      setBm25: (bm25) => set({ bm25 }),
    }),
    { name: "ir-ui" }
  )
);
```

### 7.10 Search Hook (React Query)

`src/hooks/useSearch.ts`:
```ts
import { useQuery } from "@tanstack/react-query";
import { search } from "../api/client";

export function useSearch(req: {
  dataset: string;
  mode: "basic" | "with_features";
  representation: string;
  fusion: string;
  bm25: { k1: number; b: number };
  query: string;
}) {
  return useQuery({
    queryKey: ["search", req],
    enabled: req.query.trim().length > 0,
    queryFn: () =>
      search({
        dataset_id: req.dataset,
        query: req.query,
        mode: req.mode,
        representation: req.representation,
        fusion: req.representation === "hybrid_parallel" ? req.fusion : undefined,
        bm25_k1: req.bm25.k1,
        bm25_b: req.bm25.b,
        k: 10,
      }),
    staleTime: 30_000,
  });
}
```

### 7.11 UX Polish
- Show latency (`LatencyBadge` — "⏱ 312 ms").
- Show total result count.
- Dark/light toggle (Tailwind `dark:` classes + `class` strategy in `tailwind.config.js`).
- Mobile-friendly: `grid-cols-1 md:grid-cols-2` everywhere.
- Loading skeletons (simple animated divs).
- Empty state with sample queries users can click.

### 7.12 Windows Dev Workflow

```powershell
# Terminal 1 — backend gateway (you'll start the other services as needed)
.\.venv\Scripts\Activate.ps1
uvicorn services.gateway.app.main:app --reload --port 8000

# Terminal 2 — React UI
cd services\ui
npm run dev
# open http://localhost:5173
```

### 7.13 Exit Criteria
- [ ] `npm run dev` starts Vite on `http://localhost:5173`.
- [ ] All controls listed in §7.7 are functional.
- [ ] BM25 sliders trigger a new search within ~300 ms of stopping.
- [ ] Clicking a result sends a `/log/click` to the gateway.
- [ ] RAG panel renders an answer + source doc_ids when enabled.
- [ ] `npm run build` produces `services/ui/dist/` with no TypeScript errors.
- [ ] (Optional) `docker compose up` brings the whole stack online and `http://localhost:3000/` shows the UI.

---

## Phase 8 — Additional Features (Vector Store + RAG)

> Vector Store is already implemented in Phase 3 (FAISS). This phase hardens it and adds RAG.

### 8.1 Vector Store Hardening
- [ ] Document the FAISS choice (Flat vs IVF vs HNSW) in the report.
- [ ] Provide a `rebuild_faiss.py` script.
- [ ] Add a benchmark: average query latency, recall@10 vs exact (Flat) for IVF.

### 8.2 RAG Service (port 8005)
File: `services/rag/app/`.

**Pipeline:**
1. Receive `{query, dataset_id, k}`.
2. Call `retrieval` service (top-k, default k=5).
3. Build context (truncate to fit LLM context window, e.g., 2000 tokens).
4. Format prompt with the strict template:
   ```
   You are a precise assistant. Use ONLY the context below.
   If the answer is not in the context, say "I don't know based on the given documents."
   Cite sources as [doc_id].
   --- CONTEXT ---
   {context}
   --- QUESTION ---
   {question}
   --- ANSWER ---
   ```
5. Generate with a local LLM.

**Generator options:**
- **Preferred**: Ollama running `llama3.1:8b` (good quality, local).
- **Fallback**: `google/flan-t5-base` via `transformers` (CPU-friendly, lower quality).

**Ollama setup (one-time):**
```bash
# Install Ollama (https://ollama.com)
ollama pull llama3.1:8b
ollama serve   # in another terminal
```

Endpoint: `POST /rag/answer`.

### 8.3 Faithfulness Guardrails
- Force every claim to include a `[doc_id]` citation.
- If the LLM responds "I don't know", surface that explicitly in the UI.
- Add a small manual eval set (10 Q&A) and self-score in the report.

### 8.4 Exit Criteria
- [ ] RAG returns grounded answers with citations.
- [ ] FAISS index is fast enough (< 100ms per query).
- [ ] RAG documented in the Arabic report with example outputs.

---

## Phase 9 — System Evaluation

**Goal:** Quantify every (dataset, representation, condition) combination with MAP, Recall, P@10, nDCG.

### 9.1 Data Prep
- Sample **200 test queries** with non-empty qrels per dataset.
- Save under `evaluation/queries/<dataset_id>_queries.txt` (TREC format).

### 9.2 Run Format
For every run, write a TREC file:
```
query_id Q0 doc_id rank score run_id
```
- `run_id` = `<dataset>__<model>__<condition>`, e.g. `msmarco_passage__bm25__baseline`.

### 9.3 Evaluation Script
File: `scripts/run_evaluation.py`.
- For each dataset, for each model (tfidf, bm25, embedding, hybrid_serial, hybrid_parallel), for each condition (baseline, with_features):
  - Run all test queries.
  - Write TREC run file.
  - Run `pytrec_eval` → store {MAP, Recall, P@10, nDCG}.

### 9.4 Conditions
- **C1 Baseline**: preprocessing + retriever.
- **C2 With Features**: + refinement (synonyms, spell, grammar) + personalization.

### 9.5 Output
- CSV: `evaluation/reports/summary.csv`.
- Markdown table: `evaluation/reports/summary.md`.
- Bar chart per metric (matplotlib or seaborn): `evaluation/reports/plots/`.

### 9.6 Analysis (write into Arabic report)
- Impact of each representation.
- Comparative analysis TF-IDF vs BM25 vs Embeddings vs Hybrid.
- Quantify how much features improved results.
- Justify model + parameter choices with empirical evidence.

### 9.7 Exit Criteria
- [ ] Tables produced for **both datasets, all representations, both conditions**.
- [ ] Numbers are **sensible** (e.g., BM25 on MS MARCO: MAP ≥ 0.20, P@10 ≥ 0.60 — see spec rejection note).
- [ ] Plots included in the report.

---

## Phase 10 — Hardening, Documentation & Submission

**Goal:** Polished, defensible, ready-to-ship.

### 10.1 README.md
- Project description.
- Architecture diagram.
- **Quick start (Windows / PowerShell)**:
  - `docker compose up --build` → open `http://localhost:3000/`.
  - **Dev mode**: backend on 8000, UI on 5173 (`npm run dev` in `services/ui`).
- How to run the UI (browser URL and screenshots).
- How to run evaluation (`python scripts/run_evaluation.py`).
- Dataset choice + justification.
- Work distribution (you'll be listed as the sole author; mention this in the spec context).
- Screenshots of the **React UI** in both datasets and all 4 representations.
- Citation of libraries.

### 10.2 Arabic Report (PDF)
Sections:
1. ملخص تنفيذي
2. مقدمة وخلفية نظرية
3. وصف النظام (هندسة SOA + مخطط)
4. وصف مجموعات البيانات
5. منهجية المعالجة التمهيدية
6. تمثيل الوثائق (TF-IDF, BM25, Embeddings, Hybrid)
7. الفهرسة (Inverted Index)
8. تحسين الاستعلام (Refinement)
9. المطابقة والترتيب
10. الميزات الإضافية (Vector Store, RAG)
11. واجهة المستخدم
12. التقييم والنتائج (MAP, Recall, P@10, nDCG)
13. تحليل ومناقشة
14. التحديات والدروس المستفادة
15. المراجع (APA)

### 10.3 Architecture Diagram
- Use Mermaid for the inline version in `README.md`.
- Use draw.io for a clean PNG embedded in the report.
- Show: UI → Gateway → Refinement → Preprocessing → Indexing & Retrieval → RAG → back to UI.

### 10.4 Demo Video (3–5 min)
- Show picking each dataset.
- Show all 4 representations.
- Tweak BM25 sliders live.
- Switch hybrid serial ↔ parallel.
- Enable RAG, show answer + sources.

### 10.5 Submission Checklist
- [ ] Public GitHub repo with README.
- [ ] `docker compose up` brings the system online.
- [ ] All 8 deliverables from the spec are ready (see [TEAM_DOCUMENTATION.md §15](./TEAM_DOCUMENTATION.md#15-final-submission-checklist) for a mirror).
- [ ] Demo video uploaded (unlisted YouTube or Google Drive link in README).
- [ ] Arabic report PDF committed to `reports/`.
- [ ] Final dry-run: 1 hour before submission, on a fresh machine, do `git clone && docker compose up && open UI` and confirm everything works.

---

## Appendix A — Daily/Weekly Checklist

### Daily
- [ ] 1 commit minimum, descriptive.
- [ ] Update `docs/progress.md` with what you did / what blocked you.
- [ ] Run `pytest -q` before each commit.

### Weekly (Friday afternoon)
- [ ] Run full evaluation, save the table.
- [ ] Run the UI end-to-end and screenshot.
- [ ] Write a short retro in `docs/weekly/<week>.md`.

### Two weeks before deadline
- [ ] Stop adding new features.
- [ ] Polish UI, fill the report, write the README.
- [ ] Practice the 10-min defense out loud.

### One week before deadline
- [ ] Submit early if possible (instructor usually accepts).
- [ ] Get a friend to clone and run the repo; fix any onboarding bugs they hit.

---

## Appendix B — Architecture Sketch

```
+----------------------------+
|  React UI (Vite, dev :5173)|
|  nginx-served (:3000 prod) |
+-------------+--------------+
              |  /api/*  (proxied)
              v
+----------------------------+
|   FastAPI Gateway :8000    |
+--+-----+-----+-----+---+---+
   |     |     |     |   |
   v     v     v     v   v
+-----+ +---+ +---+ +---+ +---+
|Pre" | |Idx| |Ret| |Raf| |RAG|
|proc| |   | |   | |ine| |   |
|8001| |8002| |8003| |8004| |8005
+--+--+ +---+ +---+ +---+ +---+
   |        |      |
   +--------+------+
            |
            v
+----------------------------+
| Shared Volume (./data)     |
| - processed/<dataset>/     |
| - indexes/<dataset>/       |
| - faiss/<dataset>/         |
+----------------------------+
```

(For a full draw.io version, see `docs/diagrams/architecture.png`.)

---

### Final words
This is a substantial project. **Don't try to do everything in one sprint.** Stick to the phases, hit the exit criteria, and ship a working system first — polish comes after. By Phase 6 you should already have a defensible baseline. Phases 7–10 turn it into a great one.

*Good luck. Ship it.*

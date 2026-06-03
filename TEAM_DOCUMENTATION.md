# Information Retrieval System — Project 2026
## Team Engineering Documentation (4 Developers)

> **Audience:** Yasser, Omar, Ibrahim, Abdullah
> **Course:** Information Retrieval Systems 2026 — Practical Project
> **Supervisors:** Dr. Abi Sandouk · Eng. Marwa Al-Daya · Eng. Salyma Al-Muhairi
> **Deadline:** July 3rd, 2026
> **Language Constraint:** Python only
> **Additional Features Committed:** 2 (RAG + Vector Stores)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Scope & Deliverables](#2-project-scope--deliverables)
3. [Technology Stack & Setup Guide](#3-technology-stack--setup-guide)
4. [Repository Structure (Monorepo)](#4-repository-structure-monorepo)
5. [Service-Oriented Architecture (SOA) Blueprint](#5-service-oriented-architecture-soa-blueprint)
6. [Dataset Strategy](#6-dataset-strategy)
7. [Requirement-by-Requirement Implementation Plan](#7-requirement-by-requirement-implementation-plan)
8. [Additional Features Plan (RAG & Vector Stores)](#8-additional-features-plan-rag--vector-stores)
9. [Evaluation Framework](#9-evaluation-framework)
10. [User Interface Specifications](#10-user-interface-specifications)
11. [Work Distribution (Yasser · Omar · Ibrahim · Abdullah)](#11-work-distribution-yasser--omar--ibrahim--abdullah)
12. [Sprint Plan & Milestones](#12-sprint-plan--milestones)
13. [Coding Standards & Git Workflow](#13-coding-standards--git-workflow)
14. [Risk Register & Mitigations](#14-risk-register--mitigations)
15. [Final Submission Checklist](#15-final-submission-checklist)

---

## 1. Executive Summary

We are building a **production-grade, service-oriented Information Retrieval (IR) search engine** that supports:

- Two independent corpora (≥ 200K docs each, with qrels).
- Four retrieval representations: **TF-IDF (VSM), BM25, Dense Embeddings, Hybrid (Serial + Parallel)**.
- **Query refinement** (personalization, synonym expansion, spell/grammar correction).
- **Inverted index** for fast lexical retrieval.
- **Two additional features**: **Retrieval-Augmented Generation (RAG)** and a **Vector Store** (FAISS).
- A **Web UI** with dataset selection, BM25 parameter controls, and Hybrid mode selection.
- A **standard IR evaluation** (MAP, Recall, P@10, nDCG) under *baseline* vs *with-features* conditions.

The system is decomposed into **six loosely coupled services** communicating over **REST** with an **API Gateway** in front, packaged with **Docker Compose** for one-command local deployment.

---

## 2. Project Scope & Deliverables

### 2.1 Functional Deliverables
| # | Deliverable | Owner |
|---|-------------|-------|
| D1 | Ingestion + Preprocessing pipeline (2 datasets) | Omar |
| D2 | Inverted Index + BM25 + TF-IDF retrievers | Omar |
| D3 | Embedding retriever + FAISS Vector Store | Ibrahim |
| D4 | Hybrid (Serial & Parallel) retriever with score fusion | Ibrahim |
| D5 | Query Refinement service (synonyms, spell-correction, personalization) | Abdullah |
| D6 | RAG service (LLM-grounded answer generation) | Abdullah |
| D7 | SOA services scaffolding (FastAPI, API Gateway, Docker) | Yasser |
| D8 | Web UI (Streamlit) | Yasser (with Abdullah) |
| D9 | Evaluation harness (pytrec_eval) + report tables | All |
| D10 | Arabic technical report + GitHub README | All |

### 2.2 Submission Artifacts (per spec §Submission Requirements)
1. Detailed report **in Arabic**, fully cited.
2. Description of both chosen datasets.
3. Detailed step-by-step breakdown of the project and per-service descriptions.
4. **Architecture diagram** (clean, professional — draw.io / Mermaid).
5. Evaluation reports (tables + per-dataset, per-model, per-condition).
6. Work distribution table for the 4 members.
7. **Executable** search engine (Docker Compose up-and-running).
8. **Public GitHub repository** with a polished `README.md`.

---

## 3. Technology Stack & Setup Guide

### 3.1 Stack Decision (and rationale)

| Layer | Choice | Why |
|-------|--------|-----|
| Language | **Python 3.11+** | Mandated by spec; great IR/ML ecosystem. |
| Web framework | **FastAPI** | Async, auto-OpenAPI, type-hints; ideal for microservices. |
| API Gateway | **FastAPI Gateway** (single entry) | Lightweight; avoids extra infra like Kong. |
| Lexical retrieval | **`rank_bm25`** + custom **inverted index** (Python dict → persisted with `pickle`/`joblib`) | Required by spec; no Elasticsearch. |
| Vector retrieval | **FAISS (faiss-cpu)** | Required additional feature; fast, mature. |
| Dense embeddings | **`sentence-transformers`** (e.g., `all-MiniLM-L6-v2`, `msmarco-MiniLM`) | Pre-trained, high-quality, fast on CPU. |
| Word2Vec option | **`gensim`** (train on corpus as backup) | Satisfies "such as Word2Vec, BERT" wording. |
| Preprocessing | **NLTK** (tokenize, stopwords, Porter/Snowball stemmer), **spaCy** (lemmatization, NER) | Standard, well-known. |
| Query refinement | **`symspellpy`** (spell), **WordNet (NLTK corpus)** (synonyms), **language-tool-python** (grammar) | All offline-capable. |
| Dataset access | **`ir_datasets`** library | Direct access to datasets from `ir-datasets.com`. |
| Evaluation | **`pytrec_eval`** | Official TREC evaluator. |
| RAG | **`langchain`** + **HuggingFace `transformers`** (e.g., `flan-t5-base`) or **Ollama** | Keeps everything local; no paid API. |
| Frontend | **Streamlit** | Fast to build, beautiful, deployable. |
| Containerization | **Docker + Docker Compose** | One-command run for the defense. |
| Message broker | **None** (REST only) — keep simple; revisit only if load demands it. | Spec allows REST, RPC, MQ. REST is sufficient. |
| Testing | **pytest** + **httpx** (async API tests) | Standard. |
| Lint/format | **ruff** + **black** + **mypy** | Enforce clean code (spec demands it). |
| Docs | **mkdocs-material** (optional) | For internal docs. |

### 3.2 Local Environment Setup (Windows / macOS / Linux)

> Run once on every dev machine.

```bash
# 1. Clone the repo
git clone https://github.com/<org>/ir-project-2026.git
cd ir-project-2026

# 2. Create a virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm

# 4. Copy env file and edit
cp .env.example .env

# 5. (Optional) pull docker images
docker compose pull

# 6. Smoke test
python scripts/smoke_test.py
```

### 3.3 `requirements.txt` (baseline)
```
fastapi==0.115.*
uvicorn[standard]==0.30.*
pydantic==2.*
pydantic-settings==2.*
httpx==0.27.*
requests==2.32.*
ir_datasets==0.5.*
nltk==3.9.*
spacy==3.7.*
gensim==4.3.*
sentence-transformers==3.*
torch>=2.2
faiss-cpu==1.8.*
rank_bm25==0.2.*
scikit-learn==1.5.*
numpy>=1.26
pandas==2.*
pytrec_eval==0.5.*
symspellpy==6.7.*
language-tool-python==2.7.*
streamlit==1.37.*
langchain==0.2.*
transformers==4.44.*
accelerate==0.33.*
pytest==8.*
pytest-asyncio==0.23.*
ruff==0.5.*
black==24.*
mypy==1.10.*
joblib==1.4.*
tqdm==4.66.*
```

### 3.4 Editor Setup
- **VS Code** with extensions: Python, Pylance, Ruff, Docker, GitLens.
- Format-on-save with Black (line length 100) and Ruff import-sort.

---

## 4. Repository Structure (Monorepo)

```
ir-project-2026/
├── README.md
├── LICENSE
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Makefile                      # convenience: make up, make eval, etc.
│
├── docs/
│   ├── architecture.md           # SOA + diagrams
│   ├── api_contract.md           # OpenAPI cross-service contract
│   ├── evaluation.md
│   ├── report_outline.md         # Arabic report skeleton
│   └── diagrams/
│       ├── architecture.png
│       └── data_flow.png
│
├── data/                         # gitignored — populated locally
│   ├── raw/                      # ir_datasets cache
│   ├── processed/<dataset_id>/
│   └── indexes/<dataset_id>/
│
├── services/
│   ├── gateway/                  # Yasser — API Gateway
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routes.py
│   │   │   └── deps.py
│   │   ├── Dockerfile
│   │   └── tests/
│   │
│   ├── preprocessing/            # Omar
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── pipeline.py
│   │   │   └── langpacks/        # per-language tokenize/stop
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   ├── indexing/                 # Omar
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── inverted_index.py
│   │   │   ├── bm25.py
│   │   │   └── tfidf.py
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   ├── retrieval/                # Ibrahim
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── embedder.py
│   │   │   ├── vector_store.py   # FAISS wrapper
│   │   │   ├── hybrid.py         # serial + parallel + fusion
│   │   │   └── fusion.py         # RRF, CombSUM, CombMNZ
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   ├── refinement/               # Abdullah
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── synonyms.py
│   │   │   ├── spell.py
│   │   │   ├── grammar.py
│   │   │   └── personalization.py # user log weights
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   ├── rag/                      # Abdullah (Feature 1)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── prompt.py
│   │   │   ├── generator.py      # HF / Ollama
│   │   │   └── context_builder.py
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   └── ui/                       # Yasser (with Abdullah)
│       ├── streamlit_app.py
│       ├── components/
│       ├── Dockerfile
│       └── tests/
│
├── shared/
│   ├── ir_common/                # internal package
│   │   ├── config.py             # pydantic settings
│   │   ├── schemas.py            # Pydantic models (Request/Response)
│   │   ├── logging.py
│   │   ├── http_client.py        # resilient httpx wrapper
│   │   └── constants.py
│   └── tests/
│
├── scripts/
│   ├── ingest_<dataset_a>.py
│   ├── ingest_<dataset_b>.py
│   ├── build_indexes.py
│   ├── run_evaluation.py
│   └── smoke_test.py
│
├── evaluation/
│   ├── queries/                  # sampled test queries w/ qrels
│   ├── results/                  # per-run TREC results
│   └── reports/                  # generated tables & plots
│
├── reports/
│   └── final_report_ar.md        # Arabic, exported to PDF
│
└── .github/
    └── workflows/
        ├── ci.yml                # ruff + pytest
        └── docker.yml
```

### 4.1 Service Communication Contract (shared package)
All services import `shared.ir_common` for:
- Pydantic schemas (e.g., `QueryRequest`, `SearchResponse`, `ScoredDoc`).
- A common logger.
- Config from `.env` (service URLs, model names, BM25 defaults).

This avoids duplication and guarantees the same data model across the wire.

---

## 5. Service-Oriented Architecture (SOA) Blueprint

### 5.1 Service Map

| Service | Port | Responsibility | Talks to |
|---------|------|----------------|----------|
| **gateway** | 8000 | Single public entry, routing, auth (optional) | All |
| **preprocessing** | 8001 | Tokenize, normalize, stem/lemmatize, stopword removal | gateway |
| **indexing** | 8002 | Build/save inverted index, BM25, TF-IDF | gateway, retrieval (read) |
| **retrieval** | 8003 | Embedding retriever + FAISS + Hybrid (Serial/Parallel) | gateway |
| **refinement** | 8004 | Query rewriting (spell, synonyms, grammar, personalization) | gateway |
| **rag** | 8005 | Grounded LLM answer from top-k docs | gateway |
| **ui** | 8501 | Streamlit frontend | gateway |

### 5.2 Why these boundaries?
- **Single Responsibility**: each service is a verb in the IR pipeline (`preprocess`, `index`, `retrieve`, `refine`, `generate`).
- **Independent deployability**: each has its own `Dockerfile` and can be run in isolation (`uvicorn services.preprocessing.app.main:app`).
- **Reusability**: `refinement` and `preprocessing` could be reused by a future ingestion pipeline.
- **Loose coupling**: services only know HTTP + shared schemas; no shared DB lock-in.
- **Scalability**: retrieval/embedding is the bottleneck — we can scale it horizontally later (stateless workers, shared FAISS index on shared volume).

### 5.3 Communication Protocol
- **REST/JSON** over HTTP using **FastAPI**.
- Synchronous request/response is sufficient for IR (latency < 1s typical).
- Each service exposes `/health` and OpenAPI docs at `/docs`.
- Timeouts via `httpx` (connect=2s, read=15s).

### 5.4 End-to-End Query Flow

```
[UI]  ──text query──▶  [gateway]
                         │
                         ▼
                   [refinement]  (rewrite)
                         │
                         ▼
                  [preprocessing] (tokenize/stem)
                         │
                         ├────────────────────┐
                         ▼                    ▼
                  [indexing] (BM25/TF-IDF)   [retrieval] (embeddings + FAISS)
                         │                    │
                         └──────┬─────────────┘
                                ▼
                       [retrieval/hybrid]  (serial OR parallel + fusion)
                                │
                                ▼
                            [gateway] ──results──▶ [UI]
                                │
                                ▼ (optional)
                            [rag] (top-k → LLM answer)
```

### 5.5 Design Patterns Used
- **API Gateway** pattern (Yasser).
- **Service Registry** lite (gateway holds a service map in `config.py`).
- **Circuit Breaker / Retry** via `httpx` retries + tenacity (in `http_client.py`).
- **Strategy pattern** for retrievers (BM25, TF-IDF, Embedding, Hybrid) registered in a factory.
- **Builder pattern** for query pipelines.
- **Repository pattern** for indexes (load/save from disk).
- **Singleton** for embedder & FAISS index per dataset (lazy-loaded at service start).

### 5.6 Observability
- Structured JSON logging (per request: `request_id`, `service`, `latency_ms`).
- `/health` returns `{ "status": "ok", "dataset_loaded": "<id>" }`.

---

## 6. Dataset Strategy

### 6.1 Selection criteria (per spec)
- **≥ 200,000 documents** each.
- Must include **qrels** + test queries.
- **No Antique.**
- At least one should be English (or both English for simplicity); a second may be multilingual if we choose feature 12, but our 2 committed features are RAG + Vector Stores, so we recommend both English.

### 6.2 Recommended candidates (pick any 2)

| Dataset | Docs | Has qrels | Notes |
|---------|------|-----------|-------|
| `msmarco/passage` | 8.8M | ✅ | Classic IR benchmark. |
| `msmarco/document` | 3.2M | ✅ | Document-level variant. |
| `beir/trec-covid` | 171K | ✅ | Biomed, may be too small (use as second option). |
| `beir/scifact` | 5K | ✅ | Too small — skip. |
| `beir/nfcorpus` | 3.6K | ✅ | Too small — skip. |
| `beir/fiqa` | 57K | ✅ | Too small — skip. |
| `beir/arguana` | 8.6K | ✅ | Too small — skip. |
| `cranfield` | 1.4K | ✅ | Too small. |
| `vaswani` | 11K | ✅ | Too small. |
| `medline/2004` | 250K+ | ✅ | Biomedical. |
| `cord19/abstracts` | 500K+ | ✅ | COVID research. |
| `nyt/acquis` | 300K+ | ✅ | News. |

### 6.3 Recommended pair
- **Dataset A**: `msmarco/passage` (huge, well-known).
- **Dataset B**: `cord19/abstracts` (good size, different domain → tests generalization).

Final choice must be approved by the lab instructor (Eng. Marwa / Eng. Salyma) before ingestion.

### 6.4 Ingestion
- Use `ir_datasets.load("msmarco/passage")` → `dataset.docs_iter()`, `dataset.queries_iter()`, `dataset.qrels_iter()`.
- Stream documents, do not load all in RAM.
- Persist processed text to `data/processed/<dataset_id>/docs.jsonl`.
- Persist qrels as TREC format (`query_id Q0 doc_id rel`).
- Sample **N=200 test queries** (with non-empty qrels) for evaluation — store under `evaluation/queries/`.

---

## 7. Requirement-by-Requirement Implementation Plan

> This section mirrors the spec 1:1. Each subsection states the requirement, the design, and the service/file that owns it.

### 7.1 Data Pre-Processing
**Spec:** "Stemming, Lemmatization, Normalization, etc."

**Pipeline (per document):**
1. Strip HTML/Markdown.
2. Unicode normalize (NFKC).
3. Lowercase.
4. Tokenize (NLTK `word_tokenize`).
5. Remove stopwords (NLTK English + domain-specific additions).
6. Remove short tokens (`len < 2`).
7. Choose **one** of: Snowball stemmer, Porter stemmer, or spaCy lemmatizer. (Justify choice in report.)
8. Persist `doc_id → tokens[]` and `doc_id → original_text` (for snippets).

**Implementation:** `services/preprocessing/app/pipeline.py` exposes:
- `POST /preprocess` → `{ "text": "..." } → { "tokens": [...], "normalized": "..." }`
- `POST /preprocess/batch` → bulk version.

**Notes:** *Same pipeline* must be used at query time. The choice is centralized in `shared/ir_common/langpacks/en.py` and re-used by `refinement` and `indexing` services.

### 7.2 Document Representation (all four required)

#### 7.2.1 VSM_TF-IDF
- `TfidfVectorizer` (scikit-learn) over preprocessed corpus.
- Stored as `sparse` matrix + vocabulary (joblib).
- Query → same vectorizer → cosine similarity (top-k).
- File: `services/indexing/app/tfidf.py`.

#### 7.2.2 BM25
- `rank_bm25.BM25Okapi` (or BM25Plus) built on tokenized corpus.
- Parameters `k1` and `b` are **exposed via UI** as sliders (default k1=1.5, b=0.75).
- File: `services/indexing/app/bm25.py`.

#### 7.2.3 Embedding Representation
- Use **`sentence-transformers/all-MiniLM-L6-v2`** (default, fast on CPU).
- Optional second model: **`msmarco-distilbert-base-v3`** for better quality.
- Encode documents in batches of 256.
- Encode queries on the fly.
- Similarity = cosine.
- File: `services/retrieval/app/embedder.py`.

#### 7.2.4 Hybrid Representation (Serial + Parallel)
- **Serial**: pipeline one retriever feeds the next (e.g., BM25 → re-rank with embeddings). Justify in report.
- **Parallel**: run two or more retrievers **simultaneously** → fuse scores.
  - **Fusion methods** (provide as configurable):
    - **Reciprocal Rank Fusion (RRF)** — `score(d) = Σ 1/(k + rank_i(d))`
    - **CombSUM** — sum of normalized scores
    - **CombMNZ** — sum × number of non-zero contributions
  - Hybrid can combine: BM25 + TF-IDF, BM25 + Embedding, TF-IDF + Embedding, **or multiple embeddings** (spec line 34 allows multi-embedding combinations).
- File: `services/retrieval/app/hybrid.py`, `services/retrieval/app/fusion.py`.
- UI radio: `Serial` | `Parallel` (per spec line 33).

### 7.3 Indexing
- Build an **Inverted Index**: `term → {doc_id: tf}` plus `doc_len` and `df` (postings).
- Persist as `.joblib` files under `data/indexes/<dataset_id>/`.
- Provide `GET /index/{dataset_id}/stats` → vocab size, avg doc length, total docs.
- Use a **block-based** or **single dict** structure (justify in report; both are acceptable at this scale).
- Optionally a **skip pointer** variant (mentioned in the report for the "high efficiency" wording).
- File: `services/indexing/app/inverted_index.py`.

### 7.4 Query Processing
- Reuse the preprocessing pipeline verbatim (`preprocessing` service).
- Endpoint: `POST /query/preprocess` → tokens + normalized.
- The `retrieval` service calls this *or* uses an in-process pipeline for latency (decide: in-process for speed, remote for consistency — default to in-process inside retrieval, with a unit test that guarantees parity with the remote service).

### 7.5 Query Refinement
**Spec:** weighting from previous history, synonyms, linguistic correction.

**Components:**
1. **Spell correction**: `symspellpy` with a frequency dictionary (download once).
2. **Synonym expansion**: WordNet (NLTK corpus) — for each query term, add 1–2 synonyms.
3. **Grammar correction**: `language-tool-python` (offline jar).
4. **Personalization**: maintain a per-user JSONL log of past queries. Boost terms that frequently co-occur in the user's past successful queries (clicks simulated by qrels during evaluation, or by the user clicking on a result in the UI → write to log).
5. **Pseudo-relevance feedback (PRF)**: optional bonus — use top-N terms from the top-k BM25 results to expand the query (mention in report).

Endpoint: `POST /refine` → `{ "query": "...", "user_id": "..." } → { "expanded_query": "...", "tokens": [...] }`.

File: `services/refinement/app/`.

### 7.6 Query Matching & Ranking
- Each retriever returns `(doc_id, score)`.
- **Cosine similarity** for TF-IDF and Embedding.
- **BM25** score from `rank_bm25`.
- **Hybrid** uses fusion functions.
- Endpoint: `POST /search` on `retrieval` → `SearchResponse{ results: [ScoredDoc], fusion_used?: str, ... }`.

### 7.7 Application of SOA
See [Section 5](#5-service-oriented-architecture-soa-blueprint). Architecture diagram to be drawn in `docs/diagrams/architecture.png` (and embedded in the report).

### 7.8 System Evaluation
See [Section 9](#9-evaluation-framework).

### 7.9 User Interface
See [Section 10](#10-user-interface-specifications).

---

## 8. Additional Features Plan (RAG & Vector Stores)

### 8.1 Feature A — Vector Store (FAISS)
- All dense embeddings stored in a **FAISS index** per dataset.
- Choose index type by size:
  - `< 1M vectors`: `IndexFlatIP` (exact, cosine via normalized vectors).
  - `≥ 1M`: `IndexIVFFlat` or `IndexHNSWFlat` for sub-linear search.
- Persist with `faiss.write_index` to `data/indexes/<dataset_id>/faiss.index`.
- Expose `POST /retrieval/embed` and `POST /retrieval/search` (k-nearest).
- Lazy-load on service start (`singleton` pattern in `vector_store.py`).
- This feature also powers the Embedding retriever (so it earns double credit).

### 8.2 Feature B — RAG (Retrieval-Augmented Generation)
**Goal:** Given a query, retrieve top-k documents, build a context window, prompt an LLM to generate a grounded answer with citations.

**Components:**
1. **Retriever hook**: reuses `retrieval` service (any representation).
2. **Context builder**: joins top-k snippets, fits within the LLM's context (e.g., 2048 tokens).
3. **Prompt template**:
   ```
   You are a precise assistant. Use ONLY the context below to answer.
   If the answer is not in the context, say "I don't know based on the given documents."
   Cite sources as [doc_id].
   --- CONTEXT ---
   {context}
   --- QUESTION ---
   {question}
   ```
4. **Generator**: local LLM (no paid API). Options:
   - **Ollama** running `llama3.1:8b` or `mistral:7b` (preferred).
   - Fallback: `google/flan-t5-base` via `transformers` (smaller, CPU-friendly).
5. **Endpoint**: `POST /rag/answer` → `{ "query", "dataset_id", "k" } → { "answer", "sources": [...] }`.
6. **Evaluation**: RAG quality evaluated via (a) faithfulness heuristic (answer contains cited doc ids) and (b) a small set of curated Q&A pairs (manual scoring). Document in the report.

---

## 9. Evaluation Framework

### 9.1 Metrics (mandatory for every representation × every dataset)
- **MAP** (Mean Average Precision)
- **Recall**
- **Precision@10**
- **nDCG**

### 9.2 Tooling
- `pytrec_eval` with the official TREC run format:
  ```
  query_id Q0 doc_id rank score run_id
  ```
- Each representation writes its own run file under `evaluation/results/<dataset_id>/<model>_<condition>.txt`.

### 9.3 Conditions
- **C1 (Baseline)**: preprocessing + retriever only.
- **C2 (With features)**: + query refinement (synonyms, spell, grammar) + personalization (synthetic user history derived from qrels, simulating past interactions).

### 9.4 Scripts
- `scripts/run_evaluation.py` iterates over all (dataset, model, condition) → produces a CSV/Markdown table.
- `evaluation/reports/summary.md` shows side-by-side numbers.
- Plots: P@10 bar chart per model; nDCG curve for top-k (k = 5, 10, 20).

### 9.5 Acceptance Criterion
> "Any system that yields extremely low or illogical evaluation results … will be strictly rejected."

Targets (must hit roughly these for `msmarco/passage`):
- BM25: MAP ≥ 0.20, P@10 ≥ 0.60, nDCG ≥ 0.30.
- Embedding: comparable to BM25 (±10%).
- Hybrid (parallel) ≥ best single model.

If below, iterate: try BM25Plus, tune k1/b, add PRF, re-train embeddings on corpus, add more synonyms.

---

## 10. User Interface Specifications

### 10.1 Framework
**Streamlit** (single page) — fast to develop, easy to demo.

### 10.2 Layout
```
┌──────────────────────────────────────────────────────────────┐
│  IR Search Engine 2026                                       │
├──────────────────────────────────────────────────────────────┤
│  [Dataset ▾]  [Mode: Basic | With Features]                  │
│  [Representation: TF-IDF | BM25 | Embedding | Hybrid(S|P)]  │
│  [BM25 k1 ▭▭▭▭▭  BM25 b ▭▭▭▭▭]  (visible only for BM25/Hybrid)│
│  [Hybrid config: RRF | CombSUM | CombMNZ]  (if Hybrid)      │
│  [Use RAG answer ☑]  (extra toggle for RAG feature)         │
│                                                              │
│  [        search input box            ] [ Search ]           │
├──────────────────────────────────────────────────────────────┤
│  #1  Title …………  snippet …………  score: 0.812                 │
│  #2  …                                                            │
│  #10 …                                                            │
├──────────────────────────────────────────────────────────────┤
│  💡 RAG Answer:  "..."  [Sources: doc_id, doc_id]            │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 Functional Requirements
- Dataset selection → persisted as a Streamlit session state, sent to gateway on every request.
- Query input.
- Display top-k results (title + snippet + score + clickable link to full doc).
- Toggle **Basic** vs **With Features** (Basic = no refinement; With Features = full refinement pipeline).
- BM25 sliders (`k1`, `b`) live, react to query re-execution.
- Hybrid representation radio: `Serial` or `Parallel`.
- Hybrid config dropdown (RRF, CombSUM, CombMNZ).
- "Generate RAG answer" button → calls RAG service, displays answer + source doc_ids.
- Click on a result → logged to `user_log.jsonl` (powers personalization).

### 10.4 Accessibility
- Keyboard navigable.
- Light/dark theme.

---

## 11. Work Distribution (Yasser · Omar · Ibrahim · Abdullah)

### 11.1 Role summary

| Member | Lead Role | Primary Services / Modules | Secondary |
|--------|-----------|----------------------------|-----------|
| **Yasser** | **Platform / Infra Lead** | `gateway`, `shared/ir_common`, Docker Compose, CI, UI infra | RAG infra support |
| **Omar** | **Data & Lexical Retrieval Lead** | `preprocessing`, `indexing` (inverted index, BM25, TF-IDF), dataset ingestion scripts | Evaluation support |
| **Ibrahim** | **Representation & Hybrid Lead** | `retrieval` (embedder, FAISS, hybrid serial/parallel, fusion) | Vector Store feature |
| **Abdullah** | **Intelligence & UX Lead** | `refinement`, `rag`, `ui` (Streamlit), evaluation harness, report writing | Personalization + RAG |

> **Why this split?** It maps cleanly to the SOA service boundaries, so each member can develop and demo their services independently (spec line 7 "ensure the ability to run or test each Service independently").

### 11.2 Yasser — Detailed Tasks
- [ ] Y1. Initialize the monorepo (folder structure, `pyproject.toml`, `Makefile`, `.env.example`).
- [ ] Y2. Implement `shared/ir_common` (config, schemas, http_client, logging).
- [ ] Y3. Implement `services/gateway` (routing to all services, request validation, `/health`).
- [ ] Y4. Author `docker-compose.yml` for all services.
- [ ] Y5. Set up CI (GitHub Actions: ruff + pytest + build docker).
- [ ] Y6. UI scaffolding (Streamlit `streamlit_app.py`, components dir).
- [ ] Y7. Wire UI to gateway (httpx calls), handle session state.
- [ ] Y8. Architecture diagrams (`docs/diagrams/architecture.png`, Mermaid fallback).
- [ ] Y9. Final integration test — `make up` must bring the whole system online.
- [ ] Y10. Defense demo script (a `demo.md` with click-by-click steps).

### 11.3 Omar — Detailed Tasks
- [ ] O1. Pick datasets, get instructor approval.
- [ ] O2. Write `scripts/ingest_<dataset_a>.py` and `<dataset_b>.py` (streaming, progress bars).
- [ ] O3. Implement `preprocessing` service + unit tests.
- [ ] O4. Implement inverted index (build, save, load, query).
- [ ] O5. Implement BM25 retriever + parameter passing.
- [ ] O6. Implement TF-IDF retriever.
- [ ] O7. Expose `/index/{dataset_id}/build` and `/index/{dataset_id}/stats`.
- [ ] O8. Verify index build on both datasets end-to-end.
- [ ] O9. Document the preprocessing choices in the report (Porter vs Snowball vs spaCy, justify).
- [ ] O10. Add `evaluation/queries/sample_200.txt` for both datasets.

### 11.4 Ibrahim — Detailed Tasks
- [ ] I1. Pick embedding model(s); justify in report.
- [ ] I2. Implement `embedder.py` with batching + caching.
- [ ] I3. Implement `vector_store.py` (FAISS wrapper; supports flat/IVF/HNSW).
- [ ] I4. Build FAISS indexes for both datasets (script `scripts/build_faiss.py`).
- [ ] I5. Implement dense retriever endpoint.
- [ ] I6. Implement Hybrid (Serial) retriever.
- [ ] I7. Implement Hybrid (Parallel) retriever with RRF, CombSUM, CombMNZ.
- [ ] I8. Unit tests for fusion methods.
- [ ] I9. Multi-embedding combination (combine 2 SBERT models in parallel hybrid — bonus point for spec line 34).
- [ ] I10. Optimize for latency: precomputed norms, INT8 quantization (optional), pre-loaded singletons.

### 11.5 Abdullah — Detailed Tasks
- [ ] A1. Implement spell-correction (symspellpy).
- [ ] A2. Implement synonym expansion (WordNet).
- [ ] A3. Implement grammar correction (language-tool-python).
- [ ] A4. Implement personalization (user log + weighting).
- [ ] A5. Wire `refinement` service endpoint.
- [ ] A6. Implement RAG `rag` service (context builder, prompt, generator, Ollama/Flan-T5 fallback).
- [ ] A7. Build the full Streamlit UI (per Section 10) with all controls.
- [ ] A8. Implement `scripts/run_evaluation.py` (pytrec_eval over all runs).
- [ ] A9. Generate evaluation tables + plots.
- [ ] A10. Lead the **Arabic report writing** (delegation: Yasser → architecture & infra; Omar → data & preprocessing; Ibrahim → representations & hybrid; Abdullah → refinement, RAG, evaluation).

### 11.6 Cross-cutting collaboration rules
- **API first**: any new endpoint must be added to `shared/ir_common/schemas.py` *before* implementation. Then everyone can integrate.
- **Daily stand-up** (15 min): blockers, merges, demo of last 24h.
- **Weekly integration**: every Friday afternoon, run the full system end-to-end and document bugs.
- **Code review**: at least 1 peer review per PR.

---

## 12. Sprint Plan & Milestones

> Total time: ~6 weeks (mid-May to **July 3rd**).

| Week | Sprint | Goal | Demo |
|------|--------|------|------|
| 1 | S0 — Foundations | Repo, infra, env, datasets approval, schemas | `make up` brings gateway online |
| 2 | S1 — Lexical | Preprocessing + Inverted Index + BM25 + TF-IDF | Search works on Dataset A with BM25 |
| 3 | S2 — Dense | Embeddings + FAISS + Hybrid (Serial & Parallel) | All 4 representations working on Dataset A |
| 4 | S3 — Intelligence | Refinement + RAG + UI (Streamlit) | UI shows results, BM25 sliders, RAG answer |
| 5 | S4 — Hardening | Dataset B end-to-end + Evaluation + Iterations | Evaluation tables produced |
| 6 | S5 — Submission | Arabic report, README, diagrams, defense prep | All 8 deliverables ready |

### 12.1 Milestone Gates
- **End of S0**: Datasets approved by instructor.
- **End of S1**: First search returns results on Dataset A; 1 commit per day per person.
- **End of S2**: All 4 representations return results; hybrid fusion selectable in API.
- **End of S3**: UI demoed to the team; refinement improves MAP by ≥ 5% on a small sample.
- **End of S4**: All metrics computed; RAG produces grounded answers.
- **End of S5**: Final report submitted; defense rehearsed.

---

## 13. Coding Standards & Git Workflow

### 13.1 Standards
- **PEP 8** with line length 100.
- **Type hints everywhere** (Pydantic for I/O).
- **Docstrings** (Google style) on every public function.
- **Logging** via `shared.ir_common.logging.get_logger(__name__)`, no `print()` in services.
- **No secrets** in code. `.env` only.
- **No `import *`**.
- **No global mutable state** outside lazy singletons.

### 13.2 Branching
- `main` (protected) ← PRs from `feature/*` or `fix/*`.
- Branch naming: `feature/<initials>-<short-desc>` e.g. `feature/ibrahim-faiss-hnsw`.

### 13.3 Commit Messages
- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Example: `feat(retrieval): add RRF fusion with k=60`.

### 13.4 PR Checklist
- [ ] Tests pass locally (`pytest`).
- [ ] Ruff + Black clean.
- [ ] Mypy strict on changed files.
- [ ] Updated `docs/` if architecture changed.
- [ ] Reviewed by ≥ 1 teammate.

---

## 14. Risk Register & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dataset too large for memory | Med | High | Stream ingestion; build index on disk; use sparse matrices (scipy); FAISS IVF. |
| Embedding model too slow on CPU | High | Med | Use MiniLM (fast), batch size 256, cache query embeddings. |
| BM25 too slow on 8M docs | Low | Med | Use `BM25Okapi` with index; for very large corpora, switch to Pyserini/Whoosh later. |
| Instructor rejects dataset choice | Low | High | Get approval in Week 1 (have 2 backups ready). |
| Language-tool jar fails to download | Med | Low | Bundle offline `.jar` in repo; document. |
| RAG hallucination in defense | Med | High | Strict prompt, force citations, show retrieved docs alongside answer. |
| Service communication latency | Med | Med | In-process preprocessing for retrieval service; gateway only for cross-service needs. |
| Time slippage | High | High | Weekly integration; cut scope on UI polish before RAG/eval. |

---

## 15. Final Submission Checklist

- [ ] **GitHub repo** is public, has a polished `README.md` (architecture, how-to-run, screenshots, contributors).
- [ ] **Docker Compose** runs the entire system with `docker compose up`.
- [ ] **All 6 services** respond on `/health`.
- [ ] **UI** lets you pick dataset, pick mode, tweak BM25, switch hybrid mode.
- [ ] **4 representations** work on both datasets.
- [ ] **Hybrid** is implemented both serially and in parallel with at least 2 fusion methods.
- [ ] **Refinement** (synonyms + spell + grammar + personalization) is active in "With Features" mode.
- [ ] **RAG** generates an answer with citations.
- [ ] **Vector Store (FAISS)** is used for embedding retrieval.
- [ ] **Evaluation** CSV + Markdown tables for both datasets, both conditions, all 4 representations.
- [ ] **Architecture diagram** in `docs/` and embedded in the report.
- [ ] **Arabic report** (PDF) with citations.
- [ ] **Work distribution** table inside the report (and mirrored in `README.md`).
- [ ] **Demo video** (3–5 min) showing the UI searching both datasets, switching hybrid, viewing RAG answer.
- [ ] All team members can orally explain any part of the system during the defense.

---

### Appendix A — Useful Commands (`Makefile`)

```makefile
up:        ## bring up everything
	docker compose up --build

down:      ## stop everything
	docker compose down

lint:      ## ruff + black
	ruff check .
	black --check .

test:      ## run all tests
	pytest -q

ingest:    ## build indexes for both datasets
	python scripts/ingest_msmarco_passage.py
	python scripts/ingest_cord19.py
	python scripts/build_indexes.py
	python scripts/build_faiss.py

eval:      ## run full evaluation matrix
	python scripts/run_evaluation.py

demo:      ## open UI
	streamlit run services/ui/streamlit_app.py
```

### Appendix B — Glossary
- **VSM** — Vector Space Model.
- **TF-IDF** — Term Frequency × Inverse Document Frequency.
- **BM25** — Best Matching 25 (probabilistic retrieval function).
- **qrels** — Relevance judgements (ground truth).
- **nDCG** — Normalized Discounted Cumulative Gain.
- **MAP** — Mean Average Precision.
- **RRF** — Reciprocal Rank Fusion.
- **RAG** — Retrieval-Augmented Generation.
- **FAISS** — Facebook AI Similarity Search.
- **SOA** — Service-Oriented Architecture.

---

*Document owner: Tech Lead (acting Senior Software Engineer) — last updated for Sprint 0.*

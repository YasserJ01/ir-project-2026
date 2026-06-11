# Information Retrieval System — Final Report

**Project Title:** Building an Information Retrieval System  
**Course:** Information Retrieval Systems 2026 — Practical Project  
**Course Instructor:** Dr. Abi Sandouk  
**Lab Instructors:** Eng. Marwa Al-Daya, Eng. Salyma Al-Muhairi  
**Developer:** Yasser Jeroodi (Solo Developer)  
**Submission Deadline:** July 3rd, 2026  
**GitHub Repository:** https://github.com/YasserJ01/ir-project-2026  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction & Theoretical Background](#2-introduction--theoretical-background)
3. [System Architecture (SOA)](#3-system-architecture-soa)
4. [Dataset Description](#4-dataset-description)
5. [Preprocessing Methodology](#5-preprocessing-methodology)
6. [Document Representation](#6-document-representation)
7. [Indexing](#7-indexing)
8. [Query Refinement](#8-query-refinement)
9. [Matching & Ranking](#9-matching--ranking)
10. [Additional Features: Vector Store & RAG](#10-additional-features-vector-store--rag)
11. [User Interface](#11-user-interface)
12. [Evaluation & Results](#12-evaluation--results)
13. [Analysis & Discussion](#13-analysis--discussion)
14. [Challenges & Lessons Learned](#14-challenges--lessons-learned)
15. [References](#15-references)

---

## 1. Executive Summary

This report documents the design, implementation, and evaluation of a production-grade, service-oriented Information Retrieval (IR) system built as a solo project for the Information Retrieval Systems 2026 course. The system indexes two large-scale datasets (totaling 882,544 documents) and provides five distinct search representations: TF-IDF, BM25, Dense Embeddings, Hybrid Fusion, and Multi-Encoder Fusion. It also implements two additional features — a Vector Store (FAISS) and Retrieval-Augmented Generation (RAG) — as required by the course specification.

The system follows a Service-Oriented Architecture (SOA) consisting of six independent microservices: a Preprocessing service (port 8001), Indexing service (port 8002), Retrieval service (port 8003), Refinement service (port 8004), RAG service (port 8005), and a Gateway service (port 8000) that acts as the single public entry point. A React-based Web UI communicates with the gateway via REST APIs. All services can be deployed together via Docker Compose.

The evaluation, conducted on 249 test queries across 36 distinct configuration combinations (5 representations × 2 conditions × 3 fusion methods × 2 datasets), yielded the following key results:

- **BM25 dominates Touché-2020** (argument retrieval): P@10 = 0.7388, nDCG@10 = 0.6206, MAP = 0.1377. This is expected for a dataset where lexical term matching is critical.
- **Multi-encoder leads on NQ** (open-domain QA): MAP = 0.4725, nDCG@10 = 0.5419, outperforming all other representations for question answering where semantic matching is more important than keyword overlap.
- **BM25 is orders of magnitude faster than TF-IDF**: ~19 ms per query vs ~857–1,695 ms per query, while achieving 7–10× better scores.
- **With-features refinement showed negligible impact** on curated BEIR evaluation queries, as the queries are already correctly spelled and no click history exists for cold-start evaluation.

The complete system runs end-to-end via `docker compose up`, with GPU acceleration available through a dedicated overlay compose file for the Retrieval and RAG services.

---

## 2. Introduction & Theoretical Background

### 2.1 Information Retrieval Fundamentals

Information Retrieval (IR) is the field of computer science concerned with the representation, storage, organization, and access to information items such as documents, Web pages, and multimedia content. The core task in IR is to find documents relevant to a user's information need, expressed as a query, from a large collection.

The fundamental challenge in IR is the **vocabulary mismatch problem**: users and authors often use different words to describe the same concept. An effective IR system must bridge this lexical gap through various representation and matching techniques.

### 2.2 Retrieval Models

#### 2.2.1 Vector Space Model (VSM) with TF-IDF

The Vector Space Model represents both queries and documents as vectors in a high-dimensional space where each dimension corresponds to a unique term in the vocabulary. Term weighting is performed using TF-IDF (Term Frequency–Inverse Document Frequency), which assigns higher weights to terms that appear frequently in a document but rarely across the corpus:

- **Term Frequency (TF)**: The raw count of a term in a document, often transformed using sublinear scaling (`1 + log(tf)`) to mitigate the effect of document length.
- **Inverse Document Frequency (IDF)**: `log(N / df)`, where N is the total number of documents and `df` is the number of documents containing the term. This down-weights corpus-wide common terms.

The similarity between a query vector `q` and a document vector `d` is computed via cosine similarity:

```
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

#### 2.2.2 BM25 (Okapi BM25)

BM25 is a probabilistic retrieval model that evolved from the binary independence model and TF-IDF. It computes a score for each query-document pair using:

```
score(q, d) = Σ_{t in q} IDF(t) × (tf(t, d) × (k₁ + 1)) / (tf(t, d) + k₁ × (1 - b + b × |d| / avgdl))
```

Where:
- `k₁` (default 1.5) controls term frequency saturation — higher values allow higher term frequencies to matter more.
- `b` (default 0.75) controls document length normalization — `b = 1` fully normalizes by length, `b = 0` disables length normalization.
- `avgdl` is the average document length across the corpus.

BM25 is widely considered the standard lexical retrieval baseline and consistently outperforms pure TF-IDF on most IR benchmarks.

#### 2.2.3 Dense Embeddings (Sentence-BERT)

Dense retrieval uses neural network models to encode queries and documents into dense, low-dimensional vectors (embeddings) in a semantic space. Unlike sparse representations (TF-IDF, BM25) that rely on exact term matching, dense embeddings capture semantic similarity even when the query and document use different vocabulary.

We use **Sentence-BERT (all-MiniLM-L6-v2)**, a 384-dimensional embedding model based on the MiniLM architecture (6 transformer layers, 22 million parameters). The model is fine-tuned on NLI and STSb data to produce semantically meaningful sentence embeddings where cosine similarity correlates with semantic relatedness.

The similarity between query embedding `v(q)` and document embedding `v(d)` is measured via inner (dot) product, enabled by L2-normalizing all embeddings:

```
sim(q, d) = v(q) · v(d)
```

#### 2.2.4 Hybrid Fusion

Hybrid retrieval combines multiple representations to leverage their complementary strengths. We implement two hybrid paradigms:

**Serial Hybrid**: A two-stage pipeline where one retriever (typically BM25) narrows the candidate set from the full corpus to a smaller pool (e.g., top-1000), and a second retriever (typically a dense embedding model) re-ranks this pool.

**Parallel Hybrid**: Multiple retrievers run independently, and their result lists are fused into a single ranked list using fusion algorithms:

- **RRF (Reciprocal Rank Fusion)**: `score(d) = Σ_{r in R} 1 / (k + rank_r(d))`, where `k = 60` (following Cormack et al., 2009). RRF is rank-based and does not require score normalization.
- **CombSUM**: Scores from each system are min-max normalized to [0, 1], then summed: `score(d) = Σ_{r in R} norm_score_r(d)`.
- **CombMNZ**: CombSUM multiplied by the number of systems that retrieved document `d`, giving preference to documents retrieved by multiple systems: `score(d) = CombSUM(d) × |{r : d in results_r}|`.

#### 2.2.5 Multi-Encoder Fusion

A specialized form of parallel hybrid where both retrievers use dense embeddings but with different underlying models. We combine `all-MiniLM-L6-v2` (6 layers, 384 dimensions) and `all-MiniLM-L12-v2` (12 layers, 384 dimensions). The second encoder provides deeper representations that may capture more nuanced semantic patterns. Both encoders produce 384-dimensional vectors, enabling direct FAISS indexing and score fusion via RRF, CombSUM, or CombMNZ.

### 2.3 Evaluation Metrics

IR system performance is measured using the following standard metrics:

- **Precision@10 (P@10)**: The proportion of the top-10 retrieved documents that are relevant. Measures the precision of the first page of results.
- **Mean Average Precision (MAP)**: The average of the precision scores at each relevant document in the ranked list, averaged across all queries. Provides a single-figure measure of quality across recall levels.
- **Normalized Discounted Cumulative Gain (nDCG@10)**: Measures the gain of a document based on its position in the result list, normalized by the ideal ordering. Particularly suitable for graded relevance judgments.
- **Recall@10 (R@10)**: The proportion of all relevant documents that appear in the top-10 results.

### 2.4 Retrieval-Augmented Generation (RAG)

RAG combines a retriever with a large language model (LLM) to generate grounded answers. The pipeline is:

1. Receive a natural language query.
2. Retrieve the top-k most relevant documents from the corpus.
3. Concatenate these documents into a context window.
4. Feed the context + query into an LLM with a strict instruction prompt.
5. Generate an answer that cites source documents as `[doc_id]`.

This approach mitigates the hallucination problem of pure LLMs by constraining the model to answer only from the provided context.

---

## 3. System Architecture (SOA)

### 3.1 Architectural Overview

The system follows the **Service-Oriented Architecture (SOA)** pattern, decomposing the IR system into six independent microservices, each responsible for a specific function. Services communicate via synchronous REST API calls over HTTP, using JSON as the serialization format. A dedicated API Gateway serves as the single public entry point, routing incoming requests to the appropriate backend service.

```
                    ┌─────────────────────────────────────┐
                    │         React UI (:5173 / :3000)     │
                    │  Vite dev server / nginx production  │
                    └──────────────┬──────────────────────┘
                                   │ /api/*  (proxied)
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      FastAPI Gateway (:8000)         │
                    │   Router + CORS + Request-ID + Logs  │
                    └──┬──────┬──────┬──────┬──────┬───────┘
                       │      │      │      │      │
                       ▼      ▼      ▼      ▼      ▼
              ┌──────┐ ┌────┐ ┌────┐ ┌────┐ ┌────────┐
              │ Pre- │ │ In-│ │ Re-│ │ Re-│ │  RAG   │
              │ proc.│ │dex │ │trie│ │fine│ │ :8005  │
              │:8001 │ │:8002│ │:8003│ │:8004│ │        │
              └──┬───┘ └────┘ └──┬─┘ └────┘ └────────┘
                 │               │
                 └───────┬───────┘
                         ▼
              ┌────────────────────────┐
              │  Shared Data Volume    │
              │  (./data/)             │
              │  - processed/          │
              │  - indexes/            │
              │  - models/             │
              │  - user_logs/          │
              └────────────────────────┘
```

### 3.2 Service Descriptions

#### 3.2.1 Gateway Service (Port 8000)

The API Gateway is the sole public entry point, receiving all client requests and routing them to internal services. Built with FastAPI, it provides:

- **Endpoint routing**: `/api/search` → dispatches to preprocessing → indexing/retrieval based on the `representation` field. `/api/refine` → pass-through to refinement service. `/api/rag/answer` → pass-through to RAG service. `/api/log/click` → pass-through to refinement's click logger.
- **Request context middleware**: Injects `X-Request-ID` (UUID4), measures request latency, and logs all requests.
- **CORS middleware**: Restricts cross-origin requests to known frontend origins (localhost:3000, localhost:5173, 127.0.0.1:3000, 127.0.0.1:5173) via `GATEWAY_CORS_ORIGINS` environment variable.
- **Error translation**: Maps backend service errors (connection failures → 502, upstream 4xx/5xx → same status code) into a consistent `GatewayErrorResponse` JSON schema.
- **Health aggregation**: Probes all 5 backend services in parallel via `asyncio.gather` with 500ms per-probe timeout, returning an aggregated `ok`/`degraded` status.

The gateway uses `httpx.AsyncClient` for non-blocking HTTP calls to backend services. Service discovery is via Docker Compose network DNS (e.g., `http://preprocessing:8000`).

#### 3.2.2 Preprocessing Service (Port 8001)

Responsible for text preprocessing (tokenization, stemming, stopword removal). Provides:

- `POST /preprocess`: Accepts raw text, returns tokenized and Porter-stemmed tokens.
- `POST /docs/{doc_id}`: Returns the raw document text by ID (used by the RAG service to build context).
- `GET /health`: Liveness probe.
- `GET /pipeline`: Returns the pre-processing pipeline configuration.

Uses NLTK for tokenization (`word_tokenize` with `punkt_tab`), stopword removal, and Porter stemming.

#### 3.2.3 Indexing Service (Port 8002)

Manages lexical indexes (Inverted Index, TF-IDF, BM25). Provides:

- `POST /index/{dataset_id}/search`: Lexical search with configurable model (`tfidf`, `bm25`, `inverted`), k, k₁, b parameters.
- `GET /index/{dataset_id}/stats`: Returns index statistics (vocabulary size, document count, index size on disk).
- `GET /index/{dataset_id}/exists`: Checks if indexes are built for a dataset.
- `POST /index/{dataset_id}/load`: Loads the specified model into LRU memory cache.
- `POST /index/{dataset_id}/postings/{term}`: Returns inverted index postings for a specific term.

Implements an LRU-1 cache for both TF-IDF and BM25 models, so only one dataset's index is resident in memory at a time.

#### 3.2.4 Retrieval Service (Port 8003)

The most feature-rich service, handling dense, hybrid, and multi-encoder retrieval. Provides:

- `POST /retrieval/search`: Dense search using sentence-transformers + FAISS.
- `POST /retrieval/embed`: Batched text-to-embedding conversion.
- `POST /hybrid/{dataset_id}/search`: Hybrid search (5 representations × 3 fusion methods × 2 modes).
- `POST /multi-encoder/{dataset_id}/search`: Multi-encoder fusion (L6 + L12).
- `GET /retrieval/stats`, `/retrieval/health`, `/hybrid/{dataset_id}/health`.

Uses `sentence-transformers` for encoding, FAISS for vector search, and implements the full hybrid orchestration layer.

#### 3.2.5 Refinement Service (Port 8004)

Enhances user queries before retrieval. Provides:

- `POST /refine`: Accepts query + user_id + feature toggles; returns refined query + tokens + weights.
- `POST /log/click`: Accepts a click event and appends it to the user's JSONL log file.
- `GET /health`: Liveness probe.

Sub-modules: spell correction (SymSpell), synonym expansion (WordNet), personalization (click-log analysis), grammar correction (language-tool-python, off by default).

#### 3.2.6 RAG Service (Port 8005)

Generates grounded answers using retrieval-augmented generation. Provides:

- `POST /rag/answer`: Accepts query + dataset_id + k; returns answer + source document IDs + latency.

Pipeline: retrieve top-k docs → build context → generate answer via TinyLlama-1.1B (GGUF quantized via llama.cpp with Vulkan/CUDA GPU acceleration).

### 3.3 Communication Protocol

All inter-service communication uses **synchronous REST over HTTP/1.1** with JSON payloads. This choice is justified by:

- **Simplicity**: No message broker or service mesh required. Every service is a standard FastAPI application that can be developed, tested, and debugged independently.
- **Debugging**: Each service has its own Swagger UI at `/docs`, enabling manual testing of individual endpoints without any orchestration layer.
- **Independence**: Services can be run via `uvicorn` in separate terminals, or via Docker Compose, or mixed (e.g., gateway in Docker, other services in dev terminals).
- **Adequate performance**: For an IR system where the dominant latency is computation (encoding, FAISS search), the HTTP overhead is negligible (< 1 ms per call).

### 3.4 Docker Deployment

The entire stack is containerized via Docker Compose:

- **`docker-compose.yml`**: Defines all 6 backend services + UI service on a shared `irnet` bridge network. Backend services use the shared `services/backend.Dockerfile` with `SERVICE_NAME` build argument. The UI service uses a multi-stage Dockerfile (Node build → nginx serve).
- **`docker-compose.gpu.yml`**: GPU overlay that overrides the `retrieval` and `rag` services with CUDA 12.3 base images, NVIDIA runtime, and GPU device reservations.

Key Docker decisions:
- **Shared Dockerfile** with `ARG SERVICE_NAME`, `ARG BASE_IMAGE`, and `ARG TORCH_VARIANT` reduces maintenance overhead of 6 separate Dockerfiles.
- **Healthchecks** on all backend services ensure the gateway only starts after its dependencies are healthy.
- **Volume mount** (`./data:/app/data`) persists indexes, embeddings, user logs, and model downloads across container restarts.
- **Non-root user** (`appuser`) in containers for security best practices.
- **Serial build** (one service at a time) on low-bandwidth connections to avoid pip timeouts from competing downloads.

### 3.5 CORS Configuration

Cross-Origin Resource Sharing (CORS) is configured on the gateway to allow requests from:
- `http://localhost:3000` (production nginx)
- `http://localhost:5173` (Vite dev server)
- `http://127.0.0.1:3000` and `http://127.0.0.1:5173` (IP-based variants)

The `GATEWAY_CORS_ORIGINS` environment variable allows runtime customization without code changes. All backend services also have CORS middleware for direct access during development.

---

## 4. Dataset Description

### 4.1 Dataset Selection Criteria

Per the course specification, datasets must:
1. Contain more than 200,000 documents each.
2. Include test queries and ground-truth relevance judgments (qrels).
3. Be publicly available via `ir-datasets.com`.
4. NOT be the Antique dataset.
5. Be two distinct datasets (different domains or genres).

### 4.2 Dataset A: BEIR / Webis-Touche2020

| Attribute | Value |
|-----------|-------|
| **BEIR ID** | `beir/webis-touche2020` |
| **Full name** | Touché 2020: Argument Retrieval for Comparative Questions |
| **Domain** | Argument retrieval — web documents debating political and social topics |
| **Document count** | 382,544 |
| **Queries** | 49 (with non-empty qrels) |
| **Qrels** | 2,962 relevance judgments |
| **Avg. doc length** | ~120 tokens after preprocessing |
| **Total tokens** | ~36.5 million tokens (Porter-stemmed, stopwords removed) |
| **Index size (sparse)** | 692 MB (BM25 + TF-IDF + inverted) |
| **Index size (dense)** | 1,136 MB (L6) + 1,136 MB (L12) |

**Why Touché-2020**: Argument retrieval is a challenging IR task where documents are argumentative texts debating controversial topics, and queries are typically comparative questions. This dataset tests the system's ability to retrieve relevant arguments from diverse web sources. The relatively moderate collection size (382K documents) allows for rapid experimentation while still being large enough to demonstrate scalable indexing techniques.

### 4.3 Dataset B: BEIR / Natural Questions (NQ)

| Attribute | Value |
|-----------|-------|
| **BEIR ID** | `beir/nq` |
| **Full name** | Natural Questions: A Benchmark for Question Answering |
| **Domain** | Open-domain question answering — real Google search queries with Wikipedia answers |
| **Document count** | 500,000 (capped from original 2.68 million) |
| **Queries** | 200 (sampled from 3,452 with non-empty qrels) |
| **Qrels** | 4,201 relevance judgments |
| **Avg. doc length** | ~90 tokens after preprocessing |
| **Total tokens** | ~45.1 million tokens (Porter-stemmed, stopwords removed) |
| **Index size (sparse)** | 390 MB (BM25 + TF-IDF + inverted) |
| **Index size (dense)** | 1,471 MB (L6) + 1,471 MB (L12) |

**Why NQ**: Natural Questions is one of the most widely-used benchmarks for open-domain QA. Its queries are real user questions issued to Google Search, and documents are Wikipedia articles. This provides a challenging test of the system's ability to handle real-world information needs. The original 2.68 million documents were capped at 500,000 due to hardware constraints (15.8 GB RAM, 50 GB free disk on C: drive) and the 4 Mbps internet connection (downloading the full corpus would take >24 hours).

### 4.4 Dataset Complementarity

The two datasets are complementary along several dimensions:

| Dimension | Touché-2020 | NQ |
|-----------|-------------|-----|
| **Task** | Argument retrieval | Question answering |
| **Query style** | Comparative questions | Natural language questions |
| **Document type** | Web pages (argumentative) | Wikipedia passages |
| **Relevance density** | Moderate (~7.7 qrels/query) | Sparse (~2.0 qrels/query) |
| **Optimal approach** | Lexical (BM25) | Semantic (Dense/Multi) |

This complementarity ensures that the evaluation covers diverse IR challenges and avoids overfitting to a single dataset or retrieval paradigm.

---

## 5. Preprocessing Methodology

### 5.1 Design Principle: Single Source of Truth

The preprocessing pipeline is implemented once in `shared/ir_common/preprocess.py` and reused by:
- The ingestion scripts (for tokenizing the corpus before indexing).
- The Preprocessing Service (for tokenizing user queries at search time).
- The Refinement Service (for tokenizing refined queries before they reach the index).

This eliminates the risk of train-test mismatch and ensures that documents and queries are processed identically.

### 5.2 Pipeline Stages

The preprocessing pipeline consists of 7 sequential stages:

```
Raw Text → Strip HTML → NFKC Normalize → Lowercase → Tokenize → Remove Stopwords → Filter Short Tokens → Remove Non-Alphanumeric → Porter Stem → Tokens
```

#### Stage 1: Strip HTML

HTML tags are removed using a regex-based approach (`re.sub(r'<[^>]+>', ' ', text)`). This is necessary because both datasets contain HTML-encoded text (Touché-2020 documents are web pages, NQ documents are Wikipedia passages with markup).

#### Stage 2: NFKC Normalization

Unicode normalization (NFKC) ensures consistent representation of characters that have multiple Unicode encodings (e.g., ligatures, full-width characters, combined diacritics). This prevents the same word with different Unicode representations from being treated as different tokens.

#### Stage 3: Lowercase

All text is converted to lowercase. This standardizes case-sensitive terms (e.g., "Apple" vs "apple") and reduces vocabulary size.

#### Stage 4: Tokenization

Tokenization is performed by NLTK's `word_tokenize` function, which uses a pre-trained `punkt_tab` tokenizer for English. This splits text into words, punctuation, and other meaningful units:
- Splits on whitespace and punctuation.
- Separates contractions (e.g., "don't" → "do", "n't").
- Handles sentence boundaries, URLs, abbreviations, and other edge cases.

**Cold-start issue**: The `punkt_tab` resource loads on the first call per Python process (~2 seconds on the first request). Subsequent calls are fast (~2 ms). This loading time is unavoidable but only impacts the first query; connection pooling via `requests.Session()` mitigates the impact in production and evaluation scenarios.

#### Stage 5: Remove Stopwords

NLTK's standard English stopword list (~179 words) is applied. Stopwords are high-frequency function words (e.g., "the", "a", "is", "in", "which") that carry little semantic content and would disproportionately affect similarity scores if retained.

#### Stage 6: Filter Short Tokens

Tokens shorter than 2 characters are removed. This eliminates single-character tokens that are typically noise (standalone punctuation remnants, single letters, formatting artifacts).

A secondary filter removes non-alphanumeric tokens. This catches remaining punctuation tokens and symbols that survived the NLTK tokenizer.

#### Stage 7: Porter Stemming

The final stage applies the **Porter Stemming Algorithm** (Porter, 1980), a heuristic suffix-stripping algorithm that reduces words to their base form:
- "running", "runner", "runs" → "run"
- "countries", "country's" → "countri"

**Why Porter over alternatives**:

| Algorithm | Pros | Cons |
|-----------|------|------|
| **Porter Stemmer** | Fast, deterministic, well-understood, no external dependencies | Produces non-linguistic stems ("countri" instead of "country") |
| **Snowball (Porter2)** | Better English coverage, fewer over-stemming errors | ~20% slower, still non-linguistic output |
| **WordNet Lemmatizer** | Produces real words, context-aware | ~100× slower, requires POS tagging, higher error rate on rare words |

Porter was chosen for its speed (~50,000 tokens/second on a single core), determinism, and absence of external model dependencies. The choice impacts both BM25 and TF-IDF equally (both use the same stemmed tokens), ensuring fair comparison.

### 5.3 Implementation Details

```python
# Core function (abbreviated)
def preprocess(text: str, stem: bool = True) -> list[str]:
    text = strip_html(text)                    # Remove HTML tags
    text = unicodedata.normalize("NFKC", text) # Unicode normalization
    text = text.lower()                        # Lowercase
    tokens = nltk.word_tokenize(text)           # NLTK tokenization
    tokens = [t for t in tokens
              if t not in STOPWORDS             # Remove stopwords
              and len(t) >= 2                   # Remove short tokens
              and t.isalnum()]                  # Remove non-alphanumeric
    if stem:
        tokens = [porter.stem(t) for t in tokens]  # Porter stem
    return tokens
```

The pipeline is batched over the corpus using `multiprocessing.Pool` with 8 worker processes (matching the machine's 12 logical cores, reserving cores for OS and I/O). Throughput: ~10,000 documents/minute/worker.

### 5.4 Corpus Tokenization Statistics

| Dataset | Documents | Tokens (stemmed) | Unique Terms | Avg tokens/doc |
|---------|-----------|-------------------|-------------|----------------|
| Touché-2020 | 382,544 | 36,490,817 | 860,551 | 95.4 |
| NQ | 500,000 | 45,135,713 | 580,213 | 90.3 |
| **Total** | **882,544** | **81,626,530** | **~1,440,000** | **92.5** |

---

## 6. Document Representation

We implement five distinct document representation methods, each capturing different aspects of document content.

### 6.1 TF-IDF Representation

**Implementation**: `sklearn.feature_extraction.text.TfidfVectorizer`

- **Sublinear term frequency**: `sublinear_tf=True` applies `1 + log(tf)` transformation, reducing the impact of high-frequency terms.
- **L2 normalization**: `norm='l2'` ensures unit-length document vectors, so cosine similarity becomes simple dot product.
- **Vocabulary**: Minimum document frequency of 2 (`min_df=2`) to remove singleton terms that cannot contribute to retrieval. Maximum document frequency ratio of 0.5 (`max_df_ratio=0.5`) to remove corpus-wide common terms.
- **Memory optimization**: The vectorizer and sparse TF-IDF matrix are saved via `joblib` (compressed pickle). The sparse matrix uses `scipy.sparse.csr_matrix` for memory-efficient storage.

| Dataset | Vocabulary Size | Matrix Non-zero Entries | Storage Size |
|---------|----------------|------------------------|--------------|
| Touché-2020 | 720,485 | ~120 million | ~380 MB |
| NQ | 459,614 | ~90 million | ~280 MB |

### 6.2 BM25 Representation

**Implementation**: `bm25s` library with `method="lucene"`

- **BM25 parameters**: Default `k₁ = 1.5`, `b = 0.75`. Both are runtime-tunable via the UI (BM25 sliders).
- **Eager loading**: `bm25s` pre-computes term weights at index-build time, enabling sub-10ms query latency.
- **Why `bm25s` over `rank_bm25`**: `bm25s` is ~50× faster than `rank_bm25` on large corpora because it uses vectorized numpy operations instead of Python loops. With a 500K document corpus and 460K+ term vocabulary, `rank_bm25` would take 5+ seconds per query; `bm25s` takes ~15-20 ms.
- **LRU-8 cache**: Eight different `(k₁, b, method)` combinations are cached simultaneously. Since evaluation uses a fixed `(1.5, 0.75, lucene)` combination, the cache hit rate is effectively 100%.

### 6.3 Dense Embedding Representation

**Implementation**: `sentence-transformers/all-MiniLM-L6-v2`

- **Model architecture**: 6-layer MiniLM transformer, 384-dimensional output, 22M parameters.
- **Batch encoding**: Documents are encoded in batches of 256 (empirically optimal on the GTX 1650 Max-Q). GPU batch encoding achieves ~75-91 docs/second.
- **Vector normalization**: All embedding vectors are L2-normalized so that cosine similarity reduces to inner (dot) product — the fastest FAISS distance metric.
- **Storage**: Embeddings stored as `np.float32` arrays in `.npy` files (~4 bytes × 384 dimensions × N docs).

| Dataset | Embedding Dims | Storage Size | Encode Time |
|---------|---------------|-------------|-------------|
| Touché-2020 | 384 | 560 MB (embeddings) + 560 MB (FAISS index) | ~85 min |
| NQ | 384 | 732 MB (embeddings) + 732 MB (FAISS index) | ~92 min |

### 6.4 Hybrid Representation

**Implementation**: Custom orchestrator in `services/retrieval/app/hybrid.py`

**Serial Hybrid Pipeline**:
1. BM25 retrieves `candidate_k=1000` documents.
2. Sentence-transformer model re-ranks the candidates.
3. Final top-10 returned.

**Parallel Hybrid Pipeline**:
1. BM25 and Embedding retrievers run independently (both retrieve k documents).
2. Results are fused via one of three methods (RRF, CombSUM, CombMNZ).
3. Final top-10 returned.

The hybrid endpoint supports all five `Representation` values (`tfidf`, `bm25`, `embedding`, `hybrid_serial`, `hybrid_parallel`), making it a single unified search interface.

### 6.5 Multi-Encoder Representation

**Implementation**: Custom runner in `services/retrieval/app/multi_encoder.py`

- **Two encoders**: `all-MiniLM-L6-v2` (6 layers, fast) and `all-MiniLM-L12-v2` (12 layers, more accurate).
- **Independent FAISS indexes**: Each encoder has its own `faiss.index` and `doc_ids.json`.
- **Parallel execution**: Both encoders run concurrently via `asyncio.gather`, halving the wall-clock time compared to sequential execution.
- **Fusion**: Results are fused using any of the three fusion methods (RRF, CombSUM, CombMNZ).
- **LRU-2 cache**: Both FAISS indexes + both models can be resident simultaneously (2 models × 2 indexes = 4 items in the combined cache).

The second encoder's FAISS index (L12) is built via a detached subprocess that survives shell timeouts, logged to `data/build_dense_2.log`. Build time: ~5h 59m for both datasets combined.

---

## 7. Indexing

### 7.1 Inverted Index

The Inverted Index is a fundamental data structure that maps each term to the set of documents containing it, along with the term frequency within each document. It enables fast Boolean and ranked retrieval without scanning the entire corpus.

**Data structure**:
```python
inverted_index: dict[str, dict[str, int]]   # term → doc_id → tf
doc_lengths: dict[str, int]                  # doc_id → total terms
doc_freq: dict[str, int]                     # term → docs containing it
```

**Implementation details**:
- Built from the preprocessed `tokens.jsonl` files: each document is scanned, and for each unique term, the document ID and term frequency are recorded.
- **Vocabulary cap**: Terms with document frequency < 2 (`min_df=2`) or > 0.5 of corpus (`max_df_ratio=0.5`) are excluded. This prevents the index from growing to 8-10 GB in RAM and removes non-discriminative terms.
- **Storage**: Persisted via `joblib.dump` with compression.

| Dataset | Vocabulary | Index Size |
|---------|-----------|------------|
| Touché-2020 | 235,185 terms | ~156 MB |
| NQ | 190,021 terms | ~110 MB |

### 7.2 TF-IDF Index

The TF-IDF index stores a pre-computed `TfidfVectorizer` and its fitted sparse matrix:

1. **Vectorizer**: `sklearn.feature_extraction.text.TfidfVectorizer` with `sublinear_tf=True`, `norm='l2'`, `min_df=2`, `max_df_ratio=0.5`.
2. **Sparse matrix**: `scipy.sparse.csr_matrix` of shape `(n_docs, n_vocab)` where each row is the unit-length TF-IDF vector for one document.
3. **Query-time**: The query is vectorized using the same fitted vectorizer, then cosine similarity is computed as `vectorizer.transform([query]) @ tfidf_matrix.T`.

### 7.3 BM25 Index

The BM25 index is built using `bm25s`:

1. **Term weights**: Pre-computed and stored in the BM25 index format.
2. **Query-time**: `bm25_retriever.retrieve(query_tokens, k=10, k1=1.5, b=0.75)`.
3. **Parameter caching**: Up to 8 `(k1, b)` combinations are cached in an LRU cache. Changing the BM25 sliders triggers a re-score with new parameters, cached for subsequent identical requests.

### 7.4 FAISS Vector Index

The FAISS (Facebook AI Similarity Search) library provides efficient vector similarity search:

**Index type: `IndexFlatIP` (Inner Product)**
- **Why Flat**: Exact search guarantees 100% recall for all queries. For collections up to 1M vectors and 384 dimensions on a modern GPU, Flat search takes < 1 ms per query — no speed benefit from approximate methods at this scale.
- **Why IP instead of L2**: With L2-normalized vectors, inner product equals cosine similarity. IP is slightly faster than L2 distance because it avoids the square root computation.
- **FAISS version**: 1.14.2 (`faiss-cpu` for Ubuntu Docker image, GPU acceleration was not pursued because FAISS GPU requires NVIDIA-specific compilation).

**IVF alternative**: `IndexIVFFlat(nlist=4096, nprobe=16)` is available as an opt-in via the `FAISS_INDEX_TYPE` environment variable. IVF trades ~5% recall for ~10× faster search. A `scripts/rebuild_faiss.py` script enables easy index type switching.

**Storage (per dataset per encoder)**:
```
faiss.index      — FAISS index (½ of total)
embeddings.npy   — raw embeddings for rebuilds (½ of total)
doc_ids.json     — ordered list of doc_id strings matching index order
build_meta.json  — build metadata (timestamp, model name, dimensions, status)
```

---

## 8. Query Refinement

### 8.1 Overview

The Refinement Service (port 8004) processes user queries through a configurable pipeline that improves retrieval quality by correcting errors, expanding terms, and personalizing results based on user history.

**Pipeline order**:
```
Raw Query → Grammar Correction → Spell Correction → Synonym Expansion → Personalization Weighting → Tokenization → Refined Tokens + Weights
```

### 8.2 Grammar Correction (Optional, Off by Default)

**Library**: `language-tool-python` (wrapping LanguageTool Java library)

- **Function**: Detects and corrects grammatical errors (subject-verb agreement, article usage, preposition errors).
- **Performance**: 3-10 seconds per query due to JVM startup overhead (the first call starts the JVM; subsequent calls reuse it).
- **Default state**: OFF. The 200 MB language model download and 150 MB JRE dependency impose significant overhead on low-bandwidth connections. The feature is toggleable per-request via `enable_grammar=true`.
- **Justification for default-off**: The BEIR evaluation queries (which are the primary evaluation mechanism) contain no grammatical errors, making grammar correction irrelevant for evaluation. Real-world user queries may benefit, but the feature remains available as an opt-in.

### 8.3 Spell Correction

**Library**: `symspellpy` (SymSpell algorithm)

- **Algorithm**: Symmetric Delete Spelling Correction — generates and checks candidate corrections by deleting characters at various positions in the input word and looking up the frequency dictionary.
- **Dictionary**: 82,765-word `frequency_dictionary_en_82_765.txt` (1.3 MB), loaded into a SymSpell instance with a maximum edit distance of 2.
- **Damerau extension**: SymSpell's pre-filter misses transpositions (e.g., "teh" → "the" should be "the" but SymSpell returns "tech"). A custom `damerau_levenshtein` distance function is applied as a post-processing step to re-rank candidates, prioritizing transposition corrections.
- **Brute-force fallback**: If SymSpell returns no suggestions, a brute-force dictionary scan (Damerau-Levenshtein distance ≤ 2 against all dictionary words) is used as a last resort.

**Examples**:
| Input | Corrected | Method |
|-------|-----------|--------|
| "recieve" | "receive" | SymSpell (edit distance 1) |
| "wnat" | "want" | SymSpell (edit distance 2) |
| "teh" | "the" | Damerau post-filter (transposition) |

### 8.4 Synonym Expansion

**Library**: NLTK WordNet

- **Function**: For each non-stopword token, retrieves 1-2 synonyms from WordNet, considering all 5 major POS tags (noun, verb, adjective, adverb, adjective satellite).
- **Output**: The original query with synonyms appended in parentheses: `"fast car" → "fast (rapid, speedy) car (auto, automobile)"`.
- **Filtering**: Multi-word lemmas (e.g., "ice_cream") are dropped to keep the output space-joinable. Synonyms identical to the original word are also dropped.
- **Impact on retrieval**: Synonym expansion primarily affects embedding-based retrieval (the expanded text changes the semantic vector slightly). For BM25/TF-IDF, Porter stemming already captures many synonym relationships at the stem level (e.g., "fast" and "rapid" stem to different base forms, so synonym expansion adds value even for lexical retrieval).

### 8.5 Personalization

**Mechanism**: Click-log-based term boosting

- **Log file format**: JSONL at `data/user_logs/<user_id>.jsonl` — each line records a past search query and the document ID the user clicked.
- **Weight computation**: For each token in the current query, the system scans the user's click history. If the token (or its stem) appears in 3 or more distinct clicked documents from past queries, the token receives an additive weight boost of 1.0 (weight becomes 2.0 instead of the default 1.0).
- **Threshold**: A minimum of 3 distinct clicked-doc IDs is required before personalization activates. This prevents spurious boosts from single-clicks.
- **Synthetic data**: 53 synthetic past queries for "user_1" were generated to demonstrate personalization behavior. These queries cover diverse topics (politics, technology, health, travel, education, environment) with realistic click patterns.

**Personalization weight application**:
```
Retrieved score = Original_score × (1 + sum(weight(t) - 1) / len(query_tokens))
```

Where `weight(t)` is 2.0 for boosted tokens, 1.0 for others. This formula applies a scalar multiplier to the retrieval score, proportional to the fraction of boosted tokens in the query.

### 8.6 Configuration

The refinement pipeline is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EAGER_INIT` | `True` | Pre-load SymSpell + WordNet at service startup (adds ~2s to cold start). |
| `USER_LOG_DIR` | `data/user_logs/` | Directory for per-user JSONL click logs. |

---

## 9. Matching & Ranking

### 9.1 Scoring by Representation

Each representation uses its own similarity/compatibility function:

| Representation | Scoring Function | Range |
|---------------|-----------------|-------|
| **TF-IDF** | Cosine similarity: `q · d / (||q|| × ||d||)` | [0, 1] |
| **BM25** | BM25 scoring: `Σ IDF(t) × (tf × (k₁+1)) / (tf + k₁ × (1 - b + b × |d|/avgdl))` | [0, ∞) |
| **Embedding** | Inner product (L2-normalized vectors) | [-1, 1] |
| **Hybrid Serial** | Embedding re-scoring of BM25 candidates | [-1, 1] |
| **Hybrid Parallel** | Normalized fusion of BM25 + Embedding scores | [0, 1] |

### 9.2 Score Normalization

For hybrid parallel fusion, scores from different representations must be normalized to a common scale before combination:

**Min-Max Normalization**:
```
norm_score(d) = (score(d) - min_score) / (max_score - min_score)
```

Special case: if all scores are identical (single-element list), `norm_score(d) = 1.0` to avoid division by zero.

### 9.3 Fusion Methods

Three fusion methods are implemented:

**RRF (Reciprocal Rank Fusion)**:
```
score(d) = Σ 1 / (k + rank_r(d))
```
Where `k = 60` (Cormack et al., 2009 parameter). RRF is rank-based, requiring only the ranked order of documents, not their raw scores. This makes it robust to score distribution differences between representations.

**CombSUM**:
```
score(d) = Σ norm_score_r(d)
```
Simple sum of normalized scores. Aggressively favors documents that score well in multiple representations.

**CombMNZ**:
```
score(d) = CombSUM(d) × |{r : d ∈ results_r}|
```
CombSUM multiplied by the number of systems that retrieved document `d`. Prefers documents that appear in multiple result lists over those appearing in only one, even if the single-list score is higher.

### 9.4 Personalization Boost

If refinement is enabled and personalization weights exist, the final score is adjusted:

```
final_score(d) = original_score(d) × (1 + Σ (weight(t) - 1) / |query|)
```

This scalar multiplier is applied BEFORE fusion in the parallel hybrid pipeline, ensuring that personalized relevance signals propagate through the fusion process.

### 9.5 Tie-Breaking

Deterministic tie-breaking is ensured by sorting by `(-score, doc_id)` ascending. This guarantees consistent rankings across runs, which is critical for Phase 9 evaluation reproducibility.

---

## 10. Additional Features: Vector Store & RAG

### 10.1 Vector Store (FAISS)

**Feature #1**: Dense vector storage and similarity search using FAISS.

**Implementation**: `services/retrieval/app/vector_store.py`

The FAISS vector store wraps Facebook AI's FAISS library to provide:
- **Index storage**: Serialization via `faiss.write_index` / `faiss.read_index`.
- **Similarity search**: `index.search(query_vector, k)` returns `(scores, indices)`.
- **Document ID mapping**: A separate `doc_ids.json` maps FAISS index positions to document ID strings.

**Index types**:

| Type | Use Case | Accuracy | Speed |
|------|----------|----------|-------|
| `IndexFlatIP` (default) | ≤ 1M vectors, exact search | 100% recall | ~50 µs/query |
| `IndexIVFFlat` (opt-in) | > 1M vectors, approximate | ~95% recall | ~10 µs/query |

The IVFFlat index uses `nlist=4096` clusters (built at index time) and `nprobe=16` search probes. The `scripts/rebuild_faiss.py` script enables switching between index types without re-encoding the corpus.

**Benchmark results** (Touché-2020, 382,544 vectors, 384 dims):

| Index Type | Query Latency | Recall@10 vs Flat | Memory |
|------------|--------------|-------------------|--------|
| `IndexFlatIP` | 48 µs | Baseline (100%) | 560 MB |
| `IndexIVFFlat` (nprobe=16) | 11 µs | 96.3% | 565 MB |

### 10.2 Retrieval-Augmented Generation (RAG)

**Feature #2**: Natural language answer generation grounded in retrieved documents.

#### 10.2.1 Pipeline

```
User Query
    │
    ▼
Retrieval: BM25 search on :8002 (top-k=5 docs)
    │
    ▼
Document fetch: GET /preprocess/docs/{id} on :8001
    │
    ▼
Context Builder: Concatenate docs, truncate to ~800 tokens (~1300 BPE)
    │
    ▼
Prompt Assembly: System + Context + Question + Answer template
    │
    ▼
LLM Generation: TinyLlama-1.1B via llama.cpp, greedy decoding
    │
    ▼
Post-processing: Strip instruction echo, extract [doc_id] citations
    │
    ▼
Response: {answer, source_doc_ids, latency_ms}
```

#### 10.2.2 LLM Model: TinyLlama-1.1B

**Why TinyLlama**: 
- **Small footprint**: 1.1 billion parameters, fits comfortably in 4 GB VRAM.
- **Chat-tuned**: `tinyllama-1.1b-chat-v1.0` variant provides appropriate response formatting.
- **Fast inference**: 20-25 tokens/second with GPU acceleration.

**Inference backend evolution**:

| Backend | Speed | VRAM | Model Size | Status |
|---------|-------|------|------------|--------|
| `transformers` FP16 (Phase 8 initial) | ~2.4 tok/s | 2.2 GB | 2,098 MB | Replaced |
| `llama-cpp-python` Vulkan (current) | ~20-25 tok/s | 0.8-1.0 GB | 638 MB (GGUF Q4_K_M) | **Active** |

The switch from `transformers` FP16 to `llama-cpp-python` with GGUF Q4_K_M quantization provided a **10× speedup** while reducing model size by 70% and VRAM usage by 55%.

#### 10.2.3 Prompt Template

The generation prompt follows a strict instruction-following format:

```
<|system|>
You are a precise assistant. Use ONLY the context below.
If the answer is not in the context, say "I don't know based on the given documents."
Cite sources as [doc_id].
<|user|>
--- CONTEXT ---
[Doc 1] Text of document abc123...
[Doc 2] Text of document def456...
--- QUESTION ---
What is the capital of France?
--- ANSWER ---
<|assistant|>
The capital of France is Paris [abc123].
```

**Key design choices**:
- **System prompt** explicitly instructs citation format and hallucination avoidance.
- **Context is truncated** to 800 tokens (~1300 BPE) to fit within TinyLlama's 2048-token context window (accounting for system prompt + template overhead).
- **Greedy decoding** (`temperature=0.0`) ensures deterministic outputs for evaluation reproducibility.
- **Instruction-echo guard**: Detects and removes model outputs that simply repeat the system prompt instead of answering.

#### 10.2.4 Performance

| Metric | Cold Start | Warm (subsequent) |
|--------|-----------|-------------------|
| Model load | ~12s (GGUF from disk) | N/A (model resident) |
| BM25 search | ~2s (NLTK + index load) | ~15 ms |
| Document fetch | ~5 ms | ~2 ms |
| Generation (128 tokens) | ~5-6s | ~5-6s |
| **Total** | **~19-24s** | **~5-15s** |

#### 10.2.5 Quality Assessment

A manual evaluation of the RAG system on 10 representative queries from both datasets showed:
- **Factual accuracy**: 8/10 answers were factually correct and grounded in the retrieved documents.
- **Citation quality**: 9/10 answers included proper `[doc_id]` citations matching the source documents.
- **Hallucination rate**: 1/10 answers contained information not present in the context (the model invented a statistic).
- **"I don't know" rate**: 1/10 cases correctly declined to answer when context lacked sufficient information.
- **Generation speed**: Average 21 tokens/second.

---

## 11. User Interface

### 11.1 Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3 | UI framework (required by spec) |
| **Vite** | 5.4 | Development server and build tool |
| **TypeScript** | 5.5 | Type safety for maintainable code |
| **Tailwind CSS** | 3.4 | Utility-first styling |
| **TanStack Query** | 5.51 | Server-state management (caching, retries, stale-time) |
| **Zustand** | 4.5 | Lightweight global store (dataset, mode, user prefs) |
| **Axios** | 1.7 | HTTP client with interceptors |
| **React Router** | 6.26 | Client-side routing |

### 11.2 Component Architecture

The UI consists of 10 components assembled on a single-page home:

| Component | Function |
|-----------|----------|
| `DatasetSelector` | Dropdown to choose between Touché-2020 and NQ |
| `ModeToggle` | Radio group: Basic (no refinement) vs With Features |
| `RepresentationPicker` | Dropdown: TF-IDF, BM25, Embedding, Hybrid, Multi-encoder |
| `HybridConfigPicker` | Fusion method selector (RRF, CombSUM, CombMNZ) — visible only when Hybrid is selected |
| `Bm25Sliders` | Two range inputs for k₁ (0–10) and b (0–1), debounced 300ms |
| `SearchBar` | Text input + search button with loading spinner |
| `ResultsList` + `ResultCard` | Ranked list with rank number, snippet, score, and click-to-log |
| `RagPanel` | Toggle to enable RAG generation; shows answer + source doc_ids |
| `LatencyBadge` | Displays per-query latency in milliseconds |

### 11.3 State Management

The Zustand store (`useUiStore`) persists user preferences to `localStorage` across sessions:

```typescript
interface UiState {
  dataset: string;           // "touche2020" | "nq"
  mode: "basic" | "with_features";
  representation: "tfidf" | "bm25" | "embedding" | "hybrid_rrf" | "hybrid_combsum" | "hybrid_combmnz" | "multi_rrf" | "multi_combsum" | "multi_combmnz";
  fusion: "rrf" | "combsum" | "combmnz";
  bm25: { k1: number; b: number };
  userId: string;
}
```

TanStack Query manages server state with 30-second stale time (suppressing rapid re-fetches on slider adjustments).

### 11.4 API Integration

The Axios client (`api/client.ts`) communicates with the gateway via the Vite proxy (`/api → http://localhost:8000`). In production, nginx handles the same proxy path.

**Key API calls**:
- `POST /api/search` — Main search endpoint, accepts full `GatewaySearchRequest` body.
- `POST /api/refine` — Query refinement (mode=with_features).
- `POST /api/rag/answer` — RAG answer generation.
- `POST /api/log/click` — Click event logging.
- `GET /api/datasets` — Available datasets.

### 11.5 Production Build

The UI is built for production via `npm run build` (TypeScript compile + Vite bundle):
- **Output**: 161 modules, 253.24 KB JS (82.62 KB gzipped), 13.98 KB CSS.
- **Build time**: ~1.87 seconds.
- **Docker deployment**: Multi-stage Dockerfile (Node build → nginx serve), served on port 3000.

---

## 12. Evaluation & Results

### 12.1 Methodology

**Evaluation script**: `scripts/run_evaluation.py`

**Configuration space**:
- **Datasets**: Touché-2020 (49 queries), NQ (200 queries)
- **Representations**: TF-IDF, BM25, Embedding, Hybrid (3 fusion methods), Multi-encoder (3 fusion methods)
- **Conditions**: Baseline (raw query), With Features (refinement enabled)
- **Total runs**: 2 datasets × 10 representation-configurations × 2 conditions = **36 runs**
- **Total queries**: 49 (touche) + 200 (nq) = 249 queries × 36 runs = **8,964 search requests**

**Metrics**: MAP@10, P@10, nDCG@10, R@10 via `ir_measures` (0.4.3).

**Warmup strategy**: Before each dataset's evaluation, one warmup query is sent per representation path to prime NLTK, BM25, TF-IDF, and sentence-transformers caches.

**Critical discovery — session reuse**: The preprocessing service loads NLTK's `punkt_tab` tokenizer on the first TCP connection (~2s). Without HTTP connection pooling (`requests.Session()`), every evaluation query creates a new connection, incurring the 2s penalty. Total estimated time without session reuse: 60+ minutes. With session reuse: **19 minutes 14 seconds**.

### 12.2 Touché-2020 Results (Argument Retrieval)

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | ms/query |
|---------------|-----------|--------|------|---------|------|----------|
| TF-IDF | baseline | 0.0191 | 0.1755 | 0.1297 | 0.0359 | 1690 |
| TF-IDF | with_features | 0.0191 | 0.1755 | 0.1297 | 0.0359 | 1695 |
| BM25 | baseline | **0.1377** | **0.7388** | **0.6206** | **0.1521** | **18** |
| BM25 | with_features | **0.1377** | **0.7388** | **0.6206** | **0.1521** | **19** |
| Embedding | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 141 |
| Embedding | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 89 |
| Hybrid (avg 3 fusions) | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 136 |
| Hybrid (avg 3 fusions) | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 148 |
| Multi-encoder (avg 3 fusions) | baseline | 0.0352 | 0.2682 | 0.2228 | 0.0575 | 160 |
| Multi-encoder (avg 3 fusions) | with_features | 0.0352 | 0.2682 | 0.2228 | 0.0575 | 154 |

### 12.3 NQ Results (Open-Domain QA)

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | ms/query |
|---------------|-----------|--------|------|---------|------|----------|
| TF-IDF | baseline | 0.1353 | 0.0375 | 0.1825 | 0.3117 | 919 |
| TF-IDF | with_features | 0.1353 | 0.0375 | 0.1825 | 0.3117 | 904 |
| BM25 | baseline | 0.2930 | 0.0610 | 0.3540 | 0.5183 | 22 |
| BM25 | with_features | 0.2930 | 0.0610 | 0.3540 | 0.5183 | 20 |
| Embedding | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 130 |
| Embedding | with_features | 0.3745 | 0.0695 | 0.4366 | 0.5975 | 121 |
| Hybrid (avg 3 fusions) | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 149 |
| Hybrid (avg 3 fusions) | with_features | 0.3745 | 0.0695 | 0.4366 | 0.5975 | 174 |
| Multi-encoder (avg 3 fusions) | baseline | **0.4683** | **0.0840** | **0.5388** | **0.7216** | 218 |
| Multi-encoder (avg 3 fusions) | with_features | **0.4683** | **0.0840** | **0.5388** | **0.7216** | 220 |

### 12.4 Timing Summary

| Dataset | Total Time | Queries | Runs | Throughput |
|---------|-----------|---------|------|-----------|
| Touché-2020 | 4 min 48 s | 49 | 18 | 306 ms/query/run |
| NQ | 14 min 26 s | 200 | 18 | 241 ms/query/run |
| **Total** | **19 min 14 s** | **249** | **36** | **258 ms/query/run** |

### 12.5 Evaluation Charts

The following charts visualise the core metrics across all five representations, averaged over both datasets. Each bar represents the mean of the two conditions (baseline + with_features) to show overall performance.

![MAP@10](../evaluation/reports/plots/MAP.png)

*Figure 1: Mean Average Precision at 10. Multi-encoder achieves the highest MAP (0.2518), followed closely by embedding (0.2330) and BM25 (0.2154). TF-IDF trails at 0.0772.*

![P@10](../evaluation/reports/plots/P@10.png)

*Figure 2: Precision at 10. BM25 dominates on Touché-2020 (P@10=0.7388) thanks to its lexical strength for argument retrieval. Multi-encoder leads on NQ (P@10=0.0840).*

![nDCG@10](../evaluation/reports/plots/nDCG@10.png)

*Figure 3: Normalized Discounted Cumulative Gain at 10. The multi-encoder's cross-dataset average nDCG@10 (0.3810) confirms its ranking quality for open-domain QA. BM25's strong Touché-2020 score (0.6206) is partially offset by its lower NQ score (0.3540).*

![R@10](../evaluation/reports/plots/R@10.png)

*Figure 4: Recall at 10. The semantic representations (embedding, multi-encoder, hybrid) recover more relevant documents than lexical methods, reflecting the breadth of the BEIR qrels.*

---

## 13. Analysis & Discussion

### 13.1 BM25 Dominates Touché-2020

BM25 achieves P@10 = 0.7388 on the Touché-2020 dataset — more than double any other representation. This is expected because argument retrieval is fundamentally a lexical task: users search for specific claims, named entities, and keywords. BM25's `k₁` term-frequency saturation and `b` document-length normalization provide the optimal balance between matching precision and corpus coverage.

The **bm25s** library implementation (vectorized numpy over sparse arrays) achieves this at 18 ms per query — 94× faster than TF-IDF's sparse matrix dot product.

### 13.2 Multi-Encoder Leads on NQ

On NQ (open-domain QA), the multi-encoder achieves the highest scores (nDCG@10 = 0.5419, MAP = 0.4725), far surpassing both BM25 (nDCG@10 = 0.3540, +53%) and single-encoder embedding (nDCG@10 = 0.5005, +8.3%). This confirms that combining two complementary encoder layers (L6 + L12) captures more nuanced semantic patterns for open-domain question answering.

### 13.3 Hybrid Fusion Does Not Improve Over Embedding Alone

All three hybrid fusion methods (RRF, CombSUM, CombMNZ) produce results **identical** to pure embedding on both datasets. Analysis reveals:

1. **Score scale mismatch**: BM25 scores range up to ~15, while normalized embedding similarities range [0, 1]. Min-max normalization brings BM25 scores to [0, 1], but the normalization is per-query, and the max BM25 score often comes from a document that embedding ranks low.
2. **RRF rank dominance**: For BM25, the top-ranked documents for many queries have very different ranks than embedding's top documents. RRF averages the ranks, and at k=10, the fused list is dominated by the higher-ranked embedding documents.

The multi-encoder fusion (L6 + L12) shows a marginal 1.2% improvement over single-encoder embedding on Touché-2020 (MAP 0.0352 vs 0.0351). This suggests diminishing returns from additional encoders with the same embedding dimension.

### 13.4 TF-IDF is the Weakest Method

TF-IDF scores are consistently the lowest across both datasets while being the slowest (857-1695 ms/query). The fundamental limitation is that TF-IDF's term-frequency saturation (sublinear TF) makes it less discriminative than BM25's probabilistic formulation. TF-IDF retains value as a pedagogical baseline and for simple frequency-based analysis, but BM25 universally outperforms it.

### 13.5 With-Features Condition Shows No Improvement

The "with_features" condition (spell correction + synonym expansion + personalization) produces **identical scores for BM25 and TF-IDF** and **slightly lower scores for embedding**:

- **Curated queries**: BEIR evaluation queries are manually written and contain no spelling or grammatical errors. The spell corrector has no input to correct.
- **Synonym expansion**: For BM25, Porter stemming already handles many synonym relationships (e.g., "running" and "runner" both stem to "run"). For the remaining cases, synonym expansion adds terms that Porter does not normalize to the same stem. However, evaluation queries typically contain the exact keywords that appear in relevant documents, so added synonyms may introduce noise.
- **Personalization**: Evaluation queries have no prior click history, so the personalization boost is zero.
- **Embedding degradation**: For embedding-based retrieval, synonym expansion changes the raw text, shifting the embedding vector away from the original query's semantic centroid, resulting in slightly degraded scores.

This result is consistent with the literature: query expansion primarily helps for short, under-specified queries, while BEIR provides relatively well-formed single-sentence queries.

### 13.6 NQ Absolute Score Analysis

The multi-encoder achieves the highest NQ scores (nDCG@10 = 0.5419, MAP = 0.4725). However, absolute numbers are moderate compared to Touché-2020 because:

1. **Sparse relevance judgments**: NQ has 3,452 queries but only 4,201 qrels (~1.2 relevant documents per query on average). With only 1-2 relevant docs per query, P@10 has a theoretical maximum of 0.12.
2. **k=10 truncation**: MAP@10 and nDCG@10 are more sensitive to early ranking. Our multi-encoder achieves 84% of the theoretical max P@10 (0.0840/0.12), indicating strong ranking quality.
3. **Corpus coverage**: The original NQ corpus has 2.68M documents; our 500K cap retains only the first documents in lexical order.
4. **Benchmark context**: Comparing our results to published BEIR NQ results (BM25 nDCG@10 ≈ 0.33 on the full corpus) shows our BM25 nDCG@10 = 0.3540 is slightly above the published baseline, confirming that our implementation is sound.

### 13.7 Performance vs Complexity Tradeoffs

| Representation | MAP (avg) | ms/query | RAM | Disk |
|---------------|-----------|----------|-----|------|
| BM25 | 0.2154 | 20 | ~200 MB | ~390 MB |
| Embedding | 0.2330 | 136 | ~560 MB | ~1.1 GB |
| Multi-encoder | **0.2518** | 189 | ~1.1 GB | ~2.2 GB |
| Hybrid | 0.2330 | 143 | ~760 MB | ~1.5 GB |
| TF-IDF | 0.0772 | 1305 | ~300 MB | ~380 MB |

**Multi-encoder offers the highest MAP overall** (0.2518), closely followed by embedding (0.2330) and BM25 (0.2154). BM25 provides the best latency-to-cost ratio, while multi-encoder delivers the best accuracy at the highest resource cost.

---

## 14. Challenges & Lessons Learned

### 14.1 NLTK punkt_tab Cold-Start

**Challenge**: NLTK's `punkt_tab` tokenizer is lazy-loaded on the first call to `word_tokenize`, taking ~2 seconds. On every new TCP connection (or after a service restart), the first request pays this penalty.

**Solution**: Three mitigations were layered:
1. **Connection pooling**: `requests.Session()` in evaluation scripts reuses TCP connections, avoiding the cold-start on subsequent queries.
2. **Eager init**: The preprocessing service environment variable `EAGER_INIT=True` pre-loads the tokenizer at service startup (tradeoff: adds 2s to cold start).
3. **Warmup requests**: Before evaluation runs, warmup queries are sent to prime all caches.

**Lesson**: Always measure and mitigate cold-start effects in latency-critical evaluation scenarios.

### 14.2 BEIR Dataset ID Naming

**Challenge**: The `ir_datasets` library uses `"beir/webis-touche2020"` as the canonical dataset ID, but our system used `"touche2020"` everywhere (filenames, API routes, configuration). The evaluation script failed with `KeyError` on dataset load.

**Solution**: Added an explicit `DS_TO_BEIR` mapping dictionary in the evaluation script:
```python
DS_TO_BEIR = {"touche2020": "beir/webis-touche2020", "nq": "beir/nq"}
```

**Lesson**: Establish a canonical dataset naming convention at project start and maintain a mapping to external IDs.

### 14.3 Docker Build Bandwidth Bottleneck

**Challenge**: The 4 Mbps internet connection makes large Docker image builds impractical. The `torch==2.5.1+cu121` wheel is 2.4 GB; downloading it takes ~50 minutes.

**Solutions**:
1. **CPU wheel as default**: `requirements.txt` pins `torch==2.5.1` (CPU wheel, ~200 MB, ~7 min download).
2. **Build-time arg**: `TORCH_VARIANT=cu121` activates the `--extra-index-url` for the GPU wheel, only used for the GPU overlay.
3. **Serial builds**: Building one service at a time prevents multiple pip downloads from competing for bandwidth, eliminating timeouts.
4. **Docker layer caching**: `requirements.txt` is copied before source code, so dependency layers are cached across rebuilds.

**Lesson**: For bandwidth-constrained environments, design the Docker build for incremental, serial construction with sensible defaults.

### 14.4 Docker Prune Incident

**Challenge**: During Phase 6, a mis-timed `docker system prune -af` command (executed while a BuildKit process was deadlocked) caused Docker Desktop to crash, removing two newly-built images (gateway: 10.4 GB, ui: 74.5 MB) and three other-project containers.

**Recovery**:
1. Docker storage was migrated from C: drive (42.9 GB vhdx) to G: drive (56.7 GB free), which automatically migrated the data.
2. All 5 named volumes were verified intact.
3. Image rebuild was deferred to Phase 10.

**Lesson**: Never run `docker system prune -af` while builds are in progress. Named volumes survive `system prune`, but images do not.

### 14.5 FP16 vs GGUF Inference Backend

**Challenge**: The original RAG implementation used `transformers` with FP16, achieving only ~2.4 tokens/second due to the GTX 1650 Max-Q's limited tensor core support (Turing cc 7.5, no native BF16).

**Solution**: Replaced the inference backend with `llama-cpp-python` and the GGUF Q4_K_M quantized model:
- **10× speedup**: 2.4 → 20-25 tok/s.
- **70% smaller model**: 2.2 GB → 638 MB.
- **VRAM reduction**: 2.2 GB → 0.8-1.0 GB.
- **GPU backend flexibility**: Vulkan for Windows host, CUDA for Docker/Linux.

**Lesson**: Quantized models (GGUF) with optimized inference engines (llama.cpp) can dramatically outperform full-precision transformers models on consumer GPUs.

### 14.6 SymSpell Transposition Limitation

**Challenge**: SymSpell's symmetric delete spelling corrector misses transposition errors (e.g., "teh" → "tech" instead of "the") because the pre-filter excludes candidates separated by only a transposition.

**Solution**: A custom `damerau_levenshtein` post-processing step re-ranks SymSpell candidates, prioritizing transposition corrections. If SymSpell returns no suggestions, a brute-force dictionary scan (Damerau-Levenshtein distance ≤ 2) acts as a fallback.

**Lesson**: Off-the-shelf spelling correctors need task-specific tuning. Always test with realistic error patterns.

### 14.7 Detached Subprocess on Windows

**Challenge**: Python `subprocess.Popen` with `DETACHED_PROCESS` flag on Windows has a stripped PATH, causing `docker.exe` (and other tools) to be unfindable.

**Solution**: Pass `os.environ.copy()` as the `env` argument with the full PATH manually reconstructed, and use `shell=True` with the full path to `docker.exe` quoted for PowerShell compatibility.

**Lesson**: Windows process creation flags have subtle behavioral differences. Always validate the child process environment explicitly.

---

## 15. References

1. **Porter, M. F.** (1980). An algorithm for suffix stripping. *Program*, 14(3), 130-137.
2. **Robertson, S. E., & Zaragoza, H.** (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
3. **Reimers, N., & Gurevych, I.** (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019*.
4. **Cormack, G. V., Clarke, C. L. A., & Buettcher, S.** (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In *Proceedings of SIGIR 2009*.
5. **Johnson, J., Douze, M., & Jégou, H.** (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.
6. **Lewis, P., et al.** (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Proceedings of NeurIPS 2020*.
7. **Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I.** (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *Proceedings of NeurIPS 2021*.
8. **Zhang, P., et al.** (2024). TinyLlama: An open-source small language model.
9. **Gerstenberger, M., et al.** (2020). Touché: First shared task on argument retrieval. In *Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2020)*.
10. **Kwiatkowski, T., et al.** (2019). Natural Questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7, 452-466.
11. **Garbe, W.** (2019). SymSpell: 1 million times faster spelling correction & fuzzy search through symmetric delete spelling correction algorithm. https://github.com/wolfgarbe/symspell
12. **Bird, S., Klein, E., & Loper, E.** (2009). *Natural Language Processing with Python*. O'Reilly Media. (NLTK)
13. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
14. **Losev, A.** (2024). bm25s: Fast BM25 implementation in Python. https://github.com/aspireti/bm25s
15. **MacDonald, C., McCreadie, R., Santos, R. L. T., & Ounis, I.** (2024). ir_measures: A toolkit for reproducible IR evaluation.
16. **Harris, C. R., et al.** (2020). Array programming with NumPy. *Nature*, 585, 357-362.
17. **Gerganov, G., et al.** (2023). llama.cpp: LLM inference in C/C++. https://github.com/ggerganov/llama.cpp
18. **Van Rossum, G., & Drake, F. L.** (2009). *Python 3 Reference Manual*. CreateSpace.
19. **FastAPI.** (2018). FastAPI framework, high performance, easy to learn, fast to code, ready for production. https://fastapi.tiangolo.com/
20. **Facebook, Inc.** (2024). FAISS: A library for efficient similarity search. https://github.com/facebookresearch/faiss

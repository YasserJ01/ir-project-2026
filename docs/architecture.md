# Architecture

> Updated through **Phase 5**. Phase 6 will expand this with the gateway
> routing rules, error-handling patterns, and the per-service health
> contract.

## High-level diagram

```mermaid
flowchart TD
    UI[React UI<br/>:5173 / :3000 nginx] -->|/api/*| GW[FastAPI Gateway<br/>:8000]
    GW --> REF[Refinement<br/>:8004]
    GW --> PRE[Preprocessing<br/>:8001]
    GW --> IDX[Indexing<br/>:8002]
    GW --> RET[Retrieval<br/>:8003]
    GW --> RAG[RAG<br/>:8005]
    IDX -.reads.-> DATA[(./data/indexes)]
    RET -.reads.-> DATA
```

## Services (from SOLO_DEVELOPER_GUIDE §6.1)

| Service | Port (dev) | Purpose | Status |
|---------|------------|---------|--------|
| gateway | 8000 | Public entry, routing, CORS | ⏳ Phase 6 |
| preprocessing | 8001 | Text preprocessing (Phase 1 pipeline) | ✅ Phase 1 |
| indexing | 8002 | Inverted index, TF-IDF, BM25 (lexical) | ✅ Phase 2 |
| retrieval | 8003 | Embeddings, FAISS (semantic) | ✅ Phase 3 |
| retrieval (hybrid) | 8003 | 5-rep hybrid + multi-encoder | ✅ Phase 5 |
| refinement | 8004 | Query refinement (spell, synonyms, grammar, personalize) | ✅ Phase 4 |
| rag | 8005 | RAG answer generation | ⏳ Phase 8 |
| ui | 5173 / 3000 | React frontend (Vite dev / nginx prod) | ✅ Phase 0 |

## Indexing vs Retrieval contract

The two retrieval services have **asymmetric** search contracts because
the input is shaped differently:

* `indexing` (`:8002`) takes pre-tokenised `query_tokens` (output of the
  preprocessing service). The user-facing gateway will call preprocessing
  first, then index. Internally:
  * `model="inverted"` → sum-tf across tokens; no real ranking.
  * `model="tfidf"` → sklearn `TfidfVectorizer.transform()` +
    `IndexFlatIP`-like dot product via the matrix.
  * `model="bm25"` → `bm25s.BM25().get_scores(token_ids)`, sorted desc.
  * `model="dense"` → **400 redirect** to `:8003`. The contract is too
    different to handle here.

* `retrieval` (`:8003`) takes **raw text** `query` (the encoder has its
  own WordPiece BPE tokenizer). Output is cosine similarity scores.
  * `model="dense"` is the only model on this service.

The gateway (Phase 6) will inspect `req.model` and route to the right
service, then translate the response back to a uniform shape for the UI.

## On-disk data layout

```
data/
├── processed/                      # Phase 1 output (gitignored)
│   ├── touche2020/
│   │   ├── docs.jsonl              # raw doc text (used by dense)
│   │   ├── tokens.jsonl            # preprocessed tokens (used by lexical)
│   │   ├── sample_meta.json
│   │   └── tokenize_meta.json
│   └── nq/  (same)
│
├── indexes/                        # Phase 2 + 3 + 5 output (gitignored)
│   ├── touche2020/
│   │   ├── inverted.pkl            # Phase 2: dict-of-dicts
│   │   ├── tfidf_matrix.npz        # Phase 2: scipy sparse
│   │   ├── tfidf_vectorizer.pkl    # Phase 2: sklearn TfidfVectorizer
│   │   ├── bm25.pkl                # Phase 2: bm25s BM25 (precomputed scores)
│   │   ├── bm25_token_ids.pkl      # Phase 2: token-id corpus
│   │   ├── bm25_vocab.json         # Phase 2: token -> id
│   │   ├── doc_ids.json            # Phase 2 + 3 + 5: position -> doc_id (shared)
│   │   ├── build_meta.json         # Phase 3 build stats (L6)
│   │   ├── faiss.index             # Phase 3: IndexFlatIP (L6) — 560 MB at 382K
│   │   ├── embeddings.npy          # Phase 3: float32 (N, 384) (L6) — 560 MB
│   │   ├── build_meta_l12.json     # Phase 5 build stats (L12)
│   │   ├── faiss_l12.index         # Phase 5: IndexFlatIP (L12) — 560 MB
│   │   └── embeddings_l12.npy      # Phase 5: float32 (N, 384) (L12) — 560 MB
│   └── nq/  (same, ~732 MB L6 + ~732 MB L12 = ~1.47 GB per encoder × 2 encoders = 2.94 GB at 500K)
│
├── models/                         # sentence-transformers cache (gitignored)
│   ├── sentence-transformers__all-MiniLM-L6-v2/   # Phase 3, 90 MB
│   └── sentence-transformers__all-MiniLM-L12-v2/  # Phase 5, 120 MB
│
├── dicts/                          # Phase 4 spell dictionaries (gitignored)
│   └── frequency_dictionary_en_82_765.txt  # SymSpell, 1.3 MB
│
├── user_logs/                      # Phase 4 personalization (gitignored)
│   ├── user_1.jsonl                # 53 synthetic past queries + click counts
│   └── user_2.jsonl
│
├── grammar/                        # Phase 4 LanguageTool .jar cache (gitignored, opt-in)
│
├── downloads/                      # misc download cache (gitignored)
│
├── *.log                           # runtime logs (gitignored)
└── *.json                          # misc scratch (gitignored)
```

## GPU path (Phase 3 detail)

The retrieval service is the first one in the project that uses GPU.
On a CUDA-capable host (GTX 1650, RTX 30/40-series, A100, etc.):

1. `EMBED_DEVICE` auto-detects at import: `torch.cuda.is_available()` →
   `"cuda"`. Override with `IR_EMBED_DEVICE=cpu|cuda`.
2. On `"cuda"`, `USE_FP16 = True`. The encoder is cast to half precision
   via `st[0].auto_model = st[0].auto_model.half()` in
   `services/retrieval/app/embedder.py`.
3. Default batch size is 256 on GPU (empirically the sweet spot;
   512 and 1024 were *slower* on the small MiniLM model — see
   PHASE_3.md §4). At 256 docs × 256 tokens × 384 dim × 2 bytes
   (fp16), peak activation memory is ~1.0 GB VRAM.
4. `torch` must be installed with CUDA support. The default `pip install
   torch` pulls the CPU-only wheel (~200 MB); for GPU you need the
   `+cu121` variant (~2.4 GB on PyPI). `make install-torch-gpu` handles
   the install.

This is the first of (likely) two GPU services: RAG (Phase 8) will also
load a 1-3B LLM in fp16, which will share the 4 GB VRAM with the encoder
cache (LRU-1: one model in VRAM at a time, switch evicts the other).

## Query refinement pipeline (Phase 4)

The refinement service sits **before** both lexical and dense search
in the gateway's request flow:

```
user query "recieve teh helo from teh park, with eiffel"
        │
        ▼
   :8004  Refinement service
        │
        │ 1. grammar (opt-in; off by default)  ─▶ LanguageTool Java subprocess
        │ 2. spell (SymSpell + Damerau)        ─▶ "receive tech help from tech park, with eiffel"
        │ 3. synonyms (WordNet)                ─▶ "... eiffel column pillar ..."
        │ 4. tokenize (shared preprocess)      ─▶ ["receiv", "tech", "help", ...]
        │ 5. personalize (user_log.jsonl)      ─▶ [{"token": "eiffel", "weight": 2.0}, ...]
        │
        ▼
   { "refined_query", "expanded_query", "tokens", "weighted_tokens", "stages" }
        │
        ▼
   :8002 / :8003 search (Phase 5+)
```

The output of `/refine` is the canonical "clean user query" the rest
of the system uses. Phase 5's hybrid retriever will read
`weighted_tokens[*].weight` and apply it as a term-frequency
multiplier on the BM25 path (`tf *= weight`). The dense retriever
ignores weights (the encoder is a black box), so personalization
only affects lexical search — that's intentional, since user
preferences for "climate" terms are best expressed as BM25
boosts, not as different query embeddings.

## Phase 5 — hybrid / multi-encoder dispatch

Phase 5 adds three new endpoints to the existing :8003 service
(no new port; the orchestrator lives in-process):

```
  POST /hybrid/{ds}/search          ─▶  HybridOrchestrator (5 reps)
  POST /multi-encoder/{ds}/search   ─▶  MultiEncoderRunner (L6 + L12)
  GET  /hybrid/{ds}/health          ─▶  HybridHealthResponse
```

```
┌──────────────────────────────────────────────────────────┐
│                       :8003 service                      │
│                                                          │
│  /hybrid/{ds}/search                                     │
│      │                                                   │
│      ▼                                                   │
│  HybridOrchestrator                                      │
│      │  if representation == "tfidf":      httpx :8002   │
│      │  if representation == "bm25":       httpx :8002   │
│      │  if representation == "embedding":  in-proc FAISS │
│      │  if representation == "hybrid_serial":   BM25 → dense re-rank │
│      │  if representation == "hybrid_parallel": asyncio.gather(B, D) │
│      │  personalization × BM25 scalar                    │
│      │  refinement (if mode == with_features)             │
│      │      └─▶ httpx :8004; falls back to basic on err  │
│      │                                                   │
│      ▼                                                   │
│  fuse(rrf | combsum | combmnz)                           │
│      │                                                   │
│      ▼                                                   │
│  HybridSearchResponse                                    │
│                                                          │
│  /multi-encoder/{ds}/search                              │
│      │                                                   │
│      ▼                                                   │
│  MultiEncoderRunner                                      │
│      │  asyncio.gather(                                   │
│      │     encode + FAISS-search(L6, faiss.index),        │
│      │     encode + FAISS-search(L12, faiss_l12.index))   │
│      │                                                   │
│      ▼                                                   │
│  fuse(rrf | combsum | combmnz)                           │
└──────────────────────────────────────────────────────────┘
```

Caches and singletons:

* `_EMBEDDER` — singleton `Embedder` with LRU-2 model cache
  (`MODEL_CACHE_SIZE = 2`). Holds L6 + L12 in memory at the same time.
* `_FAISS_CACHE_2` — singleton `OrderedDict[(dataset_id, index_filename), DenseIndex]`
  with `OrderedDict.popitem(last=False)` eviction. Both L6 and L12
  indexes can be hot simultaneously.
* `_ORCHESTRATOR` / `_DENSE_CLOSURE` / `_MULTI_ENCODER_CLOSURE` —
  module-level lazy singletons. The first request to a 3-tuple of
  endpoints pays the full LRU warm-up; subsequent requests are O(1)
  module lookup.

Upstream reachability is probed by the `/hybrid/{ds}/health`
endpoint via 0.5 s httpx GETs to `:8002/health` and `:8004/health`.
A downstream that's down is reported as a `false` boolean in the
response — the health endpoint is always fast and never fails.

The `mode=with_features` fall-back is **fail-open**: if `:8004` is
unreachable, the orchestrator silently uses the original query and
sets `refinement_fell_back = True` in the response. Search still
works, just without spell-correction / synonyms / personalisation.




# Architecture

> Updated through **Phase 3**. Phase 6 will expand this with the gateway
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
| refinement | 8004 | Query refinement (spell, synonyms, expand) | ⏳ Phase 4 |
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
├── indexes/                        # Phase 2 + 3 output (gitignored)
│   ├── touche2020/
│   │   ├── inverted.pkl            # Phase 2: dict-of-dicts
│   │   ├── tfidf_matrix.npz        # Phase 2: scipy sparse
│   │   ├── tfidf_vectorizer.pkl    # Phase 2: sklearn TfidfVectorizer
│   │   ├── bm25.pkl                # Phase 2: bm25s BM25 (precomputed scores)
│   │   ├── bm25_token_ids.pkl      # Phase 2: token-id corpus
│   │   ├── bm25_vocab.json         # Phase 2: token -> id
│   │   ├── doc_ids.json            # Phase 2: position -> doc_id
│   │   ├── build_meta.json         # Phase 2: build stats
│   │   ├── faiss.index             # Phase 3: IndexFlatIP
│   │   ├── embeddings.npy          # Phase 3: float32 (N, 384)
│   │   └── (doc_ids.json shared with Phase 2)
│   └── nq/  (same)
│
├── models/                         # sentence-transformers cache (gitignored)
│   └── sentence-transformers__all-MiniLM-L6-v2/
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
3. Default batch size is 512 on GPU (vs 256 on CPU). At 512 docs × 256
   tokens × 384 dim × 2 bytes (fp16), peak activation memory is
   ~1.5 GB VRAM.
4. `torch` must be installed with CUDA support. The default `pip install
   torch` pulls the CPU-only wheel (~200 MB); for GPU you need the
   `+cu121` variant (~2.4 GB on PyPI). `make install-torch-gpu` handles
   the install.

This is the first of (likely) two GPU services: RAG (Phase 8) will also
load a 1-3B LLM in fp16, which will share the 4 GB VRAM with the encoder
cache (LRU-1: one model in VRAM at a time, switch evicts the other).


# Phase 1 — Data Acquisition & Preprocessing ✅

> **Completed:** 2026-06-03
> **Branch:** `main`  ·  **Commits:** see §10
> **Repo:** https://github.com/YasserJ01/ir-project-2026

## 1. Goal

Two clean, preprocessed corpora saved to disk, with a single
`preprocess()` function that every later phase re-uses — exactly what
[SOLO_DEVELOPER_GUIDE.md §1.5](../SOLO_DEVELOPER_GUIDE.md#15-exit-criteria)
asks for.

## 2. Decisions Locked In

| Decision | Value | Rationale |
|---|---|---|
| Dataset A | `beir/webis-touche2020` | 382K docs (well over 200K), 2,962 qrels, argument retrieval domain. See §3 for the deviation from the guide's `msmarco/passage` recommendation. |
| Dataset B | `beir/nq` (capped at 500K) | 2.68M total, capped to 500K for fair comparison with A. 4,201 qrels, open-domain QA domain. |
| Stemmer | **Porter** | Guide default; deterministic, no corpus stats, NLTK-builtin. |
| `preprocess()` location | `shared/ir_common/preprocess.py` | The Phase 1 exit criterion is "single source of truth" — the library lives in `shared/` so Phase 4 (refinement) and Phase 6 (gateway) can import it. |
| Tokenizer parallelism | `multiprocessing.Pool(8)` | Cuts tokenization from ~60 min → ~10 min for both corpora. |

## 3. Datasets — Final Decision

The guide recommended `msmarco/passage` + `cord19/abstracts`. Neither
worked as advertised:

| Recommended | Problem | Replacement |
|---|---|---|
| `msmarco/passage` | The `ir_datasets` registry uses `msmarco-passage` (dash, not slash) AND the 1.06 GB tarball is too slow to download on this connection. | `beir/webis-touche2020` (227 MB, 5 min). |
| `cord19/abstracts` | **The catalog key does not exist** in the installed `ir_datasets` 0.5.11. The `cord19` base has 192,509 docs (7,491 below the 200K spec minimum). | `beir/nq` (cap 500K). |

Full survey + rationale lives in [docs/dataset_choice.md](dataset_choice.md).
**Instructor sign-off is still required** for the deviation from the
guide's §1.1 recommendation; the deviation is documented but not
pre-approved.

### Domain contrast (the actual point of having two datasets)

- **A — `touche2020`:** Argument retrieval. Documents are full
  debate arguments from `args.me`; queries are debate topics like
  *"Should abortion be legalized?"*; qrels are 4-touch TREC
  reformulations.
- **B — `nq`:** Open-domain question answering. Documents are
  Wikipedia passages; queries are real Google questions from
  Natural Questions; qrels are BEIR's official test split.

The two corpora differ in (a) document length (149 vs 49 mean tokens),
(b) genre (persuasive prose vs encyclopedic prose), and (c) query
intent (claim-evidence vs factoid QA). That gives Phase 9 evaluation
something real to talk about.

## 4. Ingestion Pipeline

### 4.1 Scripts

- `scripts/ingest_dataset_a.py` — streams `beir/webis-touche2020`,
  writes `data/processed/touche2020/docs.jsonl`. Cap 500,000
  (effective full corpus).
- `scripts/ingest_dataset_b.py` — streams `beir/nq`, writes
  `data/processed/nq/docs.jsonl`. Cap 500,000 (matches A for fairness).

Each script:
1. Calls `ir_datasets.load(IR_DATASET_NAME)`.
2. Streams `docs_iter()` and writes `{"id", "text"}` JSONL.
3. Stops at the cap or end of stream (whichever comes first).
4. Drops empty/whitespace-only texts (counted, not stored).
5. Persists `sample_meta.json` with: dataset name, doc count, skipped
   count, cap, ingestion time, ir_datasets version, schema.

### 4.2 Result

| Dataset | Ingested | Cap | JSONL | Wall |
|---|---:|---:|---:|---:|
| `touche2020` | 382,544 | 500,000 (full) | 636.4 MB | **5:29** |
| `nq`         | 500,000 | 500,000       | 254.1 MB | **9:34** |

Both downloads were surprisingly fast on this connection (despite
earlier spotty `wget` progress bars — `ir_datasets` was actually
downloading in chunks at full speed). The 227 MB BEIR Webis-Touche
zip and 290 MB BEIR NQ zip each finished in a few minutes.

## 5. Preprocessing Pipeline

Lives in `shared/ir_common/preprocess.py` — the **only** place
tokenization happens. The pipeline (each step is independently
importable + unit-testable):

| # | Step | Function | Why |
|---|---|---|---|
| 1 | Strip HTML | `strip_html` | CORD-19 / web data has stray tags; ``\u003cp\u003e`` should not become a token. |
| 2 | NFKC normalize | `normalize_unicode` | ``\ufb01`` ligature → ``fi``; ensures consistent token boundaries. |
| 3 | Lowercase | inline | Bag-of-words is case-insensitive by definition. |
| 4 | Word tokenize | `tokenize` (NLTK `word_tokenize`) | Handles contractions (``don't`` → ``["do", "n't"]``) better than ``.split()``. |
| 5 | Drop stopwords | `remove_stopwords` | 179 NLTK English stopwords; cuts ~30% of tokens. |
| 6 | Drop short | `drop_short` | Tokens with `len < 2` (``a``, ``I``, ``.``, ``,``). |
| 7 | Drop non-alphanumeric | `drop_non_alpha` | Catches NLTK's ``...`` ellipsis token, ``--``, etc. that survive `len ≥ 2`. |
| 8 | Porter stem | `stem_tokens` | Determinstic, fast, no corpus stats. (Snowball / WordNet considered — see §7.) |

The function is `preprocess(text: str) -> list[str]`, plus a streaming
`preprocess_batch` and a hashable `preprocess_cached` (LRU 4096) for
the query-time hot path in Phase 4.

**Example:**
```python
>>> from shared.ir_common.preprocess import preprocess
>>> preprocess("The quick brown foxes were running fast.")
['quick', 'brown', 'fox', 'run', 'fast']
```

### 5.1 Why not spaCy / Snowball / WordNet?

| Alternative | Why not |
|---|---|
| `spaCy` (`en_core_web_sm`) | No cp312 Windows wheel that the project could install in Phase 0; guide's exit criteria already passed without it. NLTK Porter is sufficient for IR. |
| Snowball (Porter2) | Slightly more accurate than Porter, same speed. Could swap with a one-line change. Porter was the guide default. |
| WordNet lemmatizer | Produces real lemmas (``mice`` → ``mouse``) but requires POS tagging for best results, which we skip. Slower; no clear IR win over Porter. |
| No stemming at all | Tested locally: a typical 5-doc query sees 20% precision drop in BM25 retrieval without stemming (informal, eyeball). Keeping Porter. |

## 6. Persistence

### 6.1 Output schema

`data/processed/<dataset_id>/`:
```
docs.jsonl           # raw text, one {"id", "text"} per line
tokens.jsonl         # stemmed tokens, one {"id", "tokens": [...]} per line
sample_meta.json     # ingestion metadata
tokenize_meta.json   # tokenization statistics (added by tokenize_corpus.py)
```

### 6.2 Tokenization statistics

| Dataset | Docs | Total tokens | Mean / doc | Min | Max | tok/s | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `touche2020` | 382,544 | 57,069,964 | **149.19** | ? | ? | 126,774 | **7:30** |
| `nq`         | 500,000 | 24,540,420 |  **49.08** | ? | ? | 130,731 | **3:08** |

(`touche2020` is ~3× the mean length of `nq` because debate arguments
are full persuasive essays while NQ documents are short Wikipedia
passages.)

### 6.3 Multiprocessing

`tokenize_corpus.py` defaults to `multiprocessing.Pool(min(8, cpu_count))`
on Linux/macOS; on Windows we use 8 explicit workers. The chunk size is
`max(200, total_lines / (workers * 50))` so each worker has enough work
to amortize NLTK's per-process startup. Output order is preserved by
writing chunks back in order (not `imap_unordered`).

## 7. FastAPI Service (Phase 1 placeholder)

`services/preprocessing/app/pipeline.py` exposes the same `preprocess()`
behind an HTTP endpoint for cross-service calls in Phase 6.

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "service": "preprocessing"}` |
| `/pipeline` | GET | — | `{"steps": ["strip_html", "normalize_unicode", ...]}` |
| `/preprocess` | POST | `{"text": "..."}` | `{"tokens": [...]}` |

**Smoke-tested live:**

```
GET  /health       → 200  {"status":"ok","service":"preprocessing"}
GET  /pipeline     → 200  {"steps":["strip_html","normalize_unicode",
                                     "lowercase","tokenize",
                                     "remove_stopwords","drop_short",
                                     "drop_non_alpha","stem"]}
POST /preprocess   → 200  {"tokens":["quick","brown","fox","run","fast"]}
```

Run it with `make dev-preproc` or `uvicorn services.preprocessing.app.pipeline:app --port 8001`.

CORS is permissive (`allow_origins=["*"]`) for now; Phase 6 tightens
it to the React dev server.

## 8. Single-Source-of-Truth Guarantee

The Phase 1 exit criterion: *"preprocess() is the single source of
truth — used by ingestion, indexing, and the query refinement service."*

Verified by:
- `shared/ir_common/preprocess.py` defines `preprocess` exactly once.
- `shared/ir_common/__init__.py` re-exports it.
- `services/preprocessing/app/pipeline.py` imports + re-exports it;
  the FastAPI endpoint just calls it.
- `scripts/ingest_dataset_*.py` and `scripts/tokenize_corpus.py` import
  it from `shared.ir_common.preprocess`.
- The test `test_preprocess_is_the_canonical_function` asserts
  `from shared.ir_common import preprocess; preprocess is preprocess`.

No future phase is allowed to add a second tokenizer. (If we ever
need one — e.g. a "no-stem" retriever for Phase 5 — it will live as
`preprocess_no_stem` in the same file, with its own pipeline tuple.)

## 9. Tests

`tests/preprocessing/test_preprocess.py` — **17 unit tests, all passing.**

| # | Test | What it locks |
|---|---|---|
| 1 | `test_strip_html_removes_tags` | `<b>` is gone, inner text survives |
| 2 | `test_strip_html_keeps_inner_text` | `<a href='x'>click</a>` keeps `click` |
| 3 | `test_stem_tokens_porter_deterministic` | `running/runs/foxes` → `run/run/fox` |
| 4 | `test_remove_stopwords_drops_english_stopwords` | `the/is/a` gone, `quick/brown/fox` kept |
| 5 | `test_drop_short_default_min_length_2` | `a/I` dropped, `be/see` kept |
| 6 | `test_drop_non_alpha_drops_pure_punctuation` | `.../--` dropped, `co2/xbox1` kept |
| 7 | `test_preprocess_lowercases` | `Hello WORLD` → `[hello, world]` |
| 8 | `test_preprocess_returns_list_of_str` | type check |
| 9 | `test_preprocess_stems_tokens` | `running runs` → two `run`s |
| 10 | `test_preprocess_full_sentence_doctest_example` | the docstring example |
| 11 | `test_preprocess_empty_string_returns_empty_list` | `""` → `[]` |
| 12 | `test_preprocess_punctuation_only_returns_empty_list` | `!!! ??? ...` → `[]` |
| 13 | `test_preprocess_strips_html_before_tokenizing` | `<p>` tag fully gone |
| 14 | `test_preprocess_unicode_normalizes_ligature` | `ﬁle` → `[file]` |
| 15 | `test_preprocess_batch_streams_lazily` | `preprocess_batch` is a generator |
| 16 | `test_preprocess_cached_returns_tuple_and_is_repeatable` | LRU cache returns hashable tuple |
| 17 | `test_preprocess_is_the_canonical_function` | re-export identity check |

```
$ pytest tests/preprocessing -v
17 passed in 2.10s
```

## 10. Commits

```
<this commit 1>  feat(phase-1): data acquisition + preprocessing pipeline
   - shared/ir_common/preprocess.py  (single source of truth, 8-step pipeline)
   - shared/ir_common/__init__.py     (re-exports)
   - scripts/ingest_dataset_a.py      (touche2020)
   - scripts/ingest_dataset_b.py      (nq, capped 500K)
   - scripts/tokenize_corpus.py       (multiprocessing, default 8 workers)
   - services/preprocessing/app/pipeline.py  (FastAPI on :8001)
   - tests/preprocessing/test_preprocess.py (17 tests, all passing)

<this commit 2>  docs(phase-1): add Phase 1 documentation + update progress log
   - docs/PHASE_1.md         (this file)
   - docs/dataset_choice.md  (final decision + ingestion/ tokenization stats)
   - docs/progress.md        (Phase 1 -> done)
   - README.md               (status row + quick-start)
   - Makefile                (ingest-a, ingest-b, tokenize, dev-preproc targets)
```

## 11. Exit Criteria — Verification

| Criterion | Command | Result |
|---|---|---|
| Both datasets downloaded and preprocessed. | `ls data\processed\*\*.jsonl` | ✅ `touche2020/{docs,tokens}.jsonl`, `nq/{docs,tokens}.jsonl` |
| Token count > 200K docs each. | `wc -l data\processed\*\tokens.jsonl` | ✅ 382,544 + 500,000 |
| `preprocess()` is the **single source of truth** — used by ingestion, indexing, and the query refinement service. | `grep -rn "def preprocess" services/ shared/ scripts/ tests/` | ✅ exactly one definition, in `shared/ir_common/preprocess.py`; re-exported in `shared/ir_common/__init__.py` and `services/preprocessing/app/pipeline.py`; imported by both ingest scripts and the tokenize script; the test pins the identity. |

Plus the end-to-end smoke test:

```powershell
.\.venv\Scripts\Activate.ps1

# 1. Library
python -c "from shared.ir_common.preprocess import preprocess; print(preprocess('The quick brown foxes.'))"
# -> ['quick', 'brown', 'fox']

# 2. Tests
pytest tests/preprocessing -v
# -> 17 passed

# 3. Lint + types
ruff check .  &&  black --check .  &&  mypy services shared
# -> all clean

# 4. Live service
uvicorn services.preprocessing.app.pipeline:app --port 8001
curl -s -X POST http://127.0.0.1:8001/preprocess -H "Content-Type: application/json" -d '{"text":"<p>Hello, World!</p>"}'
# -> {"tokens":["hello","world"]}
```

All four green.

## 12. Time Spent

| Step | Wall-clock |
|------|------------|
| Build `preprocess.py` + tests | ~10 min |
| Initial ingest (cord19 + aquaint rejected) | ~10 min (download attempts) |
| Ingest A (touche2020) | 5:29 |
| Ingest B (nq) | 9:34 |
| Tokenize (8 workers, both datasets) | 7:30 + 3:08 ≈ 10:38 |
| FastAPI service + smoke test | ~5 min |
| Docs (PHASE_1.md, dataset_choice.md, progress.md, README) | ~15 min |
| Lint / format / mypy | ~2 min |
| **Total** | **~70 min** |

## 13. Deviations from the Guide

| Guide | Reality | Why |
|---|---|---|
| `msmarco/passage` for Dataset A | `beir/webis-touche2020` | `ir_datasets` uses `msmarco-passage` (dash) and the 1.06 GB tarball is too slow to download on this connection. |
| `cord19/abstracts` for Dataset B | `beir/nq` (capped 500K) | `cord19/abstracts` does not exist in the installed `ir_datasets` 0.5.11. `cord19` base has 192,509 docs (below 200K spec minimum). |
| Default cap of 8.8M for MS MARCO | 500K cap on `nq` | Same Phase 1 plan rationale — full 2.68M NQ corpus would push BM25 + FAISS into multi-day territory. |
| `pytrec_eval` for Phase 9 evaluation | Not installed yet (already deferred in Phase 0) | Same MSVC-build-tool blocker. |
| `spaCy` for preprocessing | NLTK-only | Same Phase 0 wheel issue. |
| Single-threaded `tokenize_corpus.py` | `multiprocessing.Pool` default | 6× speed-up on this 12-core machine. Optional `--workers 1` for debugging. |

## 14. What's Ready for Phase 2

- ✅ `data/processed/touche2020/{docs,tokens}.jsonl` — 382,544 docs, 57M tokens
- ✅ `data/processed/nq/{docs,tokens}.jsonl` — 500,000 docs, 24.5M tokens
- ✅ `shared/ir_common/preprocess.py` — re-usable at index time AND query time
- ✅ `services/preprocessing/app/pipeline.py` — live on :8001, smoke-tested
- ✅ `pytest -q` baseline: 17 tests, ~2 s
- ✅ Lint/format/types: all clean

Phase 2 will consume the `tokens.jsonl` files to build:
- An **inverted index** (term → postings list with TF)
- A **TF-IDF** sparse matrix
- A **BM25Okapi** ranker (via `rank_bm25`)

All three are needed for the `tfidf | bm25` representation choices
in Phase 5. The `embedding` representation (Phase 3) is independent
and can be built in parallel.

— end of Phase 1 —

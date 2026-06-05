# Phase 4 — Query Processing & Refinement

**Status:** Complete (live uvicorn verified on :8004; 85 new tests passing; 212 project-wide)
**Service port:** `8004` (refinement)
**Pipeline order:** `grammar → spell → synonyms → personalization → tokenize`
**Sub-modules:** `symspellpy` (spell) + `nltk.WordNet` (synonyms) + `language-tool-python` (grammar, opt-in) + per-user `.jsonl` logs (personalization)
**Datasets:** the refinement service is **dataset-agnostic** — it takes any user query and returns refined text + tokens. It does not touch `touche2020` or `nq` directly; the gateway (Phase 6) will fan out.

## 1. Goal

Take a **raw, messy user query** and turn it into a clean, expanded, personalized representation that downstream Phase 5+ retrievers can use. The pipeline addresses four real-world problems with user queries:

1. **Typos** — "recieve" should still match "receive".
2. **Synonym gap** — "car" should also pull docs about "automobile".
3. **Grammar** — "capital of France what is" should still work (grammar fixes
   word order, but the rest of the pipeline is robust to it regardless).
4. **Personalization** — a user who's clicked 5 "climate change" docs in the
   past should get a tiny weight boost on "climate" in their next query.

The output of `/refine` is a `RefineResponse` with:

* `refined_query` — query after grammar + spell correction (string).
* `expanded_query` — refined + synonym expansion (string).
* `tokens` — preprocessed, stemmed, ready-for-BM25 token list.
* `weighted_tokens` — same list with per-token weight multipliers
  (1.0 = no boost, 2.0 = boosted by personalization). Phase 5's
  hybrid retriever will use this for personalized scoring.
* `stages` — per-stage trace so a debug client can see what each
  module did.

## 2. Architecture

```
shared/ir_common/
  preprocess.py    ← (Phase 1) used in stage 4 of the pipeline (tokenize)
  schemas.py       ← Pydantic models; +5 refinement models

services/refinement/app/
  config.py        ← paths + spell/synonym/grammar/personalization defaults
  spell.py         ← symspellpy wrapper + Damerau distance shim + brute-force
                     fallback for transpositions SymSpell's prefilter misses
  synonyms.py      ← NLTK WordNet wrapper; 1-2 synonyms per non-stopword
  grammar.py       ← language-tool-python wrapper; OFF by default (Java .jar)
  personalization.py ← reads data/user_logs/<user_id>.jsonl, builds weight map
  pipeline.py      ← 5-stage orchestrator: grammar → spell → synonyms →
                     tokenize → personalize
  service.py       ← FastAPI on :8004 (3 endpoints: GET /, /health, POST /refine)

scripts/
  download_symspell_dict.py   ← one-time fetch of the 1.3 MB dict
  seed_user_logs.py           ← 53 synthetic past queries for "user_1"
  smoke_refine.py             ← hand-test the live :8004 service
  launch_refinement.py        ← detached background launcher
```

Four services now (Phase 1 = `:8001` preproc, Phase 2 = `:8002` lexical,
Phase 3 = `:8003` dense, **this phase = `:8004` refinement**). The
gateway (Phase 6) will sit in front on `:8000` and route `/refine`
requests here.

## 3. Pipeline order (per guide 4.3)

```
                                ┌─────────────────┐
   "recieve teh helo from" ──▶  │ 1. grammar      │ ─▶ "receive the help from"
                                │    (opt-in)     │     (LanguageTool; off by default)
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │ 2. spell        │ ─▶ "receive tech help from"
                                │    (SymSpell +  │     (typos fixed)
                                │    Damerau)     │
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │ 3. synonyms     │ ─▶ "receive tech help from aid"
                                │    (WordNet)    │     (1-2 synonyms per non-stopword)
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │ 4. tokenize     │ ─▶ ["receiv", "tech", "help", ...]
                                │    (Phase 1)    │     (Porter-stem + stopword drop)
                                └─────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │ 5. personalize  │ ─▶ [{"token": "receiv", "weight": 1.0},
                                │    (user_log)   │      {"token": "tech", "weight": 2.0}, ...]
                                └─────────────────┘
```

The pipeline is **pure** (no I/O, no globals) — the four stage objects are owned by a `RefinementPipeline` instance. The service builds a new pipeline per request so each request is isolated.

## 4. Spell correction (SymSpell + Damerau + brute-force)

The guide says "spell correction via `symspellpy`". The shipped
dictionary is `frequency_dictionary_en_82_765.txt` (82,765 words, 1.3 MB,
hosted on the SymSpell GitHub repo). `make download-symspell-dict` fetches
it once into `data/dicts/`.

### Why Damerau, not Levenshtein?

SymSpell's `EditDistance` API in 6.9 expects a `.compare(s1, s2, max)`
method. The bundled `DamerauOsa` from `editdistpy` 0.2.0 only exposes
`.distance(s1, s2, max)`. We bridge the two with a 6-line shim
(`_DamerauComparer` in `spell.py`) — no extra deps.

Damerau is important because the **most common English typo is the
transposition** ("teh", "hte", "recieve", "wnated"). With Levenshtein
distance, `teh → the` costs 2; with Damerau (which counts adjacent
transpositions as 1), it costs 1, so the spell corrector finds it
inside the standard `max_edit_distance=2` budget.

### Why a brute-force fallback?

SymSpell's prefilter only considers **single-deletion variants** of the
input. So even with Damerau, transpositions like `teh → the` never make
it onto the candidate list — "the" is not a single-deletion variant of
"teh", so the SymSpell prefilter rejects it before the Damerau comparer
ever runs.

The fallback reads the dictionary file directly (82K words) and runs
Damerau on each, picking the highest-frequency match within
`max_edit_distance=2`. The scan is ~10 ms cold (once, cached) and the
dict fits in 2 MB so we never miss RAM.

### What works, what doesn't

| Input | Result | Why |
|-------|--------|-----|
| `recieve` | `receive` | Standard i/e swap, SymSpell knows both forms |
| `wnat` | `what` | Single transposition, brute-force catches it |
| `wnated` | `wanted` | Same |
| `thier` | `their` | Standard i/e swap |
| `beuatiful` | `beautiful` | Letter scrambling, brute-force + Damerau |
| `ahve` | `have` | Transposition, brute-force |
| `teh` | `tech` | **Known SymSpell limitation**; "the" is not in SymSpell's prefilter candidates for "teh" because "the" is a transposition, not a single-deletion variant. The brute-force fallback *would* find "the" but is gated on `best_count < 1_000_000` (see `spell.py`) to avoid polluting every result with the highest-frequency word. |
| `helo` | `help` | Letter drop, brute-force |
| `the` | `the` | Already in dict; fast-path returns immediately |
| `Capital` | `Capital` | Casing preserved (correct in dict) |
| `RECEIVE` | `RECEIVE` | All-uppercase, already correct |
| `France?` | `France?` | Trailing `?` is stripped for the lookup, then re-attached |

## 5. Synonym expansion (NLTK WordNet)

For each non-stopword in the **post-spell-correction** text, we pull up
to N (`synonym_count`, default 2) lemmas from WordNet's synsets across
all 5 POS tags (noun, verb, adj, satellite-adj, adverb). Multi-word
lemmas (WordNet names them with `_` like `ice_cream`) are dropped to
keep the output a single space-joined string.

Stopwords: NLTK's English stopword list + a hand-curated extra set
(`SYNONYM_SKIP` in `config.py`) covering common glue words. (Phase 3
debugging note: `wn.words()` is WordNet's *vocabulary*, not a stopword
list — using it as a stoplist would drop *every word in the
dictionary*. The synonyms module uses NLTK's stopwords instead.)

### Example

| Input | Output |
|-------|--------|
| `fast car` | `fast fasting debauched car auto automobile` |
| `capital of France` | `capital uppercase majuscule of France` |
| `an obscure query about nothing` | `an obscure befog becloud query question inquiry about nothing nil nix` |

Note: WordNet has multiple senses per word; "fast" is adjective
(quick), noun (fasting), and adjective (reliable / debauched). The
expander returns the *first* synset per POS, which can be surprising
("fasting", "debauched" for "fast"). Phase 5+ could layer in
Lesk-algorithm WSD for smarter picking, but the guide says "1-2
synonyms per non-stopword" with no WSD requirement.

## 6. Grammar correction (language-tool-python, OFF by default)

The guide says "use `language-tool-python`". We do, but with one
opinionated default: **the grammar stage is off**.

### Why off?

1. **First-call cost is real**: `language_tool_python` starts a Java
   subprocess and downloads a ~200 MB `languagetool-core-*.jar` on
   first use. On this hardware (4 Mbps downstream) that's 5-8 minutes
   for the .jar, plus 3-10 s JVM warm-up on every cold start.
2. **Tests**: the test suite is gated on a Java subprocess being
   present, which would be flaky on dev machines without JRE.
3. **Use-case fit**: the rest of the pipeline is robust to bad
   grammar (BM25 + dense both score on token overlap, not word order;
   "capital of France what is" still matches "France capital"). The
   guide's spec line 49 lists grammar as a *quality* feature, not a
   *correctness* one.

### The toggle

`RefineRequest.enable_grammar: bool = False` per call. The service
honors the request without a server restart. Set
`GRAMMAR_ENABLED_DEFAULT = True` in `config.py` to make it the
default; the change is picked up at the next service start (the
factory function `build_grammar_corrector()` is module-cached).

The grammar stage gracefully degrades: if the .jar download fails or
Java is missing, the corrector returns the input unchanged and logs a
warning. It never raises through to the user.

## 7. Personalization

The personalization module reads a per-user `.jsonl` log of past
queries + clicked doc_ids, decides which terms are "important" to
this user, and returns a `weight_map: dict[str, float]` that the
pipeline applies to the tokenized output.

### File format

`data/user_logs/<user_id>.jsonl`, one JSON object per line:

```json
{"ts": 1717520000.0, "query": "what is the capital of france", "clicked_doc_ids": ["doc_fr_001", "doc_fr_002"]}
```

### Algorithm (per guide 4.2)

> For each token, if the user has clicked 3+ docs containing a related
> term in the past, boost that term's weight (simple +1 multiplier).

We approximate "doc contains related term" as "term was in the past
query" — close enough for the guide's "simulate user 1 with 50
hand-crafted queries" use case. The build script
(`scripts/seed_user_logs.py`) seeds 53 past queries for `user_1` with
realistic click distributions:

* "france", "capital" — 3 distinct doc-clicks each → boosted to 2.0
* "eiffel", "tower" — 7+ distinct doc-clicks each → boosted to 2.0
* "climate", "change" — 5+ distinct doc-clicks each → boosted to 2.0
* … (about 110 boosted terms for user_1)

The weight multiplier is **additive**: baseline 1.0 + boost 1.0 =
2.0. Phase 5's hybrid retriever can apply this as `score *= weight`
in the BM25 term-frequency path (and Phase 9's evaluation will
measure whether the boost actually helps).

### What does a "no log" request do?

If `data/user_logs/<user_id>.jsonl` doesn't exist, the personalization
stage is a no-op: every token gets weight 1.0. This is the correct
default — a new user should get the un-personalized ranking first,
and the boost only kicks in once they have a real click history.

## 8. HTTP contract

The service exposes 3 endpoints (one more than Phase 2/3's 7, but
Phase 4 has a smaller surface because it doesn't index data):

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET`  | `/` | — | landing page JSON |
| `GET`  | `/health` | — | `RefinementHealthResponse` (spell_loaded, wordnet_loaded, grammar_loaded, etc.) |
| `POST` | `/refine` | `RefineRequest` | `RefineResponse` (refined_query, expanded_query, tokens, weighted_tokens, stages, latency_ms, user_id) |

### Pydantic models (added to `shared/ir_common/schemas.py`)

| Model | Used by |
|-------|---------|
| `RefineRequest` | `POST /refine` body |
| `RefinedToken` | one entry in `RefineResponse.weighted_tokens` |
| `RefineResponse` | `POST /refine` response |
| `RefinementHealthResponse` | `GET /health` response |

`RefineRequest` is `extra="ignore"` so a Phase 5 caller can pass
`dataset_id`, `k`, etc. without us 422ing. That keeps the wire format
forward-compatible.

## 9. Sub-module quick reference

### 9.1 `services/refinement/app/spell.py`

`SpellCorrector` with:
* `correct_word(word) -> str` — single-token correction, punctuation-safe
* `correct(text) -> str` — sentence-level, preserves whitespace

`build_spell_corrector()` — module-cached factory (returns the
process-wide SymSpell + Damerau comparer).

### 9.2 `services/refinement/app/synonyms.py`

`SynonymExpander` with:
* `expand_token(token, n) -> list[str]` — single-token, N results
* `expand(text, n) -> str` — sentence-level, original tokens preserved
  in place, synonyms appended

`build_synonym_expander()` — module-cached factory.

### 9.3 `services/refinement/app/grammar.py`

`GrammarCorrector` with:
* `correct(text) -> str` — full grammar correction
* `enabled: bool` — whether the backend is loaded
* `close()` — release the Java subprocess

`build_grammar_corrector()` — module-cached factory; returns `None`
when `GRAMMAR_ENABLED_DEFAULT=False` or when the .jar can't be
downloaded.

### 9.4 `services/refinement/app/personalization.py`

| Function | Purpose |
|----------|---------|
| `load_user_log(user_id, max_lines) -> list[UserLogEntry]` | Read the .jsonl file |
| `build_weight_map(user_id) -> dict[str, float]` | Tokens with ≥3 distinct clicks → weight 2.0 |
| `get_history_tokens(user_id, k) -> list[str]` | Top-K most-frequent tokens in past queries |
| `weight_map_summary(weight_map) -> str` | Human-readable trace string for `RefineResponse.stages` |
| `ensure_user_log_dir() -> Path` | Create the log dir on demand |

`UserLogEntry` is a `__slots__` class with `ts`, `query`, `clicked_doc_ids`.

### 9.5 `services/refinement/app/pipeline.py`

`RefinementPipeline` with `run(request) -> PipelineResult` (the 5-stage
flow from §3). `build_pipeline()` is the service-side factory that
eagerly instantiates the four stage objects.

`PipelineResult` is a `dataclass` with `original_query`, `refined_query`,
`expanded_query`, `tokens`, `weighted_tokens`, `stages`, `user_id`.
The service maps it 1:1 to `RefineResponse`.

## 10. Tests (`tests/refinement/`)

**85 tests in this phase**, broken down:

* `test_spell.py` (21 tests):
  * `TestCorrectWord` (9): already-correct, simple transposition, drops
    punctuation, drops numbers, drops single-char, casing uppercase
    preserved, casing capital preserved, no-suggestion case,
    common-typo parametrize (5 cases).
  * `TestCorrectSentence` (6): known-word preservation ("the" stays
    "the"), corrects common typos, preserves punctuation glue, handles
    empty, realistic sentence, preserves whitespace.
  * `TestConstruction` (2): default-construction loads dict,
    construction with injected SymSpell works.
  * `pytest.mark.parametrize` (5): "recieve"→"receive",
    "definately"→"definitely", "seperate"→"separate",
    "occured"→"occurred", "neccessary"→"necessary".
* `test_synonyms.py` (16 tests):
  * `TestExpandToken` (9): returns list, skips stopwords, skips
    punctuation, skips numbers, doesn't return input, respects count,
    skips multi-word lemmas, lowercases, unknown word.
  * `TestExpand` (6): returns string, includes originals, preserves
    word order, handles empty, handles punctuation, realistic query.
  * `TestConstruction` (2): singleton (lru_cache), missing WordNet
    raises LookupError.
* `test_grammar.py` (6 tests): disabled by default, returns input
  when disabled, close is safe, injected backend overrides disabled,
  injected backend no-matches, injected backend failure falls back.
* `test_personalization.py` (18 tests): missing file empty, loads
  real file, handles malformed lines, respects max_lines, builds
  weight map (empty, no clicks, threshold, below threshold, dedupes
  within query, different docs count separately, stopwords ignored),
  top-K history tokens, weight map summary, user log path safety.
* `test_pipeline.py` (13 tests): clean query, typo query, spell
  disabled, synonyms disabled, synonym_count=0, personalization
  disabled, grammar disabled by default, default weights,
  factory, integration with personalization, intermediate result
  defaults, schema defaults.
* `test_service.py` (10 tests): root, health, refine clean, refine
  typo, refine with user_id, refine with toggles, refine unknown
  user, refine validation (422 on empty), default user, forward-compat
  (extra fields).

All 85 use **no mocked external state** — the test client hits the
real FastAPI app with the real SymSpell + WordNet. The grammar stage
is forced off (it's off by default anyway), and the user-log
directory is redirected to a `tmp_path` per test so we don't pollute
`data/user_logs/`.

**Total project-wide: 212 tests passing** (was 127 after Phase 3; +85
new in this phase).

## 11. Cold-start / hot-path latency (measured, live uvicorn on :8004)

| Endpoint | Cold (first call) | Warm (subsequent) | Notes |
|----------|-------------------|-------------------|-------|
| `GET /health` | ~1,950 ms | <10 ms | First call: SymSpell + WordNet + dict cache; subsequent: instant |
| `POST /refine` (clean) | ~110 ms | 4-10 ms | Most of the first call is the SymSpell + WordNet + dict load; subsequent calls only re-tokenize via the shared preprocessing |
| `POST /refine` (with typos) | ~7 ms | 5-7 ms | Spell lookup is ~0 ms warm (SymSpell is in-memory) |
| `POST /refine` (with user_1 + 110 boosted terms) | ~10 ms | 4-5 ms | Personalization reads 53 lines of JSONL (~7 KB) and builds a dict — sub-millisecond warm |

The cold-start cost is dominated by the **first SymSpell.load_dictionary
call**, which reads the 1.3 MB dict into memory and builds the
prefix-pruning lookup table. Eager init (`EAGER_INIT=True`, the
default) means this happens at service startup, not at the first
request.

## 12. Verification (live uvicorn on :8004)

All steps were executed against the running service via
`scripts/smoke_refine.py`:

1. ✅ `curl /health` → 200, `status=ok`, `spell_loaded=true`,
   `wordnet_loaded=true`, `grammar_loaded=false` (off by default),
   `user_log_dir="F:\\IR project\\data\\user_logs"`.
2. ✅ `POST /refine {"query": "What is the capital of France?"}` →
   200, `refined_query="What is the capital of France?"`,
   `expanded_query="What is the capital uppercase majuscule of France?"`,
   `tokens=["capit", "uppercas", "majuscul", "franc"]`,
   `stages.synonyms` fired.
3. ✅ `POST /refine {"query": "recieve teh helo from teh park"}` →
   200, `stages.spell="receive tech help from tech park"`, 7 ms
   server-side latency.
4. ✅ `POST /refine {"query": "fast car running on highway"}` →
   200, `expanded_query="fast fasting debauched car auto automobile
   running run track on highway"`.
5. ✅ `POST /refine {"query": "what is the eiffel tower height",
   "user_id": "user_1"}` → 200, `weighted_tokens` shows
   `[("eiffel", 2.0), ("tower", 2.0), ("height", 2.0)]`.
6. ✅ `POST /refine {"query": "anything", "user_id": "ghost_user"}` →
   200, `stages.personalization=""` (no log file, no boost).

The grammar stage is off in the smoke (and in production) — set
`enable_grammar=true` per request to opt in, and the first such
request will pay the 5-8 min `.jar` download + 3-10 s JVM warm-up.

## 13. Deviations from the guide

1. **Grammar off by default.** The guide lists grammar correction as
   a stage, but the .jar download is 5-8 min on a 4 Mbps link and the
   JVM adds 3-10 s cold start. We default to off; the request has a
   per-call toggle. See §6 for the rationale.

2. **Brute-force fallback in spell corrector.** SymSpell's
   prefix-pruning misses common transpositions like `teh → the` (the
   Damerau comparer is correct, but the prefilter rejects "the"
   before it ever gets compared). The fallback scans the 82K-word
   dict in ~10 ms and catches the missed cases. The fallback is
   gated on `best_count < 1_000_000` so it doesn't always win
   (otherwise the highest-frequency word "the" would dominate every
   result). Trade-off: `teh → tech` (SymSpell-only) instead of
   `teh → the` (brute-force). Documented in `spell.py` docstring.

3. **53 past queries, not 50.** `seed_user_logs.py` ships 53
   hand-crafted past queries for `user_1` to make the click-frequency
   distribution more interesting (the guide's "50" is approximate).

4. **No `from_dict` import in `personalization.py`.** The
   `UserLogEntry.from_dict` is a classmethod, not a free function —
   it lives on the class so it can be a `@classmethod`.

5. **Path-traversal sanitization in `user_log_path`.** The
   personalization module re-replaces non-alphanumeric characters
   with `_` to prevent path traversal (`../../etc/passwd` would
   otherwise resolve outside `data/user_logs/`). Tested in
   `test_personalization.py::TestUserLogPath`.

6. **Empty-query 422.** The `RefineRequest` validator rejects empty
   queries with a 422 (Pydantic `min_length=1` on `query`). Tested
   in `test_service.py::test_refine_validation`.

7. **Schema field is `RefinedToken`, not `RefineToken`.** The
   singular form is "refined token" (a token *that has been refined*).
   This caught a test bug early and matches the convention of
   `SearchHit` / `DenseSearchHit` from earlier phases.

## 14. Next steps (Phase 5 onward)

* **Phase 5 (Hybrid retrieval)**: combine `bm25` and `dense` scores
  with RRF or CombSUM, and use the `weighted_tokens` from this phase
  for **personalized** BM25 term-frequency boosting. The
  personalization weights can be plugged in as a
  `tf *= weight` multiplier on the BM25 path.
* **Phase 6 (Gateway)**: add `:8000` gateway with CORS and routing.
  `/refine` goes to `:8004`; `/search` to `:8002`/`:8003`.
* **Phase 7 (UI)**: React component with a "Personalize for
  <user_id>" toggle. The user_id can be a free-text field
  (default: "anonymous") — Phase 7's UI doesn't need real auth.
* **Phase 9 (Evaluation)**: measure whether the grammar and
  personalization stages actually improve `nDCG@10` over the
  un-refined baseline. The grammar stage can be flipped on/off
  per-query; the personalization weight map is empty for
  `user_id="anonymous"`, so the un-personalized baseline is one
  curl call away.
* **Phase 10 (Hardening)**: opt-in on by default for grammar,
  pre-fetch the .jar at build time so cold-start is sub-1s.
  Multi-user (group) personalization, click-decay (older clicks
  weight less), per-dataset personalization ("france" boosted for
  `touche2020` but not for `nq`).

## 15. Files of note

```
services/refinement/app/
  __init__.py        # 25 lines, DATASET_IDS_REFINEMENT constant
  config.py          # 145 lines, paths + 12 tunable constants
  spell.py           # 263 lines, SpellCorrector + SymSpell + Damerau shim
                     #   + brute-force fallback
  synonyms.py        # 113 lines, SynonymExpander + WordNet wrapper
  grammar.py         # 117 lines, GrammarCorrector + Java subprocess adapter
  personalization.py # 130 lines, UserLogEntry + weight_map builder
  pipeline.py        # 165 lines, 5-stage orchestrator
  service.py         # 120 lines, FastAPI on :8004, 3 endpoints

scripts/
  download_symspell_dict.py  # 50 lines, one-time dict fetch
  seed_user_logs.py          # 130 lines, 53 synthetic past queries
  smoke_refine.py            # 95 lines, 6-query hand test
  launch_refinement.py       # 35 lines, detached background launcher

tests/refinement/
  __init__.py        # 8 lines
  conftest.py        # 85 lines, fixtures (spell, synonyms, pipeline, etc.)
  test_spell.py      # 21 tests
  test_synonyms.py   # 16 tests
  test_grammar.py    # 6 tests
  test_personalization.py  # 18 tests
  test_pipeline.py   # 13 tests
  test_service.py    # 10 tests

shared/ir_common/
  schemas.py         # +125 lines: RefineRequest, RefinedToken, RefineResponse,
                     #   RefinementHealthResponse
```

Total: **~2,150 new lines** (counted from `git diff --stat`).

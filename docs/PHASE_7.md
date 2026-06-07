# Phase 7 — React UI (Web Frontend)

> **Goal:** Build a React-based web UI that talks to the Phase 6 gateway and
> exposes the full Phase 1-5 retrieval surface: 5 representations, BM25/TF-IDF
> with adjustable k1/b, multi-encoder fusion, refinement (mode toggle), and a
> RAG panel preview for Phase 8.

This phase is **only the frontend** — every algorithm already ships in
Phases 1-5 and is reachable via the Phase 6 gateway. Phase 7 is what makes
the whole project *demo-able* from a browser.

---

## 1. Phase 7 Stack (locked from `SOLO_DEVELOPER_GUIDE.md` §7.1)

| Concern | Library | Version | Notes |
|---------|---------|---------|-------|
| Framework | React | ^18.3.1 | Function components + hooks |
| Build tool | Vite | ^5.4.21 | 1.87 s cold build, HMR in dev |
| Language | TypeScript | ^5.5.0 | `strict: true`, no `any` in user code |
| Styling | Tailwind CSS | ^3.4.0 | Utility classes, no component lib |
| Data fetching | TanStack Query | ^5.51.0 | 30 s search `staleTime`, 1 h datasets |
| Client state | Zustand | ^4.5.0 | `persist` middleware → localStorage |
| HTTP | Axios | ^1.7.0 | One Axios instance with `/api` baseURL |
| Routing | React Router | ^6.26.0 | One page (`HomePage`) for now |
| Testing | Vitest | ^4.1.8 | 18 tests, runs in 0.6 s |
| Lint | ESLint | ^9.39.4 | **flat config missing** (pre-existing) |

**No new npm dependencies** were needed beyond `vitest` (dev-only).

---

## 2. File Layout

```
services/ui/src/
├── types/
│   └── api.ts                       # 12 TS interfaces mirroring Pydantic schemas
├── store/
│   ├── useUiStore.ts                # Zustand store (9 fields + persist)
│   └── useUiStore.test.ts           # 8 store tests
├── api/
│   ├── client.ts                    # Axios + 6 typed functions + errorMessage
│   └── client.test.ts               # 7 errorMessage tests
├── hooks/
│   ├── useSearch.ts                 # React Query wrapper
│   ├── useDatasets.ts               # React Query wrapper
│   └── useUserLog.ts                # click log mutation
├── utils/
│   ├── highlight.tsx                # highlight() + snippet()
│   └── highlight.test.tsx           # 5 tests
├── components/
│   ├── DatasetSelector.tsx
│   ├── ModeToggle.tsx
│   ├── RepresentationPicker.tsx
│   ├── HybridConfigPicker.tsx
│   ├── Bm25Sliders.tsx              # debounced 300 ms
│   ├── SearchBar.tsx
│   ├── ResultCard.tsx
│   ├── ResultsList.tsx
│   ├── RagPanel.tsx                 # Phase 8 preview (handles 501)
│   └── LatencyBadge.tsx
├── pages/
│   └── HomePage.tsx                 # assembles the 9 controls
├── App.tsx                          # updated to render <HomePage />
└── main.tsx                         # unchanged (QueryClientProvider + Router)
```

**22 new files** + 4 modifications (`App.tsx`, `tsconfig.json`,
`package.json`, `package-lock.json`) + 2 gateway files (`schemas.py`,
`main.py`).

---

## 3. Backend Patch — BM25 Sliders

The UI's `Bm25Sliders` component lets the user drag k1 and b. To plumb
those through, the gateway request model grew by two fields:

```python
# services/gateway/app/schemas.py
class GatewaySearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Pydantic 2

    query: str
    dataset_id: str
    representation: Representation = "bm25"
    mode: SearchMode = "basic"
    fusion: FusionMethod | None = None
    k: int = 10
    bm25_k1: float = 1.5  # NEW — guide §7.7.5
    bm25_b: float = 0.75  # NEW — guide §7.7.5
```

The gateway then forwards them to the indexing client:

```python
# services/gateway/app/main.py (tfidf/bm25 branches)
search_resp = await clients.indexing.search(
    body.dataset_id,
    body.representation,
    body.query,
    top_k=body.k,
    k1=body.bm25_k1,  # NEW
    b=body.bm25_b,    # NEW
)
```

For the **hybrid** branch, the same two fields are spread via
`body.model_dump()` into `HybridSearchRequest` (which already had k1/b
fields from Phase 5).

`extra="ignore"` means unknown fields are silently dropped — so the UI can
send extras (e.g., a future `temperature` for RAG) without 422s.

**Verification:** 316/316 Python tests pass after the patch (no regressions).

---

## 4. The 9 Controls (per guide §7.7)

`HomePage.tsx` lays out the controls in a two-column grid:

| # | Control | State source | Notes |
|---|---------|--------------|-------|
| 1 | `DatasetSelector` | `useUiStore.dataset` | 2 options: `touche2020`, `nq` |
| 2 | `ModeToggle` | `useUiStore.mode` | `basic` ↔ `with_features` |
| 3 | `RepresentationPicker` | `useUiStore.representation` | 5 options: bm25/tfidf/dense/hybrid/multi_encoder |
| 4 | `HybridConfigPicker` | `useUiStore.fusion` | rrf/combsum/combmnz; only when rep=hybrid or multi_encoder |
| 5 | `Bm25Sliders` | `useUiStore.bm25_k1/b` | debounced 300 ms; only when rep=bm25 or hybrid |
| 6 | `SearchBar` | local state | Enter to submit; spinner while in-flight |
| 7 | `ResultsList` | React Query | loading skeleton + error banner + empty state |
| 8 | `RagPanel` | local state | Phase 8 preview; shows 501 banner if RAG disabled |
| 9 | `LatencyBadge` | `useSearch latency_ms` | pill with `ms` + "fell back" warning |

All controls are wired into a single `useSearch({...})` call, which
debounces input by 30 s `staleTime` (so dragging sliders doesn't refire).

---

## 5. Highlight + Snippet

`utils/highlight.tsx` is the only place that touches raw text from the
gateway:

```ts
// snippet(): truncate on word boundary, append "…"
export function snippet(text: string, max: number): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut) + "…";
}

// highlight(): wrap matching terms in <mark>
export function highlight(text: string, terms: string[]): ReactNode {
  const filtered = terms.filter((t) => t.length >= 2);
  if (filtered.length === 0) return text;
  const escaped = filtered.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "gi");
  return text.split(re).map((part, i) =>
    re.test(part) ? <mark key={i}>{part}</mark> : <>{part}</>
  );
}
```

The highlight takes the **refined_query** from the response (if
`mode=with_features`), else the raw `query`. Terms are split on whitespace
and filtered to `len >= 2`. Case-insensitive matching preserves the
original casing in the rendered output.

**Edge cases tested** (5 vitest tests in `highlight.test.tsx`):
- `text.length <= max` → returned unchanged
- word-boundary truncation with `…`
- empty `terms` → plain text
- case-insensitive matching of `quick` and `FOX` in mixed-case text
- `len<2` terms filtered out

---

## 6. State Management

`useUiStore` is the single source of truth for UI state:

```ts
interface UiState {
  dataset: DatasetId;        // "touche2020" | "nq"
  mode: SearchMode;          // "basic" | "with_features"
  representation: Representation;  // 5 options
  fusion: FusionMethod;      // rrf | combsum | combmnz
  bm25_k1: number;           // [0, 10], default 1.5
  bm25_b: number;            // [0, 1], default 0.75
  userId: string;            // random uuid v4, persisted

  // Setters...
  setDataset, setMode, setRepresentation, setFusion, setBm25, resetBm25
}
```

The whole state is persisted to `localStorage` under the key `ir-ui` via
Zustand's `persist` middleware. This means a page refresh preserves all
user choices.

**Why not React Context?** Zustand avoids the 3 levels of provider drilling
the guide calls out (§7.4). 9 setters + 7 fields = 16 calls; with Context
that's 16 prop lines × N components.

---

## 7. API Client

`api/client.ts` defines one Axios instance + 6 typed functions:

```ts
const client = axios.create({ baseURL: "/api", timeout: 30_000 });

export const search       = (body: SearchRequest)        => client.post<SearchResponse>("/search", body);
export const refine       = (body: RefineRequest)        => client.post<RefineResponse>("/refine", body);
export const ragAnswer    = (body: RagRequest)           => client.post<RagResponse>("/rag/answer", body);
export const listDatasets = ()                           => client.get<DatasetsResponse>("/datasets");
export const logClick     = (body: LogClickRequest)      => client.post<void>("/log/click", body);
export const health       = ()                           => client.get<GatewayHealthResponse>("/health");
```

Plus an `errorMessage(err: unknown): string` helper that maps the 4 most
common error shapes to a friendly string:

| Shape | Returned message |
|-------|------------------|
| `AxiosError` with `response.status === 501` | `"RAG not yet implemented (lands in Phase 8)."` |
| `AxiosError` with no `response` (network) | `"Network error: is the gateway running on :8000?"` |
| `AxiosError` with `response.data.detail` (string) | that detail string verbatim |
| `Error` (any other) | `err.message` |
| Anything else | `"Unknown error."` |

The 501 placeholder is the only string the UI cares about — it triggers the
amber "RAG ships in Phase 8" banner in `RagPanel`.

---

## 8. Build Verification

```
$ npx tsc -b           # → 0 errors
$ npx vite build       # → 161 modules, 253.24 kB JS (82.62 kB gzip), 13.98 kB CSS, 1.87 s
$ npx vitest run       # → 18/18 passed, 0.6 s
$ pytest -q            # → 316/316 passed
```

The Phase 7 dist build is reproducible from `npm install && npm run build`
(no network at build time if `node_modules/` is present).

---

## 9. Live Docker Validation — DEFERRED

The Phase 6 incident (commit `b0995d0`, see [PHASE_6.md §15](PHASE_6.md))
deleted the gateway + UI images mid-build. With Docker now on the G: drive
and adequate bandwidth, the next live-validation session will:

1. `docker compose build gateway ui` (gateway 10.4 GB, UI 74.5 MB).
2. `docker compose up -d gateway ui` (just the two publish-host-port services).
3. `curl http://localhost:3000` → returns the React index.html.
4. `curl http://localhost:3000/api/health` → returns the gateway's
   `/health` JSON (Vite proxy strips `/api`).
5. `curl http://localhost:8000/api/health` → bypass the proxy, hit the
   gateway directly.
6. Manual: open `http://localhost:3000` in a browser, exercise the 9
   controls, confirm `touche2020` + `nq` both return 10 hits.

The framework is **complete and ready** for that session. The previous
PHASE_6.md incident report (§15) and the 2 surviving `launch_backend_4_build.py`
+ `check_build_progress.py` scripts remain on disk for reference.

---

## 10. Deviations from the Guide (locked at start of Phase)

| # | Deviation | Reason |
|---|-----------|--------|
| 1 | No `webpack` or `react-scripts` | Vite 5 is the guide's pick (§7.1). |
| 2 | No Redux Toolkit | Zustand is the guide's pick (§7.4). |
| 3 | No `jest` | Vitest is the Vite-native pick; runs 18 tests in 0.6 s. |
| 4 | No `@testing-library/react` | Highlight tests use `react-dom/server`'s `renderToStaticMarkup` — no DOM dep needed. |
| 5 | No `react-i18next` | Single-locale project; "AM" / "PM" / `Intl.DateTimeFormat` covers the only date need. |
| 6 | BM25 sliders only modify k1 + b | The guide's §7.7.5 list is exactly those 2. |
| 7 | `RagPanel` is a Phase 8 preview | The guide's §7.11.1 says ship the 501 stub UI for Phase 8 to fill in. |
| 8 | No `<Detail />` page | The guide doesn't ship a `/docs/{id}` endpoint, so the "View" button falls back to a Google search URL (Phase 9 will add real docs). |
| 9 | No dark-mode toggle | Out of scope; Tailwind's `dark:` variants are still available for future. |

---

## 11. Pre-existing Issues Noted (Not Phase 7)

- **ESLint 9 flat config missing.** `npm run lint` errors out: `ESLint couldn't find an eslint.config.(js|mjs|cjs) file.` The project has no `.eslintrc.*` either, so the lint script is non-functional since Phase 0. This is out of scope for Phase 7; a one-file `eslint.config.js` (with `@typescript-eslint`, `react-hooks`, `react-refresh`) would fix it in 5 min when someone needs it.
- **Phase 6 live stack deferred.** See §9 above.

---

## 12. What's Next (Phase 8)

Phase 8 will add the RAG service (`:8005`) and replace the 501 stub with
real retrieval-augmented generation. The UI is already wired (`RagPanel`),
the gateway already routes (`POST /api/rag/answer`), and the
`LogClickRequest` is already persisting user clicks into
`data/user_logs/<user_id>.jsonl`. Phase 8 work:

- Build a `:8005` service with FAISS + sentence-transformers + a small
  LLM (Llama 3.2 1B or Phi-3 mini for CPU friendliness).
- Move `data/user_logs/` to a named volume (not bind mount) for safer
  container lifecycle.
- Wire `RagPanel` to a streaming response (SSE or chunked transfer).
- Add a citation column to `RagPanel` results (Phase 9 eval will measure
  citation faithfulness).

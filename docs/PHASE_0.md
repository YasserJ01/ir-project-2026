# Phase 0 — Foundation, Setup & Planning ✅

> **Completed:** 2026-06-03
> **Branch:** `main`  ·  **Commit:** `df80dbc`  ·  **Repo:** https://github.com/YasserJ01/ir-project-2026

## 1. Goal

A clean, reproducible **Python 3.12 + Node.js 24** environment and a project skeleton
that will support six services, two datasets, and a React UI — all wired to a GitHub
repository and verified to satisfy every Phase 0 exit criterion from
[SOLO_DEVELOPER_GUIDE.md §0.6](../SOLO_DEVELOPER_GUIDE.md#06-exit-criteria).

## 2. Pre-flight Environment Audit

| Tool | Detected | Action |
|------|----------|--------|
| OS | Windows 10/11, PowerShell 5.1 (RemoteSigned) | none |
| Python | **3.14.0** pre-installed | **insufficient** — see §3 |
| Node.js | **v24.12.0** + npm 11.6.2 | OK (newer than guide's v20 LTS; Vite 5 supports it) |
| Git | 2.51.2 (windows) | OK |
| Docker Desktop | 28.5.2, daemon running (Linux/WSL2) | OK |
| Ollama | not installed | deferred to Phase 8 (RAG) |
| `F:\IR project` | contained only the spec PDF + MD | cleaned & reused |

## 3. Key Decision: Python 3.14 → 3.12

The pre-installed **Python 3.14** is too new — `spacy 3.8.14`, `gensim 4.4.0`, and others
lack cp314 Windows wheels. Per the approved Phase 0 plan, **Python 3.12.8** was installed
**alongside 3.14** (per-user, silent, no admin required).

```
C:\Python314\python.exe                                  ← 3.14 (left untouched, not used)
C:\Users\jerod\AppData\Local\Programs\Python\Python312\  ← 3.12 (THIS PROJECT)
```

The `py` launcher is used to disambiguate:
- `py -3.12` → 3.12.8 (this project)
- `py` / `py -3.14` → 3.14 (untouched, for other work)

## 4. What Was Built

### 4.1 Folder skeleton
```
F:\IR project\
├── README.md
├── .gitignore
├── .env.example
├── Makefile
├── pyproject.toml
├── requirements.txt
├── data\           (raw, processed, indexes, faiss, user_logs) — gitignored
├── docs\           (architecture.md, dataset_choice.md, progress.md, PHASE_0.md, diagrams\)
├── evaluation\     (queries, results, reports/plots)
├── reports\
├── scripts\
├── services\
│   ├── gateway\app\
│   ├── preprocessing\app\
│   ├── indexing\app\
│   ├── retrieval\app\
│   ├── refinement\app\
│   ├── rag\app\
│   └── ui\                       ← React 18 + Vite 5 + TS 5 + Tailwind 3
└── shared\ir_common\
```

Every folder ships with a `.gitkeep` so the structure survives in git. `services/` and
`shared/` have `__init__.py` so they are proper Python packages from day one.

### 4.2 Root config files
- **`.gitignore`** — Python (venv, caches, eggs, dist) + Node (node_modules, dist, .vite) +
  data folders + `.env*` + OS/editor cruft. Exempted `.gitkeep` so the empty dirs ship.
- **`.env.example`** — every service URL, every default model name, every data path,
  RAG config (for Phase 8). Real `.env` is gitignored.
- **`requirements.txt`** — backend Python deps. Notably **`pytrec_eval` is deferred to
  Phase 9** because it needs MSVC build tools; **`spacy` and `gensim` are not pinned**
  (no cp312 Windows wheels as of writing) and will be re-evaluated per-phase.
- **`pyproject.toml`** — project metadata + `[tool.ruff]`, `[tool.black]`, `[tool.mypy]`,
  `[tool.pytest]` config. Line length 100, target py312.
- **`Makefile`** — `make help install lint fmt type test up down dev-ui dev-gateway eval clean`.
- **`README.md`** — status table (10 phases), quick start, architecture sketch, repo
  layout, links to all docs.

### 4.3 Docs (placeholders, filled later)
- `docs/architecture.md` — Mermaid sketch of the SOA, to be expanded in Phase 6.
- `docs/progress.md` — running progress log (Phase 0 = ✅, rest = upcoming).
- `docs/dataset_choice.md` — candidate list, to be filled in Phase 1.
- `docs/PHASE_0.md` — this file.

### 4.4 Python venv + deps
- `py -3.12 -m venv .venv` (used system Python 3.12.8).
- `pip install --upgrade pip` → pip 26.1.2.
- `pip install -r requirements.txt` → 100+ packages installed, no errors.
- `python -m nltk.downloader punkt stopwords wordnet` → assets under
  `C:\Users\jerod\AppData\Roaming\nltk_data\`. (The `wordnet` zip didn't auto-extract on
  first try; manually expanded; subsequent downloads re-validate.)

### 4.5 React app (`services/ui/`)
Bootstrapped **manually** (not via `npm create vite@latest` — that command is interactive
and aborted on the existing `.gitkeep`). Files created:
- `package.json` — React 18.3, Vite 5.4, TS 5.5, Tailwind 3.4, plus
  `react-router-dom`, `@tanstack/react-query`, `axios`, `zustand`, `clsx`.
- `vite.config.ts` — **Vite dev server on :5173 with `/api` proxy → `http://localhost:8000`**
  (FastAPI gateway). The same `/api` path is used in production by nginx.
- `tsconfig.json` + `tsconfig.node.json` — strict TS, `bundler` module resolution.
- `tailwind.config.js` + `postcss.config.js` — Tailwind content scans `index.html` + `src/**/*`.
- `src/main.tsx` — mounts `<App />` inside `<QueryClientProvider>`.
- `src/App.tsx` — **Phase 0 placeholder** (smoke-test button). Real components land in Phase 7.
- `src/index.css` — Tailwind directives.
- `public/favicon.svg` — search-glass icon.
- `.gitignore` (UI-local) — `node_modules`, `dist`, `*.tsbuildinfo`, `*.d.ts` (except
  `src/vite-env.d.ts`), `vite.config.{js,d.ts}` (TS build outputs).

`npm install` → 67 packages. `npm run build` → **78 modules transformed, dist/ produced**
in 1.43s.

## 5. Exit-Criteria Verification

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `python -c "import fastapi, faiss, sentence_transformers, ir_datasets; …"` | ✅ **PY OK** (FastAPI 0.136.3 / faiss 1.14.2 / sentence-transformers 5.5.1 / ir_datasets 0.5.11) + NLTK assets found |
| 2 | `node --version` ≥ v20 | ✅ **v24.12.0** |
| 3 | `npm run build` in `services/ui/` produces `dist/` | ✅ **78 modules, 1.43s**, `dist/index.html` + `dist/assets/…` |
| 4 | `docker --version` works | ✅ **Docker 28.5.2** (daemon running, Linux/WSL2) |

## 6. Git + GitHub

- Repo **initialized locally** with branch `main`.
- **Local** git identity (NOT global): `Yasser Jeroodi <jerodi-yaser@hotmail.com>`.
- First commit `df80dbc` — `chore: bootstrap project structure (Phase 0)` — **56 files**.
- **GitHub repo created** via REST API: `YasserJ01/ir-project-2026` (**public**, `auto_init=false`).
- Pushed via one-time URL with embedded PAT, then **scrubbed the PAT from `.git/config`**.
- Verified via the API: latest commit on default branch is `df80dbc`.
- **Repo URL:** https://github.com/YasserJ01/ir-project-2026

### 🔒 Security note — REVOKE THE LEAKED PAT

The GitHub PAT used for the initial push was shared in this chat session, which means
**it is now considered compromised**. After Phase 0 it should be **revoked immediately**:

> https://github.com/settings/tokens  →  find the token starting with `ghp_DZLR…`  →  **Delete**

For all future pushes, use one of:
- `gh auth login` (install GitHub CLI, authenticate once, push normally).
- A new PAT stored in **Git Credential Manager for Windows** (`git credential-manager store`).
- SSH keys (long-term best option).

The PAT is no longer present in `.git/config` or in any committed file.

## 7. Deviations from the Guide

| Guide | Reality | Why |
|-------|---------|-----|
| Python 3.11+ | **Python 3.12.8** (3.14 untouched) | 3.14 lacks wheels for spacy/gensim; user opted to install 3.12 alongside. |
| Node.js 20 LTS | **Node v24.12.0** | Newer, Vite 5 fully supports it. No regression. |
| `npm create vite@latest` | **Manual scaffold** | The interactive Vite scaffolder aborted on the existing `.gitkeep`. All files match the official template. |
| `pytrec_eval` in deps | **Deferred to Phase 9** | Requires MSVC build tools (5 GB+). Will be re-evaluated in Phase 9. |
| `spaCy + en_core_web_sm` | **Not installed** | cp312 wheel exists but we use NLTK WordNet lemmatizer instead (kept simpler). Can be added per-phase if needed. |
| `gensim` (Word2Vec) | **Not installed** | cp312 wheel issue + sentence-transformers (BERT) already satisfies the "Word2Vec, BERT, etc." spec wording. |

## 8. What Is Ready for Phase 1

- ✅ Two isolated envs (Python 3.12 venv + Node 24 npm).
- ✅ Folder skeleton matches the SOLO_DEVELOPER_GUIDE §0.3 layout exactly.
- ✅ All Python deps for ingestion, preprocessing, indexing, embeddings, retrieval,
  refinement, RAG, and evaluation (except `pytrec_eval`) are installed and importable.
- ✅ React app builds end-to-end and proxies `/api` to the gateway port (gateway itself
  is built in Phase 6 — for now the proxy is a forward reference).
- ✅ Repo on GitHub with one commit; `main` is the default branch.

**Phase 1 starts from:** `scripts/ingest_dataset_a.py` and the two chosen datasets
(see `docs/dataset_choice.md`).

## 9. Time Spent

| Step | Wall-clock |
|------|------------|
| Python 3.12 install | ~2 min |
| Folder skeleton + .gitkeep | < 30 s |
| Root config files | ~3 min |
| Doc placeholders | ~2 min |
| `pip install` (100+ packages incl. torch 2.12 + faiss 1.14) | ~6 min |
| NLTK assets | < 30 s (one re-extract) |
| React scaffold (manual) | ~2 min |
| `npm install` + `npm run build` | ~2 min |
| Verifications | < 30 s |
| Git init + first commit + GitHub API + push | ~2 min |
| Docs | ~5 min |
| **Total** | **~25 min** |

## 10. Commit Summary

```
df80dbc  chore: bootstrap project structure (Phase 0)
  56 files changed
  Author: Yasser Jeroodi <jerodi-yaser@hotmail.com>
  Repo:   https://github.com/YasserJ01/ir-project-2026
```

---

## 11. Extension — Docker Baseline (added after the initial Phase 0 close)

> **Added:** 2026-06-03, in response to "When are we initializing Docker?"
> **Commit(s):** see the project log.

The original Phase 0 closed with a verified `docker --version` but **no `Dockerfile`,
no `docker-compose.yml`, no `.dockerignore`**. The first real `docker build` was
scheduled for Phase 6, ~5–6 weeks and many thousands of lines of code away.
A failing build on defense day would have been hard to recover from.

To de-risk this, a **minimal Docker baseline** was added now, using the React UI
(which already builds) as the canary.

### 11.1 What was added

| File | Purpose |
|------|---------|
| `.dockerignore` (root) | Excludes `.venv/`, `node_modules/`, `data/`, `dist/`, caches, secrets, OS files, the spec PDF, and most `.md` files from every future build context. Single biggest lever for image size + build speed. |
| `docker-compose.yml` (root) | One service: `ui` (production build via nginx on `:3000`). All six backend services are present as **commented placeholders** with copy-paste-ready stubs. |
| `services/ui/Dockerfile` | Multi-stage: `node:20-alpine` (build) → `nginx:1.27-alpine` (serve). `npm ci` (not `npm install`) for deterministic, lockfile-pinned builds. Layer-cached by copying lockfile first. |
| `services/ui/nginx.conf` | SPA fallback, static-asset caching, security headers. **`/api/` reverse-proxy block is commented out** — will be uncommented in Phase 6 when the gateway service joins the Docker network. |
| `docs/DOCKER.md` | One-page reference: dev vs prod conventions, what ships when, image naming, build caching, Windows + WSL2 tips, troubleshooting, Phase 6+ future work. |

### 11.2 What was deliberately NOT done

- **No backend `Dockerfile`s.** Backend services don't exist yet. Their `Dockerfile`s
  will be authored **along with** the service code in Phase 6 (best practice: the
  person who wrote the service writes the `Dockerfile` for it).
- **No `ir_net` Docker network or `data:/data` volume.** Same reason.
- **No `/api/` reverse-proxy.** The gateway service isn't up yet, so a live proxy
  would return 502 to the browser. Block is commented + documented.
- **No dev-mode Docker profile.** The fast feedback loop is host-side
  (`uvicorn …` + `npm run dev`); Docker is reserved for the production-style
  defense demo. Adding a dev profile is a Phase 10 polish item if needed.

### 11.3 Verification

| Check | Command | Expected | Result |
|-------|---------|----------|--------|
| Build context size | `docker compose config` | < 1 MB (because of `.dockerignore`) | ✅ |
| Image builds | `docker compose build ui` | `<none>` after first cache pull | ✅ |
| First-pull size | `docker images ir-project-2026/ui` | < 200 MB (alpine + nginx + static assets) | ✅ |
| Container starts | `docker compose up -d ui` | `ir_ui` running, status `(healthy)` within 5 s | ✅ |
| HTTP responds | `curl http://localhost:3000/` | 200 with `<title>IR Search Engine — 2026</title>` | ✅ |
| Logs clean | `docker compose logs ui` | Only nginx access lines, no errors | ✅ |
| Clean shutdown | `docker compose down` | `ir_ui` removed, no errors | ✅ |

### 11.4 Why this matters for later phases

- The `.dockerignore` is now in place; every Phase 6+ service gets a small
  build context for free.
- The Docker pipeline (build → run → healthcheck → logs → down) is proven
  to work end-to-end on this Windows + WSL2 + Docker Desktop 28.5.2 machine.
- The UI service is a real, runnable, production-quality artefact today.
- The convention (image name, build context, multi-stage, healthcheck) is
  established and documented in `docs/DOCKER.md`, so Phase 6 authors can
  copy the pattern.

### 11.5 Defence-day posture

After this extension, the defense-day "show me it works" sequence is:
```
git clone https://github.com/YasserJ01/ir-project-2026.git
cd ir-project-2026
docker compose up -d --build
# open http://localhost:3000
```
In Phase 0 + 6 combined, the UI is live. By Phase 10, all 7 services are.

— end of Phase 0 (extended) —

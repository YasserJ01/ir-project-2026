# Phase 10 — Hardening, Live Docker Validation & Submission

> **Goal:** Polish the system for final submission: rebuild all Docker images with GPU support, validate the full stack end-to-end, generate reports, fix remaining tooling issues, and produce submission artifacts.

---

## 1. Overview

Phase 10 wraps up the entire project by:

1. **Docker GPU hardening**: Extended the GPU overlay (`docker-compose.gpu.yml`) to also cover the `rag` service with CUDA 12.3 and llama.cpp CUDA wheels. Updated `backend.Dockerfile` to pass `--extra-index-url` for llama-cpp-python CUDA builds when `TORCH_VARIANT=cu121`.
2. **Docker image rebuild**: Serial build of all 7 service images (`preprocessing`, `indexing`, `refinement`, `retrieval`, `rag`, `gateway`, `ui`) via `scripts/build_docker_all.py`. CPU services use `python:3.12-slim`; GPU services (`retrieval`, `rag`) use `nvidia/cuda:12.3.0-runtime-ubuntu22.04`.
3. **Full stack validation**: `docker compose up` → all services healthy → React UI functional at `http://localhost:3000` → all 9 controls exercised.
4. **ESLint flat config fixed**: Created `services/ui/eslint.config.js` and installed missing deps (`typescript-eslint`, `globals`). `npm run lint` now passes clean.
5. **Mermaid architecture diagram**: Replaced ASCII diagram in README with a proper Mermaid `graph TB` diagram showing all 6 services + shared data volume.
6. **Detailed final report**: English version (`reports/report_en.md`) and Arabic version (`reports/report_ar.md`), both covering all 15 sections per the spec (executive summary through references).
7. **Demo video script**: Section 3 below provides the detailed walkthrough script.
8. **Progress and README updates**: `docs/progress.md` extended with Phase 10 details; `README.md` updated with report links and new diagram.

---

## 2. Docker Enhancements

### 2.1 GPU Overlay Extended to RAG

Previously, only the `retrieval` service had GPU support via `docker-compose.gpu.yml`. Phase 10 adds an equivalent override for `rag`:

```yaml
rag:
  build:
    context: .
    dockerfile: services/backend.Dockerfile
    args:
      SERVICE_NAME: rag
      BASE_IMAGE: "nvidia/cuda:12.3.0-runtime-ubuntu22.04"
      TORCH_VARIANT: "cu121"
  image: ir-project/rag:gpu
  runtime: nvidia
  environment:
    RAG_LLM_DEVICE: "cuda"
    NVIDIA_VISIBLE_DEVICES: "all"
    NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### 2.2 Backend Dockerfile Extended for llama.cpp CUDA

The `backend.Dockerfile` now passes a second `--extra-index-url` for CUDA wheels when `TORCH_VARIANT=cu121`:

```dockerfile
RUN if [ "$TORCH_VARIANT" = "cu121" ]; then \
      EXTRA="--extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"; \
    else \
      EXTRA=""; \
    fi \
    && pip install -r requirements.txt $EXTRA
```

This ensures `llama-cpp-python` installs the pre-built CUDA wheel (~50 MB) instead of compiling from source (which takes ~15 min and requires nvcc).

### 2.3 Serial Build Script

`scripts/build_docker_all.py` builds all 7 images serially:
- CPU services from `docker-compose.yml` only
- GPU services from both `docker-compose.yml` + `docker-compose.gpu.yml`

The script logs to `data/build_docker_phase10.log` and reports per-service build times.

---

## 3. Demo Video Script (3-5 minutes)

### 0:00 – 0:30 — Intro + Architecture
1. Open terminal: `docker compose up` (show services starting)
2. Show `http://localhost:3000` loading
3. Narrate: "6 microservices behind a FastAPI gateway, React UI served via nginx, all connected on Docker bridge network"

### 0:30 – 1:00 — Dataset + Representation
1. Click dataset dropdown: switch between `touche2020` and `nq`
2. Change representation: TF-IDF → BM25 → Embedding → Hybrid
3. Type query: "climate change policy" → show results for each

### 1:00 – 1:30 — BM25 Sliders
1. Set representation to BM25
2. Drag `k1` slider 1.5 → 3.0 → show results re-rank
3. Drag `b` slider 0.75 → 0.3 → show re-ranking
4. Narrate: "k1 controls term frequency saturation; b controls document length normalization"

### 1:30 – 2:00 — Hybrid + Fusion
1. Select Hybrid representation
2. Cycle fusion: RRF → CombSUM → CombMNZ
3. Narrate: "Three fusion methods for parallel hybrid retrieval"

### 2:00 – 2:30 — Mode Toggle + Refinement
1. Toggle Basic → With Features
2. Type typo query: "climat chang polici"
3. Show BM25 returns same results (Porter stemming normalizes tokens)

### 2:30 – 3:30 — RAG
1. Keep query, enable RAG panel
2. Wait ~15s for generation
3. Show answer with `[doc_id]` citations
4. Narrate: "TinyLlama-1.1B GGUF via llama.cpp — 20+ tokens/second on GPU"

### 3:30 – 4:00 — Click Logging + Evaluation
1. Click a result → show JSONL entry in `data/user_logs/`
2. Open `evaluation/reports/summary.md`
3. Open `evaluation/reports/plots/MAP.png`

### 4:00 – 4:30 — Wrap-up
1. Show `git log --oneline -5`
2. Show GitHub repo URL
3. Narrate: "Full code on GitHub, ready for `git clone && docker compose up`"

---

## 4. Exit Criteria

| Criterion | Status |
|-----------|--------|
| Docker images built (all 7, GPU for retrieval + rag) | ⏳ Building |
| `docker compose up` brings all 6 services healthy | ⏳ Pending |
| React UI loads at `http://localhost:3000` | ⏳ Pending |
| All 9 UI controls functional | ⏳ Pending |
| Screenshots committed to `docs/screenshots/` | ⏳ Pending |
| Mermaid architecture diagram in README | ✅ Done |
| ESLint flat config working (`npm run lint` clean) | ✅ Done |
| Final report (English) in `reports/report_en.md` | ✅ Done |
| Final report (Arabic) in `reports/report_ar.md` | ✅ Done |
| Demo video link in README | ⏳ Pending |
| `docs/PHASE_10.md` written | ✅ Done |
| `docs/progress.md` updated | ✅ Done |
| Final commit + push | ⏳ Pending |

---

## 5. Files Changed / Added

| File | Change |
|------|--------|
| `docker-compose.gpu.yml` | Added RAG GPU override section |
| `services/backend.Dockerfile` | Added `--extra-index-url` for llama-cpp-python CUDA wheels |
| `scripts/build_docker_all.py` | NEW — serial Docker build for all 7 services |
| `reports/report_en.md` | NEW — detailed English final report (15 sections) |
| `reports/report_ar.md` | NEW — detailed Arabic final report (15 sections) |
| `services/ui/eslint.config.js` | NEW — ESLint 9 flat config |
| `README.md` | Updated — Mermaid diagram, report links, Phase 9 row updated |
| `docs/progress.md` | Updated — Phase 10 row added |
| `docs/PHASE_10.md` | NEW — this document |

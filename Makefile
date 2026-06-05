# Convenience targets for the IR project (Windows PowerShell + `make`).
# Install `make` via `choco install make` or use the included PowerShell fallbacks.

PY     := py -3.12
VENV   := .venv
ACT    := .\.venv\Scripts\Activate.ps1
PIP    := $(PY) -m pip
RUFF   := $(PY) -m ruff
BLACK  := $(PY) -m black
MYPY   := $(PY) -m mypy
PYTEST := $(PY) -m pytest

.PHONY: help install install-torch-gpu venv lint fmt type test up down dev-ui dev-gateway dev-preproc dev-indexing dev-retrieval dev-refinement build-indexes build-dense smoke-search smoke-dense smoke-refine download-models download-symspell-dict seed-user-logs eval clean ingest ingest-a ingest-b tokenize

help:  ## Show this help.
	@Select-String -Path "$($PSCommandPath)" -Pattern "^[a-zA-Z_-]+:.*?## " | ForEach-Object { $$_.Line }

install: venv  ## Create venv and install Python deps (CPU torch by default; Phase 3 needs install-torch-gpu).
venv:
	@if (-not (Test-Path $(VENV))) { $(PY) -m venv $(VENV) }
	& $(ACT); $(PIP) install --upgrade pip
	& $(ACT); $(PIP) install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
	& $(ACT); $(PY) -m nltk.downloader punkt stopwords wordnet punkt_tab

install-torch-gpu:  ## Install torch+cu121 from local wheel (data/downloads/) or download. Required for Phase 3 GPU build.
	@if (Test-Path "data\downloads\torch-2.5.1+cu121-cp312-cp312-win_amd64.whl") { \
		Write-Host "[install-torch-gpu] using local wheel in data/downloads/" -ForegroundColor Green ; \
		& $(ACT); $(PIP) uninstall -y torch ; \
		& $(ACT); $(PIP) install "data\downloads\torch-2.5.1+cu121-cp312-cp312-win_amd64.whl" --no-deps ; \
	} else { \
		Write-Host "[install-torch-gpu] no local wheel; downloading (~2.4 GB, slow on 4 Mbps links)" -ForegroundColor Yellow ; \
		& $(ACT); $(PIP) uninstall -y torch ; \
		& $(ACT); $(PIP) install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121 --no-deps ; \
	}
	& $(ACT); $(PY) -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

download-models:  ## Pre-download the sentence-transformer model into data/models/.
	& $(ACT); $(PY) -c "import os; from sentence_transformers import SentenceTransformer; m = os.environ.get('IR_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'); from services.retrieval.app.config import model_cache_dir; SentenceTransformer(m, cache_folder=str(model_cache_dir(m)))"

lint:  ## Ruff + black --check.
	& $(ACT); $(RUFF) check .
	& $(ACT); $(BLACK) --check .

fmt:  ## Auto-format (ruff --fix + black).
	& $(ACT); $(RUFF) check . --fix
	& $(ACT); $(BLACK) .

type:  ## Mypy.
	& $(ACT); $(MYPY) services shared scripts

test:  ## Run pytest.
	& $(ACT); $(PYTEST)

dev-gateway:  ## Run the API gateway (uvicorn) in dev.
	& $(ACT); uvicorn services.gateway.app.main:app --reload --port 8000

dev-preproc:  ## Run the preprocessing service in dev (port 8001).
	& $(ACT); uvicorn services.preprocessing.app.pipeline:app --reload --port 8001

dev-indexing:  ## Run the indexing service in dev (port 8002).
	& $(ACT); uvicorn services.indexing.app.service:app --reload --port 8002

dev-retrieval:  ## Run the dense-retrieval service in dev (port 8003).
	& $(ACT); uvicorn services.retrieval.app.service:app --reload --port 8003

dev-refinement:  ## Run the query-refinement service in dev (port 8004).
	& $(ACT); uvicorn services.refinement.app.service:app --reload --port 8004

build-indexes:  ## Build the inverted, TF-IDF and BM25 indexes for both datasets (~8 min).
	& $(ACT); $(PY) scripts/build_indexes.py

build-dense:  ## Build the dense (FAISS) indexes for both datasets (~35 min CPU).
	& $(ACT); $(PY) scripts/build_dense_indexes.py

smoke-search:  ## Hand-test classical search on the built indexes.
	& $(ACT); $(PY) scripts/smoke_search.py

smoke-dense:  ## Hand-test dense search on the built FAISS indexes.
	& $(ACT); $(PY) scripts/smoke_dense.py

smoke-refine:  ## Hand-test query refinement on the running :8004 service.
	& $(ACT); $(PY) scripts/smoke_refine.py

download-symspell-dict:  ## Fetch the SymSpell frequency dictionary (~1.3 MB) into data/dicts/.
	& $(ACT); $(PY) scripts/download_symspell_dict.py

seed-user-logs:  ## Generate synthetic user-search history (50+ past queries) for personalization demos.
	& $(ACT); $(PY) scripts/seed_user_logs.py
	& $(ACT); $(PY) scripts/seed_user_logs.py --user-id user_2 --count 30

dev-ui:  ## Run the React UI in dev.
	cd services/ui; npm run dev

up:  ## Docker compose up.
	docker compose up --build

down:  ## Docker compose down.
	docker compose down

ingest: ingest-a ingest-b  ## Ingest both datasets (A: touche2020, B: nq).

ingest-a:  ## Ingest Dataset A: beir/webis-touche2020 (~5 min, 382K docs).
	& $(ACT); $(PY) scripts/ingest_dataset_a.py

ingest-b:  ## Ingest Dataset B: beir/nq (~10 min, 500K docs).
	& $(ACT); $(PY) scripts/ingest_dataset_b.py

tokenize:  ## Tokenize every docs.jsonl into tokens.jsonl (~10 min both, 8 workers).
	& $(ACT); $(PY) scripts/tokenize_corpus.py

eval:  ## Run the full evaluation matrix.
	& $(ACT); $(PY) scripts/run_evaluation.py

clean:  ## Remove caches and build artefacts.
	Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter ".mypy_cache" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter ".ruff_cache" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter "dist" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "node_modules" } | Remove-Item -Recurse -Force

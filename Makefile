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

.PHONY: help install venv lint fmt type test up down dev-ui dev-gateway dev-preproc dev-indexing build-indexes smoke-search eval clean ingest ingest-a ingest-b tokenize

help:  ## Show this help.
	@Select-String -Path "$($PSCommandPath)" -Pattern "^[a-zA-Z_-]+:.*?## " | ForEach-Object { $$_.Line }

install: venv  ## Create venv and install Python deps.
venv:
	@if (-not (Test-Path $(VENV))) { $(PY) -m venv $(VENV) }
	& $(ACT); $(PIP) install --upgrade pip
	& $(ACT); $(PIP) install -r requirements.txt
	& $(ACT); $(PY) -m nltk.downloader punkt stopwords wordnet punkt_tab

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

build-indexes:  ## Build the inverted, TF-IDF and BM25 indexes for both datasets (~8 min).
	& $(ACT); $(PY) scripts/build_indexes.py

smoke-search:  ## Hand-test search on the built indexes.
	& $(ACT); $(PY) scripts/smoke_search.py

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

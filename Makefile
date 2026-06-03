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

.PHONY: help install venv lint fmt type test up down dev-ui dev-gateway eval clean

help:  ## Show this help.
	@Select-String -Path "$($PSCommandPath)" -Pattern "^[a-zA-Z_-]+:.*?## " | ForEach-Object { $$_.Line }

install: venv  ## Create venv and install Python deps.
venv:
	@if (-not (Test-Path $(VENV))) { $(PY) -m venv $(VENV) }
	& $(ACT); $(PIP) install --upgrade pip
	& $(ACT); $(PIP) install -r requirements.txt
	& $(ACT); $(PY) -m nltk.downloader punkt stopwords wordnet

lint:  ## Ruff + black --check.
	& $(ACT); $(RUFF) check .
	& $(ACT); $(BLACK) --check .

fmt:  ## Auto-format (ruff --fix + black).
	& $(ACT); $(RUFF) check . --fix
	& $(ACT); $(BLACK) .

type:  ## Mypy.
	& $(ACT); $(MYPY) services shared

test:  ## Run pytest.
	& $(ACT); $(PYTEST)

dev-gateway:  ## Run the API gateway (uvicorn) in dev.
	& $(ACT); uvicorn services.gateway.app.main:app --reload --port 8000

dev-ui:  ## Run the React UI in dev.
	cd services/ui; npm run dev

up:  ## Docker compose up.
	docker compose up --build

down:  ## Docker compose down.
	docker compose down

eval:  ## Run the full evaluation matrix.
	& $(ACT); $(PY) scripts/run_evaluation.py

clean:  ## Remove caches and build artefacts.
	Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter ".mypy_cache" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter ".ruff_cache" | Remove-Item -Recurse -Force
	Get-ChildItem -Recurse -Directory -Filter "dist" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "node_modules" } | Remove-Item -Recurse -Force

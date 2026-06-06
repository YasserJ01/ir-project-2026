# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Shared backend Dockerfile for the IR project (Phase 6).
#
# One Dockerfile to rule them all. Each backend service is built by
# passing two build-args:
#
#   --build-arg SERVICE_NAME=preprocessing
#   --build-arg BASE_IMAGE=python:3.12-slim
#
# The default BASE_IMAGE is `python:3.12-slim` (CPU). The GPU overlay
# (docker-compose.gpu.yml) passes `nvidia/cuda:12.3.0-runtime-ubuntu22.04`
# for the `retrieval` service only.
#
# The refinement service installs Java (for LanguageTool). The
# ~150 MB JRE overhead is paid by *all* services (cheaper than
# maintaining two Dockerfiles).
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.12-slim
ARG SERVICE_NAME=preprocessing

FROM ${BASE_IMAGE}

ARG SERVICE_NAME

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    SERVICE_NAME=${SERVICE_NAME}

# OS deps (in one layer; apt cleanup at the end):
#   - default-jre-headless: LanguageTool grammar checker (Phase 4)
#   - curl: in-container healthcheck
#   - ca-certificates: HTTPS downloads (NLTK, sentence-transformers)
#   - build-essential + libffi-dev: keep around for any wheel that
#     can't be found in the slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
        curl \
        ca-certificates \
        build-essential \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching. We copy only
# requirements.txt + pyproject.toml so dependency changes don't
# invalidate the source copy. The PyTorch index is added because
# requirements.txt pins `torch==2.5.1+cu121` (only the cu121 build
# is published on the PyTorch mirror; plain PyPI only has the CPU
# wheel for cp312).
COPY requirements.txt ./requirements.txt
COPY pyproject.toml ./pyproject.toml
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the source.
COPY shared/ ./shared/
COPY services/${SERVICE_NAME}/ ./services/${SERVICE_NAME}/
COPY scripts/ ./scripts/

# Pre-download NLTK assets (small, shared across services).
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# Create a non-root user and chown the working tree.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# Per-service defaults; the compose file overrides these.
EXPOSE 8000
STOPSIGNAL SIGTERM

# A tiny entrypoint that just execs uvicorn with the right module path.
# Each service has `app/main.py` or `app/service.py` (refinement).
# We compute the entrypoint dynamically from SERVICE_NAME.
ENTRYPOINT ["sh", "-c"]
CMD ["exec uvicorn services.${SERVICE_NAME}.app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info || exec uvicorn services.${SERVICE_NAME}.app.service:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info"]

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
ARG TORCH_VARIANT=cpu

FROM ${BASE_IMAGE}

ARG SERVICE_NAME

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
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
# invalidate the source copy.
#
# The pip flags:
#   --default-timeout=600  — give pip 10 min per request before
#                            timing out (the 4 Mbps link + 780 MB
#                            torch wheel on Linux can take 25+ min).
#   --retries=20           — let pip retry on transient network errors.
#   --extra-index-url      — only added when TORCH_VARIANT=cu121 (GPU
#                            overlay). Default is the plain CPU wheel
#                            from PyPI (~200 MB) for the default CPU stack.
COPY requirements.txt ./requirements.txt
COPY pyproject.toml ./pyproject.toml
ARG TORCH_VARIANT
# --mount=type=cache  keeps downloaded wheels across retries (4 Mbps survival).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && if [ "$TORCH_VARIANT" = "cu121" ]; then \
         EXTRA="--extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"; \
       else \
         EXTRA=""; \
       fi \
    && for i in $(seq 1 5); do \
         echo "--- pip install attempt $i/5 ---" && \
         pip install -r requirements.txt \
             --default-timeout=600 \
             --retries=5 \
             $EXTRA && break || \
         echo "=== pip install failed (attempt $i/5), retrying in 30s ===" && sleep 30; \
       done

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

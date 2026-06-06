"""Gateway configuration (Phase 6).

All backend URLs are read from environment variables so the same image
runs in dev (localhost) and in docker-compose (service-name DNS).
Defaults match the dev workflow in the Makefile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


@dataclass(frozen=True)
class GatewayConfig:
    """Service URLs + tunables for the gateway."""

    # Backend URLs. Override with env vars in production.
    preprocessing_url: str = _env("PREPROCESSING_URL", "http://localhost:8001")
    indexing_url: str = _env("INDEXING_URL", "http://localhost:8002")
    retrieval_url: str = _env("RETRIEVAL_URL", "http://localhost:8003")
    refinement_url: str = _env("REFINEMENT_URL", "http://localhost:8004")
    rag_url: str = _env("RAG_URL", "http://localhost:8005")

    # How long to wait on any single downstream call.
    downstream_timeout_s: float = float(_env("GATEWAY_DOWNSTREAM_TIMEOUT", "30"))

    # Per-probe timeout for the /health endpoint's upstream checks.
    health_probe_timeout_s: float = float(_env("GATEWAY_HEALTH_TIMEOUT", "0.5"))

    # Comma-separated CORS allow-list. Default covers Vite dev (5173) +
    # nginx production (3000).
    cors_allow_origins: tuple[str, ...] = tuple(
        s.strip()
        for s in _env(
            "GATEWAY_CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,"
            "http://127.0.0.1:5173,http://127.0.0.1:3000",
        ).split(",")
        if s.strip()
    )


CONFIG = GatewayConfig()

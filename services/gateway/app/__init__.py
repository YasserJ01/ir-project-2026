"""Phase 6: API Gateway.

The gateway is the single public entry point for the React UI. It
translates ``/api/*`` requests into httpx calls to the relevant backend
service (preprocessing :8001, indexing :8002, retrieval :8003,
refinement :8004). It is intentionally thin: no retrieval logic, no
business rules — just routing, CORS, request_id, latency, and error
translation.

The RAG service (:8005) is **not** part of this phase; the gateway
returns 501 with a forward-compat message so the React UI's RAG toggle
can be wired in Phase 8.
"""

from __future__ import annotations

__all__ = ["main", "clients", "middleware", "schemas"]

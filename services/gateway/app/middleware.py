"""Gateway middleware (Phase 6).

``RequestContextMiddleware``:
- Generates an X-Request-ID if the client didn't send one.
- Stores it in ``request.state.request_id`` for handlers + access log.
- Times the request (perf_counter).
- Logs on completion: ``[gateway] <rid> <method> <path> -> <status> in <ms>ms``.
- Echoes the request_id back in the response header.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Stamp every request with a UUID + latency log line."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        # ``request.state`` is the canonical per-request scratch space.
        request.state.request_id = rid
        # Skip the access log for hot endpoints (health probes, docs assets)
        # so the logs don't drown in noise. We do still time them so a
        # slow /health probe would show up as a > 1s log line.
        skip_log_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
        log_this = request.url.path not in skip_log_paths
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log and re-raise; FastAPI's exception handler will turn it
            # into a 500 for the client. The log line still records the
            # path + rid, so post-mortem correlation works.
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "[gateway] %s %s %s -> EXC in %.1fms",
                rid,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers[REQUEST_ID_HEADER] = rid
        if log_this:
            logger.info(
                "[gateway] %s %s %s -> %d in %.1fms",
                rid,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

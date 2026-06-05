#!/usr/bin/env python
"""Hand-test the Phase 4 query-refinement service.

This script sends a handful of test queries to the running service on
port 8004 and prints the before/after. It's a quick way to sanity-check
the pipeline end-to-end without spinning up the React UI.

Run in two terminals:
    Terminal 1:  py -3.12 -m uvicorn services.refinement.app.service:app --port 8004
    Terminal 2:  py -3.12 scripts/smoke_refine.py

Test queries (mix of clean, misspelled, and personalized):

    1. "What is the capital of France?"  -- clean
    2. "recieve teh helo from teh park"  -- lots of typos
    3. "fast car running on highway"     -- synonyms of fast/car/run
    4. "What is the eiffel tower height" -- personalization user_1
    5. "france capital population"       -- personalization user_1
"""

from __future__ import annotations

import json
import sys
import time
from urllib import request
from urllib.error import URLError

SERVICE_URL = "http://127.0.0.1:8004"

TEST_QUERIES: tuple[tuple[str, str], ...] = (
    # (label, query, user_id)
    ("clean  ", "What is the capital of France?", "anonymous"),
    ("typos  ", "recieve teh helo from teh park", "anonymous"),
    ("synonym", "fast car running on highway", "anonymous"),
    ("eiffel ", "what is the eiffel tower height", "user_1"),
    ("france ", "france capital population", "user_1"),
    ("unknown", "an obscure query about nothing", "user_42"),
)


def post_refine(query: str, user_id: str) -> dict:
    body = json.dumps(
        {
            "query": query,
            "user_id": user_id,
            "enable_spell": True,
            "enable_synonyms": True,
            "enable_grammar": False,
            "enable_personalization": True,
            "synonym_count": 2,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{SERVICE_URL}/refine",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_health() -> dict:
    with request.urlopen(f"{SERVICE_URL}/health", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    try:
        health = get_health()
    except URLError as exc:
        print(f"FAIL  Cannot reach {SERVICE_URL}: {exc}", file=sys.stderr)
        print(
            "      Start the service first: py -3.12 -m uvicorn services.refinement.app.service:app --port 8004",
            file=sys.stderr,
        )
        return 1

    print("Service health:", json.dumps(health, indent=2))
    print()
    for label, query, user_id in TEST_QUERIES:
        t0 = time.perf_counter()
        try:
            result = post_refine(query, user_id)
        except Exception as exc:
            print(f"  {label}  query={query!r}  ERROR: {exc}")
            continue
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  {label}  query={query!r}  user={user_id!r}")
        print(f"          refined:        {result['refined_query']!r}")
        print(f"          expanded:       {result['expanded_query']!r}")
        print(f"          tokens:         {result['tokens']}")
        boosted = [
            (t["token"], t["weight"]) for t in result["weighted_tokens"] if t["weight"] > 1.0
        ]
        if boosted:
            print(f"          boosted:        {boosted}")
        print(f"          stages:         {result['stages']}")
        print(f"          server latency: {result['latency_ms']} ms (round-trip {dt_ms:.0f} ms)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

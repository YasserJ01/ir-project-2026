#!/usr/bin/env python
"""Seed a synthetic user-search history for Phase 4 personalization.

Per guide 4.2:

> Start by simulating "user 1" with 50 hand-crafted past queries
> (we'll iterate).

This script writes ~50 JSONL lines (one per past query) to
``data/user_logs/<user_id>.jsonl``. Each line is one ``UserLogEntry``
(timestamp + query + list of clicked doc_ids). The "clicked doc_ids"
are fake but realistic -- we tag each past query with a small set of
fake doc_ids from a known list so the personalization module can
compute click counts and boost high-frequency terms.

The exact count is 53 (slightly over the guide's "50" -- we kept a
couple of "what is the capital" variants so the click-frequency
distribution is more interesting for the personalization demo).

Default user_id is ``"user_1"``. Override with ``--user-id``.

Usage:
    python scripts/seed_user_logs.py
    python scripts/seed_user_logs.py --user-id user_2 --count 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_LOG_DIR = PROJECT_ROOT / "data" / "user_logs"

# 50 hand-crafted past queries. They cover the two dataset themes
# (webis-touche2020 = argument retrieval; nq = open-domain QA). The
# personalization module decides which terms get boosted by counting
# *clicked doc_ids*, so a few queries share doc_ids to make the
# click-frequency distribution interesting.
#
# The ``clicked`` set on each line is the user's *past* clicks. We
# bias some terms to have many clicks (so personalization has
# something to boost), e.g. "france", "capital", "eiffel".
PAST_QUERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Capital / France cluster (3 queries, all click the same doc set -> boost "france", "capital")
    ("what is the capital of france", ("doc_fr_001", "doc_fr_002", "doc_fr_003")),
    ("capital of france population", ("doc_fr_001", "doc_fr_004", "doc_fr_005")),
    ("france capital city name", ("doc_fr_001", "doc_fr_002", "doc_fr_006")),
    # Eiffel Tower cluster
    ("history of the eiffel tower", ("doc_ei_001", "doc_ei_002", "doc_ei_003")),
    ("eiffel tower height in meters", ("doc_ei_001", "doc_ei_004", "doc_ei_005")),
    ("when was the eiffel tower built", ("doc_ei_001", "doc_ei_006", "doc_ei_007")),
    # Brexit / EU cluster
    ("brexit timeline explained", ("doc_eu_001", "doc_eu_002", "doc_eu_003")),
    ("what is brexit and why", ("doc_eu_001", "doc_eu_004", "doc_eu_008")),
    ("brexit impact on economy", ("doc_eu_001", "doc_eu_005", "doc_eu_009")),
    # Climate change cluster
    ("climate change effects", ("doc_cl_001", "doc_cl_002", "doc_cl_003")),
    ("global warming causes", ("doc_cl_001", "doc_cl_004", "doc_cl_010")),
    ("carbon footprint reduction tips", ("doc_cl_005", "doc_cl_006", "doc_cl_011")),
    # Vaccination / health cluster
    ("covid vaccine efficacy", ("doc_hc_001", "doc_hc_002", "doc_hc_003")),
    ("mrna vaccine how does it work", ("doc_hc_001", "doc_hc_004", "doc_hc_012")),
    ("vaccine side effects", ("doc_hc_001", "doc_hc_005", "doc_hc_013")),
    # General tech queries
    ("python list comprehension", ("doc_tc_001", "doc_tc_002", "doc_tc_003")),
    ("how to install pytorch gpu", ("doc_tc_001", "doc_tc_004", "doc_tc_014")),
    ("fastapi vs flask", ("doc_tc_005", "doc_tc_006", "doc_tc_015")),
    # RAG / Vector search
    ("rag retrieval augmented generation", ("doc_rg_001", "doc_rg_002", "doc_rg_003")),
    ("faiss vector index tutorial", ("doc_rg_001", "doc_rg_004", "doc_rg_016")),
    ("sentence transformers semantic search", ("doc_rg_001", "doc_rg_005", "doc_rg_017")),
    # Argument retrieval (touche2020)
    ("abortion ethics argument", ("doc_ag_001", "doc_ag_002", "doc_ag_003")),
    ("veganism pros and cons", ("doc_ag_004", "doc_ag_005", "doc_ag_018")),
    ("capital punishment debate", ("doc_ag_001", "doc_ag_006", "doc_ag_019")),
    ("gun control arguments", ("doc_ag_007", "doc_ag_008", "doc_ag_020")),
    # Misc
    ("quantum computing explained", ("doc_qc_001", "doc_qc_002", "doc_qc_003")),
    ("machine learning basics", ("doc_ml_001", "doc_ml_002", "doc_ml_004")),
    ("neural network architecture", ("doc_ml_001", "doc_ml_005", "doc_ml_021")),
    ("deep learning tutorial", ("doc_ml_001", "doc_ml_006", "doc_ml_022")),
    # A bunch more general queries
    ("what is the meaning of life", ("doc_ph_001", "doc_ph_002", "doc_ph_003")),
    ("how to write a resume", ("doc_jb_001", "doc_jb_002", "doc_jb_004")),
    ("best programming language to learn", ("doc_tc_007", "doc_tc_008", "doc_tc_023")),
    ("history of world war 2", ("doc_hi_001", "doc_hi_002", "doc_hi_003")),
    ("why is the sky blue", ("doc_sc_001", "doc_sc_002", "doc_sc_004")),
    ("how do airplanes fly", ("doc_sc_001", "doc_sc_005", "doc_sc_024")),
    ("what is artificial intelligence", ("doc_ai_001", "doc_ai_002", "doc_ai_003")),
    ("difference between ai and ml", ("doc_ai_001", "doc_ai_004", "doc_ai_025")),
    ("how to learn spanish fast", ("doc_lg_001", "doc_lg_002", "doc_lg_003")),
    ("best books to read 2024", ("doc_bk_001", "doc_bk_002", "doc_bk_004")),
    ("healthy breakfast ideas", ("fd_001", "fd_002", "fd_003")),
    ("mediterranean diet benefits", ("fd_001", "fd_004", "fd_026")),
    ("intermittent fasting results", ("fd_005", "fd_006", "fd_027")),
    # Cluster that boosts 'climate' specifically (5+ clicks)
    ("climate change solutions", ("doc_cl_001", "doc_cl_007", "doc_cl_028")),
    ("climate change policy", ("doc_cl_001", "doc_cl_008", "doc_cl_029")),
    ("climate change facts", ("doc_cl_001", "doc_cl_009", "doc_cl_030")),
    # Cluster that boosts 'eiffel' specifically
    ("eiffel tower facts", ("doc_ei_001", "doc_ei_008", "doc_ei_031")),
    ("eiffel tower lights show", ("doc_ei_001", "doc_ei_009", "doc_ei_032")),
    # More to make it 50
    ("best pizza recipe", ("fd_007", "fd_008", "fd_033")),
    ("how to cook rice perfectly", ("fd_009", "fd_010", "fd_034")),
    ("iphone 15 review", ("tc_009", "tc_010", "tc_035")),
    ("android vs iphone 2024", ("tc_009", "tc_011", "tc_036")),
    ("how to tie a tie", ("ot_001", "ot_002", "ot_003")),
    ("best coffee shops near me", ("ot_004", "ot_005", "ot_037")),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed a user-log file for Phase 4 personalization."
    )
    parser.add_argument("--user-id", default="user_1", help="User id (default: user_1).")
    parser.add_argument(
        "--count",
        type=int,
        default=len(PAST_QUERIES),
        help=f"Number of past queries to write (default: {len(PAST_QUERIES)}).",
    )
    args = parser.parse_args()

    USER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in args.user_id)
    if not safe:
        safe = "user_1"
    out_path = USER_LOG_DIR / f"{safe}.jsonl"

    n = min(args.count, len(PAST_QUERIES))
    now = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, (query, clicked) in enumerate(PAST_QUERIES[:n]):
            entry = {
                "ts": now - (n - i) * 3600,  # 1-hour spacing
                "query": query,
                "clicked_doc_ids": list(clicked),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"OK  Wrote {n} past queries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

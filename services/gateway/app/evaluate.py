"""Live evaluation runner (Phase 8b).

Reuses the gateway's own client instances to run all evaluation queries
for a given configuration and compute metrics via ``ir_measures``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from services.gateway.app.clients import GatewayClients

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
QUERIES_DIR = REPO_ROOT / "evaluation" / "queries"
TOP_K = 10

DS_TO_BEIR = {"touche2020": "beir/webis-touche2020", "nq": "beir/nq"}


def _load_queries(dataset_id: str) -> list[tuple[str, str]]:
    fname = f"{dataset_id}_queries.txt"
    path = QUERIES_DIR / fname
    if not path.exists():
        logger.warning("Query file not found: %s", path)
        return []
    queries: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            qid, text = line.split("\t", 1)
            queries.append((qid.strip(), text.strip()))
    return queries


async def _warmup(clients: GatewayClients, dataset_id: str) -> None:
    """Prime caches with one throwaway query."""
    warmup_body = {
        "query": "climate change",
        "dataset_id": dataset_id,
        "representation": "bm25",
        "k": 5,
        "mode": "basic",
        "user_id": "eval_warmup",
        "enable_grammar": False,
    }
    try:
        await clients.retrieval.hybrid_search(dataset_id, warmup_body)
    except Exception:
        pass


async def _search_query(
    clients: GatewayClients,
    query: str,
    dataset_id: str,
    rep: str | None,
    fusion: str | None,
    mode: str,
    bm25_k1: float,
    bm25_b: float,
) -> list[dict[str, Any]]:
    """Run one query through the same pipeline the gateway uses."""
    if rep in ("tfidf", "bm25"):
        try:
            tokens = await clients.preprocessing.preprocess(query)
            result = await clients.indexing.search(
                dataset_id, tokens, model=rep, k=TOP_K, k1=bm25_k1, b=bm25_b,
            )
        except Exception:
            return []
        return result.get("results", [])
    payload: dict[str, Any] = {
        "query": query,
        "dataset_id": dataset_id,
        "k": TOP_K,
        "mode": mode,
        "user_id": "eval_user",
        "enable_grammar": False,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
    }
    if rep is not None:
        payload["representation"] = rep
    if fusion is not None:
        payload["fusion"] = fusion
    try:
        result = await clients.retrieval.hybrid_search(dataset_id, payload)
    except Exception:
        return []
    return result.get("results", [])


async def _multi_search_query(
    clients: GatewayClients,
    query: str,
    dataset_id: str,
    fusion: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Run one multi-encoder query."""
    payload = {
        "query": query,
        "k": TOP_K,
        "mode": mode,
        "user_id": "eval_user",
        "fusion": fusion,
    }
    try:
        result = await clients.retrieval.multi_encoder_search(dataset_id, payload)
    except Exception:
        return []
    return result.get("results", [])


def _compute_metrics(
    dataset_id: str,
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute MAP@10, P@10, nDCG@10, R@10 via ``ir_measures``."""
    import ir_datasets
    import ir_measures as ir

    beir_id = DS_TO_BEIR.get(dataset_id, f"beir/{dataset_id}")
    ds = ir_datasets.load(beir_id)

    run_qids = set(run.keys())
    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        if qrel.query_id in run_qids:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    measures = [ir.AP @ 10, ir.P @ 10, ir.nDCG @ 10, ir.R @ 10]
    try:
        result = ir.calc(measures, qrels, run)
        agg = result.aggregated
        return {
            "MAP": round(float(agg.get(ir.AP @ 10, 0.0)), 4),
            "P@10": round(float(agg.get(ir.P @ 10, 0.0)), 4),
            "nDCG@10": round(float(agg.get(ir.nDCG @ 10, 0.0)), 4),
            "R@10": round(float(agg.get(ir.R @ 10, 0.0)), 4),
        }
    except Exception as e:
        logger.error("ir.calc failed for %s: %s", dataset_id, e)
        return {"MAP": 0.0, "P@10": 0.0, "nDCG@10": 0.0, "R@10": 0.0}


async def run_evaluation(
    clients: GatewayClients,
    dataset_id: str,
    representation: str | None,
    fusion: str | None,
    mode: str,
    bm25_k1: float,
    bm25_b: float,
    use_multi: bool,
) -> dict[str, Any]:
    """Execute a live evaluation and return aggregated metrics."""
    queries = _load_queries(dataset_id)
    if not queries:
        return {"error": f"No queries found for dataset {dataset_id}"}

    await _warmup(clients, dataset_id)

    run: dict[str, dict[str, float]] = {}
    success = 0
    errors = 0

    t0 = time.perf_counter()
    for qid, qtext in queries:
        if use_multi and fusion:
            results = await _multi_search_query(
                clients, qtext, dataset_id, fusion, mode,
            )
        else:
            results = await _search_query(
                clients, qtext, dataset_id, representation,
                fusion, mode, bm25_k1, bm25_b,
            )

        if results:
            run[qid] = {}
            for rank, hit in enumerate(results, start=1):
                doc_id = hit.get("doc_id", "?")
                score = hit.get("score", 0.0)
                run[qid][doc_id] = score
            success += 1
        else:
            errors += 1

    elapsed = time.perf_counter() - t0

    metrics = _compute_metrics(dataset_id, run)

    return {
        "dataset_id": dataset_id,
        "representation": representation or "multi",
        "condition": "baseline" if mode == "basic" else "with_features",
        "queries": len(queries),
        "success": success,
        "errors": errors,
        "time_s": round(elapsed, 1),
        "metrics": metrics,
    }

"""Phase 9 — Full system evaluation.

Evaluates all (dataset, representation, condition) combinations:
  2 datasets × (6 search reps + 3 multi-encoder reps) × 2 conditions = 36 runs

Output:
  evaluation/results/<dataset>/<run_id>.txt    TREC run files (k=10)
  evaluation/reports/summary.csv               CSV table (all metrics)
  evaluation/reports/summary.md                Markdown table
  evaluation/reports/plots/<metric>.png        Bar charts per metric

Usage:
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py touche2020
  python scripts/run_evaluation.py nq

The script uses a persistent ``requests.Session`` so HTTP connections are
reused across queries (NLTK + BM25/TF-IDF cold loads happen only once per
run group, not per query).
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_evaluation")

REPO = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO / "evaluation" / "queries"
RESULTS_DIR = REPO / "evaluation" / "results"
REPORTS_DIR = REPO / "evaluation" / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
TOP_K = 10

DATASETS = ["touche2020", "nq"]
DS_TO_BEIR = {"touche2020": "beir/webis-touche2020", "nq": "beir/nq"}

SEARCH_CONFIGS: list[dict] = [
    {"rep": "tfidf", "fusion": None, "label": "tfidf"},
    {"rep": "bm25", "fusion": None, "label": "bm25"},
    {"rep": "embedding", "fusion": None, "label": "embedding"},
    {"rep": "hybrid_serial", "fusion": "rrf", "label": "hybrid_rrf"},
    {"rep": "hybrid_serial", "fusion": "combsum", "label": "hybrid_combsum"},
    {"rep": "hybrid_serial", "fusion": "combmnz", "label": "hybrid_combmnz"},
]

MULTI_CONFIGS: list[dict] = [
    {"fusion": "rrf", "label": "multi_rrf"},
    {"fusion": "combsum", "label": "multi_combsum"},
    {"fusion": "combmnz", "label": "multi_combmnz"},
]

CONDITIONS = [
    ("basic", "basic"),
    ("with_features", "features"),
]

# Shared session for connection reuse (dramatically reduces per-query overhead)
_SESSION = requests.Session()


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


def _warmup(dataset_id: str) -> None:
    """Prime all backend caches (NLTK, BM25, TF-IDF, embedding, hybrid)."""
    logger.info("  Warmup: priming caches for %s ...", dataset_id)
    for rep in ("tfidf", "bm25", "embedding", "hybrid_rrf"):
        body = {"query": "climate change", "dataset_id": dataset_id,
                "representation": rep, "k": 5, "mode": "basic"}
        try:
            _SESSION.post(f"{GATEWAY_URL}/api/search", json=body, timeout=180)
        except requests.RequestException:
            pass


def _search_gateway(query: str, dataset_id: str, rep: str | None,
                    fusion: str | None, mode: str) -> list[dict]:
    body: dict = {
        "query": query, "dataset_id": dataset_id,
        "k": TOP_K, "mode": mode, "user_id": "eval_user", "enable_grammar": False,
    }
    if rep is not None:
        body["representation"] = rep
    if fusion is not None:
        body["fusion"] = fusion
    try:
        r = _SESSION.post(f"{GATEWAY_URL}/api/search", json=body, timeout=60)
        if r.status_code != 200:
            return []
        return r.json().get("results", [])
    except requests.RequestException:
        return []


def _multi_encoder_gateway(query: str, dataset_id: str,
                           fusion: str, mode: str) -> list[dict]:
    body = {"query": query, "k": TOP_K, "mode": mode,
            "user_id": "eval_user", "fusion": fusion}
    try:
        r = _SESSION.post(
            f"{GATEWAY_URL}/api/multi-encoder/{dataset_id}/search",
            json=body, timeout=120,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results", [])
    except requests.RequestException:
        return []


def _write_trec(results: list[dict], query_id: str, run_id: str, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rank, hit in enumerate(results, start=1):
            doc_id = hit.get("doc_id", "?")
            score = hit.get("score", 0.0)
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_id}\n")


def _compute_metrics(dataset_id: str, run_id: str) -> dict[str, float]:
    import ir_datasets
    import ir_measures as ir

    beir_id = DS_TO_BEIR.get(dataset_id, f"beir/{dataset_id}")
    ds = ir_datasets.load(beir_id)

    run_path = RESULTS_DIR / dataset_id / f"{run_id}.txt"
    if not run_path.exists():
        return {"MAP": 0.0, "P@10": 0.0, "nDCG@10": 0.0, "R@10": 0.0}

    run: dict[str, dict[str, float]] = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            run.setdefault(parts[0], {})[parts[2]] = float(parts[4])

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
        logger.error("  ir.calc failed for %s: %s", run_id, e)
        return {"MAP": 0.0, "P@10": 0.0, "nDCG@10": 0.0, "R@10": 0.0}


def _build_run_id(dataset_id: str, label: str, condition_tag: str) -> str:
    return f"{dataset_id}__{label}__{condition_tag}"


def _run_search_eval(dataset_id: str, queries: list[tuple[str, str]],
                     configs: list[dict], use_multi: bool = False) -> list[dict]:
    rows: list[dict] = []

    for cfg in configs:
        label = cfg["label"]
        fusion = cfg.get("fusion")

        for mode, cond_tag in CONDITIONS:
            run_id = _build_run_id(dataset_id, label, cond_tag)
            run_path = RESULTS_DIR / dataset_id / f"{run_id}.txt"
            run_path.parent.mkdir(parents=True, exist_ok=True)
            if run_path.exists():
                run_path.unlink()

            logger.info("  Run: %s (%d queries, mode=%s)", run_id, len(queries), mode)
            t0 = time.perf_counter()
            success = 0
            errors = 0

            for qid, qtext in queries:
                if use_multi:
                    results = _multi_encoder_gateway(qtext, dataset_id, fusion, mode)
                else:
                    results = _search_gateway(qtext, dataset_id, cfg.get("rep"), fusion, mode)

                if results:
                    _write_trec(results, qid, run_id, run_path)
                    success += 1
                else:
                    errors += 1

            elapsed = time.perf_counter() - t0
            logger.info("    %d ok, %d err, %.1fs total (%.1f ms/q)",
                        success, errors, elapsed,
                        elapsed / max(len(queries), 1) * 1000)

            metrics = _compute_metrics(dataset_id, run_id)
            row: dict = {
                "dataset": dataset_id,
                "representation": label,
                "condition": "baseline" if mode == "basic" else "with_features",
                "queries": success,
                "time_s": round(elapsed, 1),
                **metrics,
            }
            rows.append(row)
            logger.info("    MAP=%.4f  P@10=%.4f  nDCG@10=%.4f  R@10=%.4f",
                        metrics["MAP"], metrics["P@10"],
                        metrics["nDCG@10"], metrics["R@10"])

    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "representation", "condition",
                  "MAP", "P@10", "nDCG@10", "R@10", "queries", "time_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


def _write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")
        f.write("Metrics: **MAP@10, P@10, nDCG@10, R@10**\n\n")
        f.write("Conditions: **baseline** = no refinement; "
                "**with_features** = spell + synonyms + personalization\n\n")

        for ds in sorted(set(r["dataset"] for r in rows)):
            f.write(f"## Dataset: {ds}\n\n")
            f.write("| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in (r for r in rows if r["dataset"] == ds):
                f.write(f"| {r['representation']} | {r['condition']} "
                        f"| {r['MAP']:.4f} | {r['P@10']:.4f} "
                        f"| {r['nDCG@10']:.4f} | {r['R@10']:.4f} "
                        f"| {r['queries']} | {r['time_s']:.1f} |\n")
            f.write("\n")
    logger.info("Wrote %s", path)


def _plot_charts(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for metric in ["MAP", "P@10", "nDCG@10", "R@10"]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, ds in enumerate(["touche2020", "nq"]):
            ax = axes[idx]
            ds_rows = [r for r in rows if r["dataset"] == ds]
            reps = sorted(set(r["representation"] for r in ds_rows))
            x = np.arange(len(reps))
            width = 0.35

            baseline_vals = []
            features_vals = []
            for rep in reps:
                base = next((r[metric] for r in ds_rows
                             if r["representation"] == rep and r["condition"] == "baseline"), 0)
                feat = next((r[metric] for r in ds_rows
                             if r["representation"] == rep and r["condition"] == "with_features"), 0)
                baseline_vals.append(base)
                features_vals.append(feat)

            ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="#4A90D9")
            ax.bar(x + width/2, features_vals, width, label="With Features", color="#E67E22")
            ax.set_title(f"{ds} — {metric}")
            ax.set_xticks(x)
            ax.set_xticklabels(reps, rotation=30, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(PLOTS_DIR / f"{metric}.png", dpi=150)
        plt.close(fig)
        logger.info("Saved %s", PLOTS_DIR / f"{metric}.png")


def main() -> None:
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    all_rows: list[dict] = []

    for ds in datasets:
        if ds not in DATASETS:
            logger.warning("Unknown dataset %s, skipping", ds)
            continue

        queries = _load_queries(ds)
        if not queries:
            logger.warning("No queries for %s, skipping", ds)
            continue

        logger.info("Dataset: %s (%d queries)", ds, len(queries))
        _warmup(ds)

        logger.info("  Search endpoint runs ...")
        all_rows.extend(_run_search_eval(ds, queries, SEARCH_CONFIGS, use_multi=False))

        logger.info("  Multi-encoder runs ...")
        all_rows.extend(_run_search_eval(ds, queries, MULTI_CONFIGS, use_multi=True))

    if not all_rows:
        logger.error("No results collected")
        sys.exit(1)

    _write_csv(all_rows, REPORTS_DIR / "summary.csv")
    _write_markdown(all_rows, REPORTS_DIR / "summary.md")
    _plot_charts(all_rows)
    logger.info("All done. Summary: %s", REPORTS_DIR / "summary.md")


if __name__ == "__main__":
    main()

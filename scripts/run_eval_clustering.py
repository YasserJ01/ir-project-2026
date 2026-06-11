"""Evaluation: impact of cluster-aware retrieval.

Compares baseline vs cluster-boost for embedding and BM25
on both datasets. Writes comparison table + bar charts.

Usage:
  python scripts/run_eval_clustering.py

Requires:
  - All 7 services running (start_all.ps1)
  - Clusters built (scripts/build_clusters.py has been run)
  - Evaluation queries present (evaluation/queries/{{ds}}_queries.txt)
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
logger = logging.getLogger("run_eval_clustering")

REPO = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO / "evaluation" / "queries"
RESULTS_DIR = REPO / "evaluation" / "results"
REPORTS_DIR = REPO / "evaluation" / "reports"
PLOTS_DIR = REPORTS_DIR / "plots" / "clustering"

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
TOP_K = 10
CLUSTER_BOOST = 1.5

DATASETS = ["touche2020", "nq"]
DS_TO_BEIR = {"touche2020": "beir/webis-touche2020", "nq": "beir/nq"}

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
    logger.info("  Warmup: %s ...", dataset_id)
    for rep in ("embedding", "bm25"):
        try:
            _SESSION.post(
                f"{GATEWAY_URL}/api/search",
                json={"query": "climate change", "dataset_id": dataset_id,
                       "representation": rep, "k": 5, "mode": "basic"},
                timeout=180,
            )
        except requests.RequestException:
            pass


def _search_gateway(dataset_id: str, rep: str,
                     mode: str) -> list[dict]:
    body = {"query": "", "dataset_id": dataset_id,
            "representation": rep, "k": TOP_K, "mode": mode,
            "user_id": "eval_user", "enable_grammar": False}
    r = _SESSION.post(f"{GATEWAY_URL}/api/search", json=body, timeout=60)
    return r.json().get("results", []) if r.status_code == 200 else []


def _cluster_search(dataset_id: str, rep: str,
                     mode: str) -> list[dict]:
    body = {"query": "", "dataset_id": dataset_id,
            "representation": rep, "k": TOP_K, "mode": mode,
            "user_id": "eval_user", "enable_grammar": False,
            "enable_clustering": True, "cluster_boost": CLUSTER_BOOST}
    r = _SESSION.post(f"{GATEWAY_URL}/api/cluster/{dataset_id}/search",
                      json=body, timeout=180)
    return r.json().get("results", []) if r.status_code == 200 else []


def _run_one(queries: list[tuple[str, str]], dataset_id: str,
             rep: str, label: str, cluster: bool) -> dict:
    run_id = f"{dataset_id}__{rep}__{'cluster' if cluster else 'baseline'}"
    run_path = RESULTS_DIR / dataset_id / f"{run_id}.txt"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    if run_path.exists():
        run_path.unlink()

    logger.info("  Run: %s (%d queries)", run_id, len(queries))
    t0 = time.perf_counter()
    success = 0
    errors = 0

    search_fn = _cluster_search if cluster else _search_gateway

    for qid, qtext in queries:
        body = {"query": qtext}
        try:
            if cluster:
                r = _SESSION.post(
                    f"{GATEWAY_URL}/api/cluster/{dataset_id}/search",
                    json={**body, "representation": rep, "k": TOP_K,
                          "mode": "basic", "user_id": "eval_user",
                          "enable_grammar": False,
                          "enable_clustering": True,
                          "cluster_boost": CLUSTER_BOOST},
                    timeout=180,
                )
            else:
                r = _SESSION.post(
                    f"{GATEWAY_URL}/api/search",
                    json={**body, "dataset_id": dataset_id,
                          "representation": rep, "k": TOP_K,
                          "mode": "basic", "user_id": "eval_user",
                          "enable_grammar": False},
                    timeout=60,
                )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    with open(run_path, "a", encoding="utf-8") as f:
                        for rank, hit in enumerate(results, start=1):
                            doc_id = hit.get("doc_id", "?")
                            score = hit.get("score", 0.0)
                            f.write(f"{qid} Q0 {doc_id} {rank} {score} {run_id}\n")
                    success += 1
                else:
                    errors += 1
            else:
                errors += 1
        except requests.RequestException:
            errors += 1

    elapsed = time.perf_counter() - t0
    logger.info("    %d ok, %d err, %.1fs total (%.1f ms/q)",
                success, errors, elapsed,
                elapsed / max(len(queries), 1) * 1000)

    import ir_datasets
    import ir_measures as ir

    beir_id = DS_TO_BEIR.get(dataset_id, f"beir/{dataset_id}")
    ds = ir_datasets.load(beir_id)

    run: dict[str, dict[str, float]] = {}
    if run_path.exists():
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
        metrics = {
            "MAP": round(float(agg.get(ir.AP @ 10, 0.0)), 4),
            "P@10": round(float(agg.get(ir.P @ 10, 0.0)), 4),
            "nDCG@10": round(float(agg.get(ir.nDCG @ 10, 0.0)), 4),
            "R@10": round(float(agg.get(ir.R @ 10, 0.0)), 4),
        }
    except Exception as e:
        logger.error("  ir.calc failed: %s", e)
        metrics = {"MAP": 0.0, "P@10": 0.0, "nDCG@10": 0.0, "R@10": 0.0}

    logger.info("    MAP=%.4f  P@10=%.4f  nDCG@10=%.4f  R@10=%.4f",
                metrics["MAP"], metrics["P@10"],
                metrics["nDCG@10"], metrics["R@10"])

    return {
        "dataset": dataset_id,
        "representation": rep,
        "condition": "cluster" if cluster else "baseline",
        "cluster_boost": CLUSTER_BOOST if cluster else 0.0,
        "queries": success,
        "time_s": round(elapsed, 1),
        **metrics,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "representation", "condition", "cluster_boost",
                  "MAP", "P@10", "nDCG@10", "R@10", "queries", "time_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


def _write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Clustering Impact Evaluation\n\n")
        f.write(f"Cluster boost: **{CLUSTER_BOOST}×** for nearest-cluster docs.\n\n")
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
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, ds in enumerate(["touche2020", "nq"]):
            ax = axes[idx]
            ds_rows = [r for r in rows if r["dataset"] == ds]
            reps = sorted(set(r["representation"] for r in ds_rows))
            x = np.arange(len(reps))
            width = 0.35

            base_vals = []
            clust_vals = []
            for rep in reps:
                base = next((r[metric] for r in ds_rows
                             if r["representation"] == rep and r["condition"] == "baseline"), 0)
                clust = next((r[metric] for r in ds_rows
                              if r["representation"] == rep and r["condition"] == "cluster"), 0)
                base_vals.append(base)
                clust_vals.append(clust)

            ax.bar(x - width/2, base_vals, width, label="Baseline", color="#4A90D9")
            ax.bar(x + width/2, clust_vals, width, label=f"Cluster {CLUSTER_BOOST}×", color="#27AE60")
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

        for rep in ("embedding", "bm25"):
            for cluster in (False, True):
                cond_tag = "cluster" if cluster else "baseline"
                label = f"{rep} ({cond_tag})"
                logger.info("  ---- %s ----", label)
                row = _run_one(queries, ds, rep, label, cluster)
                all_rows.append(row)

    if not all_rows:
        logger.error("No results collected")
        sys.exit(1)

    _write_csv(all_rows, REPORTS_DIR / "clustering_comparison.csv")
    _write_markdown(all_rows, REPORTS_DIR / "clustering_comparison.md")
    _plot_charts(all_rows)
    logger.info("All done. Summary: %s", REPORTS_DIR / "clustering_comparison.md")


if __name__ == "__main__":
    main()

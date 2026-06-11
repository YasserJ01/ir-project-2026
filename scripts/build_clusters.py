from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from services.clustering.app.clusterer import (
    BATCH_SIZE,
    CorpusClusterer,
    build_meta_path,
    centroids_path,
    cluster_dir,
    doc_id_map_path,
    labels_path,
)

DATASETS = ("touche2020", "nq")
DEFAULT_K = 20
ELBOW_K_RANGE = [5, 10, 15, 20, 25, 30]


def load_embeddings(dataset_id: str) -> tuple[np.ndarray, list[str]]:
    idx_dir = _PROJECT_ROOT / "data" / "indexes" / dataset_id
    embeddings = np.load(str(idx_dir / "embeddings.npy"))
    doc_ids = json.loads((idx_dir / "doc_ids.json").read_text(encoding="utf-8"))
    return embeddings, doc_ids


def build_elbow(embeddings: np.ndarray) -> dict[int, float]:
    elbow = {}
    print(f"  Computing elbow for k in {ELBOW_K_RANGE} ...")
    for k in ELBOW_K_RANGE:
        t0 = time.perf_counter()
        cc = CorpusClusterer(n_clusters=k)
        cc.fit(embeddings)
        elbow[k] = round(cc.inertia_, 1)
        print(f"    k={k}: inertia={elbow[k]:.1f} ({time.perf_counter() - t0:.1f}s)")
    return elbow


def build(dataset_id: str) -> None:
    t0 = time.perf_counter()
    print(f"\nBuilding clusters for {dataset_id} ...")

    embeddings, doc_ids = load_embeddings(dataset_id)
    n_docs = embeddings.shape[0]
    print(f"  Loaded {n_docs} embeddings, dim={embeddings.shape[1]}")

    elbow = build_elbow(embeddings)

    cc = CorpusClusterer(n_clusters=DEFAULT_K)
    cc.fit(embeddings)
    sizes = cc.cluster_sizes()
    print(f"  Fitted k={DEFAULT_K}: inertia={cc.inertia_:.1f}, sizes={sizes[:5]}...")

    out_dir = cluster_dir(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(centroids_path(dataset_id)), cc.centroids)
    np.save(str(labels_path(dataset_id)), cc.labels)

    doc_id_map = {doc_ids[i]: int(cc.labels[i]) for i in range(n_docs)}
    doc_id_map_path(dataset_id).write_text(
        json.dumps(doc_id_map, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "status": "ok",
        "dataset_id": dataset_id,
        "n_clusters": DEFAULT_K,
        "total_docs": n_docs,
        "embedding_dim": int(embeddings.shape[1]),
        "inertia": round(cc.inertia_, 1),
        "elbow": elbow,
        "per_cluster_sizes": sizes,
        "batch_size": BATCH_SIZE,
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    build_meta_path(dataset_id).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.perf_counter() - t0
    print(f"  Done {dataset_id}: {n_docs} docs, {DEFAULT_K} clusters, {elapsed:.1f}s")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Mini-Batch K-Means clusters from FAISS embeddings."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Build only one dataset (default: both).",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        build(ds)

    print("\nAll done.")


if __name__ == "__main__":
    main()

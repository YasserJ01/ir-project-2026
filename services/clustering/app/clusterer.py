from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEXES_DIR = _PROJECT_ROOT / "data" / "indexes"
DEFAULT_N_CLUSTERS = 20
BATCH_SIZE = 10000


class CorpusClusterer:
    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS) -> None:
        self.n_clusters = n_clusters
        self.kmeans: MiniBatchKMeans | None = None
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.inertia_: float = 0.0

    def fit(self, embeddings: np.ndarray) -> CorpusClusterer:
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=BATCH_SIZE,
            n_init=3,
            random_state=42,
        )
        self.kmeans.fit(embeddings)
        self.centroids = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_
        self.inertia_ = float(self.kmeans.inertia_)
        return self

    def predict(self, query_embedding: np.ndarray) -> tuple[int, float]:
        dists = np.linalg.norm(self.centroids - query_embedding, axis=1)
        nearest = int(np.argmin(dists))
        return nearest, float(dists[nearest])

    def cluster_sizes(self) -> list[int]:
        return [int(np.sum(self.labels == i)) for i in range(self.n_clusters)]


def cluster_dir(dataset_id: str) -> Path:
    return INDEXES_DIR / dataset_id / "clusters"


def build_meta_path(dataset_id: str) -> Path:
    return cluster_dir(dataset_id) / "build_meta.json"


def centroids_path(dataset_id: str) -> Path:
    return cluster_dir(dataset_id) / "centroids.npy"


def labels_path(dataset_id: str) -> Path:
    return cluster_dir(dataset_id) / "labels.npy"


def doc_id_map_path(dataset_id: str) -> Path:
    return cluster_dir(dataset_id) / "doc_id_to_cluster.json"


def is_built(dataset_id: str) -> bool:
    p = build_meta_path(dataset_id)
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return meta.get("status") == "ok"
    except Exception:
        return False


def load_clusterer(dataset_id: str) -> CorpusClusterer:
    meta = json.loads(build_meta_path(dataset_id).read_text(encoding="utf-8"))
    n = int(meta["n_clusters"])
    cc = CorpusClusterer(n_clusters=n)
    cc.centroids = np.load(str(centroids_path(dataset_id)))
    cc.labels = np.load(str(labels_path(dataset_id)))
    cc.inertia_ = float(meta["inertia"])
    return cc


def load_doc_id_map(dataset_id: str) -> dict[str, int]:
    return json.loads(doc_id_map_path(dataset_id).read_text(encoding="utf-8"))

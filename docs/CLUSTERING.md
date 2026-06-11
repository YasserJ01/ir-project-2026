# Clustering — Optional Feature

Clustering (Mini-Batch K-Means, k=20) adds a score boost to documents
from the nearest cluster centroid. It is **optional** — the system works
without it.

> **Note**: Evaluation shows clustering does not improve search quality
> (see §12.6 in `reports/report_en.md`). Use it for experimentation and
> learning, not for production retrieval.

## Enable / Disable

### Native (start_all.ps1)

```powershell
# Default (clustering OFF — 6 services)
.\start_all.ps1

# With clustering (7 services)
.\start_all.ps1 -Clustering
```

### Docker Compose

```yaml
# The clustering service block exists in docker-compose.yml but is NOT
# required. The gateway does not depend on it. To use clustering in
# Docker, uncomment or add the service explicitly before starting.
services:
  clustering:
    build:
      context: .
      dockerfile: services/backend.Dockerfile
      args:
        SERVICE_NAME: clustering
    image: ir-project/clustering:latest
    container_name: ir_clustering
    restart: unless-stopped
    expose:
      - "8000"
    environment:
      CLUSTERING_RETRIEVAL_URL: "http://retrieval:8000"
    volumes:
      - ./data:/app/data
    networks: [irnet]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 5s
```

### UI Toggle

The UI always shows the Clustering toggle checkbox and boost slider.
When the clustering service is not running and the user enables the
toggle, the system silently falls back to a normal (non-clustered)
search and shows a warning badge.

## Build the Clusters

Clusters must be built from the L6 embeddings before the endpoint
can return results:

```powershell
.\.venv\Scripts\python.exe scripts/build_clusters.py
```

This builds both touche2020 (k=20, ~45s) and nq (k=20, ~45s).

## Check if Clusters Exist

```powershell
curl.exe -s http://localhost:8006/cluster/touche2020/stats
# {"dataset_id":"touche2020","built":true,"n_clusters":20,...}
```

## Rebuild with Different k

Edit `scripts/build_clusters.py` and change the `N_CLUSTERS` constant,
then re-run. The elbow sweep (k=5–30) runs automatically and logs
inertia — k=20 was chosen as the elbow point.

## Architecture

```
UI toggle ON
  → POST /api/cluster/{ds}/search (gateway)
    → POST /cluster/{ds}/search (clustering :8006)
      → GET /retrieval/embed (encode query via :8003)
      → predict nearest cluster
      → POST /hybrid/{ds}/search (downstream search via :8003)
      → boost scores × cluster_boost (default 1.5)
      → rerank
      → return
```

## Related Files

| File | Purpose |
|------|---------|
| `services/clustering/app/service.py` | FastAPI :8006 — 3 endpoints |
| `services/clustering/app/clusterer.py` | Mini-Batch K-Means wrapper |
| `scripts/build_clusters.py` | One-time cluster build |
| `scripts/run_eval_clustering.py` | Baseline vs cluster evaluation |
| `services/ui/src/components/ClusteringToggle.tsx` | UI checkbox + boost slider |
| `services/ui/src/components/ClusterBarChart.tsx` | Cluster size bar chart |
| `evaluation/reports/plots/clustering/*.png` | Comparison charts |
| `evaluation/reports/clustering_comparison.md` | Full results table |

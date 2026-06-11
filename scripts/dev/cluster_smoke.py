import requests, sys, json

host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8006"
ds = sys.argv[2] if len(sys.argv) > 2 else "touche2020"

r = requests.get(f"{host}/health", timeout=10)
print(f"Health: {r.status_code}")
r = requests.get(f"{host}/cluster/{ds}/stats", timeout=10)
s = r.json()
print(f"Stats: built={s['built']} clusters={s['n_clusters']} docs={s['total_docs']}")

r = requests.post(f"{host}/cluster/{ds}/search", json={
    "query": "climate change effects", "k": 3,
    "representation": "embedding", "enable_clustering": True, "cluster_boost": 1.5,
}, timeout=180)
d = r.json()
print(f"Cluster search: nearest={d.get('nearest_cluster_id')} hits={len(d.get('results',[]))} ms={d.get('latency_ms')}")
for h in d.get("results", []):
    print(f"  rank={h['rank']} score={h['score']:.4f} cluster={h.get('_cluster_id','?')}")

r = requests.post(f"{host}/cluster/{ds}/search", json={
    "query": "climate change effects", "k": 3,
    "representation": "bm25", "enable_clustering": True, "cluster_boost": 1.5,
}, timeout=180)
d = r.json()
print(f"Cluster BM25: nearest={d.get('nearest_cluster_id')} hits={len(d.get('results',[]))} ms={d.get('latency_ms')}")
for h in d.get("results", []):
    print(f"  rank={h['rank']} score={h['score']:.4f} cluster={h.get('_cluster_id','?')}")

print("OK")

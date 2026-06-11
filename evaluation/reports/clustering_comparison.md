# Clustering Impact Evaluation

Cluster boost: **1.5×** for nearest-cluster docs.

## Dataset: nq

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |
|---|---|---|---|---|---|---|---|
| embedding | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 200 | 16.8 |
| embedding | cluster | 0.3829 | 0.0790 | 0.4620 | 0.6775 | 200 | 75.7 |
| bm25 | baseline | 0.2930 | 0.0610 | 0.3540 | 0.5183 | 200 | 4.4 |
| bm25 | cluster | 0.2961 | 0.0610 | 0.3561 | 0.5183 | 200 | 52.1 |

## Dataset: touche2020

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |
|---|---|---|---|---|---|---|---|
| embedding | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 3.7 |
| embedding | cluster | 0.0340 | 0.2857 | 0.2217 | 0.0609 | 49 | 17.1 |
| bm25 | baseline | 0.1377 | 0.7388 | 0.6206 | 0.1521 | 49 | 1.4 |
| bm25 | cluster | 0.1331 | 0.7388 | 0.6058 | 0.1521 | 49 | 11.8 |


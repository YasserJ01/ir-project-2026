# Evaluation Summary

Metrics: **MAP@10, P@10, nDCG@10, R@10**

Conditions: **baseline** = no refinement; **with_features** = spell + synonyms + personalization

## Dataset: nq

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |
|---|---|---|---|---|---|---|---|
| tfidf | baseline | 0.1353 | 0.0375 | 0.1825 | 0.3117 | 200 | 183.7 |
| tfidf | with_features | 0.1353 | 0.0375 | 0.1825 | 0.3117 | 200 | 180.8 |
| bm25 | baseline | 0.2930 | 0.0610 | 0.3540 | 0.5183 | 200 | 4.5 |
| bm25 | with_features | 0.2930 | 0.0610 | 0.3540 | 0.5183 | 200 | 4.0 |
| embedding | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 200 | 26.0 |
| embedding | with_features | 0.3745 | 0.0695 | 0.4366 | 0.5975 | 200 | 24.2 |
| hybrid_rrf | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 200 | 29.1 |
| hybrid_rrf | with_features | 0.3745 | 0.0695 | 0.4366 | 0.5975 | 200 | 36.0 |
| hybrid_combsum | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 200 | 29.9 |
| hybrid_combsum | with_features | 0.3745 | 0.0695 | 0.4366 | 0.5975 | 200 | 33.8 |
| hybrid_combmnz | baseline | 0.4308 | 0.0790 | 0.5005 | 0.6775 | 200 | 30.6 |
| hybrid_combmnz | with_features | 0.3745 | 0.0695 | 0.4366 | 0.5975 | 200 | 34.4 |
| multi_rrf | baseline | 0.4603 | 0.0840 | 0.5331 | 0.7233 | 200 | 44.6 |
| multi_rrf | with_features | 0.4603 | 0.0840 | 0.5331 | 0.7233 | 200 | 44.0 |
| multi_combsum | baseline | 0.4721 | 0.0840 | 0.5414 | 0.7208 | 200 | 43.0 |
| multi_combsum | with_features | 0.4721 | 0.0840 | 0.5414 | 0.7208 | 200 | 44.0 |
| multi_combmnz | baseline | 0.4725 | 0.0840 | 0.5419 | 0.7208 | 200 | 41.7 |
| multi_combmnz | with_features | 0.4725 | 0.0840 | 0.5419 | 0.7208 | 200 | 42.8 |

## Dataset: touche2020

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |
|---|---|---|---|---|---|---|---|
| tfidf | baseline | 0.0191 | 0.1755 | 0.1297 | 0.0359 | 49 | 87.7 |
| tfidf | with_features | 0.0191 | 0.1755 | 0.1297 | 0.0359 | 49 | 90.1 |
| bm25 | baseline | 0.1377 | 0.7388 | 0.6206 | 0.1521 | 49 | 0.8 |
| bm25 | with_features | 0.1377 | 0.7388 | 0.6206 | 0.1521 | 49 | 1.0 |
| embedding | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 8.8 |
| embedding | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 5.3 |
| hybrid_rrf | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 5.8 |
| hybrid_rrf | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 7.1 |
| hybrid_combsum | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 5.8 |
| hybrid_combsum | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 7.1 |
| hybrid_combmnz | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 5.5 |
| hybrid_combmnz | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 7.4 |
| multi_rrf | baseline | 0.0352 | 0.2694 | 0.2233 | 0.0579 | 49 | 11.5 |
| multi_rrf | with_features | 0.0352 | 0.2694 | 0.2233 | 0.0579 | 49 | 7.4 |
| multi_combsum | baseline | 0.0351 | 0.2673 | 0.2228 | 0.0573 | 49 | 7.5 |
| multi_combsum | with_features | 0.0351 | 0.2673 | 0.2228 | 0.0573 | 49 | 7.4 |
| multi_combmnz | baseline | 0.0353 | 0.2673 | 0.2222 | 0.0574 | 49 | 8.0 |
| multi_combmnz | with_features | 0.0353 | 0.2673 | 0.2222 | 0.0574 | 49 | 7.4 |


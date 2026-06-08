# Evaluation Summary

Metrics: **MAP@10, P@10, nDCG@10, R@10**

Conditions: **baseline** = no refinement; **with_features** = spell + synonyms + personalization

## Dataset: nq

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |
|---|---|---|---|---|---|---|---|
| tfidf | baseline | 0.0078 | 0.0022 | 0.0106 | 0.0181 | 200 | 171.3 |
| tfidf | with_features | 0.0078 | 0.0022 | 0.0106 | 0.0181 | 200 | 170.8 |
| bm25 | baseline | 0.0170 | 0.0035 | 0.0205 | 0.0300 | 200 | 3.8 |
| bm25 | with_features | 0.0170 | 0.0035 | 0.0205 | 0.0300 | 200 | 3.7 |
| embedding | baseline | 0.0250 | 0.0046 | 0.0290 | 0.0393 | 200 | 22.3 |
| embedding | with_features | 0.0217 | 0.0040 | 0.0253 | 0.0346 | 200 | 21.0 |
| hybrid_rrf | baseline | 0.0250 | 0.0046 | 0.0290 | 0.0393 | 200 | 27.8 |
| hybrid_rrf | with_features | 0.0217 | 0.0040 | 0.0253 | 0.0346 | 200 | 31.6 |
| hybrid_combsum | baseline | 0.0250 | 0.0046 | 0.0290 | 0.0393 | 200 | 27.8 |
| hybrid_combsum | with_features | 0.0217 | 0.0040 | 0.0253 | 0.0346 | 200 | 31.6 |
| hybrid_combmnz | baseline | 0.0250 | 0.0046 | 0.0290 | 0.0393 | 200 | 29.0 |
| hybrid_combmnz | with_features | 0.0217 | 0.0040 | 0.0253 | 0.0346 | 200 | 31.6 |
| multi_rrf | baseline | 0.0267 | 0.0049 | 0.0309 | 0.0419 | 200 | 40.6 |
| multi_rrf | with_features | 0.0267 | 0.0049 | 0.0309 | 0.0419 | 200 | 36.8 |
| multi_combsum | baseline | 0.0274 | 0.0049 | 0.0314 | 0.0418 | 200 | 35.6 |
| multi_combsum | with_features | 0.0274 | 0.0049 | 0.0314 | 0.0418 | 200 | 36.3 |
| multi_combmnz | baseline | 0.0274 | 0.0049 | 0.0314 | 0.0418 | 200 | 36.5 |
| multi_combmnz | with_features | 0.0274 | 0.0049 | 0.0314 | 0.0418 | 200 | 38.5 |

## Dataset: touche2020

| Representation | Condition | MAP@10 | P@10 | nDCG@10 | R@10 | Queries | Time (s) |
|---|---|---|---|---|---|---|---|
| tfidf | baseline | 0.0191 | 0.1755 | 0.1297 | 0.0359 | 49 | 82.8 |
| tfidf | with_features | 0.0191 | 0.1755 | 0.1297 | 0.0359 | 49 | 83.1 |
| bm25 | baseline | 0.1377 | 0.7388 | 0.6206 | 0.1521 | 49 | 0.9 |
| bm25 | with_features | 0.1377 | 0.7388 | 0.6206 | 0.1521 | 49 | 0.9 |
| embedding | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 6.9 |
| embedding | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 4.3 |
| hybrid_rrf | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 6.6 |
| hybrid_rrf | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 7.1 |
| hybrid_combsum | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 5.8 |
| hybrid_combsum | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 7.4 |
| hybrid_combmnz | baseline | 0.0351 | 0.2857 | 0.2248 | 0.0609 | 49 | 5.7 |
| hybrid_combmnz | with_features | 0.0342 | 0.2776 | 0.2185 | 0.0591 | 49 | 6.9 |
| multi_rrf | baseline | 0.0352 | 0.2694 | 0.2233 | 0.0579 | 49 | 26.7 |
| multi_rrf | with_features | 0.0352 | 0.2694 | 0.2233 | 0.0579 | 49 | 7.1 |
| multi_combsum | baseline | 0.0351 | 0.2673 | 0.2228 | 0.0573 | 49 | 7.5 |
| multi_combsum | with_features | 0.0351 | 0.2673 | 0.2228 | 0.0573 | 49 | 7.7 |
| multi_combmnz | baseline | 0.0353 | 0.2673 | 0.2222 | 0.0574 | 49 | 7.9 |
| multi_combmnz | with_features | 0.0353 | 0.2673 | 0.2222 | 0.0574 | 49 | 7.7 |


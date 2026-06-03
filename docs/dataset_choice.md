# Dataset Choice

> **Filled in Phase 1.** Records the two datasets we will use, the qrels, and the
> justification.

## Hard constraints (from the spec)
- ≥ 200,000 documents each.
- Has **qrels** and test queries.
- **NOT** `antique`.
- One of the proposed datasets from `ir-datasets.com` may be replaced with an external
  dataset, subject to instructor approval.

## Candidates (preliminary)

| Dataset | Docs | Has qrels | Notes |
|---------|------|-----------|-------|
| `msmarco/passage` | ~8.8M | ✅ | Classic IR benchmark, well-known. |
| `msmarco/document` | ~3.2M | ✅ | Document-level variant. |
| `cord19/abstracts` | ~500K | ✅ | COVID research; different domain. |
| `medline/2004` | 250K+ | ✅ | Biomedical. |
| `nyt/acquis` | 300K+ | ✅ | News. |

## Recommended pair (pending instructor approval)

- **Dataset A:** `msmarco/passage`
- **Dataset B:** `cord19/abstracts`

## Final decision

_To be filled in Phase 1._

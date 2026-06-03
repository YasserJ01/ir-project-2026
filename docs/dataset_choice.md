# Dataset Choice

> **Filled in Phase 1.** Records the two datasets we will use, the qrels, and the
> justification. Final decision below.

## Hard constraints (from the spec)
- ≥ 200,000 documents each.
- Has **qrels** and test queries.
- **NOT** `antique`.
- One of the proposed datasets from `ir-datasets.com` may be replaced with an external
  dataset, subject to instructor approval.

## Candidates surveyed (June 2026)

| Dataset | Docs | Has qrels | Notes |
|---------|------|-----------|-------|
| `msmarco-passage` | 8,841,823 | ✅ (dev/small) | Classic IR benchmark; 1.06 GB tarball — slow to download on this connection. |
| `cord19` (base) | 192,509 | ❌ (only via trec-covid subsets) | **Under 200K spec minimum — rejected.** |
| `cord19/trec-covid` | 171,332 | ✅ | **Under 200K — rejected.** |
| `cord19/fulltext` | 192,509 | ❌ | Same as `cord19` base — rejected. |
| `aquaint/trec-robust-2005` | 1,033,461 | ✅ | Requires LDC license; corpus not freely downloadable — rejected. |
| `medline/2004` | 3,672,808 | ✅ | Very large; not freely downloadable via `ir_datasets`. |
| **`beir/webis-touche2020`** | **382,545** | ✅ (**2,962** qrels) | **Selected for Dataset A.** Argument retrieval, debate topics, ~227 MB, freely downloadable. |
| `beir/nq` | 2,681,468 | ✅ (4,201 qrels) | Natural Questions; would be huge to ingest full. |
| **`beir/nq`** | capped to **500,000** | ✅ | **Selected for Dataset B.** Open-domain QA, Wikipedia passages. |

## Final decision

### Dataset A — `beir/webis-touche2020` (capped at full corpus)
- **Docs:** 382,545 total → 382,544 ingested after 1 empty-text filter.
- **Qrels:** 2,962 (TREC-style 4-touch reformulation of args.me).
- **Domain:** argument retrieval / debate topics.
- **Storage:** 636 MB JSONL.
- **Why this is a good pick:**
  - Well above the 200K spec minimum.
  - Domain that contrasts strongly with open-domain QA — gives the report a real "domain robustness" angle.
  - Freely downloadable via `ir_datasets` (no LDC license).
  - Qrels are clean TREC format — Phase 9 evaluation works out of the box.
- **Why we did NOT use the original recommendation (`cord19/abstracts`):**
  - The catalog key `cord19/abstracts` does not exist in the current `ir_datasets` (0.5.11) registry.
  - The `cord19` base corpus is 192,509 docs — 7,491 below the spec's hard 200K floor.
  - The biomedical specialty was not worth a spec rejection.
  - **Instructor sign-off still required** for this deviation from the guide's §1.1 recommendation.

### Dataset B — `beir/nq` (capped at 500,000)
- **Docs:** 2,681,468 total → 500,000 ingested (deterministic first-500K subset, matching the MS MARCO 500K cap from the original plan).
- **Qrels:** 4,201 (BEIR/NQ official test split).
- **Domain:** open-domain question answering over Wikipedia passages (Natural Questions).
- **Storage:** 254 MB JSONL.
- **Why this is a good pick:**
  - Largest freely-downloadable BEIR dataset with qrels, so the cap is the only thing keeping it tractable.
  - Domain contrast with Dataset A (open-domain QA vs argument retrieval).
  - The Wikipedia-passage format exercises the same preprocessing pipeline as MS MARCO would have.

## Ingestion result (post-Phase 1)

| Dataset | Raw docs | Ingested | Skipped (empty) | Cap | JSONL size | Wall time |
|---------|---------:|---------:|----------------:|----:|-----------:|----------:|
| `touche2020` | 382,545 | 382,544 | 1 | 500,000 (effective full) | 636.4 MB | 5:29 |
| `nq`        | 2,681,468 | 500,000 | 0 (capped before any skip) | 500,000 | 254.1 MB | 9:34 |

Tokenization result:

| Dataset | Docs | Total tokens | Mean / doc | tok/s | Workers | Wall |
|---------|-----:|-------------:|-----------:|------:|--------:|-----:|
| `touche2020` | 382,544 | 57,069,964 | 149.19 | 126,774 | 8 | 7:30 |
| `nq`        | 500,000 | 24,540,420 |  49.08 | 130,731 | 8 | 3:08 |

(Mean tokens/doc on `touche2020` is ~3× higher than on `nq` because the source
docs are full debate arguments rather than short Wikipedia passages.)

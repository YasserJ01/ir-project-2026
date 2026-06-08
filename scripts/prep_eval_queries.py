"""Sample test queries with non-empty qrels for both datasets.

Output: evaluation/queries/<dataset_id>_queries.txt (TREC format: query_id\ttext).
"""
import logging
import random
import sys
from pathlib import Path

import ir_datasets

random.seed(42)

SAMPLES_PER_DATASET = 200
DATASETS = ["beir/webis-touche2020", "beir/nq"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "queries"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prep_eval_queries")


def _has_qrel(dataset_id: str, query_id: str, qrels: dict[str, set[str]]) -> bool:
    return query_id in qrels


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds_id in DATASETS:
        short_name = ds_id.split("/")[-1]
        logger.info("Processing %s ...", ds_id)

        ds = ir_datasets.load(ds_id)

        # Build query_id → set of doc_ids with positive relevance
        qrel_map: dict[str, set[str]] = {}
        for qrel in ds.qrels_iter():
            if qrel.relevance > 0:
                qrel_map.setdefault(qrel.query_id, set()).add(qrel.doc_id)

        # Collect all queries
        all_queries = list(ds.queries_iter())
        logger.info("  Total queries: %d", len(all_queries))
        logger.info("  Queries with positive qrels: %d", len(qrel_map))

        # Filter to queries that have at least one positive qrel
        valid = [(q.query_id, q.text) for q in all_queries if _has_qrel(ds_id, q.query_id, qrel_map)]
        logger.info("  Queries with qrels: %d", len(valid))

        if not valid:
            logger.warning("  No queries with qrels found for %s, using all queries", ds_id)
            valid = [(q.query_id, q.text) for q in all_queries]

        # Sample
        sample = valid
        if len(valid) > SAMPLES_PER_DATASET:
            sample = random.sample(valid, SAMPLES_PER_DATASET)

        # Write TREC format: query_id\ttext
        out_path = OUTPUT_DIR / f"{short_name}_queries.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for qid, text in sample:
                f.write(f"{qid}\t{text}\n")

        logger.info("  Wrote %d queries to %s", len(sample), out_path)


if __name__ == "__main__":
    main()

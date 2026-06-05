"""Query-refinement service.

Phase 4 of the IR project. Wraps four sub-modules behind a FastAPI
service on port 8004:

    - ``spell`` -- SymSpell edit-distance correction
    - ``synonyms`` -- NLTK WordNet expansion
    - ``grammar`` -- language-tool-python grammar correction (off by default)
    - ``personalization`` -- per-user click-history weighting

Public entry points:
    - ``app.service.app`` -- the FastAPI application
    - ``app.service.run`` -- CLI helper to run the service via uvicorn
"""

from __future__ import annotations

__all__ = ["DATASET_IDS_REFINEMENT"]

# A user query can be relevant to any dataset the system knows about;
# the refinement service itself is dataset-agnostic, but we keep the
# tuple here so the gateway (Phase 6) can list "supported datasets"
# uniformly.
DATASET_IDS_REFINEMENT: tuple[str, ...] = ("touche2020", "nq")

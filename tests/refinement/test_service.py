"""Integration tests for ``services.refinement.app.service`` (HTTP layer)."""

from __future__ import annotations


class TestService:
    def test_root(self, client_factory) -> None:
        client = client_factory
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "refinement"
        assert "POST /refine" in body["endpoints"]

    def test_health(self, client_factory) -> None:
        client = client_factory
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "refinement"
        # SymSpell + WordNet should be loaded (we tested them above).
        assert body["spell_loaded"] is True
        assert body["wordnet_loaded"] is True
        # Grammar is off by default.
        assert body["grammar_enabled"] is False

    def test_refine_clean(self, client_factory) -> None:
        client = client_factory
        r = client.post(
            "/refine",
            json={"query": "What is the capital of France?"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "What is the capital of France?"
        assert "capital" in body["refined_query"]
        assert isinstance(body["tokens"], list)
        assert isinstance(body["weighted_tokens"], list)
        assert "latency_ms" in body

    def test_refine_typo(self, client_factory) -> None:
        client = client_factory
        r = client.post(
            "/refine",
            json={"query": "recieve teh helo"},
        )
        assert r.status_code == 200
        body = r.json()
        # Spell should have fired.
        assert body["stages"]["spell"] != ""
        # "recieve" should have been corrected.
        assert (
            "recieve" not in body["refined_query"].lower()
            or "receive" in body["refined_query"].lower()
        )

    def test_refine_with_user_id(self, client_factory) -> None:
        client = client_factory
        r = client.post(
            "/refine",
            json={"query": "what is the eiffel tower", "user_id": "user_1"},
        )
        assert r.status_code == 200
        body = r.json()
        # Should have a personalization stage (even if empty).
        assert "personalization" in body["stages"]
        assert body["user_id"] == "user_1"

    def test_refine_toggles(self, client_factory) -> None:
        client = client_factory
        r = client.post(
            "/refine",
            json={
                "query": "fast car running",
                "enable_spell": False,
                "enable_synonyms": False,
                "enable_personalization": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        # All stages off -> stages dict is all empty.
        assert body["stages"]["spell"] == ""
        assert body["stages"]["synonyms"] == ""
        assert body["stages"]["personalization"] == ""
        # Without synonym expansion, expanded == refined.
        assert body["expanded_query"] == body["refined_query"]

    def test_refine_unknown_user(self, client_factory) -> None:
        client = client_factory
        r = client.post(
            "/refine",
            json={"query": "anything", "user_id": "ghost_user_with_no_log"},
        )
        assert r.status_code == 200
        body = r.json()
        # No log -> no boost -> stages["personalization"] is empty.
        assert body["stages"]["personalization"] == ""

    def test_refine_validation(self, client_factory) -> None:
        client = client_factory
        # Empty query is rejected.
        r = client.post("/refine", json={"query": ""})
        assert r.status_code == 422

    def test_refine_default_user(self, client_factory) -> None:
        client = client_factory
        r = client.post("/refine", json={"query": "hello world"})
        assert r.status_code == 200
        body = r.json()
        # Default user_id is "anonymous".
        assert body["user_id"] == "anonymous"

    def test_refine_forward_compat(self, client_factory) -> None:
        """A future client can send extra fields; we don't 422."""
        client = client_factory
        r = client.post(
            "/refine",
            json={
                "query": "hello world",
                "dataset_id": "touche2020",  # Phase 5+ field
                "k": 10,
            },
        )
        # 200 OK -- extra fields are silently ignored (extra="ignore").
        assert r.status_code == 200

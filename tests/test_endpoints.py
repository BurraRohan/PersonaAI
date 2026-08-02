"""
End-to-end endpoint tests.

The Groq boundary is monkeypatched, so these run offline. They prove the API
wiring works — including that /brand really goes through the ReAct agent and
returns its tool-call trace.
"""

import json
import os
import sys

import pytest

os.environ.setdefault("API_KEY", "test-key-for-pytest")
os.environ.setdefault("GROQ_API_KEY", "not-used-in-tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTH = {"Authorization": "Bearer test-key-for-pytest"}

FAKE_STRATEGY = {
    "tone": "Direct and practical",
    "content_themes": ["RAG", "MLOps"],
    "positioning_summary": "Engineer who ships.",
    "do_guidelines": ["Show real numbers"],
    "dont_guidelines": ["Avoid buzzwords"],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by a temporary database, with the LLM stubbed out."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")

    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils", "schemas"}:
            del sys.modules[module]

    import services.llm_service as llm

    monkeypatch.setattr(llm, "_call_with_retry",
                        lambda prompt, trace_id=None: json.dumps(FAKE_STRATEGY))

    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c


class TestAuthEnforcement:
    def test_missing_header_is_401(self, client):
        assert client.post("/brand", json={}).status_code == 401

    def test_wrong_key_is_403(self, client):
        res = client.post("/brand", json={}, headers={"Authorization": "Bearer nope"})
        assert res.status_code == 403

    def test_health_is_public(self, client):
        assert client.get("/health").status_code == 200


class TestBrandEndpoint:
    def _payload(self):
        return {
            "name": "Ada Lovelace",
            "role": "ML Engineer",
            "industry": "AI/ML",
            "goals": "Thought leadership",
            "preferred_tone": "Direct",
        }

    def test_agent_path_returns_trace(self, client, monkeypatch):
        """When the agent runs and saves, the response reports mode 'agent'."""
        import agents.brand_agent as ba

        class FakeAI:
            type = "ai"
            tool_calls = [{"name": "check_existing_profile", "args": {"name": "Ada Lovelace"}}]
            content = ""

        class FakeTool:
            type = "tool"
            tool_calls = []

            def __init__(self, name, content):
                self.name = name
                self.content = content

        def fake_run(agent, instruction):
            # The save tool writes a real row, exactly as the live agent would.
            saved = ba.save_brand_profile.invoke({
                "name": "Ada Lovelace", "role": "ML Engineer", "industry": "AI/ML",
                "goals": "Thought leadership", "preferred_tone": "Direct",
                "strategy_json": json.dumps(FAKE_STRATEGY),
            })
            return [
                FakeAI(),
                FakeTool("check_existing_profile", '{"exists": false}'),
                FakeTool("save_brand_profile", saved),
            ]

        # Force agent mode on: this test must not depend on the developer's .env.
        monkeypatch.setattr(ba, "AGENT_MODE", True)
        monkeypatch.setattr(ba, "run_agent", fake_run)
        monkeypatch.setattr(ba, "get_brand_agent", lambda: object())

        res = client.post("/brand", json=self._payload(), headers=AUTH)
        assert res.status_code == 200

        body = res.json()
        assert body["execution_mode"] == "agent"
        assert body["tone"] == "Direct and practical"
        assert len(body["agent_trace"]) >= 2
        assert body["agent_trace"][0]["action"] == "tool_call"

    def test_falls_back_when_agent_errors(self, client, monkeypatch):
        """A broken agent must not 500 the endpoint."""
        import agents.brand_agent as ba

        def boom(agent, instruction):
            raise RuntimeError("groq unreachable")

        monkeypatch.setattr(ba, "AGENT_MODE", True)
        monkeypatch.setattr(ba, "run_agent", boom)
        monkeypatch.setattr(ba, "get_brand_agent", lambda: object())

        res = client.post("/brand", json=self._payload(), headers=AUTH)
        assert res.status_code == 200

        body = res.json()
        assert body["execution_mode"] == "fallback_direct"
        assert body["id"] > 0
        assert any(t["action"] == "agent_error" for t in body["agent_trace"])


class TestPromptEndpoints:
    def test_prompts_are_seeded_on_startup(self, client):
        res = client.get("/prompts/predictor", headers=AUTH)
        assert res.status_code == 200
        assert len(res.json()) >= 1

    def test_rollback_activates_target_version(self, client):
        from database.db import SessionLocal
        from database.models import PromptTemplate

        db = SessionLocal()
        try:
            db.add(PromptTemplate(agent_name="brand", version=2,
                                  template="v2", is_active=True))
            db.commit()
        finally:
            db.close()

        res = client.post("/prompts/brand/rollback/1", headers=AUTH)
        assert res.status_code == 200
        assert res.json()["version"] == 1
        assert res.json()["is_active"] is True

        listed = client.get("/prompts/brand", headers=AUTH).json()
        active = [p for p in listed if p["is_active"]]
        assert len(active) == 1
        assert active[0]["version"] == 1

    def test_rollback_to_missing_version_is_404(self, client):
        assert client.post("/prompts/brand/rollback/99", headers=AUTH).status_code == 404


class TestAgentModeSwitch:
    """AGENT_MODE=0 must bypass the agent loop entirely and still work."""

    def _payload(self):
        return {
            "name": "Grace Hopper",
            "role": "Systems Engineer",
            "industry": "Computing",
            "goals": "Teaching",
            "preferred_tone": "Plainspoken",
        }

    def test_agent_mode_off_uses_direct_path(self, client, monkeypatch):
        import agents.brand_agent as ba

        def should_not_run(agent, instruction):
            raise AssertionError("run_agent was called despite AGENT_MODE=0")

        monkeypatch.setattr(ba, "AGENT_MODE", False)
        monkeypatch.setattr(ba, "run_agent", should_not_run)

        res = client.post("/brand", json=self._payload(), headers=AUTH)
        assert res.status_code == 200

        body = res.json()
        assert body["execution_mode"] == "direct"
        assert body["agent_trace"] == []
        assert body["tone"] == "Direct and practical"
        assert body["id"] > 0

    def test_direct_mode_output_matches_agent_mode_shape(self, client, monkeypatch):
        """Both modes must return the same fields, so the frontend is unaffected."""
        import agents.brand_agent as ba
        monkeypatch.setattr(ba, "AGENT_MODE", False)

        body = client.post("/brand", json=self._payload(), headers=AUTH).json()
        for field in ("id", "name", "role", "tone", "content_themes",
                      "positioning_summary", "execution_mode", "agent_trace"):
            assert field in body
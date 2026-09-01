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

# Set, not setdefault: the suite must use its own values even when the ambient
# environment (CI, a shell export, a .env already loaded) has different ones.
# setdefault silently loses that race and every authenticated request 403s.
os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

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

class TestReviewGate:
    """Human-in-the-loop: generated posts need a decision before they count."""

    def _make_post(self, client, status="pending"):
        """Create a profile and a post directly, bypassing the LLM."""
        from database.db import SessionLocal
        from database.models import BrandProfile, Post

        db = SessionLocal()
        try:
            profile = BrandProfile(
                name="Review Tester", role="SDE", industry="Auto",
                goals="Authority", preferred_tone="Direct",
                tone="Direct", content_themes='["AI"]',
                positioning_summary="Ships things.",
                do_guidelines='["Be specific"]', dont_guidelines='["Be vague"]',
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)

            post = Post(
                user_id=profile.id, topic="Test topic",
                content="Test content for review.", hashtags='["test"]',
                status=status,
            )
            db.add(post)
            db.commit()
            db.refresh(post)
            return profile.id, post.id
        finally:
            db.close()

    def test_new_posts_start_pending(self, client):
        _, post_id = self._make_post(client)
        from database.db import SessionLocal
        from database.models import Post

        db = SessionLocal()
        try:
            assert db.query(Post).filter(Post.id == post_id).first().status == "pending"
        finally:
            db.close()

    def test_approve_sets_status_and_timestamp(self, client):
        _, post_id = self._make_post(client)
        res = client.post(f"/posts/{post_id}/review",
                          json={"decision": "approve"}, headers=AUTH)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "approved"
        assert body["reviewed_at"] is not None

    def test_reject_keeps_the_post_with_a_note(self, client):
        """Rejection is a record, not a delete."""
        _, post_id = self._make_post(client)
        res = client.post(f"/posts/{post_id}/review",
                          json={"decision": "reject", "note": "Too generic"},
                          headers=AUTH)
        assert res.status_code == 200
        assert res.json()["status"] == "rejected"
        assert res.json()["review_note"] == "Too generic"

        from database.db import SessionLocal
        from database.models import Post
        db = SessionLocal()
        try:
            assert db.query(Post).filter(Post.id == post_id).first() is not None
        finally:
            db.close()

    def test_invalid_decision_is_422(self, client):
        _, post_id = self._make_post(client)
        res = client.post(f"/posts/{post_id}/review",
                          json={"decision": "maybe"}, headers=AUTH)
        assert res.status_code == 422

    def test_review_missing_post_is_404(self, client):
        res = client.post("/posts/9999/review",
                          json={"decision": "approve"}, headers=AUTH)
        assert res.status_code == 404

    def test_pending_list_only_shows_pending(self, client):
        user_id, pending_id = self._make_post(client, status="pending")

        from database.db import SessionLocal
        from database.models import Post
        db = SessionLocal()
        try:
            db.add(Post(user_id=user_id, topic="Approved one",
                        content="Already reviewed.", status="approved"))
            db.commit()
        finally:
            db.close()

        res = client.get(f"/posts/pending/{user_id}", headers=AUTH)
        assert res.status_code == 200
        ids = [p["post_id"] for p in res.json()]
        assert ids == [pending_id]

    def test_engagement_blocked_until_approved(self, client):
        """A pending post is not publishable, so it cannot have engagement."""
        _, post_id = self._make_post(client)
        res = client.post("/engagement",
                          json={"post_id": post_id, "likes": 10,
                                "comments": 2, "shares": 1},
                          headers=AUTH)
        assert res.status_code == 409
        assert "not approved" in res.json()["detail"]

    def test_engagement_allowed_after_approval(self, client):
        _, post_id = self._make_post(client)
        client.post(f"/posts/{post_id}/review",
                    json={"decision": "approve"}, headers=AUTH)

        res = client.post("/engagement",
                          json={"post_id": post_id, "likes": 10,
                                "comments": 2, "shares": 1},
                          headers=AUTH)
        assert res.status_code == 200

    def test_engagement_blocked_on_rejected(self, client):
        _, post_id = self._make_post(client, status="rejected")
        res = client.post("/engagement",
                          json={"post_id": post_id, "likes": 5,
                                "comments": 0, "shares": 0},
                          headers=AUTH)
        assert res.status_code == 409


class TestPredictDualMode:
    """/predict scores either a saved post or pasted text."""

    def _profile_and_post(self):
        from database.db import SessionLocal
        from database.models import BrandProfile, Post

        db = SessionLocal()
        try:
            profile = BrandProfile(
                name="Predict Tester", role="SDE", industry="Auto",
                goals="Authority", preferred_tone="Direct", tone="Direct",
                content_themes='["AI"]', positioning_summary="Ships.",
                do_guidelines='["Specific"]', dont_guidelines='["Vague"]',
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)

            post = Post(user_id=profile.id, topic="Saved topic",
                        content="Saved post content.", status="pending")
            db.add(post)
            db.commit()
            db.refresh(post)
            return profile.id, post.id
        finally:
            db.close()

    def test_post_id_returns_review_state(self, client, monkeypatch):
        import services.llm_service as llm
        monkeypatch.setattr(llm, "_call_with_retry",
                            lambda prompt, trace_id=None: json.dumps({
                                "overall_score": 72, "predicted_likes": "30-50",
                                "predicted_comments": "5-12", "predicted_shares": "2-6",
                                "brand_alignment": 80, "hook_strength": 70,
                                "readability": 75, "call_to_action": 65,
                                "improvement_tips": "Tighten the opening.",
                            }))

        user_id, post_id = self._profile_and_post()
        res = client.post("/predict",
                          json={"user_id": user_id, "post_id": post_id},
                          headers=AUTH)
        assert res.status_code == 200
        body = res.json()
        assert body["post_id"] == post_id
        assert body["status"] == "pending"

    def test_draft_text_has_no_post_id(self, client, monkeypatch):
        """Pasted text is scored but not approvable."""
        import services.llm_service as llm
        monkeypatch.setattr(llm, "_call_with_retry",
                            lambda prompt, trace_id=None: json.dumps({
                                "overall_score": 55, "predicted_likes": "10-20",
                                "predicted_comments": "1-4", "predicted_shares": "0-2",
                                "brand_alignment": 50, "hook_strength": 60,
                                "readability": 70, "call_to_action": 40,
                                "improvement_tips": "Add a hook.",
                            }))

        user_id, _ = self._profile_and_post()
        res = client.post("/predict",
                          json={"user_id": user_id, "draft_content": "Some scratch text."},
                          headers=AUTH)
        assert res.status_code == 200
        assert res.json()["post_id"] is None
        assert res.json()["status"] is None

    def test_neither_input_is_422(self, client):
        user_id, _ = self._profile_and_post()
        res = client.post("/predict", json={"user_id": user_id}, headers=AUTH)
        assert res.status_code == 422

    def test_post_from_another_profile_is_404(self, client):
        user_id, post_id = self._profile_and_post()
        res = client.post("/predict",
                          json={"user_id": user_id + 999, "post_id": post_id},
                          headers=AUTH)
        assert res.status_code in (404,)
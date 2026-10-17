"""
Security regression tests.

Each class pins one finding from the security review so it cannot quietly
return. All offline: the Groq boundary is stubbed.
"""

import json
import os
import sys

import pytest

os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["ADMIN_API_KEY"] = "admin-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTH = {"Authorization": "Bearer test-key-for-pytest"}
ADMIN = {"Authorization": "Bearer admin-key-for-pytest"}
XSS = '<img src=x onerror=alert(document.cookie)>'


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/sec.db")
    monkeypatch.delenv("METRICS_PUBLIC", raising=False)
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils", "schemas"}:
            del sys.modules[module]

    import services.llm_service as llm
    monkeypatch.setattr(llm, "_call_with_retry", lambda prompt, trace_id=None: json.dumps({
        "tone": "t", "content_themes": ["a"], "positioning_summary": "p",
        "do_guidelines": [], "dont_guidelines": [],
        "post_content": "Generated post.", "suggested_hashtags": ["x"],
    }))

    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _profile(name="Owner"):
    from database.db import SessionLocal
    from database.models import BrandProfile
    db = SessionLocal()
    try:
        p = BrandProfile(name=name, role="r", industry="i", goals="g", preferred_tone="t",
                         tone="t", content_themes='["x"]', positioning_summary="p",
                         do_guidelines="[]", dont_guidelines="[]")
        db.add(p); db.commit(); db.refresh(p)
        return p.id
    finally:
        db.close()


# ── Prompt injection: the model must not choose whose data it touches ──

class TestAgentToolsAreScoped:
    def test_content_tools_take_no_profile_or_topic_argument(self, client):
        from agents.content_agent import build_content_tools
        for t in build_content_tools(1, "topic"):
            assert "user_id" not in t.args
            assert "topic" not in t.args

    def test_brand_tools_take_no_personal_details(self, client):
        from agents.brand_agent import build_brand_tools
        from schemas.schemas import BrandRequest
        req = BrandRequest(name="A", role="r", industry="i", goals="g", preferred_tone="t")
        for t in build_brand_tools(req):
            assert t.args == {}

    def test_feedback_tools_take_no_profile_argument(self, client):
        from agents.feedback_agent import build_feedback_tools
        for t in build_feedback_tools(1, "summary"):
            assert "user_id" not in t.args
            assert "brand_summary" not in t.args

    def test_save_tool_writes_to_the_bound_profile_only(self, client):
        from agents.content_agent import build_content_tools
        from database.db import SessionLocal
        from database.models import Post

        victim = _profile("Victim")
        attacker = _profile("Attacker")
        save = {t.name: t for t in build_content_tools(attacker, "my topic")}["create_and_save_post"]
        post_id = json.loads(save.invoke({}))["post_id"]

        db = SessionLocal()
        try:
            post = db.query(Post).filter(Post.id == post_id).first()
            assert post.user_id == attacker
            assert post.topic == "my topic"
            assert db.query(Post).filter(Post.user_id == victim).count() == 0
        finally:
            db.close()

    def test_generate_rejects_a_post_belonging_to_another_profile(self, client, monkeypatch):
        """Even if a tool result names another profile's post, it is not returned."""
        import agents.content_agent as ca
        from database.db import SessionLocal
        from database.models import Post

        mine = _profile("Mine")
        other = _profile("Other")
        db = SessionLocal()
        try:
            foreign = Post(user_id=other, topic="secret", content="other profile's draft")
            db.add(foreign); db.commit(); db.refresh(foreign)
            foreign_id = foreign.id
        finally:
            db.close()

        class FakeTool:
            type = "tool"; tool_calls = []; name = "create_and_save_post"
            content = json.dumps({"post_id": foreign_id})

        monkeypatch.setattr(ca, "AGENT_MODE", True)
        monkeypatch.setattr(ca, "get_content_agent", lambda *a: object())
        monkeypatch.setattr(ca, "run_agent", lambda agent, instr, *args: [FakeTool()])

        res = client.post("/generate", json={"user_id": mine, "topic": "A real topic"}, headers=AUTH)
        assert res.status_code == 200
        assert res.json()["post_id"] != foreign_id
        assert res.json()["execution_mode"] == "fallback_direct"


# ── XSS via model output ──

class TestPredictionOutputIsSanitised:
    def _predict(self, client, monkeypatch, payload):
        import services.llm_service as llm
        monkeypatch.setattr(llm, "_call_with_retry", lambda prompt, trace_id=None: json.dumps(payload))
        uid = _profile()
        return client.post("/predict", json={"user_id": uid, "draft_content": "x"}, headers=AUTH)

    def test_markup_in_scores_and_ranges_is_neutralised(self, client, monkeypatch):
        res = self._predict(client, monkeypatch, {
            "overall_score": XSS, "brand_alignment": "80%;background:url(//evil)",
            "hook_strength": XSS, "readability": 70, "call_to_action": "999",
            "predicted_likes": XSS, "predicted_comments": "1-2", "predicted_shares": "abc",
            "improvement_tips": "fine",
        })
        assert res.status_code == 200
        body = res.json()
        for key in ("overall_score", "brand_alignment", "hook_strength",
                    "readability", "call_to_action"):
            assert isinstance(body[key], int) and 1 <= body[key] <= 100
        assert body["predicted_likes"] == "N/A"
        assert body["predicted_comments"] == "1-2"
        assert body["predicted_shares"] == "N/A"
        assert "<" not in json.dumps({k: v for k, v in body.items() if k != "improvement_tips"})

    def test_ranges_are_normalised(self, client, monkeypatch):
        res = self._predict(client, monkeypatch, {
            "brand_alignment": 50, "hook_strength": 50, "readability": 50, "call_to_action": 50,
            "predicted_likes": " 50 – 30 ", "predicted_comments": "2-6", "predicted_shares": "0-1",
            "improvement_tips": "",
        })
        assert res.json()["predicted_likes"] == "30-50"

    def test_frontend_never_inlines_style_attributes(self):
        js = open(os.path.join(os.path.dirname(__file__), "..", "static", "app.js"),
                  encoding="utf-8").read()
        assert 'style="' not in js


# ── Input bounds ──

class TestInputLimits:
    @pytest.mark.parametrize("payload", [
        {"name": "A" * 101, "role": "r", "industry": "i", "goals": "g", "preferred_tone": "t"},
        {"name": "A", "role": "r" * 201, "industry": "i", "goals": "g", "preferred_tone": "t"},
        {"name": "A", "role": "r", "industry": "i", "goals": "g" * 1001, "preferred_tone": "t"},
        {"name": "   ", "role": "r", "industry": "i", "goals": "g", "preferred_tone": "t"},
    ])
    def test_oversized_or_blank_profile_fields_are_rejected(self, client, payload):
        assert client.post("/brand", json=payload, headers=AUTH).status_code == 422

    def test_topic_length_is_bounded(self, client):
        uid = _profile()
        assert client.post("/generate", json={"user_id": uid, "topic": "t" * 301},
                           headers=AUTH).status_code == 422

    @pytest.mark.parametrize("topic", ["hi", "hit", "   short   ", "123456789"])
    def test_topic_needs_at_least_ten_characters(self, client, topic):
        uid = _profile()
        assert client.post("/generate", json={"user_id": uid, "topic": topic},
                           headers=AUTH).status_code == 422

    def test_short_real_topic_is_accepted(self, client):
        uid = _profile()
        res = client.post("/generate", json={"user_id": uid, "topic": "AI in EdTech"},
                          headers=AUTH)
        assert res.status_code != 422

    def test_huge_engagement_numbers_are_a_422_not_a_crash(self, client):
        assert client.post("/engagement", json={"post_id": 1, "likes": 10 ** 30},
                           headers=AUTH).status_code == 422

    def test_negative_engagement_is_rejected(self, client):
        assert client.post("/engagement", json={"post_id": 1, "likes": -1},
                           headers=AUTH).status_code == 422

    def test_profile_edit_is_bounded(self, client):
        uid = _profile()
        assert client.patch(f"/brand/{uid}", json={"name": "A" * 101},
                            headers=AUTH).status_code == 422
        assert client.patch(f"/brand/{uid}", json={"content_themes": ["x"] * 21},
                            headers=AUTH).status_code == 422

    @pytest.mark.parametrize("path", [
        "/dashboard/99999999999999999999",
        "/posts/99999999999999999999",
        "/posts/pending/0",
        "/history/-1",
    ])
    def test_out_of_range_path_ids_are_422(self, client, path):
        assert client.get(path, headers=AUTH).status_code == 422

    @pytest.mark.parametrize("query", ["limit=0", "limit=501", "limit=-5",
                                       "agent_name=" + "a" * 51, "agent_name=DROP;"])
    def test_audit_log_query_is_bounded(self, client, query):
        assert client.get(f"/audit-logs?{query}", headers=ADMIN).status_code == 422


# ── Request size ──

class TestBodySizeLimit:
    def test_large_body_is_rejected_before_parsing(self, client):
        res = client.post("/brand", content=b"{" + b" " * (70 * 1024) + b"}",
                          headers={**AUTH, "Content-Type": "application/json"})
        assert res.status_code == 413

    def test_normal_body_passes(self, client):
        res = client.post("/brand", json={"name": "A", "role": "r", "industry": "i",
                                          "goals": "g", "preferred_tone": "t"}, headers=AUTH)
        assert res.status_code != 413


# ── Information disclosure ──

class TestNoInternalDetailsLeak:
    def test_llm_errors_return_a_generic_message(self, client, monkeypatch):
        import importlib
        from fastapi import HTTPException
        import services.llm_service as llm

        # The fixture stubs _call_with_retry; reload to test the real one.
        real = importlib.reload(llm)

        class FailingClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        raise RuntimeError("upstream: key gsk_SECRET123 invalid at /internal/path")

        monkeypatch.setattr(real, "client", FailingClient)
        with pytest.raises(HTTPException) as exc:
            real._call_with_retry("prompt", trace_id="abc-123")

        assert exc.value.status_code == 502
        assert "gsk_" not in exc.value.detail
        assert "/internal" not in exc.value.detail
        assert "abc-123" in exc.value.detail

    def test_agent_errors_report_type_not_message(self, client, monkeypatch):
        import agents.brand_agent as ba

        def boom(agent, instruction, *args):
            raise RuntimeError("connection to 10.0.0.5:5432 refused, password=hunter2")

        monkeypatch.setattr(ba, "AGENT_MODE", True)
        monkeypatch.setattr(ba, "get_brand_agent", lambda *a: object())
        monkeypatch.setattr(ba, "run_agent", boom)
        body = client.post("/brand", json={"name": "A", "role": "r", "industry": "i",
                                           "goals": "g", "preferred_tone": "t"},
                           headers=AUTH).json()
        errors = [t for t in body["agent_trace"] if t["action"] == "agent_error"]
        assert errors and errors[0]["error"] == "RuntimeError"
        assert "hunter2" not in json.dumps(body)

    def test_metrics_requires_the_api_key(self, client):
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers=AUTH).status_code == 200


# ── Response headers ──

class TestSecurityHeaders:
    def test_app_responses_carry_a_strict_policy(self, client):
        h = client.get("/health").headers
        csp = h["content-security-policy"]
        assert "script-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert "frame-ancestors 'none'" in csp
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer"
        assert h["cache-control"] == "no-store"

    def test_ui_page_gets_the_strict_policy(self, client):
        assert "unsafe-inline" not in client.get("/").headers["content-security-policy"]

    def test_docs_still_load(self, client):
        res = client.get("/docs")
        assert res.status_code == 200
        assert "cdn.jsdelivr.net" in res.headers["content-security-policy"]

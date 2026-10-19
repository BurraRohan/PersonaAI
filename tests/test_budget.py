"""
Daily budget guard.

Spend is seeded straight into the audit log, so these tests control exactly
where today's total sits relative to the limit. Offline: the Groq boundary is
stubbed.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["ADMIN_API_KEY"] = "admin-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTH = {"Authorization": "Bearer test-key-for-pytest"}
ADMIN = {"Authorization": "Bearer admin-key-for-pytest"}
PROFILE = {"name": "Meera Iyer", "role": "Product Designer", "industry": "Healthtech",
           "goals": "Grow among UX designers", "preferred_tone": "Warm"}

STUB_REPLY = json.dumps({
    "tone": "warm", "content_themes": ["ux"], "positioning_summary": "p",
    "do_guidelines": [], "dont_guidelines": [],
    "post_content": "A post.", "suggested_hashtags": ["ux"],
    "performance_summary": "s", "improvement_recommendation": "r",
    "brand_alignment": 70, "hook_strength": 70, "readability": 70, "call_to_action": 70,
    "predicted_likes": "10-20", "predicted_comments": "1-2", "predicted_shares": "0-1",
    "improvement_tips": "t",
})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/budget.db")
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils", "schemas"}:
            del sys.modules[module]

    import services.llm_service as llm
    monkeypatch.setattr(llm, "_call_with_retry", lambda prompt, trace_id=None: STUB_REPLY)

    # Setup requests use the direct path; tests that need the agent path opt in.
    import agents.brand_agent as ba
    import agents.content_agent as ca
    import agents.feedback_agent as fa
    for module in (ba, ca, fa):
        monkeypatch.setattr(module, "AGENT_MODE", False)

    import utils.budget as budget
    monkeypatch.setattr(budget, "DAILY_BUDGET_USD", 1.00)
    monkeypatch.setattr(budget, "SOFT_LIMIT_PERCENT", 80)

    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _spend(amount, when=None):
    """Record LLM spend in the audit log, as a real call would."""
    from database.db import SessionLocal
    from database.models import AuditLog
    db = SessionLocal()
    try:
        db.add(AuditLog(trace_id="seed", agent_name="content", model="m", latency_ms=1,
                        status="success", prompt_tokens=1, completion_tokens=1,
                        total_tokens=2, cost_usd=amount,
                        created_at=when or datetime.now(timezone.utc)))
        db.commit()
    finally:
        db.close()


def _agents_must_not_run(monkeypatch):
    import agents.brand_agent as ba
    import agents.content_agent as ca
    import agents.feedback_agent as fa

    def refuse(*args, **kwargs):
        raise AssertionError("agent loop ran while the budget had paused agents")

    for module in (ba, ca, fa):
        monkeypatch.setattr(module, "AGENT_MODE", True)
        monkeypatch.setattr(module, "run_agent", refuse)


def _profile_with_post(client):
    profile = client.post("/brand", json=PROFILE, headers=AUTH).json()
    post = client.post("/generate", json={"user_id": profile["id"],
                                          "topic": "Designing for elderly patients"},
                       headers=AUTH).json()
    return profile["id"], post["post_id"]


# ── Under the soft limit ──

class TestUnderBudget:
    def test_agents_run_normally(self, client, monkeypatch):
        import agents.brand_agent as ba
        monkeypatch.setattr(ba, "AGENT_MODE", True)
        _spend(0.50)
        from utils.budget import agents_allowed
        assert agents_allowed() is True

    def test_spend_from_yesterday_does_not_count(self, client):
        _spend(5.00, when=datetime.now(timezone.utc) - timedelta(days=1))
        from utils.budget import budget_status
        status = budget_status()
        assert status["state"] == "ok"
        assert status["spent_today_usd"] == 0


# ── Soft limit: agents pause, the app keeps working ──

class TestSoftLimit:
    def test_agents_switch_to_direct(self, client, monkeypatch):
        # Feedback needs an approved post with engagement, set up under budget.
        uid, post_id = _profile_with_post(client)
        client.post(f"/posts/{post_id}/review", json={"decision": "approve"}, headers=AUTH)
        client.post("/engagement", json={"post_id": post_id, "likes": 10, "comments": 2,
                                         "shares": 1}, headers=AUTH)

        _agents_must_not_run(monkeypatch)
        _spend(0.85)

        profile = client.post("/brand", json=PROFILE, headers=AUTH)
        assert profile.status_code == 200
        assert profile.json()["execution_mode"] == "budget_direct"

        post = client.post("/generate", json={"user_id": uid,
                                              "topic": "Designing for elderly patients"},
                           headers=AUTH)
        assert post.status_code == 200
        assert post.json()["execution_mode"] == "budget_direct"

        fb = client.post("/feedback", json={"user_id": uid}, headers=AUTH)
        assert fb.status_code == 200
        assert fb.json()["execution_mode"] == "budget_direct"

    def test_agent_mode_off_still_reports_direct(self, client, monkeypatch):
        import agents.brand_agent as ba
        monkeypatch.setattr(ba, "AGENT_MODE", False)
        _spend(0.85)
        assert client.post("/brand", json=PROFILE, headers=AUTH).json()["execution_mode"] == "direct"


# ── Hard limit: LLM routes stop, the rest keeps working ──

class TestHardLimit:
    @pytest.mark.parametrize("path,body", [
        ("/brand", PROFILE),
        ("/generate", {"user_id": 1, "topic": "Designing for elderly patients"}),
        ("/feedback", {"user_id": 1}),
        ("/predict", {"user_id": 1, "draft_content": "A draft."}),
        ("/orchestrate", {**PROFILE, "topic": "Designing for elderly patients"}),
    ])
    def test_llm_routes_return_429(self, client, path, body):
        if path == "/predict":
            # /predict checks the budget after its cache, so it needs a real
            # profile to get that far.
            uid = client.post("/brand", json=PROFILE, headers=AUTH).json()["id"]
            body = {**body, "user_id": uid}
        _spend(1.00)
        res = client.post(path, json=body, headers=AUTH)
        assert res.status_code == 429
        assert int(res.headers["retry-after"]) > 0

    def test_message_reveals_no_money(self, client):
        _spend(1.00)
        detail = client.post("/brand", json=PROFILE, headers=AUTH).json()["detail"]
        assert "limit" in detail
        for leak in ("$", "USD", "1.0", "cost", "budget", "spent"):
            assert leak not in detail

    def test_non_llm_routes_keep_working(self, client):
        uid, post_id = _profile_with_post(client)
        _spend(1.00)
        assert client.get(f"/dashboard/{uid}", headers=AUTH).status_code == 200
        assert client.get(f"/history/{uid}", headers=AUTH).status_code == 200
        assert client.post(f"/posts/{post_id}/review", json={"decision": "approve"},
                           headers=AUTH).status_code == 200
        assert client.get("/health").status_code == 200

    def test_a_cached_score_is_still_served(self, client):
        """A cache hit costs nothing, so the budget does not block it."""
        uid = client.post("/brand", json=PROFILE, headers=AUTH).json()["id"]
        draft = {"user_id": uid, "draft_content": "A draft worth scoring twice."}
        first = client.post("/predict", json=draft, headers=AUTH)
        assert first.status_code == 200 and first.json()["cached"] is False

        _spend(1.00)
        again = client.post("/predict", json=draft, headers=AUTH)
        assert again.status_code == 200 and again.json()["cached"] is True
        assert again.json()["overall_score"] == first.json()["overall_score"]

        new = client.post("/predict", json={**draft, "draft_content": "A different draft."},
                          headers=AUTH)
        assert new.status_code == 429

    def test_auth_is_checked_before_budget(self, client):
        _spend(1.00)
        assert client.post("/brand", json=PROFILE).status_code == 401


# ── Operator view ──

class TestOperatorView:
    @pytest.mark.parametrize("spent,state", [(0.10, "ok"), (0.85, "agents_paused"),
                                             (1.20, "blocked")])
    def test_costs_reports_budget_state(self, client, spent, state):
        _spend(spent)
        budget = client.get("/costs", headers=ADMIN).json()["budget"]
        assert budget["enabled"] is True
        assert budget["state"] == state
        assert budget["daily_limit_usd"] == 1.00
        assert budget["spent_today_usd"] == pytest.approx(spent)

    def test_guard_off_when_no_budget_set(self, client, monkeypatch):
        import utils.budget as budget
        monkeypatch.setattr(budget, "DAILY_BUDGET_USD", None)
        _spend(500.00)
        assert client.post("/brand", json=PROFILE, headers=AUTH).status_code == 200
        assert client.get("/costs", headers=ADMIN).json()["budget"]["state"] == "off"


# ── Configuration ──

class TestConfiguration:
    @pytest.mark.parametrize("raw,expected", [("", None), ("0", None), ("-3", None),
                                              ("abc", None), ("0.50", 0.50)])
    def test_budget_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("DAILY_BUDGET_USD", raw)
        from utils import budget
        assert budget._read_budget() == expected

    @pytest.mark.parametrize("raw,expected", [("80", 80), ("0", 1), ("250", 100), ("x", 80)])
    def test_soft_percent_is_clamped(self, monkeypatch, raw, expected):
        monkeypatch.setenv("BUDGET_SOFT_LIMIT_PERCENT", raw)
        from utils import budget
        assert budget._read_soft_percent() == expected

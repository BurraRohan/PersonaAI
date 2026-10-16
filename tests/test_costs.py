"""
Token usage and cost tracking.

Covers the cost arithmetic, that every LLM call records its tokens, that one
user action's calls share a request id, and that cost data is reachable only
with the operator's admin key. All offline: the Groq client is faked.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["ADMIN_API_KEY"] = "admin-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTH = {"Authorization": "Bearer test-key-for-pytest"}
ADMIN = {"Authorization": "Bearer admin-key-for-pytest"}

BRAND_REPLY = json.dumps({
    "tone": "t", "content_themes": ["a"], "positioning_summary": "p",
    "do_guidelines": [], "dont_guidelines": [],
})


def _fresh_modules():
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils", "schemas"}:
            del sys.modules[module]


class FakeGroq:
    """Stands in for the OpenAI client pointed at Groq, with a usage block."""

    def __init__(self, reply=BRAND_REPLY, prompt=1000, completion=200, cached=0):
        usage = SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=usage,
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: response))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/costs.db")
    monkeypatch.setenv("AGENT_MODE", "0")
    _fresh_modules()

    import services.llm_service as llm
    monkeypatch.setattr(llm, "client", FakeGroq())

    import agents.brand_agent as ba
    monkeypatch.setattr(ba, "AGENT_MODE", False, raising=False)

    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _create_profile(client):
    res = client.post("/brand", json={"name": "A", "role": "r", "industry": "i",
                                      "goals": "g", "preferred_tone": "t"}, headers=AUTH)
    assert res.status_code == 200, res.text
    return res.json()


# ── Arithmetic ──

class TestCostMath:
    def test_input_and_output_prices(self):
        from utils.costs import cost_usd
        # 1M input at $0.15 + 1M output at $0.60
        assert cost_usd(1_000_000, 1_000_000) == pytest.approx(0.75)

    def test_cached_input_is_half_price(self):
        from utils.costs import cost_usd
        assert cost_usd(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.075)

    def test_cached_cannot_exceed_prompt(self):
        from utils.costs import cost_usd
        assert cost_usd(100, 0, cached_tokens=500) == cost_usd(100, 0, cached_tokens=100)

    def test_usage_adds_up(self):
        from utils.costs import Usage
        total = Usage(100, 20, 10) + Usage(50, 5, 0)
        assert (total.prompt_tokens, total.completion_tokens, total.cached_tokens) == (150, 25, 10)
        assert total.total_tokens == 175

    def test_reads_openai_usage_block(self):
        from utils.costs import usage_from_openai
        u = usage_from_openai({"prompt_tokens": 10, "completion_tokens": 4,
                               "prompt_tokens_details": {"cached_tokens": 3}})
        assert (u.prompt_tokens, u.completion_tokens, u.cached_tokens) == (10, 4, 3)
        assert usage_from_openai(None) is None

    def test_sums_agent_messages(self):
        from utils.costs import usage_from_messages
        msgs = [
            SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 10,
                                            "input_token_details": {"cache_read": 40}}),
            SimpleNamespace(content="tool result"),  # tool messages carry no usage
            SimpleNamespace(usage_metadata={"input_tokens": 300, "output_tokens": 30}),
        ]
        u = usage_from_messages(msgs)
        assert (u.prompt_tokens, u.completion_tokens, u.cached_tokens) == (400, 40, 40)
        assert usage_from_messages([]) is None

    def test_bad_price_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("LLM_INPUT_PRICE_PER_MILLION", "not-a-number")
        _fresh_modules()
        import utils.costs as costs
        assert costs.INPUT_PRICE_PER_MILLION == 0.15


class TestPathNormalising:
    def test_ids_are_grouped(self):
        from utils.request_context import normalise_path
        assert normalise_path("POST", "/posts/12/review") == "POST /posts/{id}/review"
        assert normalise_path("GET", "/dashboard/7") == "GET /dashboard/{id}"
        assert normalise_path("POST", "/brand") == "POST /brand"


# ── Capture ──

class TestUsageIsRecorded:
    def test_llm_call_writes_tokens_cost_and_request(self, client):
        _create_profile(client)
        logs = client.get("/audit-logs", headers=ADMIN).json()
        row = logs[0]
        assert row["agent_name"] == "brand"
        assert row["prompt_tokens"] == 1000
        assert row["completion_tokens"] == 200
        assert row["total_tokens"] == 1200
        assert row["cost_usd"] == pytest.approx((1000 * 0.15 + 200 * 0.60) / 1_000_000)
        assert row["endpoint"] == "POST /brand"
        assert row["request_id"]

    def test_separate_requests_get_separate_ids(self, client):
        _create_profile(client)
        _create_profile(client)
        ids = {row["request_id"] for row in client.get("/audit-logs", headers=ADMIN).json()}
        assert len(ids) == 2

    def test_cached_tokens_are_recorded(self, client, monkeypatch):
        import services.llm_service as llm
        monkeypatch.setattr(llm, "client", FakeGroq(prompt=1000, completion=0, cached=600))
        _create_profile(client)
        row = client.get("/audit-logs", headers=ADMIN).json()[0]
        assert row["cached_tokens"] == 600
        assert row["cost_usd"] == pytest.approx((400 * 0.15 + 600 * 0.075) / 1_000_000)

    def test_agent_loop_usage_is_logged(self, client, monkeypatch):
        """run_agent sums every model step of the loop into one audit row."""
        import utils.agent_runtime as rt

        class FakeAgent:
            def invoke(self, state, config=None):
                return {"messages": [
                    SimpleNamespace(usage_metadata={"input_tokens": 500, "output_tokens": 50}),
                    SimpleNamespace(usage_metadata={"input_tokens": 700, "output_tokens": 70}),
                ]}

        rt.run_agent(FakeAgent(), "do the thing", "content")
        row = client.get("/audit-logs?agent_name=content_agent", headers=ADMIN).json()[0]
        assert row["prompt_tokens"] == 1200
        assert row["completion_tokens"] == 120
        assert row["status"] == "success"


# ── Operator-only access ──

class TestCostsAreOperatorOnly:
    @pytest.mark.parametrize("path", ["/audit-logs", "/costs"])
    def test_no_key_is_401(self, client, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize("path", ["/audit-logs", "/costs"])
    def test_user_api_key_is_refused(self, client, path):
        assert client.get(path, headers=AUTH).status_code == 403

    @pytest.mark.parametrize("path", ["/audit-logs", "/costs"])
    def test_admin_key_is_accepted(self, client, path):
        assert client.get(path, headers=ADMIN).status_code == 200

    def test_routes_are_disabled_without_an_admin_key(self, client, monkeypatch):
        import utils.auth as auth
        monkeypatch.setattr(auth, "ADMIN_API_KEY", None)
        assert client.get("/costs", headers=ADMIN).status_code == 503

    def test_reusing_the_app_key_disables_operator_routes(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "test-key-for-pytest")
        _fresh_modules()
        import utils.auth as auth
        assert auth.ADMIN_API_KEY is None

    def test_user_responses_never_mention_cost(self, client):
        profile = _create_profile(client)
        dashboard = client.get(f"/dashboard/{profile['id']}", headers=AUTH)
        for body in (json.dumps(profile), dashboard.text):
            assert "cost_usd" not in body
            assert "tokens" not in body

    def test_ui_never_calls_operator_routes(self):
        js = open(os.path.join(os.path.dirname(__file__), "..", "static", "app.js"),
                  encoding="utf-8").read()
        assert "/costs" not in js
        assert "/audit-logs" not in js


# ── Summary ──

class TestCostSummary:
    def test_totals_and_breakdowns(self, client):
        _create_profile(client)
        _create_profile(client)
        body = client.get("/costs", headers=ADMIN).json()

        assert body["pricing"]["model"] == "openai/gpt-oss-120b"
        assert body["totals"]["llm_calls"] == 2
        assert body["totals"]["requests"] == 2
        assert body["totals"]["total_tokens"] == 2400
        assert body["totals"]["cost_usd"] == pytest.approx(2 * 0.00027, abs=1e-6)

        assert body["by_agent"][0]["agent_name"] == "brand"
        assert body["by_endpoint"][0]["endpoint"] == "POST /brand"
        assert body["by_endpoint"][0]["avg_cost_per_request_usd"] == pytest.approx(0.00027, abs=1e-6)
        assert len(body["by_day"]) == 1

    @pytest.mark.parametrize("days", [0, 366, -1])
    def test_period_is_bounded(self, client, days):
        assert client.get(f"/costs?days={days}", headers=ADMIN).status_code == 422


# ── Upgrading an existing database ──

class TestSchemaUpgrade:
    def test_old_audit_table_gains_cost_columns(self, tmp_path, monkeypatch):
        import sqlite3
        path = tmp_path / "old.db"
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE audit_logs (
            id INTEGER PRIMARY KEY, trace_id VARCHAR, agent_name VARCHAR,
            prompt_version VARCHAR, model VARCHAR, input_summary TEXT,
            output_summary TEXT, latency_ms FLOAT, status VARCHAR,
            error_message TEXT, created_at DATETIME)""")
        con.execute("INSERT INTO audit_logs (agent_name, status) VALUES ('brand', 'success')")
        con.commit(); con.close()

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
        _fresh_modules()
        from database.db import ensure_schema
        ensure_schema()

        con = sqlite3.connect(path)
        cols = {row[1] for row in con.execute("PRAGMA table_info(audit_logs)")}
        old_row = con.execute("SELECT agent_name, total_tokens FROM audit_logs").fetchone()
        con.close()
        assert {"prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens",
                "cost_usd", "request_id", "endpoint"} <= cols
        assert old_row == ("brand", None)  # history kept, marked as "no usage data"

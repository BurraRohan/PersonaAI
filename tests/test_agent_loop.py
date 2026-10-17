"""
The real agent loop, end to end.

Other tests stub run_agent. These build the actual brand agent with LangChain's
create_agent and drive it with a scripted chat model, so a library upgrade that
changes how the loop calls tools, ends, or gives up is caught here. Offline:
the scripted model stands in for Groq.
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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

AUTH = {"Authorization": "Bearer test-key-for-pytest"}
ADMIN = {"Authorization": "Bearer admin-key-for-pytest"}

PROFILE = {"name": "Karthik Menon", "role": "Data Engineer", "industry": "Fintech",
           "goals": "Credibility in streaming data", "preferred_tone": "Practical"}
STRATEGY = json.dumps({"tone": "practical", "content_themes": ["streaming"],
                       "positioning_summary": "p", "do_guidelines": ["a"],
                       "dont_guidelines": ["b"]})


class ScriptedModel(BaseChatModel):
    """Replays a fixed list of replies; the last one repeats forever."""

    script: list = []
    step: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.script[min(self.step, len(self.script) - 1)]
        object.__setattr__(self, "step", self.step + 1)
        # A fresh message each time, as a real provider would send.
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy(deep=True))])


def _tool_call(n, name, args):
    return AIMessage(
        content="", tool_calls=[{"id": f"call_{n}", "name": name, "args": args}],
        usage_metadata={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/loop.db")
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils", "schemas"}:
            del sys.modules[module]

    import services.llm_service as llm
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=STRATEGY))],
        usage=SimpleNamespace(prompt_tokens=900, completion_tokens=300,
                              prompt_tokens_details=None),
    )
    monkeypatch.setattr(llm, "client", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: reply))))

    import agents.brand_agent as ba
    monkeypatch.setattr(ba, "AGENT_MODE", True)

    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, ba, monkeypatch


def test_agent_chooses_tools_and_saves_the_profile(client):
    c, ba, monkeypatch = client
    script = [
        _tool_call(1, "check_existing_profile", {}),
        _tool_call(2, "generate_brand_strategy", {}),
        _tool_call(3, "save_brand_profile", {}),
        AIMessage(content="Profile saved.",
                  usage_metadata={"input_tokens": 1500, "output_tokens": 20, "total_tokens": 1520}),
    ]
    monkeypatch.setattr(ba, "build_agent_llm", lambda: ScriptedModel(script=script))

    res = c.post("/brand", json=PROFILE, headers=AUTH)
    assert res.status_code == 200
    body = res.json()
    assert body["execution_mode"] == "agent"

    calls = [t["tool"] for t in body["agent_trace"] if t["action"] == "tool_call"]
    assert calls == ["check_existing_profile", "generate_brand_strategy", "save_brand_profile"]

    loop = c.get("/audit-logs?agent_name=brand_agent", headers=ADMIN).json()[0]
    assert loop["status"] == "success"
    assert loop["prompt_tokens"] == 3 * 1000 + 1500


def test_runaway_loop_falls_back_instead_of_failing(client):
    """A model that never stops calling tools hits the step limit; the user
    still gets a profile, through the direct path."""
    c, ba, monkeypatch = client
    script = [_tool_call(9, "check_existing_profile", {})]
    monkeypatch.setattr(ba, "build_agent_llm", lambda: ScriptedModel(script=script))

    res = c.post("/brand", json=PROFILE, headers=AUTH)
    assert res.status_code == 200
    assert res.json()["execution_mode"] == "fallback_direct"
    assert res.json()["id"]

    loop = c.get("/audit-logs?agent_name=brand_agent", headers=ADMIN).json()[0]
    assert loop["status"] == "error"
    assert loop["error_message"] == "GraphRecursionError"


def test_model_cannot_misspell_the_saved_details(client):
    """The model once wrote "Chekadur1" for "Chekadurai" when retyping a name
    into tool arguments. The tools now ignore model-supplied details."""
    c, ba, monkeypatch = client
    typo = {"name": "Karthik Men0n", "role": "Chef", "industry": "x", "goals": "x",
            "preferred_tone": "x", "strategy_json": "{}"}
    script = [
        _tool_call(1, "generate_brand_strategy", typo),
        _tool_call(2, "save_brand_profile", typo),
        AIMessage(content="Done."),
    ]
    monkeypatch.setattr(ba, "build_agent_llm", lambda: ScriptedModel(script=script))

    body = c.post("/brand", json=PROFILE, headers=AUTH).json()
    assert body["execution_mode"] == "agent"
    assert body["name"] == "Karthik Menon"
    assert body["role"] == "Data Engineer"
    assert body["tone"] == "practical"


def test_save_needs_a_strategy_and_never_duplicates(client):
    c, ba, monkeypatch = client
    from database.db import SessionLocal
    from database.models import BrandProfile
    from schemas.schemas import BrandRequest

    tools = {t.name: t for t in ba.build_brand_tools(BrandRequest(**PROFILE))}
    assert "error" in json.loads(tools["save_brand_profile"].invoke({}))

    tools["generate_brand_strategy"].invoke({})
    first = json.loads(tools["save_brand_profile"].invoke({}))["id"]
    second = json.loads(tools["save_brand_profile"].invoke({}))["id"]
    assert first == second

    db = SessionLocal()
    try:
        assert db.query(BrandProfile).filter(BrandProfile.name == PROFILE["name"]).count() == 1
    finally:
        db.close()

"""
Test suite for PersonaAI.

Run with:  pytest -q

Everything here runs offline. No Groq calls are made: the LLM boundary is
monkeypatched, which is the point — these tests cover our logic, not the
model's output.
"""

import json
import os
import sys

import pytest

# Auth requires API_KEY at import time, so set it before importing the app.
# Set, not setdefault: the suite must use its own values even when the ambient
# environment (CI, a shell export, a .env already loaded) has different ones.
# setdefault silently loses that race and every authenticated request 403s.
os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── JSON parsing ────────────────────────────────────────────────

class TestParseJsonResponse:
    """The parser has three fallback strategies; each one needs a test."""

    def test_plain_json(self):
        from services.llm_service import _parse_json_response
        assert _parse_json_response('{"tone": "friendly"}') == {"tone": "friendly"}

    def test_json_in_markdown_fence(self):
        from services.llm_service import _parse_json_response
        raw = '```json\n{"tone": "bold"}\n```'
        assert _parse_json_response(raw) == {"tone": "bold"}

    def test_json_with_surrounding_prose(self):
        from services.llm_service import _parse_json_response
        raw = 'Sure! Here you go:\n{"tone": "wry"}\nHope that helps.'
        assert _parse_json_response(raw) == {"tone": "wry"}

    def test_whitespace_is_tolerated(self):
        from services.llm_service import _parse_json_response
        assert _parse_json_response('  \n {"a": 1} \n ') == {"a": 1}

    def test_unparseable_raises_502(self):
        from fastapi import HTTPException
        from services.llm_service import _parse_json_response
        with pytest.raises(HTTPException) as exc:
            _parse_json_response("the model refused to answer")
        assert exc.value.status_code == 502


# ── Auth ────────────────────────────────────────────────────────

class TestAuth:
    def test_correct_key_accepted(self):
        from utils.auth import check_key
        assert check_key("test-key-for-pytest") is True

    def test_wrong_key_rejected(self):
        from utils.auth import check_key
        assert check_key("wrong") is False

    def test_empty_key_rejected(self):
        from utils.auth import check_key
        assert check_key("") is False

    def test_no_insecure_default_remains(self):
        """The old code fell back to a key that was public in the source."""
        from utils.auth import check_key
        assert check_key("dev-key-change-me") is False


# ── Graph routing ───────────────────────────────────────────────

class TestOrchestratorRouting:
    """The conditional edges are the part that makes this a graph, not a list."""

    def test_brand_error_routes_to_end(self):
        from agents.orchestrator import route_after_brand
        assert route_after_brand({"error": "boom"}) == "error"

    def test_brand_success_continues(self):
        from agents.orchestrator import route_after_brand
        assert route_after_brand({"error": None}) == "continue"

    def test_no_engagement_data_skips_feedback(self):
        from agents.orchestrator import route_after_history
        assert route_after_history({"engagement_history": []}) == "no_data"

    def test_engagement_data_runs_feedback(self):
        from agents.orchestrator import route_after_history
        state = {"engagement_history": [{"likes": 10, "comments": 2, "shares": 1}]}
        assert route_after_history(state) == "has_data"

    def test_graph_compiles_with_all_nodes(self):
        from agents.orchestrator import build_orchestration_graph
        nodes = build_orchestration_graph().get_graph().nodes
        for expected in ("brand", "content", "check_history", "feedback", "skip_feedback"):
            assert expected in nodes


# ── Engagement statistics ───────────────────────────────────────

class TestEngagementStats:
    """Averages are computed in Python so the agent cannot get them wrong."""

    def test_stats_are_exact(self, monkeypatch):
        import agents.feedback_agent as fa
        history = [
            {"post_id": 1, "topic": "RAG", "likes": 100, "comments": 20, "shares": 5},
            {"post_id": 2, "topic": "ML basics", "likes": 10, "comments": 1, "shares": 0},
        ]
        monkeypatch.setattr(fa, "_load_history", lambda user_id: history)

        stats = json.loads(fa.compute_engagement_stats.invoke({"user_id": 1}))

        assert stats["entries"] == 2
        assert stats["avg_likes"] == 55.0
        assert stats["best_post"]["topic"] == "RAG"
        assert stats["worst_post"]["topic"] == "ML basics"

    def test_empty_history_returns_error_not_zero(self, monkeypatch):
        import agents.feedback_agent as fa
        monkeypatch.setattr(fa, "_load_history", lambda user_id: [])
        stats = json.loads(fa.compute_engagement_stats.invoke({"user_id": 99}))
        assert "error" in stats


# ── Agent trace extraction ──────────────────────────────────────

class FakeAIMessage:
    type = "ai"

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []
        self.content = ""


class FakeToolMessage:
    type = "tool"

    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.tool_calls = []


class TestAgentTrace:
    def test_trace_captures_calls_and_results(self):
        from utils.agent_runtime import extract_trace
        messages = [
            FakeAIMessage([{"name": "check_existing_profile", "args": {"name": "Ada"}}]),
            FakeToolMessage("check_existing_profile", '{"exists": false}'),
        ]
        trace = extract_trace(messages)
        assert [t["action"] for t in trace] == ["tool_call", "tool_result"]
        assert trace[0]["tool"] == "check_existing_profile"

    def test_find_tool_result_pulls_saved_id(self):
        from utils.agent_runtime import find_tool_result
        messages = [FakeToolMessage("save_brand_profile", '{"id": 7, "status": "saved"}')]
        assert find_tool_result(messages, "save_brand_profile", "id") == 7

    def test_find_tool_result_missing_returns_none(self):
        from utils.agent_runtime import find_tool_result
        messages = [FakeToolMessage("some_other_tool", '{"id": 7}')]
        assert find_tool_result(messages, "save_brand_profile", "id") is None

    def test_malformed_tool_output_does_not_crash(self):
        from utils.agent_runtime import find_tool_result
        messages = [FakeToolMessage("save_brand_profile", "not json at all")]
        assert find_tool_result(messages, "save_brand_profile", "id") is None


# ── Prompt versioning and rollback ──────────────────────────────

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A throwaway SQLite database for DB-backed tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.db import Base
    import database.models  # noqa: F401  (registers the tables)

    engine = create_engine(f"sqlite:///{tmp_path}/test.db",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestPromptVersioning:
    def test_active_prompt_is_returned(self, temp_db):
        from database.models import PromptTemplate
        from services.llm_service import get_active_prompt

        temp_db.add(PromptTemplate(agent_name="brand", version=1,
                                   template="v1 text", is_active=True))
        temp_db.commit()

        template, version = get_active_prompt(temp_db, "brand")
        assert template == "v1 text"
        assert version == 1

    def test_highest_active_version_wins(self, temp_db):
        from database.models import PromptTemplate
        from services.llm_service import get_active_prompt

        temp_db.add(PromptTemplate(agent_name="brand", version=1,
                                   template="old", is_active=True))
        temp_db.add(PromptTemplate(agent_name="brand", version=2,
                                   template="new", is_active=True))
        temp_db.commit()

        template, version = get_active_prompt(temp_db, "brand")
        assert version == 2
        assert template == "new"

    def test_missing_agent_returns_none(self, temp_db):
        from services.llm_service import get_active_prompt
        template, version = get_active_prompt(temp_db, "does_not_exist")
        assert template is None
        assert version == 0

    def test_predictor_prompt_is_seeded(self, temp_db):
        """Regression: the predictor template used to be looked up but never created."""
        from main import seed_default_prompts
        from services.llm_service import get_active_prompt

        seed_default_prompts(temp_db)
        template, version = get_active_prompt(temp_db, "predictor")
        assert template is not None
        assert version >= 1

    def test_all_four_agents_have_prompts(self, temp_db):
        from main import seed_default_prompts
        from services.llm_service import get_active_prompt

        seed_default_prompts(temp_db)
        for agent in ("brand", "content", "feedback", "predictor"):
            template, _ = get_active_prompt(temp_db, agent)
            assert template is not None, f"{agent} has no active prompt"

class TestPredictorPromptV2:
    """v2 exists to fix two concrete faults in v1."""

    def test_v2_is_active_and_v1_is_kept(self, temp_db):
        from main import seed_default_prompts
        from database.models import PromptTemplate

        seed_default_prompts(temp_db)
        versions = (
            temp_db.query(PromptTemplate)
            .filter(PromptTemplate.agent_name == "predictor")
            .order_by(PromptTemplate.version)
            .all()
        )
        assert [v.version for v in versions] == [1, 2]
        assert versions[0].is_active is False
        assert versions[1].is_active is True

    def test_v2_has_no_literal_example_ranges(self, temp_db):
        """v1 leaked its examples into model output; v2 must not contain them."""
        from main import seed_default_prompts
        from services.llm_service import get_active_prompt

        seed_default_prompts(temp_db)
        template, version = get_active_prompt(temp_db, "predictor")
        assert version == 2
        for leaked in ('"30-50"', '"5-12"', '"2-6"'):
            assert leaked not in template

    def test_v2_defines_score_bands(self, temp_db):
        from main import seed_default_prompts
        from services.llm_service import get_active_prompt

        seed_default_prompts(temp_db)
        template, _ = get_active_prompt(temp_db, "predictor")
        assert "SCORING RUBRIC" in template
        for band in ("90-100", "70-89", "50-69", "30-49", "1-29"):
            assert band in template


class TestPromptSeedingIsVersionAware:
    """A new prompt version in code must reach a database that already has old ones."""

    def test_new_version_is_added_to_existing_db(self, temp_db):
        from database.models import PromptTemplate
        from main import seed_default_prompts
        from services.llm_service import get_active_prompt

        # A database that predates v2.
        temp_db.add(PromptTemplate(agent_name="predictor", version=1,
                                   template="old v1", is_active=True))
        temp_db.commit()

        seed_default_prompts(temp_db)

        _, version = get_active_prompt(temp_db, "predictor")
        assert version == 2

    def test_existing_template_is_not_overwritten(self, temp_db):
        """A version already in the database keeps its own text."""
        from database.models import PromptTemplate
        from main import seed_default_prompts

        temp_db.add(PromptTemplate(agent_name="brand", version=1,
                                   template="MY EDITED PROMPT", is_active=True))
        temp_db.commit()

        seed_default_prompts(temp_db)

        row = (temp_db.query(PromptTemplate)
               .filter(PromptTemplate.agent_name == "brand",
                       PromptTemplate.version == 1)
               .first())
        assert row.template == "MY EDITED PROMPT"

    def test_only_one_version_active_per_agent(self, temp_db):
        from database.models import PromptTemplate
        from main import seed_default_prompts

        temp_db.add(PromptTemplate(agent_name="predictor", version=1,
                                   template="old v1", is_active=True))
        temp_db.commit()
        seed_default_prompts(temp_db)

        active = (temp_db.query(PromptTemplate)
                  .filter(PromptTemplate.agent_name == "predictor",
                          PromptTemplate.is_active == True)  # noqa: E712
                  .all())
        assert len(active) == 1
        assert active[0].version == 2

    def test_seeding_twice_adds_nothing(self, temp_db):
        from database.models import PromptTemplate
        from main import seed_default_prompts

        seed_default_prompts(temp_db)
        first = temp_db.query(PromptTemplate).count()
        seed_default_prompts(temp_db)
        assert temp_db.query(PromptTemplate).count() == first


class TestCompositeScore:
    """overall_score is derived in Python, not taken from the model."""

    def test_weak_dimensions_produce_a_low_score(self):
        """Real case: the model rated hook 30 / CTA 35 and still returned 73."""
        from services.llm_service import _composite_score
        weak = {"hook_strength": 30, "call_to_action": 35,
                "brand_alignment": 85, "readability": 70, "overall_score": 73}
        assert _composite_score(weak) == 50

    def test_strong_hook_lifts_the_score(self):
        from services.llm_service import _composite_score
        strong = {"hook_strength": 92, "call_to_action": 30,
                  "brand_alignment": 85, "readability": 80, "overall_score": 78}
        assert _composite_score(strong) == 73

    def test_scores_are_spread_not_clustered(self):
        from services.llm_service import _composite_score
        scores = [
            _composite_score({"hook_strength": 92, "call_to_action": 30,
                              "brand_alignment": 85, "readability": 80}),
            _composite_score({"hook_strength": 60, "call_to_action": 45,
                              "brand_alignment": 88, "readability": 80}),
            _composite_score({"hook_strength": 30, "call_to_action": 35,
                              "brand_alignment": 85, "readability": 70}),
        ]
        assert max(scores) - min(scores) >= 20

    def test_weights_sum_to_one(self):
        from services.llm_service import SCORE_WEIGHTS
        assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9

    def test_out_of_range_values_are_clamped(self):
        from services.llm_service import _composite_score
        assert 1 <= _composite_score({"hook_strength": 500, "call_to_action": -20,
                                      "brand_alignment": 50, "readability": 50}) <= 100

    def test_missing_dimensions_fall_back(self):
        from services.llm_service import _composite_score
        assert _composite_score({"overall_score": 61}) >= 1
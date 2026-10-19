"""
Prediction cache: a repeat of an unchanged /predict request costs nothing, and
any change that could alter the score produces a fresh one. Groq is faked and
counts how many predictor calls actually go out.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["ADMIN_API_KEY"] = "admin-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTH = {"Authorization": "Bearer test-key-for-pytest"}
ADMIN = {"Authorization": "Bearer admin-key-for-pytest"}
DRAFT = "Batch jobs were never the problem.\n\nWaiting for them was."


class CountingGroq:
    """Scores rise with every call, so a reused answer is easy to spot."""

    def __init__(self):
        self.predictions = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        prompt = kw["messages"][-1]["content"]
        if "hook_strength" in prompt:
            self.predictions += 1
            level = 50 + self.predictions
            text = json.dumps({"brand_alignment": level, "hook_strength": level,
                               "readability": level, "call_to_action": level,
                               "predicted_likes": "1-2", "predicted_comments": "0-1",
                               "predicted_shares": "0-1", "improvement_tips": "tip"})
        else:
            text = json.dumps({"tone": "t", "content_themes": ["a"], "positioning_summary": "p",
                               "do_guidelines": [], "dont_guidelines": []})
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=usage)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/cache.db")
    monkeypatch.setenv("AGENT_MODE", "0")
    monkeypatch.delenv("PREDICTION_CACHE_HOURS", raising=False)
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils",
                                    "schemas", "evals"}:
            del sys.modules[module]
    import services.llm_service as llm
    fake = CountingGroq()
    monkeypatch.setattr(llm, "client", fake)
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        uid = c.post("/brand", json={"name": "A", "role": "r", "industry": "i", "goals": "g",
                                     "preferred_tone": "t"}, headers=AUTH).json()["id"]
        yield SimpleNamespace(c=c, uid=uid, fake=fake, llm=llm, monkeypatch=monkeypatch)


def predict(ctx, draft=DRAFT, **extra):
    res = ctx.c.post("/predict", json={"user_id": ctx.uid, "draft_content": draft, **extra},
                     headers=AUTH)
    assert res.status_code == 200, res.text
    return res.json()


# ── Reuse ──

def test_a_repeat_is_served_from_the_cache(ctx):
    first, second = predict(ctx), predict(ctx)
    assert ctx.fake.predictions == 1
    assert (first["cached"], second["cached"]) == (False, True)
    assert first["overall_score"] == second["overall_score"]
    assert first["improvement_tips"] == second["improvement_tips"]


def test_trailing_spaces_do_not_matter(ctx):
    predict(ctx)
    assert predict(ctx, "  " + DRAFT.replace("problem.", "problem.   ") + "\n\n")["cached"]
    assert ctx.fake.predictions == 1


def test_a_changed_paragraph_break_is_a_new_draft(ctx):
    predict(ctx)
    assert not predict(ctx, DRAFT.replace("\n\n", " "))["cached"]
    assert ctx.fake.predictions == 2


def test_a_different_draft_is_scored_fresh(ctx):
    predict(ctx)
    predict(ctx, DRAFT + " Really.")
    assert ctx.fake.predictions == 2


# ── Anything that shapes the score invalidates it ──

def test_editing_the_profile_gives_a_fresh_score(ctx):
    predict(ctx)
    ctx.c.patch(f"/brand/{ctx.uid}", json={"tone": "much blunter"}, headers=AUTH)
    assert not predict(ctx)["cached"]


def test_new_engagement_gives_a_fresh_score(ctx):
    predict(ctx)
    from database.db import SessionLocal
    from database.models import Engagement, Post
    db = SessionLocal()
    try:
        post = Post(user_id=ctx.uid, topic="t", content="c", status="approved")
        db.add(post); db.commit(); db.refresh(post)
        db.add(Engagement(post_id=post.id, likes=40, comments=4, shares=1)); db.commit()
    finally:
        db.close()
    assert not predict(ctx)["cached"]


def test_switching_predictor_prompt_gives_a_fresh_score(ctx):
    predict(ctx)
    ctx.c.post("/prompts/predictor/rollback/4", headers=AUTH)
    assert not predict(ctx)["cached"]


def test_switching_predictor_model_gives_a_fresh_score(ctx):
    predict(ctx)
    ctx.monkeypatch.setattr(ctx.llm, "PREDICTOR_MODEL", "openai/gpt-oss-20b")
    assert not predict(ctx)["cached"]


def test_entries_expire(ctx):
    predict(ctx)
    from database.db import SessionLocal
    from database.models import PredictionCache
    db = SessionLocal()
    try:
        db.query(PredictionCache).update(
            {"created_at": datetime.now(timezone.utc) - timedelta(hours=25)})
        db.commit()
    finally:
        db.close()
    assert not predict(ctx)["cached"]
    assert ctx.fake.predictions == 2


# ── Safety ──

def test_cache_can_be_turned_off(ctx):
    import services.prediction_cache as pc
    ctx.monkeypatch.setattr(pc, "CACHE_HOURS", 0)
    predict(ctx), predict(ctx)
    assert ctx.fake.predictions == 2


def test_a_corrupt_entry_falls_back_to_a_fresh_score(ctx):
    predict(ctx)
    from database.db import SessionLocal
    from database.models import PredictionCache
    db = SessionLocal()
    try:
        db.query(PredictionCache).update({"result_json": "{not json"})
        db.commit()
    finally:
        db.close()
    result = predict(ctx)
    assert not result["cached"] and result["overall_score"]
    assert predict(ctx)["cached"]   # the bad entry was replaced


def test_saved_posts_keep_their_review_details_when_cached(ctx):
    from database.db import SessionLocal
    from database.models import Post
    db = SessionLocal()
    try:
        post = Post(user_id=ctx.uid, topic="t", content=DRAFT, status="pending")
        db.add(post); db.commit(); db.refresh(post)
        post_id = post.id
    finally:
        db.close()
    first = ctx.c.post("/predict", json={"user_id": ctx.uid, "post_id": post_id},
                       headers=AUTH).json()
    second = ctx.c.post("/predict", json={"user_id": ctx.uid, "post_id": post_id},
                        headers=AUTH).json()
    assert second["cached"] and second["post_id"] == post_id and second["status"] == "pending"
    assert first["overall_score"] == second["overall_score"]


def test_evaluations_bypass_the_cache(ctx):
    """The eval scripts call the predictor directly, so they always measure the model."""
    predict(ctx)
    ctx.llm.predict_engagement(DRAFT, {"tone": "t"}, [])
    ctx.llm.predict_engagement(DRAFT, {"tone": "t"}, [])
    assert ctx.fake.predictions == 3


# ── Operator view ──

def test_costs_reports_hits_and_savings(ctx):
    predict(ctx), predict(ctx), predict(ctx)
    cache = ctx.c.get("/costs", headers=ADMIN).json()["prediction_cache"]
    assert cache["enabled"] and cache["entries"] == 1 and cache["hits"] == 2
    one_call = (1000 * 0.15 + 500 * 0.60) / 1_000_000
    assert cache["saved_usd"] == pytest.approx(2 * one_call)

    logs = ctx.c.get("/audit-logs?agent_name=predictor", headers=ADMIN).json()
    assert len(logs) == 1    # cache hits are not LLM calls, so they are not logged as one


@pytest.mark.parametrize("raw,expected", [(None, 24.0), ("6", 6.0), ("0", 0.0),
                                          ("-5", 0.0), ("soon", 24.0)])
def test_cache_hours_setting(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PREDICTION_CACHE_HOURS", raising=False)
    else:
        monkeypatch.setenv("PREDICTION_CACHE_HOURS", raw)
    from services.prediction_cache import _read_hours
    assert _read_hours() == expected

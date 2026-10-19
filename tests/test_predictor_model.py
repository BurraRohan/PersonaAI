"""
Running the predictor on its own model, and the comparison that decides it.

The Groq client is faked and answers according to the model it is asked for,
so these tests show which model each call really used. Offline.
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
BIG, SMALL = "openai/gpt-oss-120b", "openai/gpt-oss-20b"

PROFILE_REPLY = {"tone": "t", "content_themes": ["a"], "positioning_summary": "p",
                 "do_guidelines": [], "dont_guidelines": []}


def prediction(level):
    return {"brand_alignment": level, "hook_strength": level, "readability": level,
            "call_to_action": level, "predicted_likes": "10-20",
            "predicted_comments": "1-3", "predicted_shares": "0-1", "improvement_tips": "x"}


class FakeGroq:
    """Replies by model; the predictor prompt gets a score, anything else a profile."""

    def __init__(self, small_offset=0):
        self.calls = []
        self.small_offset = small_offset
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw["model"])
        prompt = kw["messages"][-1]["content"]
        if "overall_score" in prompt or "hook_strength" in prompt:
            level = 60 + (self.small_offset if kw["model"] == SMALL else 0)
            # Longer drafts score higher, so ranking can be checked.
            level += min(30, prompt.count("good") * 10)
            text = json.dumps(prediction(level))
        else:
            text = json.dumps(PROFILE_REPLY)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=usage)


def _fresh():
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils",
                                    "schemas", "evals"}:
            del sys.modules[module]


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/pred.db")
    monkeypatch.setenv("AGENT_MODE", "0")
    _fresh()
    import services.llm_service as llm
    fake = FakeGroq()
    monkeypatch.setattr(llm, "client", fake)
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, llm, fake


def _profile(c):
    return c.post("/brand", json={"name": "A", "role": "r", "industry": "i",
                                  "goals": "g", "preferred_tone": "t"}, headers=AUTH).json()["id"]


# ── Pricing ──

class TestPricing:
    def test_small_model_costs_half(self):
        _fresh()
        from utils.costs import cost_usd
        assert cost_usd(1_000_000, 1_000_000, model=SMALL) == pytest.approx(0.375)
        assert cost_usd(1_000_000, 1_000_000, model=BIG) == pytest.approx(0.75)

    def test_main_model_uses_configured_prices(self, monkeypatch):
        monkeypatch.setenv("LLM_INPUT_PRICE_PER_MILLION", "1.00")
        _fresh()
        from utils.costs import prices_for
        assert prices_for(BIG)[0] == 1.00      # GROQ_MODEL follows .env
        assert prices_for(SMALL)[0] == 0.075   # others follow the table
        assert prices_for(None)[0] == 1.00

    def test_unknown_model_is_priced_like_the_main_model(self):
        _fresh()
        from utils.costs import prices_for
        assert prices_for("someone/unlisted") == prices_for(BIG)

    def test_usage_keeps_its_model(self):
        _fresh()
        from utils.costs import Usage, usage_from_openai
        u = usage_from_openai({"prompt_tokens": 10, "completion_tokens": 5}, SMALL)
        assert u.model == SMALL
        assert (Usage(1, 1) + u).model == SMALL


# ── Which model runs ──

class TestPredictorModel:
    def test_default_is_the_main_model(self, app_client):
        c, llm, fake = app_client
        uid = _profile(c)
        assert c.post("/predict", json={"user_id": uid, "draft_content": "x"},
                      headers=AUTH).status_code == 200
        assert fake.calls == [BIG, BIG]

    def test_setting_moves_only_the_predictor(self, app_client, monkeypatch):
        c, llm, fake = app_client
        monkeypatch.setattr(llm, "PREDICTOR_MODEL", SMALL)
        uid = _profile(c)
        c.post("/predict", json={"user_id": uid, "draft_content": "x"}, headers=AUTH)
        assert fake.calls == [BIG, SMALL]   # profile on 120b, score on 20b

    def test_audit_records_the_real_model_and_its_price(self, app_client, monkeypatch):
        c, llm, fake = app_client
        monkeypatch.setattr(llm, "PREDICTOR_MODEL", SMALL)
        uid = _profile(c)
        c.post("/predict", json={"user_id": uid, "draft_content": "x"}, headers=AUTH)

        rows = {r["agent_name"]: r for r in c.get("/audit-logs", headers=ADMIN).json()}
        assert rows["predictor"]["model"] == SMALL
        assert rows["predictor"]["cost_usd"] == pytest.approx((1000 * 0.075 + 500 * 0.30) / 1e6)
        assert rows["brand"]["model"] == BIG
        assert rows["brand"]["cost_usd"] == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1e6)

    def test_explicit_model_overrides_the_setting(self, app_client, monkeypatch):
        c, llm, fake = app_client
        monkeypatch.setattr(llm, "PREDICTOR_MODEL", SMALL)
        llm.predict_engagement("x", PROFILE_REPLY, [], model=BIG)
        assert fake.calls == [BIG]

    def test_costs_breaks_spend_down_by_model(self, app_client, monkeypatch):
        c, llm, fake = app_client
        monkeypatch.setattr(llm, "PREDICTOR_MODEL", SMALL)
        uid = _profile(c)
        c.post("/predict", json={"user_id": uid, "draft_content": "x"}, headers=AUTH)

        import main
        monkeypatch.setattr("utils.config.PREDICTOR_MODEL", SMALL)
        body = c.get("/costs", headers=ADMIN).json()
        by_model = {m["model"]: m for m in body["by_model"]}
        assert set(by_model) == {BIG, SMALL}
        assert by_model[SMALL]["cost_usd"] < by_model[BIG]["cost_usd"]
        assert body["pricing"]["predictor_model"] == SMALL
        assert body["pricing"]["predictor_input_usd_per_million"] == 0.075


# ── The comparison ──

def _results(n=12):
    items = []
    for k in range(n):
        items.append({"version": 1 + k % 5, "case_id": "data-streaming", "score": k * 8,
                      "post_content": "post " + "good " * (k // 4)})
    items.append({"version": 4, "case_id": "kafka-startup", "error": "HTTP 502"})
    return {"items": items}


class TestComparisonMaths:
    def test_posts_span_the_whole_range(self):
        _fresh()
        from evals.compare_predictor import pick_posts
        picked = pick_posts(_results(), 4)
        scores = [p["score"] for p in picked]
        assert scores[0] == 0 and scores[-1] == 88
        assert scores == sorted(scores)
        assert all("error" not in p for p in picked)

    @pytest.mark.parametrize("a,b,expected", [
        ([1, 2, 3], [10, 20, 30], 1.0),
        ([1, 2, 3], [30, 20, 10], 0.0),
        ([1, 2, 3], [10, 10, 30], (0.5 + 1 + 1) / 3),
        ([1, 1, 3], [10, 10, 30], 1.0),
    ])
    def test_pair_agreement(self, a, b, expected):
        _fresh()
        from evals.compare_predictor import pair_agreement
        assert pair_agreement(a, b) == pytest.approx(expected)

    def _verdict(self, gap, agreement, base_fail=0, cand_fail=0, paired=10):
        _fresh()
        from evals.compare_predictor import verdict
        return verdict({"failures": base_fail, "cost_per_score_usd": 0.0010},
                       {"failures": cand_fail, "cost_per_score_usd": 0.0005},
                       {"paired": paired, "mean_gap": gap, "max_gap": gap * 2,
                        "pair_agreement": agreement}, BIG, SMALL)

    def test_switch_when_close_and_consistent(self):
        v = self._verdict(gap=4.0, agreement=0.9)
        assert v["switch"] and v["saving"] == 0.5
        assert "PREDICTOR_MODEL=openai/gpt-oss-20b" in v["reason"]

    def test_stay_when_scores_drift(self):
        v = self._verdict(gap=12.0, agreement=0.9)
        assert not v["switch"] and "differed by 12.0" in v["reason"]

    def test_stay_when_ranking_disagrees(self):
        v = self._verdict(gap=3.0, agreement=0.6)
        assert not v["switch"] and "60%" in v["reason"]

    def test_stay_when_less_reliable(self):
        v = self._verdict(gap=1.0, agreement=1.0, cand_fail=2)
        assert not v["switch"] and "reliability" in v["reason"]

    def test_stay_when_too_few_pairs(self):
        assert not self._verdict(gap=1.0, agreement=1.0, paired=3)["switch"]


class TestComparisonRun:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/cmp.db")
        _fresh()
        from evals import compare_predictor as cp
        cp.run_eval.prepare_database()
        import services.llm_service as llm
        fake = FakeGroq(small_offset=-3)
        monkeypatch.setattr(llm, "client", fake)
        return cp, fake, tmp_path / "results"

    def test_end_to_end(self, setup):
        cp, fake, out = setup
        posts = cp.pick_posts(_results(), 6)
        report = cp.run(posts, BIG, SMALL, results_dir=out, progress=lambda m: None)

        assert fake.calls == [BIG, SMALL] * 6
        assert report["between"]["mean_gap"] == 3.0
        assert report["between"]["pair_agreement"] == 1.0
        assert report["candidate"]["cost_per_score_usd"] == pytest.approx(
            report["baseline"]["cost_per_score_usd"] / 2)
        assert report["verdict"]["switch"] is True
        assert (out / "predictor_latest.md").exists()
        assert "Verdict: switch" in (out / "predictor_latest.md").read_text(encoding="utf-8")

        from database.db import SessionLocal
        from database.models import AuditLog
        db = SessionLocal()
        try:
            tags = {r.endpoint for r in db.query(AuditLog).all()}
            assert tags == {"EVAL predictor gpt-oss-120b", "EVAL predictor gpt-oss-20b"}
        finally:
            db.close()

    def test_production_setting_is_untouched(self, setup, monkeypatch):
        cp, fake, out = setup
        import utils.config as config
        before = config.PREDICTOR_MODEL
        cp.run(cp.pick_posts(_results(), 2), BIG, SMALL, results_dir=out,
               progress=lambda m: None)
        assert config.PREDICTOR_MODEL == before

    def test_needs_eval_results_first(self, setup, monkeypatch):
        cp, fake, out = setup
        monkeypatch.setattr(cp, "RESULTS_DIR", out)
        assert cp.main([]) == 2


# ── Predictor prompt v3 ──

class TestPredictorPromptV3:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/v3.db")
        _fresh()
        from evals import run_eval
        run_eval.prepare_database()
        from database.db import SessionLocal
        session = SessionLocal()
        yield session
        session.close()

    def test_v3_fills_in_cleanly(self, db):
        """A stray brace in the template would crash every score."""
        from services.llm_service import _prompt_version
        template, version = _prompt_version(db, "predictor", 3)
        assert version == 3
        text = template.format(tone="t", content_themes="[]", positioning="p",
                               history_context="none", draft_content="DRAFT")
        assert "DRAFT" in text and "WHAT THE WRITER KNEW" in text

    def test_v3_stops_rewarding_invented_specifics_and_bait(self, db):
        from services.llm_service import _prompt_version
        template, _ = _prompt_version(db, "predictor", 3)
        assert "CANNOT score above 55" not in template   # the v2 rule that punished honesty
        assert "invented" in template
        assert "Generic bait" in template and "40 or lower" in template  # bait scored low
        assert "Never suggest adding" in template

    def test_v2_stays_active(self, db):
        from services.llm_service import get_active_prompt
        assert get_active_prompt(db, "predictor")[1] == 2

    def test_a_version_can_be_requested(self, db, monkeypatch):
        import services.llm_service as llm
        fake = FakeGroq()
        monkeypatch.setattr(llm, "client", fake)
        seen = []
        original = fake.create
        def spy(**kw):
            seen.append(kw["messages"][-1]["content"])
            return original(**kw)
        fake.chat.completions.create = spy
        llm.predict_engagement("x", PROFILE_REPLY, [], prompt_version=3)
        llm.predict_engagement("x", PROFILE_REPLY, [])
        assert "WHAT THE WRITER KNEW" in seen[0]
        assert "WHAT THE WRITER KNEW" not in seen[1]


class TestPredictorPromptV4:
    """v3 put the draft above a section heading, and the model read the heading
    as the post's last line ("WHAT THE WRITER KNEW" turned up in a tip)."""

    @pytest.fixture
    def template(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/v4.db")
        _fresh()
        from evals import run_eval
        run_eval.prepare_database()
        from database.db import SessionLocal
        from services.llm_service import _prompt_version
        db = SessionLocal()
        try:
            return _prompt_version(db, "predictor", 4)[0]
        finally:
            db.close()

    def test_fills_in_cleanly(self, template):
        text = template.format(tone="t", content_themes="[]", positioning="p",
                               history_context="none", draft_content="DRAFT")
        assert "=== POST START ===\nDRAFT\n=== POST END ===" in text

    def test_the_draft_comes_after_every_instruction(self, template):
        draft = template.index("{draft_content}")
        for heading in ("WHAT THE WRITER KNEW", "RUBRIC", "DIMENSIONS",
                        "IMPROVEMENT TIPS", "Return ONLY valid JSON"):
            assert template.index(heading) < draft, heading
        assert template.rstrip().endswith("Return ONLY the JSON object, no extra text.")

    def test_tells_the_model_headings_are_not_the_post(self, template):
        assert "never quote them" in template

    def test_keeps_v3_rules(self, template):
        assert "invented" in template and "Generic bait" in template
        assert "CANNOT score above 55" not in template


class TestPromptVerdict:
    def _v(self, old, new, base_fail=0, cand_fail=0, paired=10):
        _fresh()
        from evals.compare_predictor import prompt_verdict
        return prompt_verdict({"failures": base_fail, "agrees_with_rules": old},
                              {"failures": cand_fail, "agrees_with_rules": new},
                              {"paired": paired}, 2, 3)

    def test_switch_when_clearly_better(self):
        v = self._v(0.37, 0.80)
        assert v["switch"] and "rollback/3" in v["reason"]

    def test_keep_when_gain_is_small(self):
        v = self._v(0.60, 0.70)
        assert not v["switch"] and "+10 points" in v["reason"]

    def test_keep_when_still_near_a_coin_flip(self):
        v = self._v(0.30, 0.55)
        assert not v["switch"] and "55%" in v["reason"]

    def test_keep_when_less_reliable(self):
        assert not self._v(0.37, 0.95, cand_fail=1)["switch"]


class TestPromptComparisonRun:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/pcmp.db")
        _fresh()
        from evals import compare_predictor as cp
        cp.run_eval.prepare_database()

        def create(**kw):
            prompt = kw["messages"][-1]["content"]
            good = prompt.count("good")
            # v3 rewards better posts; v2 does the opposite, as in the real run.
            level = 40 + good * 15 if "WHAT THE WRITER KNEW" in prompt else 85 - good * 10
            text = json.dumps(prediction(level))
            usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=500,
                                    prompt_tokens_details=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)

        import services.llm_service as llm
        monkeypatch.setattr(llm, "client", SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        out = tmp_path / "results"
        out.mkdir()
        monkeypatch.setattr(cp, "RESULTS_DIR", out)
        monkeypatch.setattr(cp.run_eval, "RESULTS_DIR", out)
        (out / "latest.json").write_text(json.dumps(_results()), encoding="utf-8")
        return cp, out

    def test_cli_compares_prompt_versions(self, setup):
        cp, out = setup
        assert cp.main(["--candidate-version", "3", "--posts", "6"]) == 0
        report = json.loads((out / "predictor_latest.json").read_text(encoding="utf-8"))
        assert report["mode"] == "prompt"
        assert report["baseline_label"] == "gpt-oss-120b v2"
        assert report["candidate_label"] == "gpt-oss-120b v3"
        assert report["candidate"]["agrees_with_rules"] > report["baseline"]["agrees_with_rules"]
        assert report["verdict"]["switch"] is True
        md = (out / "predictor_latest.md").read_text(encoding="utf-8")
        assert "gpt-oss-120b v3 (candidate)" in md and "rollback/3" in md

    def test_unknown_version_is_refused(self, setup):
        cp, out = setup
        assert cp.main(["--candidate-version", "9"]) == 2

    def test_same_setup_twice_is_refused(self, setup):
        cp, out = setup
        assert cp.main(["--candidate-version", "2"]) == 2

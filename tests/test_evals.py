"""
Prompt evaluation harness: the scoring rules and the runner.

The runner tests add two throwaway prompt versions to a temporary database and
fake Groq so each version gets a known post back. Offline and free.
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

from evals.scoring import WEIGHTS, score_post  # noqa: E402

TOPIC = "Why I stopped using timestamps as IDs"
TAGS = ["SoftwareEngineering", "Databases", "DistributedSystems"]

CLEAN = """Timestamps feel unique. They are not.

Two services writing in the same millisecond will happily hand you the same key, and the database will reject one of them without ever telling you which write you lost. I stopped trusting the clock for identity, and I'd make the same call again.

Generate the ID. Let time be a column you sort by (not the thing that makes a row yours)."""

SLOP = ("🚀 In today's fast-paced world, this is not just a tool, it's a game-changer! "
        "We leveraged it to unlock seamless, robust results — cutting latency 73% on "
        "Tuesday. Experts agree. What do you think?")


# ── Scoring ──

class TestScoring:
    def test_a_clean_post_scores_high(self):
        result = score_post(CLEAN, TAGS, TOPIC)
        assert result.score >= 90, result.issues
        assert result.paragraphs == 3

    def test_slop_scores_zero_and_says_why(self):
        result = score_post(SLOP, ["motivation"], TOPIC)
        assert result.score == 0
        for check in ("em_dash", "banned_word", "negative_parallelism", "promotional_phrase",
                      "engagement_bait", "vague_authority", "invented_specific",
                      "emoji_opener", "single_block", "hashtag_count", "filler_hashtag"):
            assert check in result.counts, check

    @pytest.mark.parametrize("phrase,check", [
        ("This is a robust approach.", "banned_word"),
        ("We are navigating change.", "banned_word"),
        ("It is fast — really fast.", "em_dash"),
        ("It's not only faster but also cheaper.", "negative_parallelism"),
        ("In an era where data rules, we adapt.", "promotional_phrase"),
        ("Studies show this works.", "vague_authority"),
        ("We moved fast, ensuring maximum impact.", "tacked_on_ing"),
        ("**Lesson one** matters.", "markdown"),
        ("It took 3 days to fix.", "invented_specific"),
        ("The alert fired at 14:32.", "invented_specific"),
        ("We shipped it on Friday.", "invented_specific"),
    ])
    def test_each_rule_is_detected(self, phrase, check):
        assert check in score_post(phrase, TAGS, TOPIC).counts

    def test_numbers_from_the_topic_are_allowed(self):
        post = "A 40% drop-off is a signal.\n\nIt says the flow asks too much, too early."
        result = score_post(post, TAGS, "Why our booking flow had a 40% drop-off")
        assert "invented_specific" not in result.counts

    def test_keycap_emoji_are_a_list_not_numbers(self):
        result = score_post("Steps:\n\n1️⃣ Plan\n2️⃣ Build", TAGS, TOPIC)
        assert "numbered_emoji_list" in result.counts
        assert "invented_specific" not in result.counts

    @pytest.mark.parametrize("tags,ok", [(["a", "b"], False), (["a", "b", "c"], True),
                                         (["a", "b", "c", "d", "e", "f"], False)])
    def test_hashtag_count(self, tags, ok):
        assert ("hashtag_count" not in score_post(CLEAN, tags, TOPIC).counts) == ok

    def test_score_never_goes_below_zero(self):
        assert score_post(SLOP * 5, [], TOPIC).score == 0

    def test_every_check_has_a_weight(self):
        result = score_post(SLOP + "\n\n1️⃣ **x**", [], TOPIC)
        assert set(result.counts) <= set(WEIGHTS)


# ── Runner ──

GOOD_TEMPLATE = "EVAL-TEST-GOOD {topic} {tone}"
BAD_TEMPLATE = "EVAL-TEST-BAD {topic} {tone}"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/eval.db")
    for module in list(sys.modules):
        if module.split(".")[0] in {"main", "database", "agents", "services", "utils",
                                    "schemas", "evals"}:
            del sys.modules[module]

    from evals import run_eval
    run_eval.prepare_database()

    from database.db import SessionLocal
    from database.models import PromptTemplate
    db = SessionLocal()
    try:
        db.add_all([
            PromptTemplate(agent_name="content", version=90, template=GOOD_TEMPLATE,
                           is_active=False, description="test good"),
            PromptTemplate(agent_name="content", version=91, template=BAD_TEMPLATE,
                           is_active=False, description="test bad"),
        ])
        db.commit()
    finally:
        db.close()

    def create(**kw):
        prompt = kw["messages"][-1]["content"]
        if "EVAL-TEST-BROKEN" in prompt:
            text = "not json"
        elif "EVAL-TEST-GOOD" in prompt:
            text = json.dumps({"post_content": CLEAN, "suggested_hashtags": TAGS})
        elif "EVAL-TEST-BAD" in prompt:
            text = json.dumps({"post_content": SLOP, "suggested_hashtags": ["motivation"]})
        else:  # the predictor, when --judge is on
            text = json.dumps({"brand_alignment": 80, "hook_strength": 80, "readability": 80,
                               "call_to_action": 80, "predicted_likes": "1-2",
                               "predicted_comments": "0-1", "predicted_shares": "0-1",
                               "improvement_tips": "x"})
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=usage)

    import services.llm_service as llm
    monkeypatch.setattr(llm, "client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    cases = run_eval.load_cases(limit=2)
    return run_eval, cases, tmp_path / "results"


class TestRunner:
    def test_ranks_versions_and_writes_reports(self, harness):
        run_eval, cases, out = harness
        report = run_eval.run([90, 91], cases, results_dir=out, progress=lambda m: None)

        by_version = {r["version"]: r for r in report["summary"]}
        assert by_version[90]["quality"] > 90
        assert by_version[91]["quality"] == 0
        assert by_version[90]["tokens_per_post"] == 1500
        assert by_version[90]["cost_per_post_usd"] == pytest.approx(
            (1000 * 0.15 + 500 * 0.60) / 1_000_000)
        assert report["recommendation"]["version"] == 90

        assert (out / "latest.json").exists() and (out / "latest.md").exists()
        assert len(list(out.glob("eval-*.json"))) == 1
        assert "Recommendation: v90" in (out / "latest.md").read_text(encoding="utf-8")

    def test_calls_are_tagged_and_production_prompt_is_untouched(self, harness):
        run_eval, cases, out = harness
        _, active_before = run_eval.content_versions()
        run_eval.run([90], cases, results_dir=out, progress=lambda m: None)
        _, active_after = run_eval.content_versions()
        assert active_before == active_after == 5

        from database.db import SessionLocal
        from database.models import AuditLog
        db = SessionLocal()
        try:
            rows = db.query(AuditLog).filter(AuditLog.endpoint == "EVAL content v90").all()
            assert len(rows) == len(cases)
            assert all(r.prompt_version == 90 and r.cost_usd > 0 for r in rows)
        finally:
            db.close()

    def test_judge_adds_a_predictor_score(self, harness):
        run_eval, cases, out = harness
        report = run_eval.run([90], cases, judge=True, results_dir=out, progress=lambda m: None)
        assert report["summary"][0]["judge_mean"] == 80
        assert all(i["judge_cost_usd"] > 0 for i in report["items"])

    def test_a_failed_generation_is_recorded_not_fatal(self, harness):
        run_eval, cases, out = harness
        from database.db import SessionLocal
        from database.models import PromptTemplate
        db = SessionLocal()
        db.add(PromptTemplate(agent_name="content", version=92, is_active=False,
                              template="EVAL-TEST-BROKEN {topic}", description="broken"))
        db.commit(); db.close()

        report = run_eval.run([92, 90], cases, results_dir=out, progress=lambda m: None)
        by_version = {r["version"]: r for r in report["summary"]}
        assert by_version[92]["errors"] == len(cases) and by_version[92]["posts"] == 0
        assert by_version[90]["posts"] == len(cases)

    def test_stops_when_the_budget_is_used_up(self, harness, monkeypatch):
        run_eval, cases, out = harness
        import utils.budget as budget
        monkeypatch.setattr(budget, "budget_status", lambda db=None: {"state": "blocked"})
        with pytest.raises(run_eval.BudgetReached):
            run_eval.run([90], cases, results_dir=out, progress=lambda m: None)
        run_eval.run([90], cases[:1], ignore_budget=True, results_dir=out,
                     progress=lambda m: None)

    def test_failures_count_against_a_version(self, harness):
        """A real run: v4 scored 100 twice but failed once; v5 never failed.
        Judged on successes alone v4 looked best, which was wrong."""
        run_eval, _, _ = harness

        def items(version, scores):
            out = []
            for s in scores:
                if s is None:
                    out.append({"version": version, "error": "HTTPException"})
                else:
                    out.append({"version": version, "score": s, "counts": {}, "words": 120,
                                "total_tokens": 3000, "cost_usd": 0.0012, "latency_ms": 900})
            return out

        summary = run_eval.summarise(items(4, [100, 100, None]) + items(5, [88, 97, 84]),
                                     [4, 5], judge=False)
        by_version = {r["version"]: r for r in summary}
        assert by_version[4]["quality"] == pytest.approx(66.7)
        assert by_version[4]["quality_when_succeeded"] == 100
        assert by_version[4]["succeeded"] == "2/3"
        assert by_version[4]["quality_min"] == 0

        rec = run_eval.recommend(summary, active=5)
        assert rec["version"] == 5
        assert "Small sample" in rec["reason"]

    def test_a_negligible_cost_gap_does_not_beat_a_higher_score(self, harness):
        """A real full run: v4 96.8 at $0.00170, v5 97 at $0.00171. A
        hundredth of a cent per post is noise; the higher score should win."""
        run_eval, _, _ = harness
        summary = [
            {"version": 4, "posts": 8, "errors": 0, "quality": 96.8, "cost_per_post_usd": 0.00170},
            {"version": 5, "posts": 8, "errors": 0, "quality": 97.0, "cost_per_post_usd": 0.00171},
        ]
        rec = run_eval.recommend(summary, active=1)
        assert rec["version"] == 5
        assert "cost more" not in rec["reason"]

    def test_rescore_rebuilds_the_report_without_calls(self, harness, monkeypatch):
        run_eval, cases, out = harness
        run_eval.run([90, 91], cases, results_dir=out, progress=lambda m: None)
        saved = json.loads((out / "latest.json").read_text(encoding="utf-8"))
        saved["recommendation"] = {"version": 91, "reason": "stale"}
        (out / "latest.json").write_text(json.dumps(saved), encoding="utf-8")

        import services.llm_service as llm
        monkeypatch.setattr(llm, "client", None)  # any Groq call would now fail
        assert run_eval.rescore(out) == 0
        rebuilt = json.loads((out / "latest.json").read_text(encoding="utf-8"))
        assert rebuilt["recommendation"]["version"] == 90
        assert rebuilt["items"] == saved["items"]
        assert "Recommendation: v90" in (out / "latest.md").read_text(encoding="utf-8")

    def test_unknown_version_is_rejected(self, harness):
        run_eval, _, _ = harness
        assert run_eval.main(["--versions", "999"]) == 2

    def test_ties_go_to_the_cheaper_version(self, harness):
        run_eval, _, _ = harness
        summary = [
            {"version": 1, "posts": 4, "errors": 0, "quality": 90.0, "cost_per_post_usd": 0.0009},
            {"version": 2, "posts": 4, "errors": 0, "quality": 89.0, "cost_per_post_usd": 0.0004},
            {"version": 3, "posts": 4, "errors": 0, "quality": 70.0, "cost_per_post_usd": 0.0001},
        ]
        assert run_eval.recommend(summary, active=1)["version"] == 2

    def test_a_full_tie_keeps_the_active_version(self, harness):
        run_eval, _, _ = harness
        summary = [{"version": v, "posts": 4, "errors": 0, "quality": 90.0, "cost_per_post_usd": 0.0004}
                   for v in (1, 5)]
        rec = run_eval.recommend(summary, active=5)
        assert rec["version"] == 5
        assert "same cost" in rec["reason"] and "already the active" in rec["reason"]


class TestCases:
    def test_cases_file_is_well_formed(self):
        from evals.run_eval import load_cases
        cases = load_cases()
        assert len(cases) >= 6
        assert len({c["id"] for c in cases}) == len(cases)
        for c in cases:
            assert len(c["topic"]) >= 10
            for key in ("tone", "content_themes", "positioning_summary",
                        "do_guidelines", "dont_guidelines"):
                assert key in c["profile"], (c["id"], key)

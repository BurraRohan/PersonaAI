"""
Prompt evaluation harness.

Runs every chosen content-prompt version against the same fixed test cases
(evals/cases.json), scores each post with the rule-based checks in
evals/scoring.py, and reports quality next to tokens and cost per version.
Optionally, the engagement predictor scores each post as a second, model-based
opinion (--judge).

    python -m evals.run_eval                     all versions, all cases
    python -m evals.run_eval --versions 3 5      compare two versions
    python -m evals.run_eval --cases 3 --judge   quick run with the predictor
    python -m evals.run_eval --repeat 2          each case twice, for variance

These are real Groq calls. Each is written to the audit log like any other, so
it counts toward the daily budget and appears in /costs, tagged with the
endpoint "EVAL content vN" to keep it apart from user traffic. A full run of
five versions over eight cases is 40 calls, roughly 3 to 4 US cents at
gpt-oss-120b prices; --judge roughly doubles that.

Production is never touched: each call names its prompt version explicitly,
so the active version stays whatever it was.

Results go to evals/results/: a timestamped JSON file for the run, plus
latest.json and latest.md, which the Streamlit dashboard reads.
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from evals.scoring import score_post  # noqa: E402

CASES_FILE = HERE / "cases.json"
RESULTS_DIR = HERE / "results"


class BudgetReached(RuntimeError):
    pass


# ── Setup ───────────────────────────────────────────────────────

def prepare_database() -> None:
    """Create tables, apply additive migrations and seed prompts, as the API does."""
    from database.db import Base, SessionLocal, engine, ensure_schema
    import database.models  # noqa: F401  (registers the tables)
    from main import seed_default_prompts

    Base.metadata.create_all(bind=engine)
    ensure_schema()
    db = SessionLocal()
    try:
        seed_default_prompts(db)
    finally:
        db.close()


def load_cases(path: Path = CASES_FILE, limit: Optional[int] = None) -> list:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return cases[:limit] if limit else cases


def content_versions() -> tuple[list, Optional[int]]:
    """All content prompt versions in the database, and the active one."""
    from database.db import SessionLocal
    from database.models import PromptTemplate

    db = SessionLocal()
    try:
        rows = (db.query(PromptTemplate.version, PromptTemplate.is_active)
                .filter(PromptTemplate.agent_name == "content")
                .order_by(PromptTemplate.version).all())
        return [v for v, _ in rows], next((v for v, active in rows if active), None)
    finally:
        db.close()


def _usage_for(request_id: str) -> dict:
    """Tokens, cost and latency recorded in the audit log for one call."""
    from sqlalchemy import func
    from database.db import SessionLocal
    from database.models import AuditLog

    db = SessionLocal()
    try:
        row = (db.query(func.sum(AuditLog.prompt_tokens), func.sum(AuditLog.completion_tokens),
                        func.sum(AuditLog.total_tokens), func.sum(AuditLog.cost_usd),
                        func.sum(AuditLog.latency_ms))
               .filter(AuditLog.request_id == request_id).one())
    finally:
        db.close()
    return {
        "prompt_tokens": int(row[0] or 0), "completion_tokens": int(row[1] or 0),
        "total_tokens": int(row[2] or 0), "cost_usd": float(row[3] or 0.0),
        "latency_ms": round(float(row[4] or 0.0)),
    }


def _tagged_call(endpoint: str, fn: Callable):
    """Run fn under its own request id and an EVAL endpoint tag."""
    from utils.request_context import endpoint_var, request_id_var

    request_id = str(uuid.uuid4())
    rid, ep = request_id_var.set(request_id), endpoint_var.set(endpoint)
    try:
        return fn(), request_id
    finally:
        request_id_var.reset(rid)
        endpoint_var.reset(ep)


def _check_budget(ignore: bool) -> None:
    if ignore:
        return
    from utils.budget import budget_status
    if budget_status()["state"] == "blocked":
        raise BudgetReached("The daily budget is used up. Raise DAILY_BUDGET_USD, "
                            "wait for midnight UTC, or pass --ignore-budget.")


# ── Run ─────────────────────────────────────────────────────────

def run(versions: list, cases: list, judge: bool = False, repeat: int = 1,
        ignore_budget: bool = False, results_dir: Path = RESULTS_DIR,
        progress: Callable = print) -> dict:
    """Evaluate the given content prompt versions and write the results."""
    from services.llm_service import generate_linkedin_post, predict_engagement
    from utils.config import GROQ_MODEL

    _, active = content_versions()
    started = datetime.now(timezone.utc)
    items = []
    total = len(versions) * len(cases) * repeat
    done = 0

    for version in versions:
        for case in cases:
            for attempt in range(1, repeat + 1):
                done += 1
                _check_budget(ignore_budget)
                item = {"version": version, "case_id": case["id"], "topic": case["topic"],
                        "attempt": attempt}
                try:
                    result, rid = _tagged_call(
                        f"EVAL content v{version}",
                        lambda: generate_linkedin_post(case["profile"], case["topic"],
                                                       prompt_version=version))
                    post = result.get("post_content", "") or ""
                    tags = result.get("suggested_hashtags", []) or []
                    item.update(post_content=post, hashtags=tags, **_usage_for(rid))
                    item.update(score_post(post, tags, case["topic"]).to_dict())

                    if judge:
                        _check_budget(ignore_budget)
                        prediction, jid = _tagged_call(
                            f"EVAL predictor v{version}",
                            lambda: predict_engagement(post, case["profile"], []))
                        item["judge_score"] = prediction.get("overall_score")
                        item["judge_cost_usd"] = _usage_for(jid)["cost_usd"]
                except BudgetReached:
                    raise
                except Exception as exc:  # one bad generation must not end the run
                    status = getattr(exc, "status_code", None)
                    item["error"] = f"HTTP {status}" if status else type(exc).__name__
                items.append(item)

                label = (f"error {item['error']}" if "error" in item
                         else f"score {item['score']:>3}  ${item['cost_usd']:.5f}")
                progress(f"[{done:>3}/{total}] v{version}  {case['id']:<20} {label}")

    report = {
        "run_id": started.strftime("%Y%m%d-%H%M%S"),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": GROQ_MODEL,
        "active_version": active,
        "versions": versions,
        "cases": len(cases),
        "repeat": repeat,
        "judge": judge,
        "summary": summarise(items, versions, judge),
        "items": items,
    }
    report["total_cost_usd"] = round(sum(i.get("cost_usd", 0) + i.get("judge_cost_usd", 0)
                                         for i in items), 6)
    report["recommendation"] = recommend(report["summary"], active)
    write_results(report, Path(results_dir))
    return report


# ── Summary ─────────────────────────────────────────────────────

def summarise(items: list, versions: list, judge: bool) -> list:
    rows = []
    for version in versions:
        mine = [i for i in items if i["version"] == version]
        ok = [i for i in mine if "error" not in i]
        if not ok:
            rows.append({"version": version, "posts": 0, "errors": len(mine),
                         "quality": 0.0, "succeeded": f"0/{len(mine)}"})
            continue
        scores = [i["score"] for i in ok]
        # A failed generation delivers nothing to the user, so it scores 0.
        # Without this, a version that fails often is judged only on its
        # successes and can look better than a reliable one.
        all_scores = scores + [0] * (len(mine) - len(ok))
        affected = Counter(check for i in ok for check in i["counts"])
        row = {
            "version": version,
            "posts": len(ok),
            "errors": len(mine) - len(ok),
            "succeeded": f"{len(ok)}/{len(mine)}",
            "quality": round(statistics.mean(all_scores), 1),
            "quality_min": min(all_scores),
            "quality_spread": round(statistics.pstdev(all_scores), 1),
            "quality_when_succeeded": round(statistics.mean(scores), 1),
            "words_mean": round(statistics.mean(i["words"] for i in ok)),
            "tokens_per_post": round(statistics.mean(i["total_tokens"] for i in ok)),
            "cost_per_post_usd": round(statistics.mean(i["cost_usd"] for i in ok), 6),
            "latency_ms_mean": round(statistics.mean(i["latency_ms"] for i in ok)),
            # Share of posts that broke each rule at least once.
            "issue_rates": {check: round(n / len(ok), 2)
                            for check, n in affected.most_common()},
        }
        if judge:
            judged = [i["judge_score"] for i in ok if isinstance(i.get("judge_score"), int)]
            row["judge_mean"] = round(statistics.mean(judged), 1) if judged else None
        rows.append(row)
    return rows


SMALL_SAMPLE = 5
# Costs within this share of the cheapest count as equal. A difference of a
# hundredth of a cent per post is noise, not a saving worth losing points for.
COST_TOLERANCE = 0.05


def recommend(summary: list, active: Optional[int]) -> dict:
    """Highest quality wins, with failed generations counted as 0.

    Versions within 2 points of the best count as a tie, and the cheaper one
    wins. Costs within 5% of each other count as equal, and then the higher
    score wins. On a full tie the active version is kept, since switching gains
    nothing.
    """
    scored = [r for r in summary if r.get("posts")]
    if not scored:
        return {"version": None, "reason": "No version produced a usable post."}
    top = max(r["quality"] for r in scored)
    contenders = [r for r in scored if r["quality"] >= top - 2]
    cheapest = min(r["cost_per_post_usd"] for r in contenders)
    affordable = [r for r in contenders
                  if r["cost_per_post_usd"] <= cheapest * (1 + COST_TOLERANCE)]
    best = max(affordable, key=lambda r: (r["quality"], r["version"] == active))

    reason = (f"v{best['version']} scored {best['quality']} at "
              f"${best['cost_per_post_usd']:.5f} per post")
    reason += (f", with {best['errors']} failed generation(s)." if best["errors"]
               else ", with no failures.")
    others = [r for r in contenders if r is not best]
    if others:
        pricier = [r for r in others if r not in affordable]
        if pricier:
            reason += " Other versions within 2 points cost more."
        else:
            reason += (" Other versions scored within 2 points at about the same cost;"
                       " the higher score wins.")
    if active is not None and best["version"] != active:
        current = next((r for r in scored if r["version"] == active), None)
        if current:
            reason += (f" The active version, v{active}, scored {current['quality']}"
                       f" at ${current['cost_per_post_usd']:.5f}.")
    elif best["version"] == active:
        reason += " It is already the active version."

    attempts = min(r["posts"] + r["errors"] for r in summary)
    if attempts < SMALL_SAMPLE:
        reason += (f" Small sample ({attempts} posts per version): run the full set "
                   "before switching versions.")
    return {"version": best["version"], "reason": reason}


# ── Output ──────────────────────────────────────────────────────

def write_results(report: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, ensure_ascii=False)
    (results_dir / f"eval-{report['run_id']}.json").write_text(body, encoding="utf-8")
    (results_dir / "latest.json").write_text(body, encoding="utf-8")
    (results_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    judge = report["judge"]
    repeats = f" × {report['repeat']} repeats" if report["repeat"] > 1 else ""
    lines = [
        "# Prompt evaluation",
        "",
        f"Run {report['run_id']} · model `{report['model']}` · {report['cases']} cases"
        f"{repeats} · total cost ${report['total_cost_usd']:.4f}",
        "",
        f"**Recommendation: v{report['recommendation']['version']}.** "
        f"{report['recommendation']['reason']}",
        "",
        "| Version | Quality | Lowest | Spread | Succeeded | " + ("Predictor | " if judge else "")
        + "Words | Tokens/post | Cost/post |",
        "| --- | --- | --- | --- | --- | " + ("--- | " if judge else "") + "--- | --- | --- |",
    ]
    for r in report["summary"]:
        if not r.get("posts"):
            lines.append(f"| v{r['version']} | 0 | — | — | {r['succeeded']} | "
                         + ("— | " if judge else "") + "— | — | — |")
            continue
        active = " (active)" if r["version"] == report["active_version"] else ""
        lines.append(
            f"| v{r['version']}{active} | {r['quality']} | {r['quality_min']} | "
            f"{r['quality_spread']} | {r['succeeded']} | "
            + (f"{r.get('judge_mean')} | " if judge else "")
            + f"{r['words_mean']} | {r['tokens_per_post']} | "
            f"${r['cost_per_post_usd']:.5f} |")
    lines += ["", "Quality is the mean rule-check score with failed generations counted as 0."]

    lines += ["", "## Most common problems", "",
              "Share of each version's posts that broke the rule at least once.", ""]
    for r in report["summary"]:
        rates = r.get("issue_rates") or {}
        top = ", ".join(f"{k.replace('_', ' ')} {int(v * 100)}%"
                        for k, v in list(rates.items())[:5]) or "none"
        lines.append(f"- **v{r['version']}**: {top}")

    lines += ["", "## How quality is scored", "",
              "Each post starts at 100 and loses points for breaking a rule the content "
              "prompt states: em dashes, banned AI-tell words, invented numbers or dates, "
              "no paragraph breaks, flat sentence rhythm, emoji and hashtag limits. "
              "The checks are in `evals/scoring.py`."]
    return "\n".join(lines) + "\n"


def print_summary(report: dict) -> None:
    judge = report["judge"]
    print()
    header = f"{'Version':<10}{'Quality':>8}{'Lowest':>8}{'OK':>6}" \
        + (f"{'Judge':>7}" if judge else "") + f"{'Words':>7}{'Tokens':>8}{'Cost/post':>11}"
    print(header)
    print("-" * len(header))
    for r in report["summary"]:
        name = f"v{r['version']}" + ("*" if r["version"] == report["active_version"] else "")
        if not r.get("posts"):
            print(f"{name:<10}{0:>8}{'-':>8}{r['succeeded']:>6}" + (f"{'-':>7}" if judge else "")
                  + f"{'-':>7}{'-':>8}{'-':>11}")
            continue
        print(f"{name:<10}{r['quality']:>8}{r['quality_min']:>8}{r['succeeded']:>6}"
              + (f"{r.get('judge_mean') or '-':>7}" if judge else "")
              + f"{r['words_mean']:>7}{r['tokens_per_post']:>8}"
              f"{'$' + format(r['cost_per_post_usd'], '.5f'):>11}")
    print("\n* active version. Quality counts a failed generation as 0.")
    print(f"\nRecommendation: {report['recommendation']['reason']}")
    print(f"Total cost of this run: ${report['total_cost_usd']:.4f}")
    print(f"Report: {RESULTS_DIR / 'latest.md'}")


# ── CLI ─────────────────────────────────────────────────────────

def _quiet_logs() -> None:
    """Keep the console to progress lines. The app logs every request at INFO,
    which buries the results; warnings and errors still show."""
    for name in ("services", "agents", "utils", "main", "database", "httpx",
                 "openai", "openai._base_client"):
        logging.getLogger(name).setLevel(logging.WARNING)


class _RateLimitNotice(logging.Filter):
    """Turn retry log lines into one short notice; let real problems through.

    Both the OpenAI client and llm_service log when Groq's rate limit makes them
    wait. During an eval that is expected, so it becomes a single indented line.
    """
    WAIT = re.compile(r"Retrying (?:request to \S+ )?in ([\d.]+)\s*s")

    def filter(self, record):
        match = self.WAIT.search(record.getMessage())
        if match:
            print(f"        waiting on Groq's rate limit, retrying in {float(match.group(1)):.0f}s")
            return False
        return record.levelno >= logging.WARNING


def rescore(results_dir: Path = None) -> int:
    """Recompute summary and recommendation from latest.json. No Groq calls.

    The saved per-post results are the evidence; the summary is derived from
    them. When the ranking rules change, this rebuilds the report without
    paying for the generations again.
    """
    results_dir = Path(results_dir or RESULTS_DIR)
    latest = results_dir / "latest.json"
    if not latest.exists():
        print(f"No saved results at {latest}. Run an evaluation first.")
        return 2
    report = json.loads(latest.read_text(encoding="utf-8"))
    report["summary"] = summarise(report["items"], report["versions"], report["judge"])
    report["recommendation"] = recommend(report["summary"], report["active_version"])
    body = json.dumps(report, indent=2, ensure_ascii=False)
    latest.write_text(body, encoding="utf-8")
    (results_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    print_summary(report)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare content prompt versions.")
    parser.add_argument("--versions", type=int, nargs="+",
                        help="Prompt versions to compare (default: all)")
    parser.add_argument("--cases", type=int, help="Use only the first N test cases")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Generate each case this many times (default 1)")
    parser.add_argument("--judge", action="store_true",
                        help="Also score each post with the engagement predictor")
    parser.add_argument("--ignore-budget", action="store_true",
                        help="Run even if the daily budget is used up")
    parser.add_argument("--rescore", action="store_true",
                        help="Rebuild the summary and report from the last run's saved "
                             "results, with no Groq calls")
    args = parser.parse_args(argv)

    if args.rescore:
        return rescore()

    _quiet_logs()
    prepare_database()
    notice = _RateLimitNotice()
    for name in ("openai._base_client", "services.llm_service"):
        logging.getLogger(name).addFilter(notice)
    logging.getLogger("openai._base_client").setLevel(logging.INFO)
    available, active = content_versions()
    versions = args.versions or available
    missing = [v for v in versions if v not in available]
    if missing:
        print(f"Unknown content prompt version(s): {missing}. Available: {available}")
        return 2

    cases = load_cases(limit=args.cases)
    calls = len(versions) * len(cases) * max(1, args.repeat) * (2 if args.judge else 1)
    print(f"Evaluating versions {versions} on {len(cases)} cases "
          f"({calls} Groq calls). Active version: v{active}.")
    print("Groq's free tier limits tokens per minute, so expect pauses between calls.\n")
    try:
        report = run(versions, cases, judge=args.judge, repeat=max(1, args.repeat),
                     ignore_budget=args.ignore_budget)
    except BudgetReached as exc:
        print(f"\nStopped: {exc}")
        return 1
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

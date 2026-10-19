"""
Does a cheaper model score posts the way the current one does?

The engagement predictor rates drafts, and a smaller model costs half as much
per call. That is only worth it if the smaller model judges the same posts the
same way. This script reuses posts from the last prompt evaluation
(evals/results/latest.json), so nothing new is generated, and asks both models
to score each one.

    python -m evals.compare_predictor                    20b against the current model
    python -m evals.compare_predictor --posts 6          quicker
    python -m evals.compare_predictor --candidate-version 3
                                                         a new predictor prompt, same model

It reports, per model: how many scores succeeded, cost and time per score, and
how often its ranking of posts agrees with the rule-based quality scores. And
between the two: the average gap in score and how often they put the same
pair of posts in the same order.

There are two kinds of comparison, with different questions:

  Model     same prompt, cheaper model. Does it judge posts like the current
            one? It should agree closely, because the point is saving money
            without changing what users see.
  Prompt    same model, new prompt version. Does it judge posts better? It
            should agree more with the rule checks, because the point is a
            change in what users see.

Production is not touched: the comparison names each model and prompt version
per call. To switch the app afterwards, set PREDICTOR_MODEL in .env, or
activate a prompt with POST /prompts/predictor/rollback/<version>.

Results go to evals/results/predictor_latest.json and predictor_latest.md.
"""

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from evals import run_eval  # noqa: E402

RESULTS_DIR = run_eval.RESULTS_DIR
DEFAULT_CANDIDATE = "openai/gpt-oss-20b"

# When is the candidate good enough to switch to? These are judgement calls,
# stated here so they can be argued with rather than buried in code.
MAX_MEAN_GAP = 8.0          # average difference in overall score, in points
MIN_PAIR_AGREEMENT = 0.80   # share of post pairs both models order the same way
MIN_PAIRED = 5              # fewer paired results than this is not evidence

# For a new prompt version: it should agree with the rule checks clearly more
# often than the current one, and well above a coin flip.
MIN_RULE_AGREEMENT = 0.65
MIN_RULE_GAIN = 0.15


# ── Picking posts ───────────────────────────────────────────────

def pick_posts(results: dict, n: int) -> list:
    """n posts spread evenly across the rule-score range, worst to best.

    A predictor that only ever sees good posts cannot show whether it tells
    good from bad, so the sample deliberately spans the whole range.
    """
    usable = [i for i in results["items"] if "error" not in i and i.get("post_content")]
    usable.sort(key=lambda i: (i["score"], i["version"], i["case_id"]))
    if len(usable) <= n:
        return usable
    if n == 1:
        return [usable[-1]]
    step = (len(usable) - 1) / (n - 1)
    return [usable[round(k * step)] for k in range(n)]


# ── Measures ────────────────────────────────────────────────────

def pair_agreement(a: list, b: list) -> float:
    """Share of item pairs that a and b put in the same order.

    Both calling a pair equal is agreement; only one of them doing so counts half.
    """
    pairs = list(combinations(range(len(a)), 2))
    if not pairs:
        return 1.0
    total = 0.0
    for i, j in pairs:
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            total += 1.0
        elif da == 0 or db == 0:
            total += 0.5
        elif (da > 0) == (db > 0):
            total += 1.0
    return total / len(pairs)


def summarise_model(rows: list) -> dict:
    ok = [r for r in rows if "error" not in r]
    out = {"scored": f"{len(ok)}/{len(rows)}", "failures": len(rows) - len(ok)}
    if ok:
        out.update(
            mean_score=round(statistics.mean(r["overall_score"] for r in ok), 1),
            cost_per_score_usd=round(statistics.mean(r["cost_usd"] for r in ok), 6),
            latency_ms_mean=round(statistics.mean(r["latency_ms"] for r in ok)),
            agrees_with_rules=round(pair_agreement([r["overall_score"] for r in ok],
                                                   [r["rule_score"] for r in ok]), 2),
        )
    return out


def compare(baseline_rows: list, candidate_rows: list) -> dict:
    paired = [(b, c) for b, c in zip(baseline_rows, candidate_rows)
              if "error" not in b and "error" not in c]
    if not paired:
        return {"paired": 0}
    gaps = [abs(b["overall_score"] - c["overall_score"]) for b, c in paired]
    return {
        "paired": len(paired),
        "mean_gap": round(statistics.mean(gaps), 1),
        "max_gap": max(gaps),
        "pair_agreement": round(pair_agreement([b["overall_score"] for b, _ in paired],
                                               [c["overall_score"] for _, c in paired]), 2),
    }


def verdict(baseline: dict, candidate: dict, between: dict,
            baseline_name: str, candidate_name: str) -> dict:
    """Switch only if the candidate is as reliable and judges posts the same way."""
    short = candidate_name.split("/")[-1]
    if candidate["failures"] > baseline["failures"]:
        return {"switch": False, "reason":
                f"{short} failed {candidate['failures']} time(s) against "
                f"{baseline['failures']} for the current model. A failed score blocks a "
                "post from being approved, so reliability comes first."}
    if between.get("paired", 0) < MIN_PAIRED:
        return {"switch": False, "reason":
                f"Only {between.get('paired', 0)} posts were scored by both models; "
                f"at least {MIN_PAIRED} are needed to judge."}

    saving = 0.0
    if baseline.get("cost_per_score_usd"):
        saving = 1 - candidate["cost_per_score_usd"] / baseline["cost_per_score_usd"]
    agree = f"{between['pair_agreement']:.0%}"
    if between["mean_gap"] <= MAX_MEAN_GAP and between["pair_agreement"] >= MIN_PAIR_AGREEMENT:
        return {"switch": True, "saving": round(saving, 2), "reason":
                f"{short} scored within {between['mean_gap']} points of the current model "
                f"on average and ordered {agree} of post pairs the same way, at "
                f"{saving:.0%} lower cost per score. Set PREDICTOR_MODEL={candidate_name}."}
    problems = []
    if between["mean_gap"] > MAX_MEAN_GAP:
        problems.append(f"scores differed by {between['mean_gap']} points on average "
                        f"(limit {MAX_MEAN_GAP:g})")
    if between["pair_agreement"] < MIN_PAIR_AGREEMENT:
        problems.append(f"only {agree} of post pairs were ordered the same way "
                        f"(needs {MIN_PAIR_AGREEMENT:.0%})")
    return {"switch": False, "saving": round(saving, 2), "reason":
            f"{short} would save {saving:.0%} per score, but " + " and ".join(problems)
            + ". Keep the current model."}


def prompt_verdict(baseline: dict, candidate: dict, between: dict,
                   baseline_version: int, candidate_version: int) -> dict:
    """Switch prompt versions only if the new one judges posts more like the rules."""
    if candidate["failures"] > baseline["failures"]:
        return {"switch": False, "reason":
                f"v{candidate_version} failed {candidate['failures']} time(s) against "
                f"{baseline['failures']} for v{baseline_version}. A failed score blocks a "
                "post from being approved, so reliability comes first."}
    if between.get("paired", 0) < MIN_PAIRED or "agrees_with_rules" not in candidate:
        return {"switch": False, "reason":
                f"Only {between.get('paired', 0)} posts were scored by both versions; "
                f"at least {MIN_PAIRED} are needed to judge."}

    old, new = baseline["agrees_with_rules"], candidate["agrees_with_rules"]
    gain = new - old
    if new >= MIN_RULE_AGREEMENT and gain >= MIN_RULE_GAIN:
        return {"switch": True, "reason":
                f"v{candidate_version} agreed with the writing rules on {new:.0%} of post "
                f"pairs, up from {old:.0%} for v{baseline_version}. Activate it with "
                f"POST /prompts/predictor/rollback/{candidate_version}."}
    problems = []
    if new < MIN_RULE_AGREEMENT:
        problems.append(f"it agreed with the writing rules on only {new:.0%} of post pairs "
                        f"(needs {MIN_RULE_AGREEMENT:.0%})")
    if gain < MIN_RULE_GAIN:
        problems.append(f"that is {gain * 100:+.0f} points against v{baseline_version} "
                        f"(needs +{MIN_RULE_GAIN * 100:.0f})")
    return {"switch": False, "reason":
            f"v{candidate_version} is not clearly better: " + " and ".join(problems)
            + f". Keep v{baseline_version}."}


def active_predictor_version() -> int:
    from database.db import SessionLocal
    from database.models import PromptTemplate
    db = SessionLocal()
    try:
        row = (db.query(PromptTemplate.version)
               .filter(PromptTemplate.agent_name == "predictor",
                       PromptTemplate.is_active == True)  # noqa: E712
               .order_by(PromptTemplate.version.desc()).first())
        return row[0] if row else 0
    finally:
        db.close()


def _label(model: str, version) -> str:
    short = model.split("/")[-1]
    return f"{short} v{version}" if version else short


# ── Run ─────────────────────────────────────────────────────────

def _score(post: dict, profile: dict, model: str, version, label: str) -> dict:
    from services.llm_service import predict_engagement

    row = {"version": post["version"], "case_id": post["case_id"],
           "rule_score": post["score"]}
    try:
        prediction, rid = run_eval._tagged_call(
            f"EVAL predictor {label}",
            lambda: predict_engagement(post["post_content"], profile, [], model=model,
                                       prompt_version=version))
        row["overall_score"] = prediction.get("overall_score")
        row.update({k: v for k, v in run_eval._usage_for(rid).items()
                    if k in ("total_tokens", "cost_usd", "latency_ms")})
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        row["error"] = f"HTTP {status}" if status else type(exc).__name__
    return row


def run(posts: list, baseline_model: str, candidate_model: str,
        ignore_budget: bool = False, results_dir: Path = None,
        progress: Callable = print, baseline_version: int = None,
        candidate_version: int = None) -> dict:
    """Score the same posts with two setups (model and prompt version)."""
    profiles = {c["id"]: c["profile"] for c in run_eval.load_cases()}
    setups = [("current", baseline_model, baseline_version),
              ("candidate", candidate_model, candidate_version)]
    rows = {name: [] for name, _, _ in setups}
    total, done = len(posts) * 2, 0

    # Alternate setups post by post, so both meet the same rate-limit conditions.
    for post in posts:
        for name, model, version in setups:
            done += 1
            run_eval._check_budget(ignore_budget)
            label = _label(model, version)
            row = _score(post, profiles[post["case_id"]], model, version, label)
            rows[name].append(row)
            result = (f"error {row['error']}" if "error" in row else
                      f"predictor {row['overall_score']:>3}  ${row['cost_usd']:.5f}")
            progress(f"[{done:>2}/{total}] {label:<16} post v{post['version']} "
                     f"{post['case_id']:<20} rule {post['score']:>3}  {result}")

    baseline = summarise_model(rows["current"])
    candidate = summarise_model(rows["candidate"])
    between = compare(rows["current"], rows["candidate"])
    mode = "prompt" if baseline_model == candidate_model else "model"
    if mode == "prompt":
        decision = prompt_verdict(baseline, candidate, between,
                                  baseline_version, candidate_version)
    else:
        decision = verdict(baseline, candidate, between, baseline_model, candidate_model)
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "mode": mode,
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "baseline_label": _label(baseline_model, baseline_version),
        "candidate_label": _label(candidate_model, candidate_version),
        "posts": len(posts),
        "baseline": baseline,
        "candidate": candidate,
        "between": between,
        "verdict": decision,
        "rows": rows,
    }
    write_results(report, Path(results_dir or RESULTS_DIR))
    return report


# ── Output ──────────────────────────────────────────────────────

def _model_line(name: str, s: dict) -> str:
    if "mean_score" not in s:
        return f"| {name} | {s['scored']} | — | — | — | — |"
    return (f"| {name} | {s['scored']} | {s['mean_score']} | {s['agrees_with_rules']:.0%} | "
            f"${s['cost_per_score_usd']:.5f} | {s['latency_ms_mean'] / 1000:.1f}s |")


def render_markdown(r: dict) -> str:
    b = r["between"]
    lines = [
        "# Predictor comparison", "",
        f"Run {r['run_id']} · {r['posts']} posts from the last prompt evaluation, "
        "spread from the lowest to the highest rule score", "",
        f"**Verdict: {'switch' if r['verdict']['switch'] else 'keep the current model'}.** "
        f"{r['verdict']['reason']}", "",
        "| Model | Scored | Mean score | Agrees with rule checks | Cost per score | Time per score |",
        "| --- | --- | --- | --- | --- | --- |",
        _model_line(f"{r['baseline_label']} (current)", r["baseline"]),
        _model_line(f"{r['candidate_label']} (candidate)", r["candidate"]),
        "",
    ]
    if b.get("paired"):
        lines.append(f"Between the two, on {b['paired']} posts both scored: an average gap "
                     f"of {b['mean_gap']} points (largest {b['max_gap']}), and the same order "
                     f"for {b['pair_agreement']:.0%} of post pairs."
                     + (" A new prompt is meant to change the scores, so a large gap is "
                        "expected; what counts is agreement with the rule checks."
                        if r.get("mode") == "prompt" else ""))
    lines += ["", "\"Agrees with rule checks\" is the share of post pairs a model ranks the "
              "same way as the rule-based quality score from `evals/scoring.py`: whether it "
              "tells better-written posts from worse ones.", "",
              (f"Thresholds for switching to a new prompt: agreement with the rule checks "
               f"of at least {MIN_RULE_AGREEMENT:.0%}, at least "
               f"{MIN_RULE_GAIN * 100:.0f} points above the current version, and no more "
               "failures." if r.get("mode") == "prompt" else
               f"Thresholds for switching models: average gap at most {MAX_MEAN_GAP:g} "
               f"points, the same order for at least {MIN_PAIR_AGREEMENT:.0%} of pairs, "
               "and no more failures than the current model.")]
    return "\n".join(lines) + "\n"


def write_results(report: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "predictor_latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "predictor_latest.md").write_text(render_markdown(report), encoding="utf-8")


def print_summary(r: dict) -> None:
    print()
    header = f"{'Model':<24}{'Scored':>8}{'Mean':>7}{'Agrees':>8}{'Cost/score':>12}{'Time':>7}"
    print(header)
    print("-" * len(header))
    for name, s in ((r["baseline_label"], r["baseline"]), (r["candidate_label"], r["candidate"])):
        if "mean_score" not in s:
            print(f"{name:<24}{s['scored']:>8}")
            continue
        print(f"{name:<24}{s['scored']:>8}{s['mean_score']:>7}{s['agrees_with_rules']:>8.0%}"
              f"{'$' + format(s['cost_per_score_usd'], '.5f'):>12}"
              f"{s['latency_ms_mean'] / 1000:>6.1f}s")
    b = r["between"]
    if b.get("paired"):
        print(f"\nBetween the two: average gap {b['mean_gap']} points (largest {b['max_gap']}), "
              f"same order for {b['pair_agreement']:.0%} of post pairs.")
        if r.get("mode") == "prompt":
            print("(A new prompt is meant to change the scores, so a large gap is expected; "
                  "what counts is agreement with the rule checks.)")
    print("Agrees = how often a model ranks a pair of posts the same way as the rule checks.")
    print(f"\nVerdict: {'SWITCH' if r['verdict']['switch'] else 'KEEP CURRENT'}. "
          f"{r['verdict']['reason']}")
    print(f"Report: {RESULTS_DIR / 'predictor_latest.md'}")


# ── CLI ─────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare predictor models on the same posts.")
    parser.add_argument("--posts", type=int, default=10,
                        help="How many posts to score with each model (default 10)")
    parser.add_argument("--candidate",
                        help=f"Model to test (default {DEFAULT_CANDIDATE}, or the current "
                             "model when --candidate-version is given)")
    parser.add_argument("--baseline", help="Model to compare against (default GROQ_MODEL)")
    parser.add_argument("--candidate-version", type=int,
                        help="Predictor prompt version to test")
    parser.add_argument("--baseline-version", type=int,
                        help="Predictor prompt version to compare against (default: active)")
    parser.add_argument("--ignore-budget", action="store_true",
                        help="Run even if the daily budget is used up")
    args = parser.parse_args(argv)

    latest = RESULTS_DIR / "latest.json"
    if not latest.exists():
        print("No prompt evaluation results found. Run `python -m evals.run_eval` first; "
              "this comparison scores the posts it generated.")
        return 2

    run_eval._quiet_logs()
    run_eval.prepare_database()
    notice = run_eval._RateLimitNotice()
    for name in ("openai._base_client", "services.llm_service"):
        logging.getLogger(name).addFilter(notice)
    logging.getLogger("openai._base_client").setLevel(logging.INFO)

    from utils.config import GROQ_MODEL, PREDICTOR_MODEL
    baseline = args.baseline or PREDICTOR_MODEL or GROQ_MODEL
    if args.candidate:
        candidate = args.candidate
    elif args.candidate_version:
        candidate = baseline
    else:
        candidate = DEFAULT_CANDIDATE
    base_version = args.baseline_version or active_predictor_version()
    cand_version = args.candidate_version or base_version

    from services.llm_service import _prompt_version
    for v in {base_version, cand_version}:
        try:
            _prompt_version(None, "predictor", v)
        except ValueError:
            print(f"There is no predictor prompt version {v}. Start the API once so new "
                  "versions are added to the database.")
            return 2
    if (baseline, base_version) == (candidate, cand_version):
        print(f"Both sides are the same: {_label(baseline, base_version)}.")
        return 2

    posts = pick_posts(json.loads(latest.read_text(encoding="utf-8")), max(2, args.posts))
    print(f"Scoring {len(posts)} posts with {_label(baseline, base_version)} (current) and "
          f"{_label(candidate, cand_version)} (candidate): {len(posts) * 2} Groq calls.")
    print("Groq's free tier limits tokens per minute, so expect pauses between calls.\n")
    try:
        report = run(posts, baseline, candidate, ignore_budget=args.ignore_budget,
                     baseline_version=base_version, candidate_version=cand_version)
    except run_eval.BudgetReached as exc:
        print(f"\nStopped: {exc}")
        return 1
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

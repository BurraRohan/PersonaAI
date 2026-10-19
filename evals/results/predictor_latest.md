# Predictor comparison

Run 20261019-092425 · 10 posts from the last prompt evaluation, spread from the lowest to the highest rule score

**Verdict: switch.** v4 agreed with the writing rules on 66% of post pairs, up from 38% for v2. Activate it with POST /prompts/predictor/rollback/4.

| Model | Scored | Mean score | Agrees with rule checks | Cost per score | Time per score |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-120b v2 (current) | 10/10 | 64.1 | 38% | $0.00061 | 10.5s |
| gpt-oss-120b v4 (candidate) | 10/10 | 67.4 | 66% | $0.00065 | 12.1s |

Between the two, on 10 posts both scored: an average gap of 10.1 points (largest 22), and the same order for 57% of post pairs. A new prompt is meant to change the scores, so a large gap is expected; what counts is agreement with the rule checks.

"Agrees with rule checks" is the share of post pairs a model ranks the same way as the rule-based quality score from `evals/scoring.py`: whether it tells better-written posts from worse ones.

Thresholds for switching to a new prompt: agreement with the rule checks of at least 65%, at least 15 points above the current version, and no more failures.

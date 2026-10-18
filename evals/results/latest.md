# Prompt evaluation

Run 20261018-065300 · model `openai/gpt-oss-120b` · 8 cases · total cost $0.0458

**Recommendation: v5.** v5 scored 97 at $0.00171 per post, with no failures. Other versions scored within 2 points at about the same cost; the higher score wins. It is already the active version.

| Version | Quality | Lowest | Spread | Succeeded | Words | Tokens/post | Cost/post |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 38.2 | 6 | 18.9 | 8/8 | 188 | 1496 | $0.00076 |
| v2 | 63.1 | 28 | 15.7 | 8/8 | 140 | 1827 | $0.00071 |
| v3 | 81.2 | 73 | 5.4 | 8/8 | 144 | 2397 | $0.00085 |
| v4 | 96.8 | 89 | 4.5 | 8/8 | 154 | 3811 | $0.00170 |
| v5 (active) | 97 | 92 | 2.4 | 8/8 | 161 | 4066 | $0.00171 |

Quality is the mean rule-check score with failed generations counted as 0.

## Most common problems

Share of each version's posts that broke the rule at least once.

- **v1**: similar neighbours 100%, invented specific 88%, emoji opener 88%, list of three 88%, em dash 50%
- **v2**: invented specific 100%, similar neighbours 88%, flat rhythm 75%, no short sentence 75%, single block 38%
- **v3**: no short sentence 100%, flat rhythm 88%, similar neighbours 88%, list of three 38%
- **v4**: similar neighbours 25%, list of three 25%, negative parallelism 12%, em dash 12%
- **v5**: list of three 62%, similar neighbours 38%

## How quality is scored

Each post starts at 100 and loses points for breaking a rule the content prompt states: em dashes, banned AI-tell words, invented numbers or dates, no paragraph breaks, flat sentence rhythm, emoji and hashtag limits. The checks are in `evals/scoring.py`.

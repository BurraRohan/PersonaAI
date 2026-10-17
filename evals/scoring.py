"""
Rule-based scoring for generated LinkedIn posts.

Every check mirrors a rule the content prompt states outright (see the v5
template in main.py): no em dashes, no listed AI-tell words, no invented
numbers or dates, real paragraphs, sentence lengths that vary, 3-5 hashtags,
two or three emoji at most. Scoring is plain Python with no model involved,
so the same post always gets the same score and every point lost can be
traced to a named check.

A post starts at 100 and loses points per violation. The result is clamped to
0-100. Weights reflect how badly each problem reads to a human: a single block
of text or an invented metric costs more than one stray emoji.
"""

import re
import statistics
from dataclasses import dataclass, field

# Points lost per occurrence (or once, for structural checks).
WEIGHTS = {
    "em_dash": 5,
    "banned_word": 5,
    "negative_parallelism": 8,
    "promotional_phrase": 8,
    "engagement_bait": 5,
    "vague_authority": 5,
    "tacked_on_ing": 4,
    "list_of_three": 3,
    "markdown": 5,
    "numbered_emoji_list": 5,
    "invented_specific": 8,
    "single_block": 15,
    "flat_rhythm": 10,
    "similar_neighbours": 2,      # per pair, capped below
    "no_short_sentence": 5,
    "too_many_emoji": 5,
    "emoji_opener": 5,
    "question_opener": 5,
    "hashtag_count": 5,
    "filler_hashtag": 3,
    "too_short": 10,
    "too_long": 5,
}
SIMILAR_NEIGHBOURS_CAP = 10

_I = re.IGNORECASE

BANNED_WORDS = re.compile(
    r"\b(delv\w*|leverag\w*|robust\w*|seamless\w*|landscapes?|testament|elevat\w*|"
    r"unlock\w*|harness\w*|navigat\w*|realms?|tapestr\w*|crucial\w*|pivotal\w*|"
    r"underscor\w*|showcas\w*|foster\w*|intricat\w*|vibrant\w*|game[- ]changer\w*|"
    r"dive deep|deep dive)\b", _I)

NEGATIVE_PARALLELISM = re.compile(
    r"\bnot (?:just|only|merely)\b[^.!?\n]{1,80}?\b(?:it'?s|it is|but)\b", _I)

PROMOTIONAL = re.compile(
    r"in today'?s (?:fast[- ]paced )?world|in today'?s fast[- ]paced|in an era (?:where|of)|"
    r"as we navigate|looking ahead|the road to|ever[- ]evolving|evolving landscape|"
    r"at the end of the day|buckle up|let that sink in", _I)

ENGAGEMENT_BAIT = re.compile(
    r"what do you think\??|would love (?:your|to hear) (?:thoughts|from you)|"
    r"\bthoughts\?|let me know in the comments|\bagree\?", _I)

VAGUE_AUTHORITY = re.compile(
    r"experts (?:agree|say)|studies (?:show|suggest)|research (?:shows|suggests)|"
    r"industry leaders", _I)

TACKED_ON_ING = re.compile(
    r",\s+(?:highlighting|ensuring|reflecting|underscoring|showcasing|emphasi[sz]ing|"
    r"demonstrating|signaling|signalling|paving)\b", _I)

LIST_OF_THREE = re.compile(
    r"\b[\w'-]+(?: [\w'-]+){0,2}, [\w'-]+(?: [\w'-]+){0,2},? (?:and|or) [\w'-]+", _I)

MARKDOWN = re.compile(r"\*\*|```|^\s*(?:[-*•]|#{1,6})\s", re.MULTILINE)

NUMBER = re.compile(r"\d[\d,.:/]*%?(?![\uFE0F\u20E3])")  # not keycap emoji like 1️⃣
WEEKDAY = re.compile(r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b", _I)

EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF☀-➿⭐⭕⃣]")

FILLER_HASHTAGS = {"motivation", "success", "inspiration", "hustle", "mindset",
                   "mondaymotivation", "goals"}

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class PostScore:
    score: int
    words: int
    paragraphs: int
    sentences: int
    emoji: int
    hashtags: int
    rhythm_variation: float
    counts: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score, "words": self.words, "paragraphs": self.paragraphs,
            "sentences": self.sentences, "emoji": self.emoji, "hashtags": self.hashtags,
            "rhythm_variation": self.rhythm_variation, "counts": self.counts,
            "issues": self.issues,
        }


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_END.split(text)]
    return [p for p in parts if re.search(r"[A-Za-z]", p)]


def _invented_specifics(text: str, topic: str) -> list[str]:
    """Numbers, times and weekdays in the post that the topic did not supply."""
    allowed = {n.strip(".,:") for n in NUMBER.findall(topic)}
    found = []
    for n in NUMBER.findall(text):
        cleaned = n.strip(".,:")
        if cleaned and cleaned not in allowed:
            found.append(cleaned)
    topic_days = {d.lower() for d in WEEKDAY.findall(topic)}
    found += [d for d in WEEKDAY.findall(text) if d.lower() not in topic_days]
    return found


def score_post(post: str, hashtags: list, topic: str) -> PostScore:
    """Score one post against the content prompt's own rules."""
    post = post or ""
    hashtags = [str(h) for h in (hashtags or [])]
    counts: dict = {}
    issues: list = []

    def hit(check: str, n: int, detail: str = ""):
        if n:
            counts[check] = counts.get(check, 0) + n
            issues.append(f"{check.replace('_', ' ')}: {detail or n}")

    # ── Phrases the prompt bans ──
    hit("em_dash", post.count("—"), f"{post.count(chr(0x2014))} em dash(es)")
    words_found = [m.group(0).lower() for m in BANNED_WORDS.finditer(post)]
    hit("banned_word", len(words_found), ", ".join(sorted(set(words_found))))
    hit("negative_parallelism", len(NEGATIVE_PARALLELISM.findall(post)))
    promo = [m.group(0) for m in PROMOTIONAL.finditer(post)]
    hit("promotional_phrase", len(promo), ", ".join(promo))
    bait = [m.group(0) for m in ENGAGEMENT_BAIT.finditer(post)]
    hit("engagement_bait", len(bait), ", ".join(bait))
    hit("vague_authority", len(VAGUE_AUTHORITY.findall(post)))
    hit("tacked_on_ing", len(TACKED_ON_ING.findall(post)))
    hit("list_of_three", len(LIST_OF_THREE.findall(post)))
    hit("markdown", len(MARKDOWN.findall(post)), "bold, bullets, headings or code fences")
    if "\u20E3" in post:
        hit("numbered_emoji_list", 1, "numbered emoji such as 1\ufe0f\u20e3")

    invented = _invented_specifics(post, topic)
    hit("invented_specific", len(invented), ", ".join(invented))

    # ── Structure ──
    paragraphs = [p for p in re.split(r"\n\s*\n", post.strip()) if p.strip()]
    if len(paragraphs) < 2:
        hit("single_block", 1, "no blank line between paragraphs")

    sentences = _sentences(post)
    lengths = [len(s.split()) for s in sentences]
    variation = 0.0
    if len(lengths) >= 2 and statistics.mean(lengths) > 0:
        variation = statistics.pstdev(lengths) / statistics.mean(lengths)
    if len(lengths) >= 3 and variation < 0.35:
        hit("flat_rhythm", 1, f"sentence lengths barely vary ({variation:.2f})")
    similar = sum(1 for a, b in zip(lengths, lengths[1:]) if abs(a - b) <= 2)
    if similar:
        counts["similar_neighbours"] = similar
        issues.append(f"similar neighbours: {similar} pair(s) of near-equal sentences")
    if lengths and min(lengths) > 5:
        hit("no_short_sentence", 1, "no sentence of five words or fewer")

    emoji = len(EMOJI.findall(post))
    if emoji > 3:
        hit("too_many_emoji", 1, f"{emoji} emoji")
    if post.strip() and EMOJI.match(post.strip()):
        hit("emoji_opener", 1, "post opens with an emoji")
    if sentences and sentences[0].rstrip().endswith("?"):
        hit("question_opener", 1, "opens with a question")

    if not 3 <= len(hashtags) <= 5:
        hit("hashtag_count", 1, f"{len(hashtags)} hashtags")
    filler = [h for h in hashtags if h.lstrip("#").lower() in FILLER_HASHTAGS]
    hit("filler_hashtag", len(filler), ", ".join(filler))

    word_count = len(post.split())
    if word_count < 50:
        hit("too_short", 1, f"{word_count} words")
    elif word_count > 350:
        hit("too_long", 1, f"{word_count} words")

    # ── Total ──
    lost = 0
    for check, n in counts.items():
        if check == "similar_neighbours":
            lost += min(n * WEIGHTS[check], SIMILAR_NEIGHBOURS_CAP)
        elif check in {"numbered_emoji_list", "single_block", "flat_rhythm", "no_short_sentence", "too_many_emoji",
                       "emoji_opener", "question_opener", "hashtag_count", "too_short",
                       "too_long"}:
            lost += WEIGHTS[check]
        else:
            lost += n * WEIGHTS[check]

    return PostScore(
        score=max(0, min(100, 100 - lost)),
        words=word_count,
        paragraphs=len(paragraphs),
        sentences=len(sentences),
        emoji=emoji,
        hashtags=len(hashtags),
        rhythm_variation=round(variation, 2),
        counts=counts,
        issues=issues,
    )

"""
PersonaAI – Personal Branding Intelligence Agent
FastAPI application entry-point.

Integrates: LangChain agents, LangGraph orchestration, Prometheus metrics,
API key auth, rate limiting, prompt versioning, and audit logging.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from database.db import engine, get_db, Base
from typing import Annotated, Optional
from fastapi import Path, Query

# Path and query parameters are bounded so an out-of-range id is a 422, not an
# integer overflow inside SQLite that surfaces as a 500.
PathID = Annotated[int, Path(ge=1, le=2_147_483_647)]
AgentName = Annotated[str, Path(min_length=1, max_length=50, pattern=r"^[a-z_]+$")]
from database.models import Post, Engagement, PromptTemplate
from schemas.schemas import (
    BrandRequest, BrandResponse,
    GenerateRequest, GenerateResponse,
    EngagementRequest, EngagementResponse,
    FeedbackRequest, FeedbackResponse,
    OrchestrateRequest, OrchestrateResponse,
    PromptTemplateResponse,
)
from agents.brand_agent import create_brand_profile
from agents.content_agent import generate_post
from agents.feedback_agent import get_feedback
from agents.orchestrator import run_full_workflow
from utils.auth import verify_admin_key, verify_api_key
from utils.budget import budget_status, enforce_budget
from services.prediction_cache import stats as prediction_cache_stats
from utils.rate_limiter import limiter
from utils.observability import setup_logging, setup_prometheus

from schemas.schemas import PredictRequest, PredictResponse, ReviewRequest, PostEditRequest, BrandEditRequest
from services.llm_service import predict_engagement
from database.models import BrandProfile

# ── Logging ─────────────────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ── Seed Default Prompts ────────────────────────────────────────

def seed_default_prompts(db: Session):
    """Insert any default prompt template that is not already in the database.

    Checked per (agent_name, version) rather than "is the table empty", so a new
    prompt version added in code reaches an existing database on next start.
    Templates already present are never overwritten — a version that has been
    rolled back stays rolled back.
    """
    defaults = [
        PromptTemplate(
            agent_name="brand",
            version=1,
            template="""You are a personal-branding strategist.
Given the following information about a professional, create a structured
LinkedIn personal brand profile. Return ONLY valid JSON with these keys:

- "tone": a 1-2 sentence description of the ideal posting tone
- "content_themes": a list of 4-6 content theme strings
- "positioning_summary": a 2-3 sentence positioning statement
- "do_guidelines": a list of 4-5 things this person SHOULD do on LinkedIn
- "dont_guidelines": a list of 4-5 things this person should AVOID on LinkedIn

Professional details:
  Name: {name}
  Role: {role}
  Industry: {industry}
  Goals: {goals}
  Preferred Tone: {preferred_tone}

Return ONLY the JSON object, no extra text.""",
            is_active=True,
            description="Default brand profile generation prompt v1",
        ),
        PromptTemplate(
            agent_name="content",
            version=1,
            template="""You are a LinkedIn content creator.
Using the brand profile below, write ONE engaging LinkedIn post about the
given topic. Return ONLY valid JSON with these keys:

- "post_content": the full post text (use line breaks, emojis where appropriate, 150-300 words)
- "suggested_hashtags": a list of 3-5 relevant hashtags (without #)

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}
  Do: {do_guidelines}
  Don't: {dont_guidelines}

Topic: {topic}

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v1 — asked for an 'engaging' post, produced influencer-voice output",
        ),
        PromptTemplate(
            agent_name="content",
            version=2,
            template="""Write a LinkedIn post the way a real person would: someone
who actually did the thing and is telling a colleague about it.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}
  Do: {do_guidelines}
  Don't: {dont_guidelines}

Topic: {topic}

WHAT YOU MAY AND MAY NOT INVENT

Write only from the topic and the brand profile above. You were not there and
you do not have the logs, so do not invent specifics you were not given:

- No timestamps, dates or times ("at 14:32 UTC yesterday").
- No metrics, percentages or counts that were not supplied.
- No log lines, stack traces, error strings or code blocks.
- No named tools, versions or systems that are not in the topic or profile.
- No quoted dialogue, and no colleagues, customers or teams by name.

Concrete does not mean fabricated. Write about the shape of the problem, what
made it hard, and what the reader should take from it. A post that says "this
kind of bug only appears when two sessions race" is specific and honest. A post
that says "at 14:32 UTC our partner app threw a 500" is a fabrication unless
you were told so.

HOW TO WRITE IT

Open with the substance, not a wind-up. Lead with the actual claim, the
counterintuitive part, or the mistake — whatever the reader most needs to hear
first. Never open with a rhetorical question, never with "In today's fast-paced
world", never with the topic restated as a headline.

Vary your sentence length. Short one. Then a longer one that takes its time and
earns the space it uses. That unevenness is most of what makes writing sound
human.

Have a point of view. Say what you think, admit what you got wrong, mention the
part that still bothers you. A post with no opinion in it reads like a press
release.

Use "I". First person is not unprofessional, it is honest.

Let the length follow the idea. If the point lands in 90 words, stop at 90.
Padding to hit a word count is the most obvious tell there is.

DO NOT WRITE LIKE AN LLM

These are the specific patterns that make a post recognisably machine-written.
Avoid every one of them:

- Em dashes. Use commas, full stops, or brackets instead.
- "It's not just X, it's Y" and "Not only... but also". Just say the thing.
- Groups of three. Two examples, or four, or one. Not always three.
- These words: delve, leverage, robust, seamless, landscape, testament,
  elevate, unlock, harness, navigate, realm, tapestry, crucial, pivotal,
  underscore, showcase, foster, intricate, vibrant, game-changer, dive deep.
- "-ing" clauses tacked on the end to sound thoughtful: "...highlighting the
  importance of", "...ensuring maximum impact", "...reflecting a broader shift".
- Vague authority: "experts agree", "studies show", "industry leaders say".
  Either name the source or drop the claim.
- Bold text for emphasis, and bullet lists where every item starts with a
  bolded label and a colon.
- Grand closing statements about what this means for the future of the industry.
- Ending with a generic engagement bait question like "What do you think?" or
  "Would love to hear your thoughts". If you ask something, ask something only
  someone who read the post could answer.

EMOJI: use them where they fit the brand's tone, the way the person themselves
would. Two or three across a post is normal on LinkedIn. Each one should carry
meaning — a specific object, a reaction — rather than sit there as decoration.
Do not open the post with one, and do not put one at the start of every line.

HASHTAGS: 3-5, lowercase in the JSON, no marketing filler like
#motivation or #success.

Return ONLY valid JSON with these keys:
- "post_content": the post text as plain prose, with a blank line between
  paragraphs. No markdown, no code fences, no bold, no bullet characters.
- "suggested_hashtags": a list of 3-5 hashtags (without the # symbol)

Before you return it, reread the post and ask: does this sound like a person,
or does it sound like a language model imitating a person? If it is the second
one, rewrite it.

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v2 — natural voice, bans LLM writing tells",
        ),
        PromptTemplate(
            agent_name="content",
            version=3,
            template="""Write a LinkedIn post the way a real person would: someone
who actually did the thing and is telling a colleague about it.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}
  Do: {do_guidelines}
  Don't: {dont_guidelines}

Topic: {topic}

WHAT YOU MAY AND MAY NOT INVENT

This is the most important rule here, and it overrides every other instruction
below. You were not there. You have no logs, no metrics, no incident, and no
colleagues. Everything you know is the topic line and the brand profile.

Never write any of the following unless it appears verbatim in the topic:

- A timestamp, date, or time of day. Not "14:32 UTC", not "last Tuesday",
  not "at 2am".
- A number describing something that happened: latency figures, error rates,
  percentages, throughput, revenue, how many users were affected.
- A log line, stack trace, error message, query, or block of code.
- A named library, tool, version, or internal service.
- A person: no colleagues, no "shout out to", no quoted conversation.
- A specific incident at all. No outage, no alert firing, no pager, no
  debugging session, no "we shipped a fix".

If the topic is phrased as personal experience ("why I stopped doing X"), write
the reasoning that leads someone to that position. Do not manufacture the
experience that produced it.

Write about the pattern, not an event. This is honest and still specific:

  "Time-based IDs look unique until two nodes generate one in the same
  millisecond. Clock skew makes that a matter of when, not if."

This is fabrication, and it is the single worst thing you can do here:

  "At 02:13 UTC our order service threw a duplicate-key error and throughput
  rose 12% after we switched to ULID."

Before returning the post, reread it and delete every sentence containing a
detail you were not given. If that leaves the post shorter, good.

HOW TO WRITE IT

Open with the substance, not a wind-up. Lead with the actual claim, the
counterintuitive part, or the mistake — whatever the reader most needs to hear
first. Never open with a rhetorical question, never with "In today's fast-paced
world", never with the topic restated as a headline.

Vary your sentence length. Short one. Then a longer one that takes its time and
earns the space it uses. That unevenness is most of what makes writing sound
human.

Have a point of view. Say what you think, admit what you got wrong, mention the
part that still bothers you. A post with no opinion in it reads like a press
release.

Use "I". First person is not unprofessional, it is honest.

Let the length follow the idea. If the point lands in 90 words, stop at 90.
Padding to hit a word count is the most obvious tell there is.

DO NOT WRITE LIKE AN LLM

These are the specific patterns that make a post recognisably machine-written.
Avoid every one of them:

- Em dashes. Use commas, full stops, or brackets instead.
- "It's not just X, it's Y" and "Not only... but also". Just say the thing.
- Groups of three. Two examples, or four, or one. Not always three.
- These words: delve, leverage, robust, seamless, landscape, testament,
  elevate, unlock, harness, navigate, realm, tapestry, crucial, pivotal,
  underscore, showcase, foster, intricate, vibrant, game-changer, dive deep.
- "-ing" clauses tacked on the end to sound thoughtful: "...highlighting the
  importance of", "...ensuring maximum impact", "...reflecting a broader shift".
- Vague authority: "experts agree", "studies show", "industry leaders say".
  Either name the source or drop the claim.
- Bold text for emphasis, and bullet lists where every item starts with a
  bolded label and a colon.
- Grand closing statements about what this means for the future of the industry.
- Ending with a generic engagement bait question like "What do you think?" or
  "Would love to hear your thoughts". If you ask something, ask something only
  someone who read the post could answer.

EMOJI: use them where they fit the brand's tone, the way the person themselves
would. Two or three across a post is normal on LinkedIn. Each one should carry
meaning — a specific object, a reaction — rather than sit there as decoration.
Do not open the post with one, and do not put one at the start of every line.

HASHTAGS: 3-5, lowercase in the JSON, no marketing filler like
#motivation or #success.

Return ONLY valid JSON with these keys:
- "post_content": the post text as plain prose, with a blank line between
  paragraphs. No markdown, no code fences, no bold, no bullet characters.
- "suggested_hashtags": a list of 3-5 hashtags (without the # symbol)

Before you return it, reread the post and ask: does this sound like a person,
or does it sound like a language model imitating a person? If it is the second
one, rewrite it.

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v3 — anti-fabrication rules, emoji guidance relaxed",
        ),
        PromptTemplate(
            agent_name="content",
            version=4,
            template="""Write a LinkedIn post the way a real person would: someone who has thought hard
about this and is telling a colleague what they concluded.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}
  Do: {do_guidelines}
  Don't: {dont_guidelines}

Topic: {topic}

RHYTHM — this is what separates writing that breathes from writing that drones.

Sentence lengths must vary sharply. No two sentences in a row may be close to
the same length. Somewhere in the post, write a sentence of five words or
fewer. Somewhere else, let one run long and take its time getting where it is
going, the way people actually talk when they are working something out.

Like this:

  Timestamps feel unique. They are not. Two services writing in the same
  millisecond will happily hand you the same key, and the database will reject
  one of them without ever telling you which write you lost.

Not like this:

  Relying on the wall clock gives the illusion of uniqueness. Clock drift
  between machines means ordering is not guaranteed. Switching to a generated
  identifier decouples uniqueness from time.

The second version is not wrong. It is just flat, because every sentence is the
same length and nothing surprises the reader.

Use paragraph breaks. Two to four short paragraphs, with a blank line between
them. A single block of text is exhausting to read on a phone.

VOICE

Have a position and commit to it. Say what you think, admit what you got wrong,
name the part that still annoys you. Hedging everything reads like a committee
wrote it.

Use "I". Use contractions. Address the reader as "you" when it fits.

Open with the sharpest thing you have — the claim, the mistake, the
counterintuitive part. Never open with a rhetorical question, never with "In
today's fast-paced world", never by restating the topic as a headline.

Let the length follow the idea. If the point lands in 90 words, stop at 90.
Padding to reach a word count is the most obvious tell there is.

WHAT YOU MAY NOT INVENT

You were not there. You have no logs, no metrics, no incident, no colleagues.
Everything you know is the topic line and the brand profile above.

Never write any of the following unless it appears verbatim in the topic:

- A timestamp, date, or time of day. Not "14:32 UTC", not "last Tuesday".
- A number describing something that happened: latency figures, error rates,
  percentages, throughput, revenue, users affected.
- A log line, stack trace, error message, query, or block of code.
- A named library, tool, version, or internal service.
- A person: no colleagues, no "shout out to", no quoted conversation.
- A specific incident: no outage, no alert firing, no pager, no "we shipped
  a fix on Friday".

If the topic is phrased as personal experience ("why I stopped doing X"), write
the reasoning that leads someone to that position. Do not manufacture the
incident that produced it.

This constraint costs you nothing. Vivid writing comes from rhythm, a clear
opinion, and precise language — not from invented specifics. "Two services
writing in the same millisecond will happily hand you the same key" is vivid
and entirely honest.

DO NOT WRITE LIKE AN LLM

- Em dashes. Use commas, full stops, or brackets.
- "It's not just X, it's Y" and "Not only... but also".
- Groups of three. Use two examples, or four, or one.
- These words: delve, leverage, robust, seamless, landscape, testament,
  elevate, unlock, harness, navigate, realm, tapestry, crucial, pivotal,
  underscore, showcase, foster, intricate, vibrant, game-changer, dive deep.
- "-ing" clauses tacked on to sound thoughtful: "...highlighting the importance
  of", "...ensuring maximum impact", "...reflecting a broader shift".
- Vague authority: "experts agree", "studies show", "industry leaders say".
- Bold text, and bullet lists where every item is a bolded label plus a colon.
- Grand closing statements about the future of the industry.
- Generic engagement bait: "What do you think?", "Would love your thoughts".
  If you ask something, ask what only a reader of this post could answer.

EMOJI

Two or three, placed where a person would actually put one: beside the thing it
refers to, or after a line that earns a reaction. Never a row of them at the
end. Never one opening the post. Never one starting every line. If none of them
would land, use none.

OUTPUT

Return ONLY valid JSON with these keys:
- "post_content": plain prose, blank line between paragraphs. No markdown, no
  code fences, no bold, no bullet characters.
- "suggested_hashtags": 3-5 hashtags without the # symbol, no filler like
  motivation or success.

Before returning it, read the post back and check two things. First: does any
sentence contain a detail you were not given? Delete it. Second: are two
consecutive sentences roughly the same length? Rewrite one of them.

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v4 — rhythm and voice rules given equal weight to the fabrication guard",
        ),
        PromptTemplate(
            agent_name="content",
            version=5,
            template="""Write a LinkedIn post the way a real person would: someone who has thought hard
about this and is telling a colleague what they concluded.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}
  Do: {do_guidelines}
  Don't: {dont_guidelines}

Topic: {topic}

RHYTHM — this is what separates writing that breathes from writing that drones.

Sentence lengths must vary sharply. No two sentences in a row may be close to
the same length. Somewhere in the post, write a sentence of five words or
fewer. Somewhere else, let one run long and take its time getting where it is
going, the way people actually talk when they are working something out.

Like this:

  Timestamps feel unique. They are not. Two services writing in the same
  millisecond will happily hand you the same key, and the database will reject
  one of them without ever telling you which write you lost.

Not like this:

  Relying on the wall clock gives the illusion of uniqueness. Clock drift
  between machines means ordering is not guaranteed. Switching to a generated
  identifier decouples uniqueness from time.

The second version is not wrong. It is just flat, because every sentence is the
same length and nothing surprises the reader.

Use paragraph breaks. Two to four short paragraphs, with a blank line between
them. A single block of text is exhausting to read on a phone. This is not
optional — a wall of text is the fastest way to lose the reader.

ENERGY

Be warm. Be a bit blunt. Write like you actually care about this, because a
post with no pulse gets scrolled past no matter how correct it is.

Things that are welcome:

  Direct address. "If you are still doing this, stop."
  Mild exasperation. "This one cost me a weekend."
  Enthusiasm about a fix that genuinely worked.
  A short aside in brackets, the way people actually think.
  Humour, if it fits the brand's tone.

What is NOT energy, and reads as machine-written every time:

  "Looking ahead, the road to tomorrow's X is being paved by..."
  "In an era where...", "As we navigate the evolving landscape of..."
  Numbered emoji lists: 1️⃣ 2️⃣ 3️⃣
  Anything that sounds like a conference brochure.

The difference is simple. Warmth comes from having a real reaction to a real
idea. Promotional register comes from dressing up an ordinary point in
important-sounding language. Do the first, never the second.

VOICE

Have a position and commit to it. Say what you think, admit what you got wrong,
name the part that still annoys you. Hedging everything reads like a committee
wrote it.

Use "I". Use contractions. Address the reader as "you" when it fits.

Open with the sharpest thing you have — the claim, the mistake, the
counterintuitive part. Never open with a rhetorical question, never with "In
today's fast-paced world", never by restating the topic as a headline.

Let the length follow the idea. If the point lands in 90 words, stop at 90.
Padding to reach a word count is the most obvious tell there is.

WHAT YOU MAY NOT INVENT

You were not there. You have no logs, no metrics, no incident, no colleagues.
Everything you know is the topic line and the brand profile above.

Never write any of the following unless it appears verbatim in the topic:

- A timestamp, date, or time of day. Not "14:32 UTC", not "last Tuesday".
- A number describing something that happened: latency figures, error rates,
  percentages, throughput, revenue, users affected.
- A log line, stack trace, error message, query, or block of code.
- A named library, tool, version, or internal service.
- A person: no colleagues, no "shout out to", no quoted conversation.
- A specific incident: no outage, no alert firing, no pager, no "we shipped
  a fix on Friday".
- A result you are claiming: "the change eliminated the collisions", "latency
  stayed the same", "confidence grew", "we saved a week". You do not know what
  happened after. Write what you expect and why, not what you achieved.

If the topic is phrased as personal experience ("why I stopped doing X"), write
the reasoning that leads someone to that position. Do not manufacture the
incident that produced it.

This constraint costs you nothing. Vivid writing comes from rhythm, a clear
opinion, and precise language — not from invented specifics. "Two services
writing in the same millisecond will happily hand you the same key" is vivid
and entirely honest.

DO NOT WRITE LIKE AN LLM

- Em dashes. Use commas, full stops, or brackets.
- "It's not just X, it's Y" and "Not only... but also".
- Groups of three. Use two examples, or four, or one.
- These words: delve, leverage, robust, seamless, landscape, testament,
  elevate, unlock, harness, navigate, realm, tapestry, crucial, pivotal,
  underscore, showcase, foster, intricate, vibrant, game-changer, dive deep.
- "-ing" clauses tacked on to sound thoughtful: "...highlighting the importance
  of", "...ensuring maximum impact", "...reflecting a broader shift".
- Vague authority: "experts agree", "studies show", "industry leaders say".
- Bold text, and bullet lists where every item is a bolded label plus a colon.
- Grand closing statements about the future of the industry.
- Generic engagement bait: "What do you think?", "Would love your thoughts".
  If you ask something, ask what only a reader of this post could answer.

EMOJI

Two or three, placed where a person would actually put one: beside the thing it
refers to, or after a line that earns a reaction. Never a row of them at the
end. Never one opening the post. Never one starting every line. If none of them
would land, use none.

OUTPUT

Return ONLY valid JSON with these keys:
- "post_content": plain prose, blank line between paragraphs. No markdown, no
  code fences, no bold, no bullet characters.
- "suggested_hashtags": 3-5 hashtags without the # symbol, no filler like
  motivation or success.

Before returning it, read the post back and check three things.

First: does any sentence contain a detail you were not given, including a
result you are claiming? Delete it.

Second: are two consecutive sentences roughly the same length? Rewrite one.

Third: is the post a single block? If there is no blank line in it, break it
into paragraphs now. This is the check most often skipped.

Return ONLY the JSON object, no extra text.""",
            is_active=True,
            description="v5 — v4 rhythm plus explicit warmth, promotional register still banned",
        ),
        PromptTemplate(
            agent_name="feedback",
            version=2,
            template="""You are a sharp LinkedIn personal-branding strategist who gives blunt, data-backed advice.

Analyze the engagement history below. Your job is to find patterns, identify what worked and what flopped, and give specific next steps.

Rules:
- Reference exact post topics and their numbers (likes, comments, shares) by name
- Compare the best-performing post against the worst-performing post and explain WHY one worked better
- Every recommendation must tie back to a specific data point from the history
- Do NOT give generic advice like "engage with your audience", "be consistent", "collaborate with others", or "post regularly"
- Instead give concrete next steps like "Your RAG post got 3x more comments than your ML basics post — write a follow-up series breaking down RAG architecture step by step"

Return ONLY valid JSON with these keys:

- "performance_summary": a 3-4 sentence summary comparing posts with exact numbers, identifying the best and worst performers and why
- "improvement_recommendation": a 3-5 sentence recommendation where every sentence references a specific post topic or number from the data

Brand context: {brand_summary}

Engagement History (most recent first):
{engagement_history}

Return ONLY the JSON object, no extra text.""",
            is_active=True,
            description="Feedback prompt v2 - data-driven, no generic advice",
        ),
    ]

    # Predictor v1 — inactive, kept so rollback has a real prior version.
    # Its literal example ranges were copied into model output instead of being
    # read as format hints, and it had no rubric, so scores barely varied.
    defaults.append(
        PromptTemplate(
            agent_name="predictor",
            version=1,
            template="""You are a LinkedIn content performance analyst.
Analyze the draft post below and predict how it will perform.
Use the brand profile and past engagement data for context.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}

Past Performance: {history_context}

Draft Post:
{draft_content}

Return ONLY valid JSON with these keys:
- "overall_score": integer 1-100 (overall predicted performance)
- "predicted_likes": string range like "30-50"
- "predicted_comments": string range like "5-12"
- "predicted_shares": string range like "2-6"
- "brand_alignment": integer 1-100 (how well it matches the brand)
- "hook_strength": integer 1-100 (how strong the opening line is)
- "readability": integer 1-100 (how easy it is to read)
- "call_to_action": integer 1-100 (how well it drives engagement)
- "improvement_tips": string with 3-4 specific actionable tips, separated by newlines

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v1 — anchored on example ranges, no scoring rubric",
        )
    )

    defaults.append(
        PromptTemplate(
            agent_name="predictor",
            version=2,
            template="""You are a demanding LinkedIn content analyst. Most posts
are mediocre, and your scores must reflect that. Grade honestly, not kindly.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}

Past Performance: {history_context}

Draft Post:
{draft_content}

SCORING RUBRIC — apply it strictly. Use the full range.

  90-100  Exceptional. A specific, surprising claim backed by a concrete number
          or a first-hand story nobody else could tell.
  70-89   Strong. Clearly on-brand, has a real insight, but the hook or the
          ending is soft.
  50-69   Average. Competent and readable, but generic — could have been written
          by anyone in this industry.
  30-49   Weak. Vague, obvious, or announcement-style ("excited to share",
          "thoughts on the future of X"). No specific detail.
  1-29    Poor. Off-brand, incoherent, or purely promotional.

A post with no numbers, no story and no concrete claim CANNOT score above 55,
no matter how well written it is. Do not inflate. If it is generic, say so with
the score.

Score each dimension independently — they should rarely all be within 10 points
of each other:
  brand_alignment  How closely the themes and tone match the profile above.
  hook_strength    Judge the FIRST LINE alone. Would it stop a scroll?
  readability      Sentence length, paragraph breaks, jargon density.
  call_to_action   Does the ending invite a reply, or just trail off?

ENGAGEMENT RANGES — derive these from the past-performance figures above.
If the history shows an average, scale it by how this post scores: a 40-scoring
post should land well below that average, a 90-scoring post well above it. If
there is no history, base the ranges on a small account (single-digit to low
double-digit likes) rather than assuming reach. Never reuse a range from an
example; compute it from the numbers you were given.

The four dimension scores are what matter most — the overall score is derived
from them, so grade each dimension carefully and independently.

Return ONLY valid JSON with these keys:
- "overall_score": integer 1-100, your own overall read (recalculated from the
  dimensions afterwards, so do not agonise over it)
- "predicted_likes": string range, e.g. "<low>-<high>"
- "predicted_comments": string range
- "predicted_shares": string range
- "brand_alignment": integer 1-100
- "hook_strength": integer 1-100
- "readability": integer 1-100
- "call_to_action": integer 1-100
- "improvement_tips": string, 3-4 specific tips separated by newlines, each
  naming something in THIS post rather than giving generic advice

Return ONLY the JSON object, no extra text.""",
            is_active=True,
            description="v2 — explicit rubric, per-dimension criteria, history-derived ranges",
        )
    )

    # v3 judges posts by the standards the content prompt writes to. v2 capped
    # any post without numbers or a first-hand story at 55, but the content
    # prompt forbids inventing either, so honest posts were marked down and
    # posts with made-up statistics were rewarded. Seeded inactive: activate it
    # with POST /prompts/predictor/rollback/3 once evals/compare_predictor.py
    # shows it agrees better with the writing rules.
    defaults.append(
        PromptTemplate(
            agent_name="predictor",
            version=3,
            template="""You are a demanding LinkedIn content analyst predicting how real
readers will respond to a draft. Most posts are mediocre, and your scores must
reflect that. Grade honestly, not kindly. Use the full range.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}

Past Performance: {history_context}

Draft Post:
{draft_content}

WHAT THE WRITER KNEW

This draft was written from a topic line and the brand profile above, nothing
else. Any precise figure, date, time, named tool, colleague or claimed result
that does not appear in the brand profile was almost certainly invented.
Readers who notice an invented detail stop trusting the whole post. Never
reward such details; treat them as a credibility risk. A post earns a high
score through a sharp claim, clear reasoning and a voice that sounds like a
person, not through figures.

RUBRIC

  90-100  Exceptional. A specific, arguable position, reasoned clearly, in a
          voice that could only be this person. The reader learns something.
  70-89   Strong. On-brand with a real point, but the opening or the ending
          is softer than it should be.
  50-69   Average. Competent and readable, but generic: anyone in this
          industry could have written it.
  30-49   Weak. Vague, obvious, announcement-style ("excited to share"), or
          written in promotional register.
  1-29    Poor. Off-brand, incoherent, or a sales pitch.

DIMENSIONS — score each independently; they should rarely all sit within 10
points of each other.

  hook_strength  The FIRST LINE alone. Strong: it opens with the claim, the
                 mistake or the counterintuitive point. Weak: an emoji opener,
                 a rhetorical question, "In today's fast-paced world", hype, or
                 the topic restated as a headline. A hook built on an invented
                 statistic is weak, however eye-catching.

  call_to_action Does the ending give a reader a reason to respond? Strong: a
                 question only a reader of THIS post could answer, or a clear
                 position to agree or argue with. Generic bait ("What do you
                 think?", "Thoughts?", "Agree?", "Drop a comment below") scores
                 40 or lower. A post that ends on a strong claim without any
                 question can still score 60-70.

  readability    Sentence lengths that vary, real paragraph breaks, plain
                 words. Walls of text, numbered emoji lists, bold labels and
                 stacked buzzwords all hurt.

  brand_alignment How closely the themes and tone match the profile above.

Lower the relevant score for writing that reads as machine-made, because
readers scroll past it: em dashes used for drama, "not just X, it's Y",
reflexive groups of three, and words like leverage, seamless, robust, unlock,
elevate, game-changer or landscape.

ENGAGEMENT RANGES — derive these from the past-performance figures above.
If the history shows an average, scale it by how this post scores: a 40-scoring
post should land well below that average, a 90-scoring post well above it. If
there is no history, base the ranges on a small account (single-digit to low
double-digit likes) rather than assuming reach. Never reuse a range from an
example; compute it from the numbers you were given.

IMPROVEMENT TIPS — name something in THIS post. Never suggest adding
statistics, results or stories the writer was not given, and never suggest
adding emoji or a generic closing question.

Return ONLY valid JSON with these keys:
- "overall_score": integer 1-100, your own overall read (recalculated from the
  dimensions afterwards, so do not agonise over it)
- "predicted_likes": string range, e.g. "<low>-<high>"
- "predicted_comments": string range
- "predicted_shares": string range
- "brand_alignment": integer 1-100
- "hook_strength": integer 1-100
- "readability": integer 1-100
- "call_to_action": integer 1-100
- "improvement_tips": string, 3-4 specific tips separated by newlines

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v3 — judges hook and call-to-action by the content prompt's standards; invented specifics and engagement bait no longer rewarded",
        )
    )

    # v4 is v3 with the draft moved to the end, between explicit markers. In v3
    # the draft sat directly above a section heading, and the model sometimes
    # read the heading as the post's closing line, then scored and quoted it.
    defaults.append(
        PromptTemplate(
            agent_name="predictor",
            version=4,
            template="""You are a demanding LinkedIn content analyst predicting how real
readers will respond to a draft. Most posts are mediocre, and your scores must
reflect that. Grade honestly, not kindly. Use the full range. The draft
itself comes last, between two marker lines.

Brand Profile:
  Tone: {tone}
  Content Themes: {content_themes}
  Positioning: {positioning}

Past Performance: {history_context}

WHAT THE WRITER KNEW

This draft was written from a topic line and the brand profile above, nothing
else. Any precise figure, date, time, named tool, colleague or claimed result
that does not appear in the brand profile was almost certainly invented.
Readers who notice an invented detail stop trusting the whole post. Never
reward such details; treat them as a credibility risk. A post earns a high
score through a sharp claim, clear reasoning and a voice that sounds like a
person, not through figures.

RUBRIC

  90-100  Exceptional. A specific, arguable position, reasoned clearly, in a
          voice that could only be this person. The reader learns something.
  70-89   Strong. On-brand with a real point, but the opening or the ending
          is softer than it should be.
  50-69   Average. Competent and readable, but generic: anyone in this
          industry could have written it.
  30-49   Weak. Vague, obvious, announcement-style ("excited to share"), or
          written in promotional register.
  1-29    Poor. Off-brand, incoherent, or a sales pitch.

DIMENSIONS — score each independently; they should rarely all sit within 10
points of each other.

  hook_strength  The FIRST LINE alone. Strong: it opens with the claim, the
                 mistake or the counterintuitive point. Weak: an emoji opener,
                 a rhetorical question, "In today's fast-paced world", hype, or
                 the topic restated as a headline. A hook built on an invented
                 statistic is weak, however eye-catching.

  call_to_action Judge the post's own last lines, inside the markers. Does
                 the ending give a reader a reason to respond? Strong: a
                 question only a reader of THIS post could answer, or a clear
                 position to agree or argue with. Generic bait ("What do you
                 think?", "Thoughts?", "Agree?", "Drop a comment below") scores
                 40 or lower. A post that ends on a strong claim without any
                 question can still score 60-70.

  readability    Sentence lengths that vary, real paragraph breaks, plain
                 words. Walls of text, numbered emoji lists, bold labels and
                 stacked buzzwords all hurt.

  brand_alignment How closely the themes and tone match the profile above.

Lower the relevant score for writing that reads as machine-made, because
readers scroll past it: em dashes used for drama, "not just X, it's Y",
reflexive groups of three, and words like leverage, seamless, robust, unlock,
elevate, game-changer or landscape.

ENGAGEMENT RANGES — derive these from the past-performance figures above.
If the history shows an average, scale it by how this post scores: a 40-scoring
post should land well below that average, a 90-scoring post well above it. If
there is no history, base the ranges on a small account (single-digit to low
double-digit likes) rather than assuming reach. Never reuse a range from an
example; compute it from the numbers you were given.

IMPROVEMENT TIPS — name something in THIS post. Never suggest adding
statistics, results or stories the writer was not given, and never suggest
adding emoji or a generic closing question.

Return ONLY valid JSON with these keys:
- "overall_score": integer 1-100, your own overall read (recalculated from the
  dimensions afterwards, so do not agonise over it)
- "predicted_likes": string range, e.g. "<low>-<high>"
- "predicted_comments": string range
- "predicted_shares": string range
- "brand_alignment": integer 1-100
- "hook_strength": integer 1-100
- "readability": integer 1-100
- "call_to_action": integer 1-100
- "improvement_tips": string, 3-4 specific tips separated by newlines

THE DRAFT POST

Everything between the two marker lines is the post, and nothing else is. The
headings above are your instructions, not part of the post: never quote them,
score them or mention them in your tips.

=== POST START ===
{draft_content}
=== POST END ===

Return ONLY the JSON object, no extra text.""",
            is_active=False,
            description="v4 — v3 rules with the draft fenced off at the end, so instructions are never read as part of the post",
        )
    )

    existing_keys = {
        (row.agent_name, row.version)
        for row in db.query(PromptTemplate.agent_name, PromptTemplate.version).all()
    }

    added = 0
    for pt in defaults:
        if (pt.agent_name, pt.version) in existing_keys:
            continue

        # A newly seeded active version supersedes the current active one for
        # that agent, so two versions are never active at once.
        if pt.is_active:
            (
                db.query(PromptTemplate)
                .filter(PromptTemplate.agent_name == pt.agent_name,
                        PromptTemplate.is_active == True)  # noqa: E712
                .update({"is_active": False})
            )

        db.add(pt)
        added += 1

    if added:
        db.commit()
        logger.info("Seeded %d new prompt template(s)", added)


# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creating database tables …")
    Base.metadata.create_all(bind=engine)

    # Bring an older database up to date without wiping it.
    from database.db import ensure_schema
    added_columns = ensure_schema()
    if added_columns:
        logger.info("Added missing columns: %s", ", ".join(added_columns))

    # Seed default prompts
    from database.db import SessionLocal
    db = SessionLocal()
    try:
        seed_default_prompts(db)
    finally:
        db.close()

    logger.info("PersonaAI is ready.")
    yield
    logger.info("Shutting down PersonaAI.")


# ── App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="PersonaAI",
    description="LinkedIn Personal Branding Intelligence Agent – with LangChain agents, "
                "LangGraph orchestration, Prometheus metrics, and prompt versioning.",
    version="2.0.0",
    lifespan=lifespan,
)

# Fix 6: Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
# The UI is served by this same app, so a wildcard origin is unnecessary.
# "*" combined with allow_credentials=True is also rejected outright by browsers.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security headers on every response, and a cap on request body size.
from utils.security import SecurityHeadersMiddleware, BodySizeLimitMiddleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

# Tag each request so every LLM call it triggers can be costed together.
from utils.request_context import RequestContextMiddleware
app.add_middleware(RequestContextMiddleware)

# Prometheus metrics (behind the API key unless METRICS_PUBLIC=1)
setup_prometheus(app)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PersonaAI API v2.0.0 – visit /docs for Swagger UI"}


# ── Original Endpoints (with auth + rate limiting) ──────────────

@app.post("/brand", response_model=BrandResponse, tags=["Brand Strategy"],
          dependencies=[Depends(verify_api_key), Depends(enforce_budget)])
@limiter.limit("10/minute")
def create_brand(request: Request, body: BrandRequest, db: Session = Depends(get_db)):
    """Create a structured brand profile using AI."""
    return create_brand_profile(db, body)


@app.post("/generate", response_model=GenerateResponse, tags=["Content Generation"],
          dependencies=[Depends(verify_api_key), Depends(enforce_budget)])
@limiter.limit("10/minute")
def generate_content(request: Request, body: GenerateRequest, db: Session = Depends(get_db)):
    """Generate a LinkedIn post aligned with the stored brand profile."""
    return generate_post(db, body)


@app.post("/engagement", response_model=EngagementResponse, tags=["Engagement Tracking"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def log_engagement(request: Request, body: EngagementRequest, db: Session = Depends(get_db)):
    """Log engagement metrics for a generated post."""
    post = db.query(Post).filter(Post.id == body.post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    # Engagement only makes sense for posts that were actually published, and
    # a post is only publishable once a human has approved it.
    if post.status != "approved":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Post {post.id} is {post.status}, not approved. "
                "Review it in the Evaluate tab before logging engagement."
            ),
        )

    engagement = Engagement(
        post_id=body.post_id,
        likes=body.likes,
        comments=body.comments,
        shares=body.shares,
    )
    db.add(engagement)
    db.commit()
    db.refresh(engagement)

    logger.info("Engagement logged for post_id=%d", body.post_id)
    return engagement


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"],
          dependencies=[Depends(verify_api_key), Depends(enforce_budget)])
@limiter.limit("10/minute")
def feedback(request: Request, body: FeedbackRequest, db: Session = Depends(get_db)):
    """Get AI-powered feedback based on engagement history."""
    return get_feedback(db, body)


# ── Fix 2: LangGraph Orchestrated Endpoint ──────────────────────

@app.post("/orchestrate", response_model=OrchestrateResponse, tags=["Orchestration"],
          dependencies=[Depends(verify_api_key), Depends(enforce_budget)])
@limiter.limit("5/minute")
def orchestrate(request: Request, body: OrchestrateRequest):
    """Run the full LangGraph orchestrated workflow: brand → content → feedback."""
    result = run_full_workflow(
        name=body.name,
        role=body.role,
        industry=body.industry,
        goals=body.goals,
        preferred_tone=body.preferred_tone,
        topic=body.topic,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return OrchestrateResponse(
        brand_profile_id=result.get("brand_profile_id", 0),
        brand_tone=result.get("brand_context", {}).get("tone") if result.get("brand_context") else None,
        post_id=result.get("post_id", 0),
        post_content=result.get("post_content", ""),
        suggested_hashtags=result.get("suggested_hashtags", []),
        feedback_summary=result.get("feedback_summary"),
        feedback_available=result.get("feedback_available", False),
        workflow_steps=result.get("workflow_steps", []),
    )


# ── Fix 5: Prompt Version Management ───────────────────────────

@app.get("/prompts/{agent_name}", response_model=list[PromptTemplateResponse],
         tags=["Prompt Versioning"], dependencies=[Depends(verify_api_key)])
def list_prompts(agent_name: AgentName, request: Request, db: Session = Depends(get_db)):
    """List all prompt versions for a given agent."""
    templates = (
        db.query(PromptTemplate)
        .filter(PromptTemplate.agent_name == agent_name)
        .order_by(PromptTemplate.version.desc())
        .all()
    )
    return templates


@app.post("/prompts/{agent_name}/rollback/{version}", response_model=PromptTemplateResponse,
          tags=["Prompt Versioning"], dependencies=[Depends(verify_api_key)])
def rollback_prompt(agent_name: AgentName, version: PathID, request: Request, db: Session = Depends(get_db)):
    """Roll back to a specific prompt version by deactivating others."""
    target = (
        db.query(PromptTemplate)
        .filter(PromptTemplate.agent_name == agent_name, PromptTemplate.version == version)
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail=f"Prompt version {version} not found for {agent_name}")

    # Deactivate all versions
    db.query(PromptTemplate).filter(
        PromptTemplate.agent_name == agent_name
    ).update({"is_active": False})

    # Activate the target
    target.is_active = True
    db.commit()
    db.refresh(target)

    logger.info("Rolled back %s prompt to version %d", agent_name, version)
    return target


# ── Fix 5: Audit Log Endpoint ──────────────────────────────────

@app.get("/audit-logs", tags=["Operator"], dependencies=[Depends(verify_admin_key)])
def get_audit_logs(
    request: Request,
    agent_name: Annotated[Optional[str], Query(max_length=50, pattern=r"^[a-z_]+$")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    db: Session = Depends(get_db),
):
    """Recent LLM calls with latency, token usage and cost. Admin key required."""
    from database.models import AuditLog

    query = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if agent_name:
        query = query.filter(AuditLog.agent_name == agent_name)
    logs = query.limit(limit).all()

    return [
        {
            "id": log.id,
            "trace_id": log.trace_id,
            "request_id": log.request_id,
            "endpoint": log.endpoint,
            "agent_name": log.agent_name,
            "prompt_version": log.prompt_version,
            "model": log.model,
            "latency_ms": log.latency_ms,
            "status": log.status,
            "error_message": log.error_message,
            "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens,
            "cached_tokens": log.cached_tokens,
            "total_tokens": log.total_tokens,
            "cost_usd": log.cost_usd,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]


@app.get("/costs", tags=["Operator"], dependencies=[Depends(verify_admin_key)])
def get_costs(
    request: Request,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    db: Session = Depends(get_db),
):
    """Token usage and cost summary for the operator. Admin key required.

    Cost per call is stored when the call is made, so these totals reflect the
    prices that applied at the time. Calls logged before usage tracking existed
    have no token data and are counted separately rather than as zero.
    """
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func
    from database.models import AuditLog
    from utils.costs import prices_for
    from utils.config import GROQ_MODEL, PREDICTOR_MODEL

    since = datetime.now(timezone.utc) - timedelta(days=days)
    window = db.query(AuditLog).filter(AuditLog.created_at >= since)
    with_usage = window.filter(AuditLog.total_tokens.isnot(None))

    def money(value) -> float:
        return round(float(value or 0), 6)

    totals = with_usage.with_entities(
        func.count(AuditLog.id),
        func.count(func.distinct(AuditLog.request_id)),
        func.sum(AuditLog.prompt_tokens),
        func.sum(AuditLog.cached_tokens),
        func.sum(AuditLog.completion_tokens),
        func.sum(AuditLog.total_tokens),
        func.sum(AuditLog.cost_usd),
    ).one()

    by_agent = (
        with_usage.with_entities(
            AuditLog.agent_name,
            func.count(AuditLog.id),
            func.sum(AuditLog.total_tokens),
            func.sum(AuditLog.cost_usd),
        )
        .group_by(AuditLog.agent_name)
        .order_by(func.sum(AuditLog.cost_usd).desc())
        .all()
    )

    by_endpoint = (
        with_usage.filter(AuditLog.endpoint.isnot(None))
        .with_entities(
            AuditLog.endpoint,
            func.count(func.distinct(AuditLog.request_id)),
            func.sum(AuditLog.total_tokens),
            func.sum(AuditLog.cost_usd),
        )
        .group_by(AuditLog.endpoint)
        .order_by(func.sum(AuditLog.cost_usd).desc())
        .all()
    )

    by_model = (
        with_usage.with_entities(
            AuditLog.model,
            func.count(AuditLog.id),
            func.sum(AuditLog.total_tokens),
            func.sum(AuditLog.cost_usd),
        )
        .group_by(AuditLog.model)
        .order_by(func.sum(AuditLog.cost_usd).desc())
        .all()
    )

    by_day = (
        with_usage.with_entities(
            func.date(AuditLog.created_at),
            func.sum(AuditLog.total_tokens),
            func.sum(AuditLog.cost_usd),
        )
        .group_by(func.date(AuditLog.created_at))
        .order_by(func.date(AuditLog.created_at))
        .all()
    )

    return {
        "period_days": days,
        "budget": budget_status(db),
        "prediction_cache": prediction_cache_stats(db),
        "pricing": {
            "model": GROQ_MODEL,
            "input_usd_per_million": prices_for(GROQ_MODEL)[0],
            "output_usd_per_million": prices_for(GROQ_MODEL)[1],
            "predictor_model": PREDICTOR_MODEL or GROQ_MODEL,
            "predictor_input_usd_per_million": prices_for(PREDICTOR_MODEL or GROQ_MODEL)[0],
            "predictor_output_usd_per_million": prices_for(PREDICTOR_MODEL or GROQ_MODEL)[1],
        },
        "totals": {
            "llm_calls": totals[0] or 0,
            "requests": totals[1] or 0,
            "prompt_tokens": int(totals[2] or 0),
            "cached_tokens": int(totals[3] or 0),
            "completion_tokens": int(totals[4] or 0),
            "total_tokens": int(totals[5] or 0),
            "cost_usd": money(totals[6]),
            "calls_without_usage": window.filter(AuditLog.total_tokens.is_(None)).count(),
        },
        "by_agent": [
            {"agent_name": name, "calls": calls, "total_tokens": int(tokens or 0),
             "cost_usd": money(cost)}
            for name, calls, tokens, cost in by_agent
        ],
        "by_endpoint": [
            {"endpoint": ep, "requests": reqs, "total_tokens": int(tokens or 0),
             "cost_usd": money(cost),
             "avg_cost_per_request_usd": money((cost or 0) / reqs) if reqs else 0.0}
            for ep, reqs, tokens, cost in by_endpoint
        ],
        "by_model": [
            {"model": m, "calls": calls, "total_tokens": int(tokens or 0),
             "cost_usd": money(cost)}
            for m, calls, tokens, cost in by_model
        ],
        "by_day": [
            {"date": str(day), "total_tokens": int(tokens or 0), "cost_usd": money(cost)}
            for day, tokens, cost in by_day
        ],
    }

# ── Post History Endpoint ───────────────────────────────

@app.get("/history/{user_id}", tags=["Post History"],
         dependencies=[Depends(verify_api_key)])
def get_post_history(user_id: PathID, request: Request, db: Session = Depends(get_db)):
    """Get all posts and engagement data for a brand profile."""
    import json as _json
    from database.models import BrandProfile

    profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    posts = (
        db.query(Post)
        .filter(Post.user_id == user_id)
        .order_by(Post.created_at.desc())
        .all()
    )

    total_likes = 0
    total_comments = 0
    total_shares = 0

    post_list = []
    for post in posts:
        # Get engagement for this post
        likes = 0
        comments = 0
        shares = 0
        for eng in post.engagements:
            likes += eng.likes
            comments += eng.comments
            shares += eng.shares

        total_likes += likes
        total_comments += comments
        total_shares += shares

        # Parse hashtags
        try:
            hashtags = _json.loads(post.hashtags) if post.hashtags else []
        except Exception:
            hashtags = []

        post_list.append({
            "post_id": post.id,
            "topic": post.topic,
            "content": post.content,
            "hashtags": hashtags,
            "status": post.status,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "created_at": post.created_at.isoformat() if post.created_at else None,
        })

    pending_count = sum(1 for p in posts if p.status == "pending")
    approved_count = sum(1 for p in posts if p.status == "approved")
    rejected_count = sum(1 for p in posts if p.status == "rejected")

    return {
        "user_id": user_id,
        "brand_name": profile.name,
        "total_posts": len(posts),
        "pending_posts": pending_count,
        "approved_posts": approved_count,
        "rejected_posts": rejected_count,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "total_shares": total_shares,
        "posts": post_list,
    }

# ── Engagement Predictor ────────────────────────────────────────────────

# The budget is checked inside the route, after the cache: a cached score costs
# nothing, so it is still served once the daily budget is used up.
@app.post("/predict", tags=["Predictor"], response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def predict(request: Request, body: PredictRequest, db: Session = Depends(get_db)):
    """Predict engagement for a draft post before publishing."""
    import json as _json
    from database.models import BrandProfile
    from schemas.schemas import PredictRequest
    from services.llm_service import predict_engagement

    profile = db.query(BrandProfile).filter(BrandProfile.id == body.user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    brand_context = {
        "tone": profile.tone,
        "content_themes": _json.loads(profile.content_themes) if profile.content_themes else [],
        "positioning_summary": profile.positioning_summary,
    }

    posts = db.query(Post).filter(Post.user_id == body.user_id).all()
    history = []
    for post in posts:
        for eng in post.engagements:
            history.append({
                "likes": eng.likes,
                "comments": eng.comments,
                "shares": eng.shares,
            })

    # Two input modes. A post_id scores a saved post and makes it approvable;
    # draft_content alone scores scratch text that is not in the database.
    scored_post = None
    if body.post_id is not None:
        scored_post = (
            db.query(Post)
            .filter(Post.id == body.post_id, Post.user_id == body.user_id)
            .first()
        )
        if not scored_post:
            raise HTTPException(
                status_code=404,
                detail=f"Post {body.post_id} not found for brand profile {body.user_id}.",
            )
        draft_content = scored_post.content
    elif body.draft_content:
        draft_content = body.draft_content
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either post_id (to score a saved post) or draft_content.",
        )

    # A repeat of an unchanged request is answered from the cache: same score,
    # no Groq call. See services/prediction_cache.py for what counts as unchanged.
    from services import prediction_cache
    model, prompt_version = prediction_cache.predictor_setup(db)
    key = prediction_cache.cache_key(draft_content, brand_context, history,
                                     model, prompt_version)
    result = prediction_cache.lookup(db, key)
    cached = result is not None
    if not cached:
        enforce_budget()
        result = predict_engagement(draft_content, brand_context, history, db=db)
        prediction_cache.store(db, key, result, model, prompt_version)

    return {
        "cached": cached,
        "overall_score": result.get("overall_score", 50),
        "predicted_likes": result.get("predicted_likes", "N/A"),
        "predicted_comments": result.get("predicted_comments", "N/A"),
        "predicted_shares": result.get("predicted_shares", "N/A"),
        "brand_alignment": result.get("brand_alignment", 50),
        "hook_strength": result.get("hook_strength", 50),
        "readability": result.get("readability", 50),
        "call_to_action": result.get("call_to_action", 50),
        "improvement_tips": result.get("improvement_tips", ""),
        "post_id": scored_post.id if scored_post else None,
        "status": scored_post.status if scored_post else None,
    }


# ── Profile Management ──────────────────────────────────

@app.delete("/brand/{user_id}", tags=["Brand Strategy"],
            dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def delete_brand_profile(user_id: PathID, request: Request,
                         db: Session = Depends(get_db)):
    """Delete a brand profile and everything belonging to it.

    Posts and engagement records are removed with the profile: leaving them
    behind would orphan rows that the dashboard and feedback queries still
    join against. Audit logs are deliberately kept — they record what the
    system did, not what the profile contained, and that history should
    survive the profile being removed.
    """
    from database.models import BrandProfile, Engagement

    profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    posts = db.query(Post).filter(Post.user_id == user_id).all()
    post_ids = [p.id for p in posts]

    engagement_count = 0
    if post_ids:
        engagement_count = (
            db.query(Engagement)
            .filter(Engagement.post_id.in_(post_ids))
            .delete(synchronize_session=False)
        )
        db.query(Post).filter(Post.user_id == user_id).delete(
            synchronize_session=False
        )

    name = profile.name
    db.delete(profile)
    db.commit()

    logger.info("Deleted profile %d (%s): %d posts, %d engagement records",
                user_id, name, len(post_ids), engagement_count)

    return {
        "deleted": True,
        "user_id": user_id,
        "name": name,
        "posts_deleted": len(post_ids),
        "engagements_deleted": engagement_count,
    }


@app.get("/posts/{post_id}", tags=["Content Generation"],
         dependencies=[Depends(verify_api_key)])
def get_post(post_id: PathID, request: Request, db: Session = Depends(get_db)):
    """Fetch one post in full, with its engagement totals."""
    import json as _json

    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    try:
        hashtags = _json.loads(post.hashtags) if post.hashtags else []
    except _json.JSONDecodeError:
        hashtags = []

    likes = sum(e.likes for e in post.engagements)
    comments = sum(e.comments for e in post.engagements)
    shares = sum(e.shares for e in post.engagements)

    return {
        "post_id": post.id,
        "user_id": post.user_id,
        "topic": post.topic,
        "content": post.content,
        "hashtags": hashtags,
        "status": post.status,
        "review_note": post.review_note,
        "reviewed_at": post.reviewed_at,
        "original_content": post.original_content,
        "edited_at": post.edited_at,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "created_at": post.created_at,
    }


@app.patch("/posts/{post_id}", tags=["Review"],
           dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def edit_post(post_id: PathID, body: PostEditRequest, request: Request,
              db: Session = Depends(get_db)):
    """Edit a post's text or hashtags before it is reviewed.

    Only pending posts can be edited. An approved post may already have
    engagement logged against it, and changing its text afterwards would make
    those numbers describe a post that no longer exists.

    The first edit keeps the model's original text in `original_content`, so
    the record shows what the AI wrote and what the reviewer changed.
    """
    import json as _json
    from datetime import datetime, timezone

    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    if post.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Post {post.id} is {post.status}. Only pending posts can be edited.",
        )

    if body.content is None and body.hashtags is None:
        raise HTTPException(status_code=422, detail="Nothing to change.")

    if body.content is not None:
        new_content = body.content.strip()
        if not new_content:
            raise HTTPException(status_code=422, detail="Post content cannot be empty.")
        if new_content != post.content:
            if post.original_content is None:
                post.original_content = post.content
            post.content = new_content
            post.edited_at = datetime.now(timezone.utc)

    if body.hashtags is not None:
        cleaned = [h.strip().lstrip("#") for h in body.hashtags if h.strip()]
        post.hashtags = _json.dumps(cleaned)
        post.edited_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(post)
    logger.info("Post %d edited by reviewer", post.id)

    return {
        "post_id": post.id,
        "content": post.content,
        "hashtags": _json.loads(post.hashtags) if post.hashtags else [],
        "status": post.status,
        "edited": post.original_content is not None,
        "edited_at": post.edited_at,
    }


@app.patch("/brand/{user_id}", tags=["Brand Strategy"],
           dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def edit_brand_profile(user_id: PathID, body: BrandEditRequest, request: Request,
                       db: Session = Depends(get_db)):
    """Update any field of a brand profile. Only fields that are sent change.

    Posts already generated keep their text; the new tone and themes apply to
    posts generated from here on.
    """
    import json as _json
    from database.models import BrandProfile

    profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=422, detail="Nothing to change.")

    list_fields = {"content_themes", "do_guidelines", "dont_guidelines"}
    for field, value in updates.items():
        if value is None:
            continue
        if field in list_fields:
            setattr(profile, field, _json.dumps([v.strip() for v in value if v.strip()]))
        else:
            value = value.strip()
            if field == "name" and not value:
                raise HTTPException(status_code=422, detail="Name cannot be empty.")
            setattr(profile, field, value)

    db.commit()
    db.refresh(profile)
    logger.info("Profile %d updated: %s", profile.id, ", ".join(sorted(updates)))

    return {
        "id": profile.id,
        "name": profile.name,
        "role": profile.role,
        "industry": profile.industry,
        "goals": profile.goals,
        "preferred_tone": profile.preferred_tone,
        "tone": profile.tone,
        "content_themes": _json.loads(profile.content_themes) if profile.content_themes else [],
        "positioning_summary": profile.positioning_summary,
        "do_guidelines": _json.loads(profile.do_guidelines) if profile.do_guidelines else [],
        "dont_guidelines": _json.loads(profile.dont_guidelines) if profile.dont_guidelines else [],
        "updated_fields": sorted(updates),
    }


# ── Human-in-the-Loop Review ────────────────────────────

@app.get("/posts/pending/{user_id}", tags=["Review"],
         dependencies=[Depends(verify_api_key)])
def list_pending_posts(user_id: PathID, request: Request, db: Session = Depends(get_db)):
    """List posts for a profile that are still awaiting a human decision."""
    from database.models import BrandProfile

    profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    posts = (
        db.query(Post)
        .filter(Post.user_id == user_id, Post.status == "pending")
        .order_by(Post.created_at.desc())
        .all()
    )
    return [
        {
            "post_id": p.id,
            "topic": p.topic,
            "content": p.content,
            "status": p.status,
            "created_at": p.created_at,
        }
        for p in posts
    ]


@app.post("/posts/{post_id}/review", tags=["Review"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def review_post(post_id: PathID, body: ReviewRequest, request: Request,
                db: Session = Depends(get_db)):
    """Approve or reject a generated post.

    Nothing is deleted on rejection: the post stays in the database marked
    "rejected", so there is a record of what the system produced and what a
    human chose not to use.
    """
    from datetime import datetime, timezone

    decision = body.decision.strip().lower()
    if decision not in {"approve", "reject"}:
        raise HTTPException(
            status_code=422,
            detail='decision must be "approve" or "reject".',
        )

    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    post.status = "approved" if decision == "approve" else "rejected"
    post.reviewed_at = datetime.now(timezone.utc)
    post.review_note = body.note
    db.commit()
    db.refresh(post)

    logger.info("Post %d marked %s", post.id, post.status)

    return {
        "post_id": post.id,
        "status": post.status,
        "reviewed_at": post.reviewed_at,
        "review_note": post.review_note,
    }

# ── Dashboard Endpoint ──────────────────────────────────

@app.get("/dashboard/{user_id}", tags=["Dashboard"],
         dependencies=[Depends(verify_api_key)])
def get_dashboard(user_id: PathID, request: Request, db: Session = Depends(get_db)):
    """Get complete dashboard data for a brand profile — brand info, stats, and post history."""
    import json as _json
    from database.models import BrandProfile

    profile = db.query(BrandProfile).filter(BrandProfile.id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Brand profile not found.")

    posts = (
        db.query(Post)
        .filter(Post.user_id == user_id)
        .order_by(Post.created_at.desc())
        .all()
    )

    total_likes = 0
    total_comments = 0
    total_shares = 0
    best_topic = None
    best_engagement = 0

    post_list = []
    for post in posts:
        likes = 0
        comments = 0
        shares = 0
        for eng in post.engagements:
            likes += eng.likes
            comments += eng.comments
            shares += eng.shares

        total_likes += likes
        total_comments += comments
        total_shares += shares

        total_eng = likes + comments + shares
        if total_eng > best_engagement:
            best_engagement = total_eng
            best_topic = post.topic

        try:
            hashtags = _json.loads(post.hashtags) if post.hashtags else []
        except Exception:
            hashtags = []

        post_list.append({
            "post_id": post.id,
            "topic": post.topic,
            "content": post.content,
            "hashtags": hashtags,
            "status": post.status,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "created_at": post.created_at.isoformat() if post.created_at else None,
        })

    pending_count = sum(1 for p in posts if p.status == "pending")
    approved_count = sum(1 for p in posts if p.status == "approved")
    rejected_count = sum(1 for p in posts if p.status == "rejected")

    num_posts = len(posts) if posts else 1  # avoid division by zero

    # Parse brand profile fields
    try:
        content_themes = _json.loads(profile.content_themes) if profile.content_themes else []
    except Exception:
        content_themes = []

    return {
        "user_id": user_id,
        "name": profile.name,
        "role": profile.role,
        "industry": profile.industry,
        "goals": profile.goals,
        "preferred_tone": profile.preferred_tone,
        "tone": profile.tone,
        "positioning_summary": profile.positioning_summary,
        "content_themes": content_themes,
        "do_guidelines": _json.loads(profile.do_guidelines) if profile.do_guidelines else [],
        "dont_guidelines": _json.loads(profile.dont_guidelines) if profile.dont_guidelines else [],
        "total_posts": len(posts),
        "pending_posts": pending_count,
        "approved_posts": approved_count,
        "rejected_posts": rejected_count,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "total_shares": total_shares,
        "avg_likes": round(total_likes / num_posts, 1) if posts else 0,
        "avg_comments": round(total_comments / num_posts, 1) if posts else 0,
        "avg_shares": round(total_shares / num_posts, 1) if posts else 0,
        "best_topic": best_topic,
        "posts": post_list,
    }

# ── Health Check ────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0"}
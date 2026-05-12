"""
LLM Service – wraps Groq API (Llama 3.3 70B) for all AI-powered generation.
Includes: retry logic, JSON parsing, audit logging, prompt versioning.
"""

import json
import logging
import os
import time
import uuid

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import HTTPException
from openai import OpenAI
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configure Groq ──────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds


# ── Audit Logging ───────────────────────────────────────────────

def _log_audit(db: Session, trace_id: str, agent_name: str, prompt_version: int,
               input_summary: str, output_summary: str, latency_ms: float,
               status: str = "success", error_message: str = None):
    """Log an LLM call to the audit_logs table."""
    from database.models import AuditLog

    try:
        log = AuditLog(
            trace_id=trace_id,
            agent_name=agent_name,
            prompt_version=prompt_version,
            input_summary=input_summary[:500] if input_summary else None,
            output_summary=output_summary[:500] if output_summary else None,
            model=MODEL,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.warning("Failed to write audit log: %s", e)


# ── Prompt Versioning ───────────────────────────────────────────

def get_active_prompt(db: Session, agent_name: str) -> tuple[str, int]:
    """Retrieve the active prompt template for an agent. Returns (template, version)."""
    from database.models import PromptTemplate

    template = (
        db.query(PromptTemplate)
        .filter(PromptTemplate.agent_name == agent_name, PromptTemplate.is_active == True)
        .order_by(PromptTemplate.version.desc())
        .first()
    )
    if template:
        return template.template, template.version
    return None, 0


# ── Core LLM Call with Retry ────────────────────────────────────

def _call_with_retry(prompt: str, trace_id: str = None) -> str:
    """Call Groq with exponential backoff retry for rate limits."""
    if not trace_id:
        trace_id = str(uuid.uuid4())

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only, no extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] Rate limited (attempt %d/%d). Retrying in %ds…",
                    trace_id, attempt, MAX_RETRIES, delay,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    continue
                raise HTTPException(
                    status_code=429,
                    detail=f"LLM rate limit exceeded after {MAX_RETRIES} retries. Please wait and try again.",
                )
            else:
                logger.error("[%s] LLM API error: %s", trace_id, e)
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM service error: {e}",
                )


# ── JSON Parser ─────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Extract the first JSON object from model output."""
    import re

    cleaned = text.strip()

    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*\n(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse LLM response as JSON: %s", text[:200])
    raise HTTPException(
        status_code=502,
        detail="LLM returned an invalid response. Please try again.",
    )


# ── Public Functions ────────────────────────────────────────────

def generate_brand_profile(
    name: str, role: str, industry: str, goals: str, preferred_tone: str,
    db: Session = None,
) -> dict:
    """Generate a structured brand profile."""
    trace_id = str(uuid.uuid4())
    start = time.time()

    # Try versioned prompt first
    prompt_version = 0
    template = None
    if db:
        template, prompt_version = get_active_prompt(db, "brand")

    if template:
        prompt = template.format(
            name=name, role=role, industry=industry,
            goals=goals, preferred_tone=preferred_tone,
        )
    else:
        prompt = f"""You are a personal-branding strategist.
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

Return ONLY the JSON object, no extra text."""

    logger.info("[%s] Generating brand profile for %s", trace_id, name)
    text = _call_with_retry(prompt, trace_id)
    result = _parse_json_response(text)
    latency = (time.time() - start) * 1000

    if db:
        _log_audit(db, trace_id, "brand", prompt_version,
                   f"name={name}, role={role}", text[:200], latency)

    return result


def generate_linkedin_post(brand_profile: dict, topic: str, db: Session = None) -> dict:
    """Generate a LinkedIn post aligned with the brand profile."""
    trace_id = str(uuid.uuid4())
    start = time.time()

    prompt_version = 0
    template = None
    if db:
        template, prompt_version = get_active_prompt(db, "content")

    if template:
        prompt = template.format(
            tone=brand_profile.get("tone", ""),
            content_themes=json.dumps(brand_profile.get("content_themes", [])),
            positioning=brand_profile.get("positioning_summary", ""),
            do_guidelines=json.dumps(brand_profile.get("do_guidelines", [])),
            dont_guidelines=json.dumps(brand_profile.get("dont_guidelines", [])),
            topic=topic,
        )
    else:
        prompt = f"""You are a LinkedIn content creator.
Using the brand profile below, write ONE engaging LinkedIn post about the
given topic. Return ONLY valid JSON with these keys:

- "post_content": the full post text (use line breaks, emojis where
  appropriate, keep it between 200-400 words)
- "suggested_hashtags": a list of 3-5 relevant hashtags (without #)

Brand Profile:
  Tone: {brand_profile.get('tone', '')}
  Content Themes: {json.dumps(brand_profile.get('content_themes', []))}
  Positioning: {brand_profile.get('positioning_summary', '')}
  Do: {json.dumps(brand_profile.get('do_guidelines', []))}
  Don't: {json.dumps(brand_profile.get('dont_guidelines', []))}

Topic: {topic}

Return ONLY the JSON object, no extra text."""

    logger.info("[%s] Generating LinkedIn post on topic: %s", trace_id, topic)
    text = _call_with_retry(prompt, trace_id)
    result = _parse_json_response(text)
    latency = (time.time() - start) * 1000

    if db:
        _log_audit(db, trace_id, "content", prompt_version,
                   f"topic={topic}", text[:200], latency)

    return result


def generate_feedback(engagement_history: list[dict], brand_summary: str,
                      db: Session = None) -> dict:
    """Generate strategic feedback based on engagement history."""
    trace_id = str(uuid.uuid4())
    start = time.time()

    prompt_version = 0
    template = None
    if db:
        template, prompt_version = get_active_prompt(db, "feedback")

    if template:
        prompt = template.format(
            brand_summary=brand_summary,
            engagement_history=json.dumps(engagement_history, indent=2),
        )
    else:
        prompt = f"""You are a LinkedIn personal-branding coach.
Analyze the engagement history below and provide actionable feedback.
Return ONLY valid JSON with these keys:

- "performance_summary": a 2-3 sentence summary of how the content
  is performing, referencing specific numbers
- "improvement_recommendation": a 3-5 sentence actionable recommendation
  for improving future LinkedIn content strategy

Brand context: {brand_summary}

Engagement History (most recent first):
{json.dumps(engagement_history, indent=2)}

Return ONLY the JSON object, no extra text."""

    logger.info("[%s] Generating feedback for %d posts", trace_id, len(engagement_history))
    text = _call_with_retry(prompt, trace_id)
    result = _parse_json_response(text)
    latency = (time.time() - start) * 1000

    if db:
        _log_audit(db, trace_id, "feedback", prompt_version,
                   f"posts={len(engagement_history)}", text[:200], latency)

    return result

def predict_engagement(draft_content: str, brand_profile: dict,
                       engagement_history: list, db: Session = None) -> dict:
    """Predict engagement for a draft post based on brand and history."""
    trace_id = str(uuid.uuid4())
    start = time.time()

    prompt_version = 0
    if db:
        _, prompt_version = get_active_prompt(db, "predictor")

    avg_likes = 0
    avg_comments = 0
    avg_shares = 0
    if engagement_history:
        avg_likes = sum(h.get("likes", 0) for h in engagement_history) / len(engagement_history)
        avg_comments = sum(h.get("comments", 0) for h in engagement_history) / len(engagement_history)
        avg_shares = sum(h.get("shares", 0) for h in engagement_history) / len(engagement_history)

    history_context = f"Average engagement: {avg_likes:.0f} likes, {avg_comments:.0f} comments, {avg_shares:.0f} shares per post."
    if not engagement_history:
        history_context = "No previous engagement data available."

    prompt = f"""You are a LinkedIn content performance analyst.
Analyze the following draft LinkedIn post and predict how it will perform.
Use the brand profile and past engagement data for context.

Brand Profile:
  Tone: {brand_profile.get('tone', '')}
  Content Themes: {json.dumps(brand_profile.get('content_themes', []))}
  Positioning: {brand_profile.get('positioning_summary', '')}

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
- "improvement_tips": string with 3-4 specific actionable tips to improve the post, separated by newlines

Return ONLY the JSON object, no extra text."""

    logger.info("[%s] Predicting engagement for draft", trace_id)
    text = _call_with_retry(prompt, trace_id)
    result = _parse_json_response(text)
    latency = (time.time() - start) * 1000

    if db:
        _log_audit(db, trace_id, "predictor", prompt_version,
                   f"draft={draft_content[:100]}", text[:200], latency)

    return result

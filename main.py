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
from utils.auth import verify_api_key
from utils.rate_limiter import limiter
from utils.observability import setup_logging, setup_prometheus

from schemas.schemas import PredictRequest, PredictResponse, ReviewRequest
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
            is_active=True,
            description="Default content generation prompt v1",
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

# Fix 4: Prometheus metrics
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
          dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def create_brand(request: Request, body: BrandRequest, db: Session = Depends(get_db)):
    """Create a structured brand profile using AI."""
    return create_brand_profile(db, body)


@app.post("/generate", response_model=GenerateResponse, tags=["Content Generation"],
          dependencies=[Depends(verify_api_key)])
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
          dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def feedback(request: Request, body: FeedbackRequest, db: Session = Depends(get_db)):
    """Get AI-powered feedback based on engagement history."""
    return get_feedback(db, body)


# ── Fix 2: LangGraph Orchestrated Endpoint ──────────────────────

@app.post("/orchestrate", response_model=OrchestrateResponse, tags=["Orchestration"],
          dependencies=[Depends(verify_api_key)])
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
def list_prompts(agent_name: str, request: Request, db: Session = Depends(get_db)):
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
def rollback_prompt(agent_name: str, version: int, request: Request, db: Session = Depends(get_db)):
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

@app.get("/audit-logs", tags=["Observability"], dependencies=[Depends(verify_api_key)])
def get_audit_logs(
    request: Request,
    agent_name: str = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Retrieve recent audit logs with optional agent name filter."""
    from database.models import AuditLog

    query = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if agent_name:
        query = query.filter(AuditLog.agent_name == agent_name)
    logs = query.limit(limit).all()

    return [
        {
            "id": log.id,
            "trace_id": log.trace_id,
            "agent_name": log.agent_name,
            "prompt_version": log.prompt_version,
            "model": log.model,
            "latency_ms": log.latency_ms,
            "status": log.status,
            "error_message": log.error_message,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]

# ── Post History Endpoint ───────────────────────────────

@app.get("/history/{user_id}", tags=["Post History"],
         dependencies=[Depends(verify_api_key)])
def get_post_history(user_id: int, request: Request, db: Session = Depends(get_db)):
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
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "created_at": post.created_at.isoformat() if post.created_at else None,
        })

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

@app.post("/predict", tags=["Predictor"],
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

    result = predict_engagement(draft_content, brand_context, history, db=db)

    return {
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


# ── Human-in-the-Loop Review ────────────────────────────

@app.get("/posts/pending/{user_id}", tags=["Review"],
         dependencies=[Depends(verify_api_key)])
def list_pending_posts(user_id: int, request: Request, db: Session = Depends(get_db)):
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
def review_post(post_id: int, body: ReviewRequest, request: Request,
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
def get_dashboard(user_id: int, request: Request, db: Session = Depends(get_db)):
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
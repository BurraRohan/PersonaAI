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

from schemas.schemas import PredictRequest, PredictResponse
from services.llm_service import predict_engagement
from database.models import BrandProfile

# ── Logging ─────────────────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ── Seed Default Prompts ────────────────────────────────────────

def seed_default_prompts(db: Session):
    """Insert default prompt templates if none exist."""
    existing = db.query(PromptTemplate).first()
    if existing:
        return

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

    for pt in defaults:
        db.add(pt)
    db.commit()
    logger.info("Seeded %d default prompt templates", len(defaults))


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    result = predict_engagement(body.draft_content, brand_context, history, db=db)

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
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "created_at": post.created_at.isoformat() if post.created_at else None,
        })

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
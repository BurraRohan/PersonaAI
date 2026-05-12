"""
SQLAlchemy ORM models for PersonaAI.
Tables: brand_profiles, posts, engagements, prompt_templates, audit_logs.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from database.db import Base


class BrandProfile(Base):
    """Stores a user's personal brand strategy profile."""

    __tablename__ = "brand_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    role = Column(String(200), nullable=False)
    industry = Column(String(100), nullable=False)
    goals = Column(Text, nullable=False)
    preferred_tone = Column(String(100), nullable=False)

    # LLM-generated brand strategy fields
    tone = Column(Text, nullable=True)
    content_themes = Column(Text, nullable=True)
    positioning_summary = Column(Text, nullable=True)
    do_guidelines = Column(Text, nullable=True)
    dont_guidelines = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    posts = relationship("Post", back_populates="brand_profile")


class Post(Base):
    """Stores generated LinkedIn posts."""

    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("brand_profiles.id"), nullable=False)
    topic = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    hashtags = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    brand_profile = relationship("BrandProfile", back_populates="posts")
    engagements = relationship("Engagement", back_populates="post")


class Engagement(Base):
    """Stores engagement metrics for a post."""

    __tablename__ = "engagements"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    post = relationship("Post", back_populates="engagements")


# ── Fix 5: Prompt Versioning ───────────────────────────────────

class PromptTemplate(Base):
    """Stores versioned prompt templates for each agent."""

    __tablename__ = "prompt_templates"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(50), nullable=False, index=True)  # brand, content, feedback
    version = Column(Integer, nullable=False, default=1)
    template = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    """Logs every LLM call for traceability and debugging."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(36), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False)
    prompt_version = Column(Integer, nullable=True)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    model = Column(String(100), nullable=True)
    latency_ms = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="success")  # success, error, retry
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

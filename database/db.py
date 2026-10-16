"""
Database configuration and session management.
Uses SQLite with SQLAlchemy for persistent storage.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./personaai.db")

# Make sure the directory holding the SQLite file exists before connecting.
if DATABASE_URL.startswith("sqlite:///"):
    db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columns added after the first release. create_all() only creates missing
# tables, never missing columns, so an existing database would otherwise break
# on the next query. Each entry is (table, column, SQL type).
_ADDED_COLUMNS = [
    ("posts", "status", "VARCHAR(20) NOT NULL DEFAULT 'approved'"),
    ("posts", "reviewed_at", "DATETIME"),
    ("posts", "review_note", "TEXT"),
    ("posts", "original_content", "TEXT"),
    ("posts", "edited_at", "DATETIME"),
    ("audit_logs", "prompt_tokens", "INTEGER"),
    ("audit_logs", "completion_tokens", "INTEGER"),
    ("audit_logs", "cached_tokens", "INTEGER"),
    ("audit_logs", "total_tokens", "INTEGER"),
    ("audit_logs", "cost_usd", "FLOAT"),
    ("audit_logs", "request_id", "VARCHAR(36)"),
    ("audit_logs", "endpoint", "VARCHAR(100)"),
]


def ensure_schema() -> list[str]:
    """Add any columns an older database is missing. Returns what was added.

    Posts that predate the review gate get status 'approved', so existing data
    stays visible and analysable rather than vanishing into a pending queue.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    added = []

    with engine.begin() as conn:
        for table, column, sql_type in _ADDED_COLUMNS:
            if table not in tables:
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
                added.append(f"{table}.{column}")

    return added

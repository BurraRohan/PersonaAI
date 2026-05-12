"""
PersonaAI – Streamlit Observability Dashboard
Run: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="PersonaAI Dashboard", layout="wide")
st.title("PersonaAI – Observability Dashboard")

DB_PATH = "personaai.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ── Sidebar ─────────────────────────────────────────────────────

st.sidebar.header("Filters")
refresh = st.sidebar.button("Refresh Data")

# ── Metrics Overview ────────────────────────────────────────────

st.header("Overview")

try:
    conn = get_connection()

    col1, col2, col3, col4 = st.columns(4)

    profiles = pd.read_sql("SELECT COUNT(*) as count FROM brand_profiles", conn)
    col1.metric("Brand Profiles", profiles["count"][0])

    posts = pd.read_sql("SELECT COUNT(*) as count FROM posts", conn)
    col2.metric("Posts Generated", posts["count"][0])

    engagements = pd.read_sql("SELECT COUNT(*) as count FROM engagements", conn)
    col3.metric("Engagement Records", engagements["count"][0])

    try:
        audit = pd.read_sql("SELECT COUNT(*) as count FROM audit_logs", conn)
        col4.metric("LLM Calls Logged", audit["count"][0])
    except Exception:
        col4.metric("LLM Calls Logged", "N/A")

    # ── Engagement Trends ───────────────────────────────────────

    st.header("Engagement Trends")

    try:
        eng_data = pd.read_sql("""
            SELECT e.created_at, e.likes, e.comments, e.shares, p.topic
            FROM engagements e
            JOIN posts p ON e.post_id = p.id
            ORDER BY e.created_at
        """, conn)

        if not eng_data.empty:
            eng_data["created_at"] = pd.to_datetime(eng_data["created_at"])
            eng_data["total_engagement"] = eng_data["likes"] + eng_data["comments"] + eng_data["shares"]

            st.line_chart(eng_data.set_index("created_at")[["likes", "comments", "shares"]])

            st.subheader("Engagement by Post")
            st.dataframe(eng_data[["topic", "likes", "comments", "shares", "total_engagement"]], width="stretch")
        else:
            st.info("No engagement data yet. Log some engagement via the API.")
    except Exception as e:
        st.warning(f"Could not load engagement data: {e}")

    # ── Audit Logs ──────────────────────────────────────────────

    st.header("Audit Logs (LLM Calls)")

    try:
        audit_data = pd.read_sql("""
            SELECT trace_id, agent_name, prompt_version, model,
                   latency_ms, status, error_message, created_at
            FROM audit_logs
            ORDER BY created_at DESC
            LIMIT 100
        """, conn)

        if not audit_data.empty:
            # Latency stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Latency", f"{audit_data['latency_ms'].mean():.0f} ms")
            col2.metric("Max Latency", f"{audit_data['latency_ms'].max():.0f} ms")
            error_rate = (audit_data["status"] != "success").sum() / len(audit_data) * 100
            col3.metric("Error Rate", f"{error_rate:.1f}%")

            # Agent breakdown
            st.subheader("Calls by Agent")
            agent_counts = audit_data["agent_name"].value_counts()
            st.bar_chart(agent_counts)

            # Log table
            st.subheader("Recent Logs")
            st.dataframe(audit_data, use_container_width=True)
        else:
            st.info("No audit logs yet. Make some API calls first.")
    except Exception as e:
        st.info("Audit logs table not available yet. Run the API to create it.")

    # ── Prompt Versions ─────────────────────────────────────────

    st.header("Prompt Templates")

    try:
        prompts = pd.read_sql("""
            SELECT agent_name, version, is_active, description, created_at
            FROM prompt_templates
            ORDER BY agent_name, version DESC
        """, conn)

        if not prompts.empty:
            st.dataframe(prompts, use_container_width=True)
        else:
            st.info("No prompt templates found.")
    except Exception as e:
        st.info("Prompt templates table not available yet.")

    conn.close()

except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.info("Make sure PersonaAI has been run at least once to create the database.")

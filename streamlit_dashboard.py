"""
PersonaAI – Streamlit Observability Dashboard
Run: streamlit run streamlit_dashboard.py
"""

import os

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="PersonaAI Dashboard", layout="wide")
st.title("PersonaAI – Observability Dashboard")

DB_PATH = os.getenv("PERSONAAI_DB_PATH", "personaai.db")


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
            st.dataframe(audit_data, width="stretch")
        else:
            st.info("No audit logs yet. Make some API calls first.")
    except Exception as e:
        st.info("Audit logs table not available yet. Run the API to create it.")

    # ── Token Usage & Cost ──────────────────────────────────────
    # Operator-only view. The user-facing app never shows any of this;
    # the API exposes it only on admin-key routes (/costs, /audit-logs).

    st.header("Token Usage & Cost")

    try:
        usage = pd.read_sql("""
            SELECT request_id, endpoint, agent_name, prompt_tokens, cached_tokens,
                   completion_tokens, total_tokens, cost_usd, created_at
            FROM audit_logs
            WHERE total_tokens IS NOT NULL
        """, conn)

        if not usage.empty:
            usage["created_at"] = pd.to_datetime(usage["created_at"])
            requests = usage["request_id"].nunique()
            total_cost = usage["cost_usd"].sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cost", f"${total_cost:.4f}")
            col2.metric("Total Tokens", f"{int(usage['total_tokens'].sum()):,}")
            col3.metric("User Actions", requests)
            col4.metric("Avg Cost per Action",
                        f"${total_cost / requests:.5f}" if requests else "N/A")

            # Rows named "<agent>_agent" are the ReAct reasoning loop; the rest
            # are single structured calls (direct mode, or made by agent tools).
            is_loop = usage["agent_name"].str.endswith("_agent")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens", f"{int(usage['prompt_tokens'].sum()):,}",
                        help=f"{int(usage['cached_tokens'].sum()):,} cached (billed at half price)")
            col2.metric("Output Tokens", f"{int(usage['completion_tokens'].sum()):,}")
            col3.metric("Spent in Agent Loops",
                        f"{usage.loc[is_loop, 'cost_usd'].sum() / total_cost * 100:.0f}%"
                        if total_cost else "N/A")

            left, right = st.columns(2)
            with left:
                st.subheader("Cost by Agent")
                st.bar_chart(usage.groupby("agent_name")["cost_usd"].sum()
                             .sort_values(ascending=False))
            with right:
                st.subheader("Cost per Day")
                per_day = usage.set_index("created_at")["cost_usd"].resample("D").sum()
                per_day.index = per_day.index.strftime("%d %b")  # label days, not a timeline
                st.bar_chart(per_day)

            st.subheader("Cost by Endpoint")
            by_endpoint = (
                usage.dropna(subset=["endpoint"])
                .groupby("endpoint")
                .agg(requests=("request_id", "nunique"),
                     total_tokens=("total_tokens", "sum"),
                     cost_usd=("cost_usd", "sum"))
                .sort_values("cost_usd", ascending=False)
            )
            by_endpoint["avg_cost_per_request"] = by_endpoint["cost_usd"] / by_endpoint["requests"]
            st.dataframe(by_endpoint.style.format({
                "cost_usd": "${:.6f}", "avg_cost_per_request": "${:.6f}",
                "total_tokens": "{:,.0f}",
            }), width="stretch")

            st.subheader("Most Expensive Actions")
            per_request = (
                usage.dropna(subset=["request_id"])
                .groupby("request_id")
                .agg(endpoint=("endpoint", "first"),
                     llm_calls=("agent_name", "count"),
                     total_tokens=("total_tokens", "sum"),
                     cost_usd=("cost_usd", "sum"),
                     at=("created_at", "min"))
                .sort_values("cost_usd", ascending=False)
                .head(10)
            )
            st.dataframe(per_request.style.format({
                "cost_usd": "${:.6f}", "total_tokens": "{:,.0f}",
            }), width="stretch")
        else:
            st.info("No token usage recorded yet. Generate a profile or post to see costs.")

        untracked = pd.read_sql(
            "SELECT COUNT(*) AS n FROM audit_logs WHERE total_tokens IS NULL", conn)["n"][0]
        if untracked:
            st.caption(f"{untracked} older LLM call(s) were logged before usage tracking "
                       "and are not included in these totals.")
    except Exception:
        st.info("Token usage columns not available yet. Restart the API once to add them.")

    # ── Prompt Versions ─────────────────────────────────────────

    st.header("Prompt Templates")

    try:
        prompts = pd.read_sql("""
            SELECT agent_name, version, is_active, description, created_at
            FROM prompt_templates
            ORDER BY agent_name, version DESC
        """, conn)

        if not prompts.empty:
            st.dataframe(prompts, width="stretch")
        else:
            st.info("No prompt templates found.")
    except Exception as e:
        st.info("Prompt templates table not available yet.")

    conn.close()

except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.info("Make sure PersonaAI has been run at least once to create the database.")
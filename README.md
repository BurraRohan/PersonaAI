# PersonaAI – Personal Branding Intelligence Agent

A multi-agent system that helps professionals build a consistent LinkedIn personal brand through an end-to-end feedback loop.

Built with **Python, FastAPI, LangChain, LangGraph, Groq (Llama 3.3 70B), SQLAlchemy and Streamlit.**

---

## What It Does

PersonaAI structures the guesswork out of personal branding. It builds a brand profile, writes posts aligned to that profile, scores a draft before you publish it, records how the post actually performed, and turns that history into concrete strategy advice.

Feedback quality improves as engagement history accumulates — the model is given more real data to reason over. Nothing is trained or fine-tuned.

```
Define Brand → Generate Post → Predict Engagement → Publish → Log Metrics → Get Feedback → Improve
```

---

## Agent Execution

`/brand`, `/generate` and `/feedback` each run a ReAct agent built with `create_react_agent`. The agent selects its own tools at runtime; the API does not call them in a fixed order.

Every response carries two fields that make this inspectable:

| Field            | Meaning                                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `execution_mode` | `agent` when the agent ran and persisted its work; `direct` when `AGENT_MODE=0` was set deliberately; `fallback_direct` when the agent failed and the single-call path rescued the request |
| `agent_trace`    | Ordered list of the tool calls the model chose, and what each returned                                                                                                                     |

A real trace from `POST /brand`, captured from a live run against Groq:

```json
"execution_mode": "agent",
"agent_trace": [
  {
    "step": 1,
    "action": "tool_call",
    "tool": "check_existing_profile",
    "args": { "name": "Demo User" }
  },
  {
    "step": 2,
    "action": "tool_result",
    "tool": "check_existing_profile",
    "result": "{\"exists\": false}"
  },
  {
    "step": 3,
    "action": "tool_call",
    "tool": "generate_brand_strategy",
    "args": {
      "name": "Demo User",
      "role": "SDE-II",
      "industry": "Automobile",
      "goals": "Build authority in automotive AI",
      "preferred_tone": "Professional and approachable"
    }
  },
  {
    "step": 4,
    "action": "tool_result",
    "tool": "generate_brand_strategy",
    "result": "{\"tone\": \"The ideal tone is professional and approachable...\", \"content_themes\": [\"Automotive AI Trends\", \"Innovation in Autonomous Vehicles\", ...]}"
  },
  {
    "step": 5,
    "action": "tool_call",
    "tool": "save_brand_profile",
    "args": {
      "name": "Demo User",
      "strategy_json": "{ ...the JSON returned by step 4... }"
    }
  },
  {
    "step": 6,
    "action": "tool_result",
    "tool": "save_brand_profile",
    "result": "{\"id\": 5, \"status\": \"saved\"}"
  }
]
```

The model was not told to call those tools in that order. It checked for an
existing profile first, saw none, generated a strategy, then persisted it —
choosing each step from the tool descriptions alone. Tool results longer than
400 characters are truncated in the trace to keep responses readable.

### Tools available to each agent

| Agent    | Tools                                                                                |
| -------- | ------------------------------------------------------------------------------------ |
| Brand    | `check_existing_profile`, `generate_brand_strategy`, `save_brand_profile`            |
| Content  | `fetch_brand_profile`, `list_recent_topics`, `create_and_save_post`                  |
| Feedback | `fetch_engagement_history`, `compute_engagement_stats`, `generate_strategy_feedback` |

`compute_engagement_stats` does its arithmetic in Python rather than asking the model to average numbers, so the statistics the agent reasons over are exact.

### Agent mode

Running the agent loop costs roughly 3-6 Groq calls per request, because the
model has to see each tool's output before choosing the next one. On Groq's
free tier that adds up quickly while developing.

`AGENT_MODE` in `.env` controls this:

| Value         | Behaviour                                         | `execution_mode` in the response |
| ------------- | ------------------------------------------------- | -------------------------------- |
| `1` (default) | Runs the ReAct agents; `agent_trace` is populated | `agent`                          |
| `0`           | One direct LLM call; `agent_trace` is empty       | `direct`                         |

Endpoints, request bodies and response fields are identical either way, so the
frontend does not care which mode is active. Restart the server after changing it.

### Why there is a fallback

A rate limit or a malformed tool call would otherwise surface as a 500. When the agent path fails, the endpoint completes through a single deterministic LLM call and reports `execution_mode: "fallback_direct"`, with the failure recorded in the trace.

---

## Orchestration Graph

`POST /orchestrate` runs the whole pipeline through a LangGraph `StateGraph` with conditional edges. Routing is decided at runtime, not hardcoded.

![Orchestration graph](assets/graph.png)

`check_history` exists because feedback on a profile with no logged engagement is meaningless. Asking a model for a data-driven analysis of zero data invites it to invent numbers. The graph routes to `skip_feedback` instead and says so; `feedback_available` in the response tells the caller which branch ran.

Render the live diagram:

```python
from agents.orchestrator import render_graph_mermaid
print(render_graph_mermaid())
```

---

## Architecture

```
PersonaAI/
├── main.py                  # FastAPI entry point, routes, prompt seeding
├── agents/
│   ├── brand_agent.py       # ReAct agent: brand profile creation
│   ├── content_agent.py     # ReAct agent: LinkedIn post generation
│   ├── feedback_agent.py    # ReAct agent: engagement analysis
│   └── orchestrator.py      # LangGraph StateGraph with conditional edges
├── services/
│   └── llm_service.py       # Groq wrapper: retries, JSON parsing, audit logging
├── database/
│   ├── models.py            # SQLAlchemy ORM models
│   └── db.py                # Engine, session, dependency
├── schemas/
│   └── schemas.py           # Pydantic request/response models
├── utils/
│   ├── agent_runtime.py     # Agent execution + tool-call trace capture
│   ├── auth.py              # API key authentication
│   ├── rate_limiter.py      # Request rate limiting
│   └── observability.py     # Prometheus metrics + structured logging
├── static/                  # Frontend UI (6-tab dashboard)
├── tests/                   # pytest suite (33 tests, no network required)
├── streamlit_dashboard.py   # Observability dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.template
```

---

## Features

- **3 ReAct agents** — each chooses its own tools at runtime, with the tool-call trace returned to the caller
- **Conditional graph routing** — the orchestrator branches on error state and on data availability
- **Engagement prediction** — scores a draft out of 100 across brand alignment, hook strength, readability and call-to-action before publishing
- **Feedback loop** — analyses recorded engagement and recommends specific changes, referencing actual post topics and numbers
- **Prompt versioning** — templates stored in the database, with rollback to any prior version
- **Audit logging** — every LLM call recorded with a trace ID, model, prompt version, latency and status

- **API security** — bearer token auth, per-endpoint rate limiting, HTML escaping on all rendered model output
- **Prometheus metrics** — request counts, latency histograms and error rates at `/metrics`
- **Streamlit dashboard** — engagement trends and audit log inspection

![Streamlit observability dashboard](assets/streamlit.png)

- **Docker deployment** — `docker-compose` with a named volume and health checks

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free Groq API key → [console.groq.com/keys](https://console.groq.com/keys)

### Setup

```bash
cd PersonaAI

python -m venv venv
# source venv/bin/activate     # macOS / Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# cp .env.template .env        # macOS / Linux
copy .env.template .env    # Windows
```

Open `.env` and set both keys. `API_KEY` is required — the app refuses to start without it rather than falling back to a default value. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Run

```bash
uvicorn main:app --reload
```

|                    |                               |
| ------------------ | ----------------------------- |
| App UI             | http://127.0.0.1:8000         |
| Swagger docs       | http://127.0.0.1:8000/docs    |
| Prometheus metrics | http://127.0.0.1:8000/metrics |

The UI asks for your API key once and keeps it in `localStorage`, so it survives tab closes and browser restarts. It is never written into the source. To be asked again, type `resetApiKey()` in the browser console.

### Tests

```bash
pytest -q
```

33 tests, all offline — the Groq boundary is stubbed, so the suite covers this project's logic rather than the model's output.

### Streamlit dashboard

```bash
streamlit run streamlit_dashboard.py
```

### Docker

```bash
docker-compose up --build
```

---

## API Endpoints

Every endpoint except `/health` and `/metrics` requires an `Authorization: Bearer <your-api-key>` header.

| Method | Endpoint                                   | Description                                                | Rate limit |
| ------ | ------------------------------------------ | ---------------------------------------------------------- | ---------- |
| POST   | `/brand`                                   | Create a brand profile (brand agent)                       | 10/min     |
| POST   | `/generate`                                | Generate a LinkedIn post (content agent)                   | 10/min     |
| POST   | `/predict`                                 | Score a draft before publishing                            | 10/min     |
| POST   | `/engagement`                              | Log likes, comments and shares for a post                  | 30/min     |
| POST   | `/feedback`                                | Strategy feedback from engagement history (feedback agent) | 10/min     |
| POST   | `/orchestrate`                             | Run the full graph: brand → content → feedback             | 5/min      |
| GET    | `/dashboard/{user_id}`                     | Brand info, totals, averages, best topic, post list        | —          |
| GET    | `/history/{user_id}`                       | All posts and engagement for a profile                     | —          |
| GET    | `/prompts/{agent_name}`                    | List prompt versions for an agent                          | —          |
| POST   | `/prompts/{agent_name}/rollback/{version}` | Activate a specific prompt version                         | —          |
| GET    | `/audit-logs`                              | Recent LLM calls, filterable by `agent_name`               | —          |
| GET    | `/metrics`                                 | Prometheus metrics                                         | —          |
| GET    | `/health`                                  | Health check                                               | —          |

Valid `agent_name` values: `brand`, `content`, `feedback`, `predictor`.

---

## Environment Variables

| Variable            | Description                                             | Required | Default                                       |
| ------------------- | ------------------------------------------------------- | -------- | --------------------------------------------- |
| `GROQ_API_KEY`      | Groq API key for Llama 3.3 70B                          | Yes      | —                                             |
| `API_KEY`           | Bearer token protecting the API                         | Yes      | — (app will not start)                        |
| `DATABASE_URL`      | SQLAlchemy connection string                            | No       | `sqlite:///./personaai.db`                    |
| `AGENT_MODE`        | `1` runs the ReAct agents, `0` uses one direct LLM call | No       | `1`                                           |
| `ALLOWED_ORIGINS`   | Comma-separated CORS origins                            | No       | `http://localhost:8000,http://127.0.0.1:8000` |
| `PERSONAAI_DB_PATH` | SQLite path used by the Streamlit dashboard             | No       | `personaai.db`                                |

---

## Tech Stack

**Backend** — Python, FastAPI, SQLAlchemy, SQLite, Pydantic v2

**Agents** — LangChain tools, LangGraph (`create_react_agent` + `StateGraph`), Groq (Llama 3.3 70B)

**Observability** — Prometheus, Streamlit, database-backed audit logs

**Security** — Bearer token auth, SlowAPI rate limiting, output escaping

**Deployment** — Docker, docker-compose, Gunicorn with Uvicorn workers

---

## Known Limitations

Stated plainly, because they shape how the results should be read:

- **Engagement prediction is unvalidated.** The scores come from the model's judgement, not a regression fitted to real LinkedIn data. Treat them as a structured second opinion, not a forecast. Validating them would require a labelled dataset of posts and their actual performance.
- **No publishing integration.** Posts are generated and stored; you copy them to LinkedIn yourself. There is no LinkedIn API connection.
- **Engagement figures are entered by hand.** Nothing verifies that the numbers logged via `/engagement` match reality.
- **Single-tenant auth.** One shared API key protects the whole instance. There are no user accounts, and any holder of the key can read every profile.
- **SQLite concurrency.** Fine for a single instance; a multi-worker deployment under write load should move to PostgreSQL.
- **Agent reliability varies.** Llama 3.3 70B occasionally skips a tool call in a three-step chain. The fallback path covers this, but `execution_mode` is worth checking when a run looks unusual.

---

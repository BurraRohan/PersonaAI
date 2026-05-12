# PersonaAI – Personal Branding Intelligence Agent

A multi-agent AI system that helps professionals build a consistent LinkedIn personal brand through an end-to-end feedback loop.

Built with **Python, FastAPI, LangChain, LangGraph, Groq (Llama 3.3 70B), SQLAlchemy, and Streamlit.**

---

## What It Does

**PersonaAI** takes the guesswork out of personal branding on LinkedIn. It creates a structured brand profile, generates posts aligned to your brand strategy, predicts engagement before you publish, tracks real performance metrics, and gives AI-powered feedback that gets smarter with more data.

### The Workflow

```
Define Brand → Generate Post → Predict Engagement → Publish → Log Metrics → Get Feedback → Improve
```

---

## Architecture

```
PersonaAI/
├── main.py                  # FastAPI entry point
├── agents/
│   ├── brand_agent.py       # Brand profile creation (LangChain + ReAct)
│   ├── content_agent.py     # LinkedIn post generation
│   ├── feedback_agent.py    # Engagement analysis & recommendations
│   └── orchestrator.py      # LangGraph StateGraph orchestration
├── services/
│   └── llm_service.py       # Groq LLM wrapper (Llama 3.3 70B)
├── database/
│   ├── models.py            # SQLAlchemy ORM models
│   └── db.py                # Engine, session, dependency
├── schemas/
│   └── schemas.py           # Pydantic request/response models
├── utils/
│   ├── auth.py              # API key authentication
│   ├── rate_limiter.py      # Request rate limiting
│   └── observability.py     # Prometheus metrics + structured logging
├── static/                  # Frontend UI (6-tab dashboard)
├── streamlit_dashboard.py   # Monitoring dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.template
```

---

## Features

- **4 AI Agents + 1 Orchestrator** — Brand, Content, Feedback, and Predictor agents, all coordinated through a LangGraph StateGraph
- **Engagement Prediction** — scores your post out of 100 with a rating breakdown before you publish
- **Feedback Loop** — analyzes past engagement data and recommends strategy improvements
- **Dashboard** — performance overview, post history, averages, and best-performing topics
- **Prompt Versioning** — version-controlled prompt templates with rollback support
- **Audit Logging** — every LLM call logged with trace IDs
- **API Security** — Bearer token authentication + per-endpoint rate limiting
- **Prometheus Metrics** — request counts, latency histograms, error rates at `/metrics`
- **Streamlit Monitoring** — visual observability dashboard for engagement trends
- **Docker Deployment** — containerized with `docker-compose`
- **Human-in-the-Loop** — no auto-posting; you review everything before it goes live

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free Groq API key → [console.groq.com/keys](https://console.groq.com/keys)

### Setup

```bash
cd PersonaAI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

copy .env.template .env
# Open .env and add your GROQ_API_KEY and set an API_KEY
```

### Run

```bash
uvicorn main:app --reload
```

- App UI → **http://127.0.0.1:8000**
- Swagger Docs → **http://127.0.0.1:8000/docs**
- Prometheus Metrics → **http://127.0.0.1:8000/metrics**

### Streamlit Dashboard

```bash
streamlit run streamlit_dashboard.py
```

### Docker

```bash
docker-compose up --build
```

---

## API Endpoints

All endpoints require `Authorization: Bearer <your-api-key>` header.

| Method | Endpoint       | Description                                        |
| ------ | -------------- | -------------------------------------------------- |
| POST   | `/brand`       | Create a brand profile                             |
| POST   | `/generate`    | Generate a LinkedIn post                           |
| POST   | `/predict`     | Predict engagement score before publishing         |
| POST   | `/engagement`  | Log engagement metrics (likes, comments, shares)   |
| POST   | `/feedback`    | Get AI-powered strategy feedback                   |
| POST   | `/orchestrate` | Run the full pipeline (brand → content → feedback) |
| GET    | `/metrics`     | Prometheus metrics                                 |
| GET    | `/health`      | Health check                                       |
| GET    | `/audit-logs`  | View LLM call audit trail                          |
| GET    | `/prompts`     | View prompt version history                        |

---

## Environment Variables

| Variable       | Description                    | Required                                    |
| -------------- | ------------------------------ | ------------------------------------------- |
| `GROQ_API_KEY` | Groq API key for Llama 3.3 70B | Yes                                         |
| `API_KEY`      | Bearer token for endpoint auth | Yes                                         |
| `DATABASE_URL` | SQLite connection string       | No (defaults to `sqlite:///./personaai.db`) |

---

## Tech Stack

**Backend** — Python, FastAPI, SQLAlchemy, SQLite, Pydantic

**AI/Agents** — LangChain (ReAct agents with tool-use), LangGraph (StateGraph orchestration), Groq (Llama 3.3 70B)

**Observability** — Prometheus, Streamlit

**Security** — Bearer token auth, SlowAPI rate limiting, audit logging

**Deployment** — Docker, docker-compose, Gunicorn

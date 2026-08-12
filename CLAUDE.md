# CLAUDE.md — Capstone Idea 3: Employee Onboarding & Lifecycle Agent

SDAIA "Advanced Agentic AI Systems Engineering" capstone (cohort 9–13 Aug 2026,
Riyadh). Official idea #3 from the Day-5 slides, built from scratch in
**English**. Standalone git repo.

## ⛔ Safety rule — overrides everything

**Never delete or modify any file outside this project folder**
(`SDAIA-Advanced-Agentic-AI-Systems-Engineering-Idea-3`). Everything outside it
is read-only. The sibling folder `SDAIA-Advanced-Agentic-AI-Systems-Engineering-Idea-1`
belongs to a teammate's tool — **never touch it, not even reads you can avoid**.
Never read personal/sensitive documents (IDs, IBANs, salaries, HR files of real
people — rule R021). All sample data here is synthetic.

## What this is

Multi-agent HR onboarding system: when a candidate is marked **hired**, a
coordinator drives specialized agents — profile analysis from the resume,
personalized training-plan generation (with a Reflexion-style reviewer),
contract drafting from Jinja2 templates, and IT provisioning through real
tool calls. Risky steps pause at a human-approval gate and **resume across
days/processes** via a persistent **PostgresSaver** checkpointer (Docker).

## Binding decisions (from the owner interview, 2026-08-13)

1. Success = official rubric, target 90+. Out of scope: OCR, GUI, real HR
   system integrations.
2. Autonomous loop: plan → 3 parallel critics → fix → execute/verify → review.
   No plan-approval pause.
3. Checkpointer: **PostgresSaver in Docker** (port 5433, db `hr_agent`).
   SqliteSaver stays as a documented fallback only.
4. Push to GitHub continuously (incremental history is graded).
5. Everything user-facing in the repo is **English**.

## Proven by spike (don't re-litigate)

- `PostgresSaver.from_conn_string(dsn)` **is** a context manager (unlike
  SqliteSaver!). Call `saver.setup()` once — idempotent.
- `interrupt()` + `Command(resume=...)` survives a fresh connection/process;
  pre-interrupt state survives intact.
- Windows needs `psycopg[binary]` (plain `psycopg` lacks libpq).

## Environment

- venv: `C:\Users\abdul\sdaia-agents-venv` (Python 3.12, langgraph 1.x,
  langgraph-checkpoint-postgres + psycopg[binary] installed 2026-08-13).
- Run tests: `PYTHONIOENCODING=utf-8 <venv>/python -X utf8 -m pytest -q`
- LLM: provider chain via env (`LLM_BASE_URL`/`LLM_API_KEY`, `_2` for the
  second provider). Captured on Mistral; Gemini as fallback. temperature=0.
- Postgres for dev: `docker run -d --name idea3-pg -e POSTGRES_PASSWORD=capstone
  -e POSTGRES_DB=hr_agent -p 5433:5432 postgres:16-alpine`

## Working rules

- TDD: red test first, then minimal green code. Evidence captured from clean
  state only — never run two captures concurrently.
- No secrets in code or logs; `.env` only, redact keys centrally.
- Honest attribution everywhere: if the system forces a step, the trace says
  so — never credit the model with a choice it didn't make.
- Relative paths in every persisted artifact (no `C:\Users\...` leaks).
- Every doc claim must have a wired implementation or it gets deleted.

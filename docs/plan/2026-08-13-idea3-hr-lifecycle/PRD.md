# PRD — Employee Onboarding & Lifecycle Agent (Capstone Idea 3)

Run folder: `docs/plan/2026-08-13-idea3-hr-lifecycle/` · Status: **OPEN**

## Problem

When HR marks a candidate as *hired*, a chain of slow manual work starts:
reading the resume, designing a training plan, drafting the contract notice,
and asking IT to provision accounts. This system automates the chain **without
giving up governance**: a human approves before anything binding happens, every
step is auditable, and the whole flow can pause and resume across days.

## Success criteria

- Official rubric ≥ 90 (all six deliverables, none under 40%).
- Integration test green: hired-candidate JSON in → paused at approval →
  resumed from a **fresh process** → contract rendered + IT ticket written.
- Evidence captured live for every rubric row (executed notebook + raw logs).

Out of scope: OCR, GUI, real HRIS/AD integration, Arabic content (English run).

## Run configuration (locked at interview)

| Decision | Value | Why | Rejected alternative |
|---|---|---|---|
| Idea | #3 HR lifecycle | Owner: minimal security surface so Fable 5 critics work freely | #2 SOC (all security) |
| Language | English | Owner preference | Arabic (previous project) |
| Checkpointer | **PostgresSaver** in Docker (port 5433) | Slide names it; Docker proven working; differentiates from previous project | SqliteSaver (fallback only) |
| Plan-approval pause | None — proceed after critique | Owner sleeps; pattern proven | Pause on PRD |
| Push | Continuous to fresh GitHub repo | Rubric grades incremental history | Local-only |
| Models | Critics/review = `fable`; slice executors + spikes = `opus`; mechanical checks = `haiku`; main loop = Opus 5 | Owner directive | — |
| Target score | 90+ | Owner | max-effort extras (Grafana etc.) |

## Architecture (agents & coordination)

**Coordination strategy: centralized** — the LangGraph `StateGraph` is the
coordinator; agents communicate only through typed state (Pydantic contracts).

| Agent (node) | Role | Output contract |
|---|---|---|
| `intake` | Validate + guard the hired-candidate file (untrusted input!) | `CaseFile` |
| `profile_analyst` | Parse resume text → structured profile | `CandidateProfile` |
| `training_planner` | **Plan-and-Execute**: design personalized onboarding plan | `TrainingPlan` |
| `plan_reviewer` | **Reflexion**: critique plan vs. role/policies; revise loop (bounded ×1) | `ReviewVerdict` |
| `contract_drafter` | Fill Jinja2 contract-notice template from typed fields | `ContractDraft` |
| `hr_gate` | **HITL interrupt**: HR approves/rejects before anything binding | resume decision |
| `it_provisioner` | **ReAct + MCP-style tools**: create account, allocate equipment | `ProvisionResult` |
| `notifier` | Render + write welcome/notification docs | files on disk |

Conditional edges: unknown/invalid intake → quarantine; incomplete profile →
re-extract loop (bounded ×2); plan review verdict routes revise/approve;
rejection at gate → offboard path. Tool use via the schema-validated
`ToolCall`/dispatch registry pattern (proven in prior project, re-implemented
fresh in English).

**Production element (slide-mandated)**: pause at `hr_gate`, resume days later
from a new process — PostgresSaver, proven by spike.

Security (light, per owner): PII masking (both digit scripts) before LLM calls,
resume-text injection guard (resumes are untrusted!), role boundary (HR tools
registry has no finance tools — attempted call is refused + audited), budget
guard. Observability: Prometheus counters, per-case JSON traces with hash
chain + independent verifier, token/latency/cost meter per node & per case,
static HTML dashboard.

## Slices

| # | Slice | Test (name → acceptance) | Group | Mechanical? |
|---|---|---|---|---|
| 1 | Contracts & schemas (`src/schemas.py`) | `test_schemas.py` → typed contracts validate/reject correctly; audit event hash chain | core | no |
| 2 | LLM provider chain (`src/llm.py`) | `test_llm_layer.py` → failover 401/402/403/429 across providers, secret redaction, usage meter | core | no |
| 3 | Tool registry MCP-style (`src/tools.py`) | `test_tools.py` → declared inputSchema, validated dispatch, execution log, **role boundary: finance tool absent + refusal audited** | core | no |
| 4 | ReAct loop (`src/agents/react.py`) | `test_react.py` → earliest-match wins, bounded, scratchpad memory, forced-call honesty flag | core | no |
| 5 | Guardrails (`src/guardrails/`) | `test_guardrails.py` → injection flag on resume, PII masking both digit scripts, budget, size | core | no |
| 6 | Real agents + prompts (`src/agents/real.py`) | `test_agents.py` → JSON parsing, citation validation, per-agent contracts (stubbed LLM) | agents | no |
| 7 | State graph (`src/graph/build.py`) | `test_graph_paths.py` → all conditional paths, bounded loops, multi-event hash chain, effects called | agents | no |
| 8 | Postgres persistence (`src/checkpointing.py`) | `test_checkpoint_resume.py` → resume from subprocess against Docker PG; sqlite fallback | prod | no |
| 9 | Effects + templates (`src/effects.py`, `templates/`) | `test_effects.py` → contract rendered from Jinja2, IT ticket row written, relative paths | prod | partly |
| 10 | Observability (`src/observability/`) | `test_observability.py` → trace writer, chain verifier CLI, metrics snapshot, dashboard render wired | prod | partly |
| 11 | Pipeline + CLI (`main.py`, `src/pipeline.py`) | `test_pipeline.py` → guard order, unique thread ids, per-case budget; CLI: run/resume/attack/verify-traces | prod | no |
| 12 | FastAPI service (`src/app.py`) | `test_service_api.py` → token gates (503 closed default), rate limit, per-request isolation, HITL cycle | deploy | no |
| 13 | Docker (Dockerfile + **docker-compose.yml**: app+postgres) | build log + live HTTP evidence | deploy | partly |
| 14 | Executed notebook `capstone.ipynb` | notebook runs top-to-bottom with saved outputs per rubric row | evidence | no |
| 15 | Evidence capture + docs (README EN, live-run, rubric-check, pentest) | `verify-traces` exit 0; every number machine-checked against files | evidence | partly |

Integration test (`tests/test_integration_e2e.py`, written RED now, xfail-marked):
hired JSON → paused `awaiting_approval` → subprocess resume approve →
`contract.md` exists + IT ticket row exists + trace chain intact.

## Project constraints

- CLAUDE.md safety rules (no writes outside folder; Idea-1 untouchable).
- Evidence from clean state; single capture at a time; unique thread ids.
- No fabricated numbers anywhere; docs numbers machine-verified before push.

## Assumptions & spikes

| Assumption | Status |
|---|---|
| PostgresSaver CM + setup() + cross-process resume | **PROVEN** (spike_postgres.py, 2026-08-13) — delete spike after baseline |
| psycopg needs [binary] on Windows | PROVEN (install log) |
| Mistral medium handles English JSON agent outputs at temp 0 | Low risk (stronger in EN than AR); verify at slice 6 live probe |
| docker-compose app+pg networking | Verify at slice 13 (compose `depends_on` + healthcheck) |

## Progress log

| Time | Slice | Status | Transitions | Spikes | SHA |
|---|---|---|---|---|---|
| 2026-08-13 01:30 | PRD drafted | — | — | 1 (postgres, passed) | — |

## Documented shortfalls

(none yet)

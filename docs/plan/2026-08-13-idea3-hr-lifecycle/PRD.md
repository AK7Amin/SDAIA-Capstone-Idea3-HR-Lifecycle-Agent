# Employee Onboarding & Lifecycle Agent (Capstone Idea 3)

**Date**: 2026-08-13 · **Folder**: `docs/plan/2026-08-13-idea3-hr-lifecycle/`

## Run settings

| Item | Value |
|---|---|
| Strongest available model | `opus` (main loop runs on Opus 5) — pinned for assumption spikes and diagnostic spikes |
| Critics/review model | `fable` — **owner override** of the default map ("نستفيد من فيبل 5 والباقي على اوبس 5") |
| Plan-approval pause | No — proceed directly after critique round 1 |
| Push at boundaries | Yes — continuous to `AK7Amin/SDAIA-Capstone-Idea3-HR-Lifecycle-Agent` (owner approved; rubric grades incremental history) |
| Execution mode | Parallel for slices marked independent; central commits by the main loop only |
| Critics count | 3 — architecture & correctness (fixed), rubric/requirements fit (fixed), evidence & TDD (open angle: this project is graded on evidence). Security not a separate critic: owner scoped it out ("بلا سيكيوريتي واجد"); its rubric slice is covered inside the two fixed angles |

## Problem

When HR marks a candidate as *hired*, a chain of slow manual work starts:
reading the resume, designing a training plan, drafting the contract notice,
and asking IT to provision accounts. Each handoff loses context and nothing is
auditable. This system automates the chain **without giving up governance**.

## Success criteria

- Official rubric ≥ 90 — all six deliverables, none under its 40% floor.
- Integration test `tests/test_integration_e2e.py::test_full_onboarding_cycle_pauses_resumes_and_produces_artifacts`
  green without its xfail mark (lifted by the final slice).
- Evidence captured live for every rubric row: executed notebook + raw logs +
  machine-verified numbers.

## Out of scope

OCR, GUI, real HRIS/Active-Directory integration, Arabic-language content
(English run; Arabic-digit PII masking still included — lesson carried over),
Grafana/Redis extras.

## Locked decisions (from the interview)

| Decision | Why | Rejected alternative |
|---|---|---|
| Idea #3 HR lifecycle | Minimal security surface so Fable critics work freely | #2 SOC (all security) |
| English codebase & docs | Owner preference | Arabic (previous project) |
| **PostgresSaver** in Docker (port 5433, db `hr_agent`) | Slide names it for idea 3; Docker proven tonight; differentiates from previous capstone | SqliteSaver as primary (kept as documented fallback) |
| No plan-approval pause | Owner sleeps; pattern proven last run | Pause on PRD |
| Continuous push | Rubric: "kept documented and continuously updated" | Local-only |
| Target 90+, rubric-only scope | Owner | Max-effort extras |

## Project constraints (from CLAUDE.md)

- Never write outside the project folder; sibling `...-Idea-1` is untouchable.
- Synthetic data only; never read real personal documents (R021).
- Evidence from clean state; unique thread ids; single capture at a time.
- Honest attribution (`decision_source` pattern); relative paths in artifacts;
  no secrets outside `.env`; central redaction.
- TDD: red before green; every doc claim wired or deleted.

## Implementation decisions

**Coordination: centralized** — the LangGraph `StateGraph` is the coordinator;
agents communicate only through typed Pydantic contracts in shared state.

Agents (nodes): `intake` (guard + validate hired-candidate JSON — untrusted
input), `profile_analyst` (resume → `CandidateProfile`), `training_planner`
(**Plan-and-Execute** → `TrainingPlan`), `plan_reviewer` (**Reflexion** critique,
revise loop bounded ×1), `contract_drafter` (Jinja2 + typed fields →
`ContractDraft`), `hr_gate` (**HITL interrupt**), `it_provisioner` (**ReAct** with
MCP-style validated tools → `ProvisionResult`), `notifier` (welcome docs).

Conditional edges: invalid intake → quarantine; incomplete profile → re-extract
(bounded ×2); review verdict routes revise/approve; gate rejection → offboard.
Named patterns: Plan-and-Execute, Reflexion, ReAct — all three live in traces.

Tooling: schema-declared tools (`inputSchema`), validated `ToolCall`, single
dispatcher, execution log. **Role boundary**: the HR registry simply has no
finance tools; an attempted call is refused and audited.

Security (scoped light): PII masking (both digit scripts), resume-injection
guard, budget guard, size guard. Observability: Prometheus counters, per-case
hash-chained JSON traces + independent `verify-traces`, token/latency/cost
meter (per node, case, provider), static HTML dashboard rendered every run.

## Assumptions & spikes

| Assumption | Risky? | Spike result |
|---|---|---|
| PostgresSaver: CM construction, setup(), cross-process resume | Yes | **PROVEN** 2026-08-13 (`spike_postgres.py`): `from_conn_string` IS a CM (opposite of SqliteSaver), `setup()` idempotent, resume on fresh connection keeps state. Delete spike after baseline |
| psycopg needs `[binary]` on Windows | Yes | PROVEN — plain install lacks libpq |
| Long-lived saver for service/CLI (outside `with`) | Yes | Verify in slice 8: `PostgresSaver(Connection.connect(dsn))` non-CM path or pool; failure mode when PG is down must be a clean error |
| Mistral-medium handles English JSON agent output @ temp 0 | Low | Probe at slice 6 |
| docker-compose app+pg networking & healthcheck | Medium | Verify at slice 13 |

## Slices

| # | Slice | Group | Files | Test | Acceptance | Mechanical? | Independent of |
|---|---|---|---|---|---|---|---|
| 1 | Contracts & schemas | core | `src/schemas.py` | `test_schemas.py` | Typed contracts validate/reject; frozen `AuditEvent` + `verify_chain` | no | — |
| 2 | LLM provider chain | core | `src/llm.py` | `test_llm_layer.py` | Failover 401/402/403/429 across providers; redaction; meter per node/case/provider | no | 3,4,5 |
| 3 | MCP-style tool registry | core | `src/tools.py` | `test_tools.py` | Declared `inputSchema`; validated dispatch by name; execution log; finance-tool refusal audited | no | 2,5 |
| 4 | ReAct loop | core | `src/agents/react.py` | `test_react.py` | Earliest-match wins; bounded; scratchpad; `forced_first_call` honesty flag | no | 2,5 |
| 5 | Guardrails | core | `src/guardrails/` | `test_guardrails.py` | Injection flag on resume text; PII both digit scripts; budget; size | no | 2,3,4 |
| 6 | Agents & prompts | agents | `src/agents/real.py`, `prompts.py` | `test_agents.py` | Stubbed-LLM: JSON parsing, per-agent contracts, citation validation | no | — |
| 7 | State graph | agents | `src/graph/build.py`, `src/state.py` | `test_graph_paths.py` | Every conditional path + bounded loops; multi-event hash chain; effects invoked | no | — |
| 8 | Postgres persistence | prod | `src/checkpointing.py` | `test_checkpoint_resume.py` | Subprocess resume vs Docker PG; allow-list serializer; sqlite fallback; clean error when PG down | no | 9,10 |
| 9 | Effects & templates | prod | `src/effects.py`, `templates/` | `test_effects.py` | Contract rendered via Jinja2; IT ticket row; relative paths returned | partly | 8,10 |
| 10 | Observability | prod | `src/observability/` | `test_observability.py` | Trace writer; independent chain verifier; metrics snapshot; dashboard wired into run | partly | 8,9 |
| 11 | Pipeline & CLI | prod | `main.py`, `src/pipeline.py` | `test_pipeline.py` | Guard order; unique thread ids; per-case budget; CLI run/resume/attack/verify-traces | no | — |
| 12 | FastAPI service | deploy | `src/app.py` | `test_service_api.py` | Token gates closed-by-default (503); rate limit; per-request isolation; full HITL cycle | no | 13 |
| 13 | Docker & compose | deploy | `Dockerfile`, `docker-compose.yml` | build+run logs | app+postgres compose up; healthz; no `.env` baked | partly | 12 |
| 14 | Executed notebook | evidence | `capstone.ipynb` | manual run | Runs top-to-bottom, one section per rubric row, outputs saved | no | 15 |
| 15 | Evidence & docs | evidence | `README.md`, `reports/` | `verify-traces` exit 0 | Grader map with clickable links; every number machine-checked; xfail lifted → e2e green | partly | 14 |

## Test decisions

- **Seams**: agent callables injected into the graph (`AgentDeps` dataclass) —
  stubs in tests, real LLM wrappers in production; `LLMLayer._post` for provider
  tests; `Effects` protocol (Null vs File) for artifact tests; `registry.dispatch`
  for tool tests. Critique round 1 approval stands in for the skipped pause.
- **Not tested & why**: model output *quality* (prompt engineering — probed live,
  not unit-tested); Docker build itself (evidence log, not pytest); notebook
  execution (manual artifact).
- **Codebase precedent**: sibling `capstone-doc-lifecycle/tests/` — same style:
  Arabic-comment-free English tests here, stub-injected, no network in `pytest -q`.

## Progress log

| Slice | Status | Transitions | Diag. spikes | Last hypothesis | Red SHA | Green SHA |
|---|---|---|---|---|---|---|
| 1–15 | not started | 0/5 | 0/2 | — | | |

**Closing status**: (declared at phase-3 exit: SUCCESS / PARTIAL)

## Documented shortfalls

(none yet)

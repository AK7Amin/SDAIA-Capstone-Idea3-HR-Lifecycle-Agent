# Employee Onboarding & Lifecycle Agent

A multi-agent system that runs the HR onboarding lifecycle end to end: when a
candidate is marked **hired**, specialized agents analyze the résumé, design a
personalized training plan, critique and revise it, draft the contract notice,
**pause for human approval**, then provision IT accounts and issue the
documents — with tamper-evident audit trails, input/output guardrails, and
state that survives a restart.

> **Training programme**: SDAIA Academy — *Advanced Agentic AI Systems
> Engineering*, cohort **9–13 August 2026**, Riyadh.
> Capstone **Idea 3** (Employee Onboarding & Lifecycle).
> SDAIA Academy on GitHub: <https://github.com/SDAIAAcademy>

## For the grader — where every deliverable is proven

Each row links to code **and** to captured output from a real run.

| Rubric deliverable | Implementation | Live evidence |
|---|---|---|
| **D1** Agentic reasoning & tool use (15) | [`src/tools.py`](src/tools.py) (MCP-style declared schemas), [`src/agents/react.py`](src/agents/react.py) (ReAct), [`src/agents/real.py`](src/agents/real.py) | [`capstone.ipynb`](capstone.ipynb) §2–3 · [`reports/react/`](reports/react/) — per-step tool **arguments and observations** |
| **D2** Graph orchestration (20) | [`src/graph/build.py`](src/graph/build.py) — 10 nodes, 4 conditional routers, 2 bounded loops | [`capstone.ipynb`](capstone.ipynb) §4 (mermaid from the compiled graph) · [`tests/test_graph_paths.py`](tests/test_graph_paths.py) — every path asserted |
| **D3** Multi-agent & roles (20) | [`src/agents/real.py`](src/agents/real.py) (5 named roles), [`src/schemas.py`](src/schemas.py) (typed messages) | every trace shows each role's decision as its own contract |
| **D4** Security, guardrails, observability (20) | [`src/guardrails/`](src/guardrails/), [`src/observability/`](src/observability/) | [`reports/logs/01-live-run.log`](reports/logs/01-live-run.log) — injection removed, PII masked · [`reports/dashboard.html`](reports/dashboard.html) · notebook §6 tamper demo |
| **D5** Persistence, HITL, cloud (20) | [`src/checkpointing.py`](src/checkpointing.py) (PostgresSaver), [`src/app.py`](src/app.py), [`docker-compose.yml`](docker-compose.yml) | [`reports/logs/02-hitl-resume.log`](reports/logs/02-hitl-resume.log) — resume from a **new process** · [`reports/logs/05-compose-up.log`](reports/logs/05-compose-up.log) — **restart survival** |
| **D6** Documentation & evidence (5) | this README + [`capstone.ipynb`](capstone.ipynb) | [`reports/logs/`](reports/logs/) — five raw captures |

Verify it yourself in one minute — no keys, no Docker, no network:

```bash
pip install -r requirements-dev.txt
pytest -q                      # 473 tests
python main.py verify-traces   # recomputes every audit hash chain from disk
```

## Architecture

Course vocabulary: **nodes**, **edges**, **state**, **agents**, **tools**.

The **coordinator is the state graph itself** (centralized coordination): each
agent is a node, agents never call each other, and every message between them
is a typed Pydantic contract living in shared **state**. Conditional **edges**
route on those contracts; two loops are bounded and terminate by design.

```
intake ──valid──> profile_analyst ──complete──> training_planner ──> plan_reviewer
   │                    │  ▲                                              │
   │invalid             │  └───── re-extract (max 2) ────────┐      revise│(max 1)
   ▼                    │incomplete & exhausted              │            ▼
quarantine <────────────┘                                    └──── contract_drafter
                                                                          │
                                                          ┌───────────────┘
                                                          ▼
                                              hr_gate  ── interrupt() ── pause
                                               │ approve        │ reject
                                               ▼                ▼
                                        it_provisioner       offboard
                                               │
                                               ▼
                                            notifier  (contract.md, welcome.md)
```

| Agent (node) | Role | Reasoning pattern | Output contract |
|---|---|---|---|
| `intake` | validate & guard untrusted input | — | `CaseState` seed |
| `profile_analyst` | résumé → structured profile | extraction | `CandidateProfile` |
| `training_planner` | design onboarding plan | **Plan-and-Execute** | `TrainingPlan` |
| `plan_reviewer` | critique & demand revision | **Reflexion** | `ReviewVerdict` |
| `contract_drafter` | fill contract fields (state only) | template-fill | `ContractDraft` |
| `hr_gate` | human approval | HITL `interrupt()` | `GateDecision` |
| `it_provisioner` | create accounts & equipment | **ReAct** + tools | `ProvisionResult` |
| `notifier` | render and write documents | — | files on disk |

**Tools** are declared MCP-style (`name` / `description` / `inputSchema`),
dispatched **by name** after validation, and every call — including refusals —
is recorded. The HR registry contains no finance tools: an attempt to call
`payroll_adjust` is refused and audited (role boundary).

## Governance & safety properties

- **Nothing binding is written before a human approves.** The contract exists
  only in state until `notifier` runs after the gate — asserted by tests.
- **Tamper-evident audit trail**: every event carries the previous event's
  hash. `verify-traces` recomputes chains *and* each event's own digest, so
  editing a summary while keeping the hashes is caught too.
- **Honest attribution**: when the system forces a policy lookup the trace says
  `forced_first_call=True` — the model is never credited with a choice it did
  not make.
- **Untrusted résumés**: prompt-injection lines are detected (Unicode-normalized,
  zero-width evasion folds away), removed, and *listed* — never silently
  dropped. PII is masked in both Arabic-Indic and Latin digits.
- **Cost control**: per-case budget guard, per-node/provider token metering,
  service-side rate limiting.

## Running it

### Prerequisites
- Python 3.11+ (developed on 3.12), Docker (for Postgres), any
  OpenAI-compatible provider key.

### Setup
```bash
python -m venv .venv && . .venv/Scripts/Activate.ps1   # Windows
pip install -r requirements-dev.txt
cp .env.example .env        # then fill in the keys
docker run -d --name idea3-pg -e POSTGRES_PASSWORD=capstone \
  -e POSTGRES_DB=hr_agent -p 5433:5432 postgres:16-alpine
```

| Variable | Meaning |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | primary provider (any OpenAI-compatible endpoint) |
| `LLM_API_KEY_FALLBACK` | second key for the SAME provider, rotated to before the chain moves on |
| `LLM_API_KEY_2`, `LLM_BASE_URL_2`, `LLM_MODEL_2` | second provider — automatic failover on 401/402/403/429 *and* on a dead endpoint |
| `POSTGRES_DSN` | checkpointer DSN (compose uses `postgres:5432`) |
| `API_TOKEN` / `APPROVAL_API_TOKEN` | service auth; **unset = endpoint closed (503)** |
| `MAX_LLM_CALLS_PER_CASE` | per-case budget guard (default 12) |
| `HR_AGENT_STUBS=1` | deterministic stub agents — demos and tests, no model calls |

### Commands
```bash
python main.py run                          # process every case in sample_candidates/
python main.py resume <thread_id> approve   # or reject — resumes a paused case
python main.py attack                       # injection scenario, guarded vs unguarded
python main.py verify-traces                # independent audit-chain verification
python main.py demo-failover                # provider chain failing over
```

Expected output of `run`: one line per candidate with its status
(`awaiting_approval`, `quarantined`), the guard findings, and a resume hint;
then a verification table showing every trace chain-verified.

### Service (Docker)
```bash
docker compose up --build          # app + postgres, healthchecked
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/process -H "X-Api-Token: $API_TOKEN" \
     -H "Content-Type: application/json" -d '{"case": {...}}'
```
`POST /process`, `POST /resume` (separate privileged token), `GET /metrics`
(Prometheus), `GET /healthz`.

## Evidence from the live run (committed)

| Metric | Value |
|---|---|
| Candidates processed | 5 (clean hire, injected résumé, sparse, PII-heavy, invalid intake) |
| Model calls / tokens | 24 calls · 18,554 tokens (Mistral) |
| Traces, all chain-verified | 5/5 |
| Tests | **473** offline + 4 Docker-marked |

## Repository layout

```
src/          schemas, llm, tools, guardrails, agents/, graph/, observability/,
              effects, checkpointing, pipeline, app
tests/        13 suites; default run is offline (no keys, no Docker, no network)
templates/    Jinja2 contract & welcome documents
policies/     synthetic HR handbook (POL-001…POL-005)
sample_candidates/  five synthetic cases, one per designed path
reports/      traces/, react/, logs/, metrics-snapshot.json, dashboard.html
docs/plan/    PRD, critique round 1, run log — how this was built
capstone.ipynb  executed evidence notebook
```

## Cost optimization

Model tiering by task: the whole pipeline runs on one mid-tier model
(`mistral-medium`) because every agent returns short structured JSON; the
second provider exists for availability, not capability. The budget guard caps
calls per case (12), the Reflexion loop is bounded to one revision, and the
re-extract loop to two attempts — so a pathological case cannot spend without
limit. Per-node token accounting in `reports/metrics-snapshot.json` shows where
the budget actually goes (`training_planner` and `plan_reviewer` dominate).

## Not built (declared honestly)

OCR, a web UI, real HRIS/Active-Directory integration, Grafana/Redis. The
presentation deck is produced separately. Everything claimed above is wired and
tested; if a feature is not in the code, it is not in this README.

## License & attribution

Training project for the SDAIA Academy programme named above. All candidates,
policies and documents are **synthetic**.

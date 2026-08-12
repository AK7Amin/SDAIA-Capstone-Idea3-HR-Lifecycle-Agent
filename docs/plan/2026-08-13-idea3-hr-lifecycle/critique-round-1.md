# Critique Round 1 — Idea 3 HR Lifecycle

| Critic | Angle | Verdict |
|---|---|---|
| 1 | Rubric & requirements fit | NEEDS-FIXES |
| 2 | Architecture & state-graph correctness | NEEDS-FIXES |
| 3 | TDD discipline & evidence credibility | NEEDS-FIXES |

All three critics ran on `fable` (owner override). Every item below carries a
written decision; PRD and the red test were amended before the baseline commit.

## Blockers (all accepted)

### [B1] Evidence deferred to slices 14–15 = floor-rule kill switch — Critic 1
- **Where**: slice table; D6 evidence plan.
- **Impact**: unattended run stalling at slice ~12 → D6 = 0 → whole submission fails the 40% floor.
- **Decision**: ACCEPTED → notebook is a **living artifact**: skeleton committed with baseline; evidence cells appended, executed, committed at **each group boundary** (core/agents/prod/deploy). "Evidence for finished slices captured" is now part of each group's definition-of-done.

### [B2] PostgresSaver lifetime & thread-safety unaddressed for service — Critic 2
- **Where**: slices 8/12; spike only proved the CM path.
- **Impact**: single psycopg connection is not thread-safe → concurrent requests corrupt/crash; saver dies at end of `with`.
- **Decision**: ACCEPTED → `psycopg_pool.ConnectionPool(dsn, kwargs={autocommit, prepare_threshold=0, row_factory=dict_row})` + `PostgresSaver(pool)` opened in FastAPI lifespan; CLI keeps CM. `psycopg-pool` added explicitly to requirements (verified importable). Slice 8 tests the **service-style** construction too.

### [B3] Red test's stub seam is fictional — Critics 2+3
- **Where**: `test_integration_e2e.py` imported `build_graph_with_stubs`, never used it; `checkpoint_db` typed as sqlite path.
- **Impact**: at lift time either production runs stubs silently (simulation = zero) or e2e hits network.
- **Decision**: ACCEPTED → red test amended NOW: `graph=` parameter through `process_case`/`resume_case`; checkpointer factory not a path; `LLM_API_KEY` monkeypatch-deleted so an accidental real-agent path fails loudly. E2E file is **append-only** until the marker lift.

### [B4] `pytest -q` cannot import `src` — strict xfail permanently muzzled — Critic 3
- **Where**: no conftest/pyproject; bare pytest lacks repo root on sys.path.
- **Impact**: ModuleNotFoundError swallowed by xfail forever; "not built yet" indistinguishable from "misconfigured".
- **Decision**: ACCEPTED → `pyproject.toml` with `pythonpath=["."]` + marker config; verified bare `pytest -q` xfails for the **right** reason.

### [B5] strict xfail detonates at slice 11, not 15 — Critics 2+3
- **Decision**: ACCEPTED → marker lift moved into **slice 11's acceptance**; e2e then guards slices 12–15 as a live regression.

## Major (accepted — folded into slice acceptance criteria)

| id | Item | Decision folded into |
|---|---|---|
| M1 (C1) | Rubric demands *shown* attacks/failover, not unit tests | Notebook cell spec (attack CLI blocked; dead-URL provider 1 → provider 2 answers live; PII before/after; metrics after) |
| M2 (C1) | Page-2 GitHub mandates unowned (attribution, SDAIAAcademy link, expected output) | Slice 15 checklist; `.gitignore`/`.env.example` already in first commit |
| M3 (C1) | No single grader entry point | README rubric-evidence map, first section (proven pattern) |
| M4 (C1) | No pre-declared cut line | Cut order declared: slice 12 first (compose still satisfies D5), dashboard second, verifier CLI third; 14–15 never cut |
| M5 (C1) | Restart proof illegible in one kernel | Notebook resumes via `subprocess` printing **PIDs** in captured output |
| M6 (C2) | `quarantine`/`offboard` routed to but don't exist | Added as real terminal nodes with audit events + contracts; slice 7 asserts their events |
| M7 (C2) | `interrupt()` re-execution semantics | `hr_gate` = `interrupt()` as FIRST statement, zero pre-effects; pause event + `awaiting_approval` synthesized by pipeline on `__interrupt__`; typed `GateDecision` |
| M8 (C2) | Loop exhaustion undefined | Re-extract ×2 exhausted → quarantine. Revise ×1 exhausted → proceed to gate with `reviewer_concerns` attached (human decides; honest, governance-aligned). Counters written only by owning nodes |
| M9 (C2) | Contract file written pre-approval violates governance | `contract_drafter` produces **state-only** draft; file writes happen post-gate in `notifier`; test asserts outbox empty while `awaiting_approval` |
| M10 (C2) | PG-down failure mode | Fail fast, actionable message, exit 2; sqlite only via explicit `--checkpointer sqlite`, never auto-fallback |
| M11 (C2) | audit_trail reducer + hash stability across serializer | Slice 1: `Annotated[list, add]` reducer; canonical sorted-keys hashing; round-trip test through the actual checkpointer serializer |
| M12 (C3) | E2E blind to on-disk observability (prior defect could recur) | E2E asserts trace **file** exists, verifies chain from file, dashboard/metrics artifact contains this thread_id |
| M13 (C3) | E2E invents contracts nowhere frozen | `FileEffects(root)`, `it_tickets()`, `outbox/<id>/contract.md`, status literals — frozen verbatim in slices 1/9 acceptance |
| M14 (C3) | `.env` loading has no structural prevention (prior defect #2) | Slice 11 test: entrypoint loads `.env` from file; slice 15: fresh-clone-follow-README transcript captured |
| M15 (C3) | Docker PG test can skip silently | `@pytest.mark.docker`; default addopts `-m "not docker"`; evidence run executes `pytest -m docker -rs` and **fails on SKIPPED**; spike kept until slice 8 green |
| M16 (C3) | Verifier has no negative controls | Slice 10 red tests: tampered byte → nonzero naming broken link; duplicate thread → nonzero; CLI uses the same `verify_chain` as e2e |
| M17 (C3) | Citation validation has no corpus; chromadb dead dep | Policies = plain markdown fixture, loaded verbatim (honest at this scale); **chromadb dropped** from requirements; slice 6 red test: nonexistent policy id → downgrade |
| M18 (C3) | Rubric deliverables never enumerated; presentation gap | Deliverable→slice→artifact map added to PRD; `demo.md` script in slice 15; PPTX explicitly out of scope (owner does it after this run) |
| M19 (C3) | Requirements can't support the plan | Provider layer uses stdlib `urllib` (stated); `jupyter`/`nbclient`/`ipykernel` added to dev requirements |

## Minor (all accepted, cheap)

`reasoning_pattern` field emitted in trace events (C1) · `plan_reviewer` added to
e2e expected nodes incl. one revise cycle in stubs (C1+C2+C3) · mermaid diagram
in notebook + README (C1) · welcome-doc assertion in effects test (C1) ·
architecture write-up section using the five course words (C1) · cost-optimization
README subsection (C1) · 503 + full authorized HTTP cycle both captured (C1) ·
it_provisioner idempotency keyed (case_id, tool) + double-invoke test (C2) ·
Windows/UTF-8/subprocess env bullets on slices 8/9/13; DSN env-driven from
slice 8 (C2) · sqlite fallback constructed explicitly with
`check_same_thread=False` (C2) · per-request objects via `config["configurable"]`
(C2) · pytestmark moved from module to function (C3) · honesty-flag **fallback
branch** test named in slice 4 (C3) · determinism policy line: default suite =
zero network/Docker/keys (C3).

## Rejected

(none — every item accepted; M8's "force-approve" alternative rejected in favor
of proceed-to-gate-with-concerns, and M17's "wire ChromaDB" alternative rejected
in favor of dropping the dependency.)

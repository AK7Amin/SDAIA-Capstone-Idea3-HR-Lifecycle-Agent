# Demo script — Employee Onboarding & Lifecycle Agent

Four minutes, five beats. Everything below is a command that produces the
output shown in `reports/logs/` — no slideware.

## Before the room

- [ ] `docker start idea3-pg` (or `docker compose up -d postgres`) — the
      checkpointer must be reachable or the CLI exits 2 by design.
- [ ] `pytest -q` once: 473 passing, zero network, zero keys.
- [ ] Open `reports/logs/01-live-run.log` in a second window as the fallback: if the
      provider rate-limits mid-demo, show the captured run instead of improvising.
- [ ] **Never run two captures at once** — two runs of one case make two threads
      by design, but a half-finished capture makes a confusing story.

## 1 · The batch (60s)

```bash
python main.py run
```

Point at three things as they scroll:
- **CAND-002** — `guard: prompt injection detected`, two lines removed and
  *printed*. One of them tried to make the agent call `payroll_adjust`.
- **CAND-004** — `guard: PII masked (EMAIL, IBAN, NATIONAL_ID, PHONE)`, including
  a phone written in Arabic-Indic digits.
- **CAND-005** — `quarantined`: invalid intake never reaches an agent.

Every case ends `awaiting_approval`: nothing binding was written.

## 2 · Governance (30s)

```bash
ls outbox/            # empty — no contract exists before a human approves
```

## 3 · The human gate, from a different process (60s)

```bash
python main.py resume <thread-id> approve      # a NEW OS process
```

Show: `completed`, the ReAct line
(`decision_source=model, forced_first_call=True`), the documents, and then

```bash
cat reports/react/<thread-id>.json | head -30
```

— the tool call with its **arguments and the real observation**, not a summary.
Then `resume <another-thread> reject` → `offboarded`, no documents.

## 4 · The audit trail cannot be forged quietly (45s)

```bash
python main.py verify-traces        # 5/5 chain verified, exit 0
```

Edit one `summary` string in any trace file, re-run: the verifier names the
event index and exits non-zero — it recomputes each event's digest, so editing
the text while keeping the hashes is caught too. Undo the edit.

## 5 · It is a deployable service (45s)

```bash
docker compose up -d --wait
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/process -d '{"case": {...}}'   # 401 without the token
docker compose restart app                                     # kill the app
curl -s -X POST localhost:8000/resume -H "X-Approval-Token: …" -d '{...}'
```

The last call resumes a case paused **before** the restart: state lives in the
Postgres volume, not the process. Captured in `reports/logs/05-compose-up.log`.

## The one-sentence opening

"An HR onboarding system where five specialised agents talk only through typed
contracts on a LangGraph state graph, every risky step waits for a human, and
every claim in the README has a captured run behind it."

## Questions to expect

- **"Is the model really calling tools?"** → `reports/react/*.json`: arguments in,
  observation out, per step. And the registry refuses `payroll_adjust` because
  the HR role simply has no finance tool.
- **"What if the model chose nothing?"** → the system forces the policy read and
  labels it `forced_first_call=true`. It never claims the model chose.
- **"What if Postgres is down?"** → exit code 2 with the fix quoted. It never
  falls back to sqlite silently: a quiet durability downgrade is worse than a stop.

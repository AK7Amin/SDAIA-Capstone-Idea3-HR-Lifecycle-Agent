"""End-to-end integration test — the executable success criterion.

Written RED before implementation. APPEND-ONLY until the xfail marker is
lifted (slice 11's exit criterion) — executors must never edit this file to
fit their code; the code must grow to fit this file.

Frozen contracts encoded here (critique round 1, B3/M13):
- `build_graph_with_stubs(effects, checkpointer) -> compiled graph`
- `process_case(case_file, graph) -> {"status", "thread_id"}`
- `resume_case(thread_id, decision, graph) -> {"status", "audit_trail"}`
- `FileEffects(root)`; `effects.it_tickets(candidate_id) -> list`
- Artifacts: `outbox/<id>/contract.md`, `outbox/<id>/welcome.md`
- Status literals: "awaiting_approval", "completed"
- On-disk evidence: `reports/traces/<thread_id>.json`, metrics/dashboard
  artifacts produced BY THIS RUN (must contain this thread_id).

Stubbed agents + sqlite checkpointer: zero network, zero Docker, zero keys.
"""
import json


def test_full_onboarding_cycle_pauses_resumes_and_produces_artifacts(
    tmp_path, monkeypatch
):
    # An accidental real-agent path must fail loudly, not call a provider.
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from src.effects import FileEffects
    from src.pipeline import build_graph_with_stubs, process_case, resume_case
    from src.schemas import verify_chain

    candidate = {
        "candidate_id": "CAND-001",
        "name": "Sara Alqahtani",
        "role": "Data Engineer",
        "start_date": "2026-09-01",
        "resume_text": "5 years building ETL pipelines with Spark and Airflow.",
    }
    case_file = tmp_path / "intake" / "CAND-001.json"
    case_file.parent.mkdir(parents=True)
    case_file.write_text(json.dumps(candidate), encoding="utf-8")

    effects = FileEffects(tmp_path)

    def make_graph():
        """Fresh graph per phase = new-process semantics on one sqlite file."""
        return build_graph_with_stubs(
            effects=effects, checkpoint_db=tmp_path / "state.sqlite"
        )

    # ---- Phase 1: run until the human gate ----
    result = process_case(case_file, graph=make_graph())
    assert result["status"] == "awaiting_approval"
    thread_id = result["thread_id"]

    # Governance: NOTHING binding exists on disk before approval (M9).
    assert not (tmp_path / "outbox").exists() or not any(
        (tmp_path / "outbox").rglob("*")
    ), "binding artifact written before human approval"

    # ---- Phase 2: resume from a FRESH graph instance ----
    resumed = resume_case(thread_id, "approve", graph=make_graph())
    assert resumed["status"] == "completed"

    # Real artifacts, not state flips:
    contract = tmp_path / "outbox" / "CAND-001" / "contract.md"
    welcome = tmp_path / "outbox" / "CAND-001" / "welcome.md"
    assert contract.exists() and "Sara Alqahtani" in contract.read_text(encoding="utf-8")
    assert welcome.exists()
    assert effects.it_tickets("CAND-001"), "IT provisioning ticket missing"

    # Audit trail: full path incl. the Reflexion reviewer and the gate.
    trail = resumed["audit_trail"]
    nodes = [e.node for e in trail]
    for expected in ("intake", "profile_analyst", "training_planner",
                     "plan_reviewer", "contract_drafter", "hr_gate",
                     "it_provisioner", "notifier"):
        assert expected in nodes, f"node {expected} missing from audit trail"
    assert verify_chain(trail)

    # On-disk observability produced BY THIS RUN (M12 — prior-project defect):
    trace_file = tmp_path / "reports" / "traces" / f"{thread_id}.json"
    assert trace_file.exists(), "per-case trace file not written"
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace["chain_intact"] is True
    assert [e["node"] for e in trace["events"]].count("intake") == 1

    metrics_file = tmp_path / "reports" / "metrics-snapshot.json"
    assert metrics_file.exists(), "metrics snapshot not written"
    assert thread_id in metrics_file.read_text(encoding="utf-8"), (
        "metrics artifact is stale — not produced by this run"
    )

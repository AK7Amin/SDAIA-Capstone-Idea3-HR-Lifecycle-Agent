"""End-to-end integration test — the executable success criterion.

Written RED before any implementation (xfail until the final slice lifts it).
Flow under test:
    hired-candidate JSON → intake/profile/plan/review/draft →
    PAUSED at hr_gate (awaiting_approval) →
    resumed with "approve" from a FRESH graph instance (new-process semantics) →
    contract file rendered + IT provisioning ticket written + trace chain intact.

Uses stubbed agents (no LLM calls) and sqlite checkpointer (no Docker) so it
runs anywhere; the Postgres path has its own dedicated test in
test_checkpoint_resume.py.
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.xfail(
    reason="RED by design: implementation not built yet (lifted by final slice)",
    strict=True,
)


def test_full_onboarding_cycle_pauses_resumes_and_produces_artifacts(tmp_path):
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
    checkpoint_db = tmp_path / "state.sqlite"

    # Phase 1: run until the human gate.
    result = process_case(case_file, effects=effects, checkpoint_db=checkpoint_db)
    assert result["status"] == "awaiting_approval"
    thread_id = result["thread_id"]

    # Phase 2: resume from a FRESH graph instance (new-process semantics).
    resumed = resume_case(thread_id, "approve", effects=effects, checkpoint_db=checkpoint_db)
    assert resumed["status"] == "completed"

    # Real artifacts, not state flips:
    contract = tmp_path / "outbox" / "CAND-001" / "contract.md"
    assert contract.exists()
    assert "Sara Alqahtani" in contract.read_text(encoding="utf-8")
    assert effects.it_tickets("CAND-001"), "IT provisioning ticket missing"

    # Audit trail: full path visible incl. the gate, chain tamper-evident.
    trail = resumed["audit_trail"]
    nodes = [e.node for e in trail]
    for expected in ("intake", "profile_analyst", "training_planner",
                     "contract_drafter", "hr_gate", "it_provisioner", "notifier"):
        assert expected in nodes, f"node {expected} missing from audit trail"
    assert verify_chain(trail)

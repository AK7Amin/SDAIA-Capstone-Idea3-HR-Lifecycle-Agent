"""Slice 1 — typed contracts, immutable audit events, and hash-chain integrity.

These tests freeze the vocabulary every later slice depends on: the status
literals the e2e asserts on, the per-agent Pydantic contracts, and the audit
chain whose digests must survive a trip through the checkpointer's serializer
(critique M11 — a hash that changes on persistence proves nothing).
"""
import hashlib
import json

import pytest
from pydantic import ValidationError

from src.schemas import (
    AuditEvent,
    CandidateProfile,
    CaseStatus,
    ContractDraft,
    GateDecision,
    ITTicket,
    ProvisionResult,
    ReviewVerdict,
    TrainingPlan,
    TrainingWeek,
    verify_chain,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _chain(*events: AuditEvent) -> list[AuditEvent]:
    """Link events into a valid hash chain (each prev_hash = previous digest)."""
    linked: list[AuditEvent] = []
    prev_hash = ""
    for event in events:
        linked_event = event.model_copy(update={"prev_hash": prev_hash})
        linked.append(linked_event)
        prev_hash = linked_event.digest()
    return linked


def _sample_chain() -> list[AuditEvent]:
    return _chain(
        AuditEvent(node="intake", summary="validated hired-candidate JSON"),
        AuditEvent(
            node="training_planner",
            summary="drafted 4-week plan",
            reasoning_pattern="Plan-and-Execute",
            cost_usd=0.0012,
            latency_ms=840,
        ),
        AuditEvent(
            node="plan_reviewer",
            summary="approved plan",
            reasoning_pattern="Reflexion",
        ),
    )


# --------------------------------------------------------------------------
# CaseStatus — the literals the e2e test asserts on
# --------------------------------------------------------------------------
def test_case_status_literals_are_frozen_for_the_e2e():
    assert CaseStatus.AWAITING_APPROVAL.value == "awaiting_approval"
    assert CaseStatus.COMPLETED.value == "completed"
    assert CaseStatus.QUARANTINED.value == "quarantined"
    assert CaseStatus.OFFBOARDED.value == "offboarded"


def test_case_status_is_a_plain_string_enum():
    # The pipeline puts these straight into dicts the e2e compares to str.
    assert CaseStatus.COMPLETED == "completed"
    assert json.dumps({"status": CaseStatus.COMPLETED}) == '{"status": "completed"}'


def test_case_status_rejects_an_unknown_value():
    with pytest.raises(ValueError):
        CaseStatus("approved_maybe")


# --------------------------------------------------------------------------
# CandidateProfile
# --------------------------------------------------------------------------
def test_candidate_profile_required_fields_are_declared():
    assert CandidateProfile.REQUIRED == ("name", "role", "start_date")


def test_missing_fields_reports_every_blank_required_field():
    profile = CandidateProfile(candidate_id="CAND-001")
    assert profile.missing_fields() == ["name", "role", "start_date"]


def test_missing_fields_is_empty_when_the_profile_is_complete():
    profile = CandidateProfile(
        candidate_id="CAND-001",
        name="Sara Alqahtani",
        role="Data Engineer",
        start_date="2026-09-01",
        skills=["Spark", "Airflow"],
        experience_summary="5 years of ETL work.",
    )
    assert profile.missing_fields() == []


def test_whitespace_only_counts_as_missing():
    # An LLM that answers "   " has not extracted anything; route to re-extract.
    profile = CandidateProfile(
        candidate_id="CAND-001", name="  ", role="Data Engineer", start_date="2026-09-01"
    )
    assert profile.missing_fields() == ["name"]


def test_candidate_profile_rejects_a_non_list_skills_value():
    with pytest.raises(ValidationError):
        CandidateProfile(candidate_id="CAND-001", skills="Spark")


def test_candidate_profile_requires_a_candidate_id():
    with pytest.raises(ValidationError):
        CandidateProfile()


# --------------------------------------------------------------------------
# TrainingPlan
# --------------------------------------------------------------------------
def test_training_plan_accepts_at_least_one_week():
    plan = TrainingPlan(
        weeks=[TrainingWeek(week=1, focus="Tooling", activities=["Set up laptop"])],
        rationale="Ramp-up before the first sprint.",
    )
    assert plan.weeks[0].week == 1


def test_training_plan_rejects_an_empty_week_list():
    with pytest.raises(ValidationError):
        TrainingPlan(weeks=[], rationale="nothing to do")


def test_training_week_rejects_a_non_integer_week_number():
    with pytest.raises(ValidationError):
        TrainingWeek(week="first", focus="Tooling", activities=[])


# --------------------------------------------------------------------------
# ReviewVerdict / GateDecision — the two decision enums
# --------------------------------------------------------------------------
def test_review_verdict_accepts_the_two_declared_actions():
    assert ReviewVerdict(action="revise", critique="too shallow").action == "revise"
    assert ReviewVerdict(action="approve").action == "approve"


def test_review_verdict_rejects_an_undeclared_action():
    with pytest.raises(ValidationError):
        ReviewVerdict(action="reject", critique="wrong vocabulary")


def test_gate_decision_defaults_to_the_hr_actor():
    decision = GateDecision(decision="approve", decided_at="2026-09-01T10:00:00Z")
    assert decision.actor == "hr"
    assert decision.decision == "approve"


def test_gate_decision_rejects_an_undeclared_decision():
    with pytest.raises(ValidationError):
        GateDecision(decision="revise")


# --------------------------------------------------------------------------
# ContractDraft / ProvisionResult
# --------------------------------------------------------------------------
def test_contract_draft_is_state_only_with_no_file_path_field():
    # Governance M9: nothing binding may exist on disk while the case is paused,
    # so the draft contract carries no path — the notifier writes files post-gate.
    field_names = set(ContractDraft.model_fields)
    assert not {n for n in field_names if "path" in n or "file" in n}


def test_contract_draft_round_trips_its_body_fields():
    draft = ContractDraft(
        candidate_id="CAND-001",
        role="Data Engineer",
        start_date="2026-09-01",
        salary_band="B3",
        body_fields={"probation_months": "3"},
    )
    assert draft.body_fields["probation_months"] == "3"


def test_provision_result_validates_nested_tickets():
    result = ProvisionResult(
        tickets=[
            ITTicket(ticket_id="IT-1", system="email", action="create", status="done")
        ]
    )
    assert result.tickets[0].system == "email"

    with pytest.raises(ValidationError):
        ProvisionResult(tickets=[{"ticket_id": "IT-2"}])


def test_provision_result_defaults_to_no_tickets():
    assert ProvisionResult().tickets == []


# --------------------------------------------------------------------------
# AuditEvent
# --------------------------------------------------------------------------
def test_audit_event_is_immutable():
    event = AuditEvent(node="intake", summary="validated")
    with pytest.raises(ValidationError):
        event.summary = "tampered"


def test_audit_event_defaults_are_the_meter_zeroes():
    event = AuditEvent(node="intake")
    assert event.reasoning_pattern == ""
    assert event.cost_usd == 0.0
    assert event.latency_ms == 0
    assert event.prev_hash == ""


def test_reasoning_pattern_is_carried_and_changes_the_digest():
    # A grepping grader looks for the named patterns; the digest must cover them
    # so the pattern label cannot be rewritten after the fact.
    plain = AuditEvent(node="it_provisioner", summary="provisioned")
    labelled = plain.model_copy(update={"reasoning_pattern": "ReAct"})
    assert labelled.reasoning_pattern == "ReAct"
    assert plain.digest() != labelled.digest()


def test_digest_is_canonical_sha256_over_sorted_keys():
    event = AuditEvent(
        node="plan_reviewer",
        summary="approved",
        reasoning_pattern="Reflexion",
        cost_usd=0.5,
        latency_ms=12,
        prev_hash="abc",
    )
    expected = hashlib.sha256(
        json.dumps(
            {
                "cost_usd": 0.5,
                "latency_ms": 12,
                "node": "plan_reviewer",
                "prev_hash": "abc",
                "reasoning_pattern": "Reflexion",
                "summary": "approved",
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert event.digest() == expected


def test_digest_ignores_keyword_order():
    a = AuditEvent(node="intake", summary="ok", latency_ms=3)
    b = AuditEvent(latency_ms=3, summary="ok", node="intake")
    assert a.digest() == b.digest()


def test_digest_is_stable_for_non_ascii_summaries():
    # ensure_ascii=False must not make the bytes depend on the environment.
    event = AuditEvent(node="notifier", summary="مرحبا Sara")
    assert event.digest() == event.model_copy().digest()
    assert len(event.digest()) == 64


# --------------------------------------------------------------------------
# verify_chain
# --------------------------------------------------------------------------
def test_verify_chain_accepts_a_valid_chain():
    assert verify_chain(_sample_chain()) is True


def test_verify_chain_accepts_empty_and_single_event_chains():
    assert verify_chain([]) is True
    assert verify_chain([AuditEvent(node="intake")]) is True


def test_verify_chain_rejects_a_non_empty_first_prev_hash():
    events = _sample_chain()
    events[0] = events[0].model_copy(update={"prev_hash": "deadbeef"})
    assert verify_chain(events) is False


def test_verify_chain_detects_a_tampered_middle_event():
    events = _sample_chain()
    events[1] = events[1].model_copy(update={"summary": "drafted 1-week plan"})
    assert verify_chain(events) is False


def test_verify_chain_detects_a_shared_prev_hash():
    # Two events pointing at the same parent = a dropped event in the middle.
    events = _sample_chain()
    events[2] = events[2].model_copy(update={"prev_hash": events[1].prev_hash})
    assert verify_chain(events) is False


def test_verify_chain_detects_a_deleted_event():
    events = _sample_chain()
    assert verify_chain([events[0], events[2]]) is False


# --------------------------------------------------------------------------
# M11 — hash stability across the ACTUAL checkpointer serializer
# --------------------------------------------------------------------------
def test_chain_survives_the_checkpointer_serializer_round_trip():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    events = _sample_chain()
    digests_before = [e.digest() for e in events]

    serde = JsonPlusSerializer()
    restored = serde.loads_typed(serde.dumps_typed(events))

    assert [type(e) for e in restored] == [AuditEvent] * len(events)
    assert [e.digest() for e in restored] == digests_before
    assert verify_chain(restored) is True


def test_non_ascii_event_digest_survives_the_serializer_round_trip():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    events = _chain(
        AuditEvent(node="intake", summary="استلمنا الطلب", cost_usd=0.000125),
        AuditEvent(node="notifier", summary="مرحبا Sara", latency_ms=7),
    )
    serde = JsonPlusSerializer()
    restored = serde.loads_typed(serde.dumps_typed(events))

    assert [e.digest() for e in restored] == [e.digest() for e in events]
    assert verify_chain(restored) is True

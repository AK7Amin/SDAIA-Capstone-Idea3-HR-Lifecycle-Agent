"""Slice 7 — the state graph: every conditional path, both bounded loops.

Written RED before `src/graph/build.py` and `src/state.py` exist. Offline by
design: the five agents are injected closures (`AgentDeps`), the effects layer
is a spy, and the checkpointer is `InMemorySaver` — zero network, zero Docker,
zero keys.

Five properties are locked here because each one is a governance claim the
project makes out loud:

1. **The gate pauses before anything binding happens** (M9). At the pause the
   spy has recorded no contract and no welcome file — and it still has not
   recorded them after a SECOND invoke of the same thread, because LangGraph
   re-runs the whole gate node on every resume attempt (M7).
2. **The graph never invents `awaiting_approval`.** That status is synthesized
   by the pipeline layer when it sees `__interrupt__`; the graph writes only
   terminal statuses.
3. **Both loops are bounded and exhaust into a real node**: re-extraction into
   `quarantine`, Reflexion into `contract_drafter` *carrying the reviewer's
   concerns* — an exhausted critic must not be silently dropped, the human at
   the gate has to see what it complained about.
4. **The hash chain survives multi-event nodes.** A node emitting two audit
   events must chain them to each other, not hang both off the same parent.
   That defect (two siblings sharing one `prev_hash`) passes a naive
   "every event has a prev_hash" check and fails `verify_chain`; the
   `it_provisioner` tests below are that bug, frozen.
5. **Counters are written only by their owning node.** A path that never
   reaches the planner leaves `revise_count` at zero.
"""
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph import (
    MAX_EXTRACT_ATTEMPTS,
    MAX_REVISIONS,
    AgentDeps,
    audit_chain,
    build_graph,
)
from src.schemas import (
    AuditEvent,
    CandidateProfile,
    ContractDraft,
    GateAction,
    ITTicket,
    ProvisionResult,
    ReviewAction,
    ReviewVerdict,
    TrainingPlan,
    TrainingWeek,
    verify_chain,
)

# --------------------------------------------------------------------------
# synthetic case data (R021: synthetic only, never a real person's file)
# --------------------------------------------------------------------------
META = {
    "candidate_id": "CAND-001",
    "name": "Sara Alqahtani",
    "role": "Data Engineer",
    "start_date": "2026-09-01",
}
MASKED_RESUME = "5 years of ETL pipelines. Reach me at [EMAIL] or [PHONE]."

COMPLETE = CandidateProfile(
    candidate_id="CAND-001",
    name="Sara Alqahtani",
    role="Data Engineer",
    start_date="2026-09-01",
    skills=["Spark", "Airflow"],
    experience_summary="5 years of ETL pipelines.",
)
INCOMPLETE = CandidateProfile(candidate_id="CAND-001", name="Sara Alqahtani")

APPROVE = ReviewVerdict(action=ReviewAction.APPROVE, critique="Solid plan.")
REVISE = ReviewVerdict(
    action=ReviewAction.REVISE,
    critique="Week 1 has no Spark work.",
    concerns=["no Spark in week 1", "no mentor named"],
)


# --------------------------------------------------------------------------
# injected agents — closures that replay a script and record their inputs
# --------------------------------------------------------------------------
def _replaying(sequence):
    """Return a picker over `sequence` whose LAST item repeats forever.

    Repeating the tail is what makes "the extractor never gets it right" and
    "the critic is never satisfied" one-liners instead of long scripts.
    """
    items = list(sequence)

    def pick(index: int):
        return items[min(index, len(items) - 1)]

    return pick


class Agents:
    """The five injected agents plus the call log each one writes."""

    def __init__(self, profiles=(COMPLETE,), verdicts=(APPROVE,), ticket_count=2):
        self._profiles = _replaying(profiles)
        self._verdicts = _replaying(verdicts)
        self._ticket_count = ticket_count
        self.analyze_calls: list[tuple[str, dict]] = []
        self.plan_calls: list[tuple[CandidateProfile, str]] = []
        self.review_calls: list[TrainingPlan] = []
        self.draft_calls: list[CandidateProfile] = []
        self.provision_calls: list[CandidateProfile] = []

    def analyze_profile(self, masked_resume, candidate_meta):
        self.analyze_calls.append((masked_resume, dict(candidate_meta)))
        return self._profiles(len(self.analyze_calls) - 1)

    def plan_training(self, profile, critique):
        self.plan_calls.append((profile, critique))
        return TrainingPlan(
            weeks=[
                TrainingWeek(
                    week=1, focus="Data platform tour", activities=["Read runbooks"]
                )
            ],
            rationale=f"revision {len(self.plan_calls)}",
        )

    def review_plan(self, profile, plan):
        self.review_calls.append(plan)
        return self._verdicts(len(self.review_calls) - 1)

    def draft_contract(self, profile):
        self.draft_calls.append(profile)
        return ContractDraft(
            candidate_id=profile.candidate_id,
            role=profile.role,
            start_date=profile.start_date,
            salary_band="B3",
        )

    def provision_it(self, profile):
        self.provision_calls.append(profile)
        systems = ["email", "laptop", "vpn"][: self._ticket_count]
        return ProvisionResult(
            tickets=[
                ITTicket(
                    ticket_id=f"IT-{index + 1}",
                    system=system,
                    action="create_account",
                    status="done",
                )
                for index, system in enumerate(systems)
            ]
        )

    def as_deps(self, effects=None) -> AgentDeps:
        return AgentDeps(
            analyze_profile=self.analyze_profile,
            plan_training=self.plan_training,
            review_plan=self.review_plan,
            draft_contract=self.draft_contract,
            provision_it=self.provision_it,
            effects=effects,
        )


class SpyEffects:
    """Duck-typed stand-in for `src.effects.FileEffects` that touches no disk."""

    def __init__(self):
        self.contracts: list[tuple] = []
        self.welcomes: list[tuple] = []
        self.tickets: list[tuple] = []
        self.quarantines: list[tuple] = []
        self.order: list[str] = []

    def write_contract(self, case_id, draft, profile):
        self.order.append("write_contract")
        self.contracts.append((case_id, draft, profile))
        return f"outbox/{case_id}/contract.md"

    def write_welcome(self, case_id, profile):
        self.order.append("write_welcome")
        self.welcomes.append((case_id, profile))
        return f"outbox/{case_id}/welcome.md"

    def provision_tickets(self, case_id, result):
        self.order.append("provision_tickets")
        self.tickets.append((case_id, result))

    def quarantine_case(self, case_id, reason):
        self.order.append("quarantine_case")
        self.quarantines.append((case_id, reason))


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def make_app(*, profiles=(COMPLETE,), verdicts=(APPROVE,), ticket_count=2):
    """Compile a graph over stub agents; return `(app, agents, effects)`."""
    agents = Agents(profiles=profiles, verdicts=verdicts, ticket_count=ticket_count)
    effects = SpyEffects()
    app = build_graph(agents.as_deps(effects), InMemorySaver())
    return app, agents, effects


def thread():
    """A fresh thread id per test — never reuse one across cases."""
    return {"configurable": {"thread_id": f"case-{uuid.uuid4()}"}}


def start_state(**overrides):
    """The production-shaped input: masked resume + metadata, no audit seed."""
    state = {"candidate_meta": dict(META), "masked_resume": MASKED_RESUME}
    state.update(overrides)
    return state


def nodes_of(state) -> list[str]:
    return [event.node for event in state["audit_trail"]]


def interrupt_payload(result) -> dict:
    assert "__interrupt__" in result, "graph did not pause at the human gate"
    return result["__interrupt__"][0].value


# --------------------------------------------------------------------------
# 1. happy path up to the pause — governance ordering (M9)
# --------------------------------------------------------------------------
def test_happy_path_pauses_at_the_gate():
    app, agents, effects = make_app()
    config = thread()

    result = app.invoke(start_state(), config)

    assert "__interrupt__" in result
    assert nodes_of(result) == [
        "intake",
        "profile_analyst",
        "training_planner",
        "plan_reviewer",
        "contract_drafter",
    ]
    assert verify_chain(result["audit_trail"])
    assert len(agents.draft_calls) == 1


def test_nothing_binding_is_written_before_the_human_approves():
    app, _agents, effects = make_app()

    app.invoke(start_state(), thread())

    assert effects.contracts == []
    assert effects.welcomes == []
    assert effects.tickets == []
    assert effects.order == []


def test_graph_does_not_invent_the_awaiting_approval_status():
    """M7: the pause status belongs to the pipeline, not to the graph."""
    app, _agents, _effects = make_app()

    result = app.invoke(start_state(), thread())

    assert result.get("status", "") == ""


def test_gate_payload_shows_the_human_what_is_being_approved():
    app, _agents, _effects = make_app()

    payload = interrupt_payload(app.invoke(start_state(), thread()))

    assert payload["case_id"] == "CAND-001"
    assert payload["candidate"]["name"] == "Sara Alqahtani"
    assert payload["contract"]["role"] == "Data Engineer"
    assert payload["training_weeks"][0]["week"] == 1
    assert payload["reviewer_concerns"] == []
    assert "question" in payload


def test_second_invoke_of_a_paused_thread_still_fires_no_effects():
    """Node re-execution safety: the gate body must start with `interrupt()`."""
    app, _agents, effects = make_app()
    config = thread()

    first = app.invoke(start_state(), config)
    again = app.invoke(None, config)

    assert "__interrupt__" in again
    assert effects.order == []
    assert nodes_of(again) == nodes_of(first), "the gate committed state twice"


# --------------------------------------------------------------------------
# 2. resume — approve
# --------------------------------------------------------------------------
def test_resume_approve_completes_the_case():
    app, agents, effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    assert final["status"] == "completed"
    assert final["gate"].decision is GateAction.APPROVE
    for expected in (
        "intake",
        "profile_analyst",
        "training_planner",
        "plan_reviewer",
        "contract_drafter",
        "hr_gate",
        "it_provisioner",
        "notifier",
    ):
        assert expected in nodes_of(final), f"node {expected} missing from the trail"
    assert verify_chain(final["audit_trail"])


def test_resume_approve_invokes_every_effect_exactly_once():
    app, agents, effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    app.invoke(Command(resume={"decision": "approve"}), config)

    assert len(effects.contracts) == 1
    assert len(effects.welcomes) == 1
    assert len(effects.tickets) == 1
    assert effects.contracts[0][0] == "CAND-001"
    assert len(agents.provision_calls) == 1


def test_provisioning_happens_before_the_documents_are_written():
    app, _agents, effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    app.invoke(Command(resume={"decision": "approve"}), config)

    assert effects.order == ["provision_tickets", "write_contract", "write_welcome"]


def test_a_bare_string_decision_is_accepted_and_typed():
    app, _agents, _effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume="approve"), config)

    assert final["status"] == "completed"
    assert final["gate"].decision is GateAction.APPROVE


def test_an_unreadable_decision_is_refused_rather_than_guessed():
    app, _agents, effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    with pytest.raises(ValueError, match="maybe"):
        app.invoke(Command(resume={"decision": "maybe"}), config)

    assert effects.order == []


# --------------------------------------------------------------------------
# 3. resume — reject
# --------------------------------------------------------------------------
def test_resume_reject_offboards_and_writes_nothing():
    app, _agents, effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "reject"}), config)

    assert final["status"] == "offboarded"
    assert final["gate"].decision is GateAction.REJECT
    assert "offboard" in nodes_of(final)
    assert "it_provisioner" not in nodes_of(final)
    assert "notifier" not in nodes_of(final)
    assert effects.contracts == [] and effects.welcomes == []
    assert verify_chain(final["audit_trail"])


# --------------------------------------------------------------------------
# 4. invalid intake -> quarantine
# --------------------------------------------------------------------------
def test_invalid_intake_quarantines_the_case():
    app, agents, effects = make_app()
    broken = dict(META)
    del broken["role"]

    final = app.invoke(start_state(candidate_meta=broken), thread())

    assert final["status"] == "quarantined"
    assert nodes_of(final) == ["intake", "quarantine"]
    assert agents.analyze_calls == [], "an invalid case reached the agents"
    assert len(effects.quarantines) == 1
    assert "role" in effects.quarantines[0][1]
    assert verify_chain(final["audit_trail"])


def test_quarantine_reason_names_every_missing_key():
    app, _agents, effects = make_app()
    broken = {"candidate_id": "CAND-777"}

    app.invoke(start_state(candidate_meta=broken), thread())

    reason = effects.quarantines[0][1]
    for key in ("name", "role", "start_date"):
        assert key in reason


def test_a_case_that_dies_at_intake_touches_no_other_counter():
    app, _agents, _effects = make_app()
    broken = dict(META, name="   ")

    final = app.invoke(start_state(candidate_meta=broken), thread())

    assert final["status"] == "quarantined"
    assert final.get("extract_attempts", 0) == 0
    assert final.get("revise_count", 0) == 0


# --------------------------------------------------------------------------
# 5. bounded re-extraction loop
# --------------------------------------------------------------------------
def test_re_extraction_recovers_on_the_second_attempt():
    app, agents, _effects = make_app(profiles=(INCOMPLETE, COMPLETE))
    config = thread()

    result = app.invoke(start_state(), config)

    assert "__interrupt__" in result
    assert len(agents.analyze_calls) == 2
    assert result["extract_attempts"] == 2
    assert nodes_of(result).count("profile_analyst") == 2
    assert verify_chain(result["audit_trail"])


def test_re_extraction_is_bounded_and_exhausts_into_quarantine():
    app, agents, effects = make_app(profiles=(INCOMPLETE,))

    final = app.invoke(start_state(), thread())

    assert len(agents.analyze_calls) == MAX_EXTRACT_ATTEMPTS == 2
    assert final["status"] == "quarantined"
    assert final["extract_attempts"] == 2
    assert agents.plan_calls == []
    assert nodes_of(final).count("profile_analyst") == 2
    assert "quarantine" in nodes_of(final)
    assert "start_date" in effects.quarantines[0][1]
    assert verify_chain(final["audit_trail"])


def test_the_analyst_only_ever_sees_the_masked_resume():
    app, agents, _effects = make_app()

    final = app.invoke(start_state(), thread())

    assert agents.analyze_calls[0][0] == MASKED_RESUME
    assert "resume_text" not in final, "raw resume text leaked into graph state"


def test_intake_drops_unmasked_resume_text_riding_along_in_the_metadata():
    """Whatever the caller hands over, only the masked copy is persisted."""
    app, agents, _effects = make_app()
    payload = dict(META, resume_text="Call me on 0500000000, my IBAN is SA00")

    final = app.invoke(start_state(candidate_meta=payload), thread())

    assert "resume_text" not in final["candidate_meta"]
    assert "resume_text" not in agents.analyze_calls[0][1]
    assert final["candidate_meta"]["candidate_id"] == "CAND-001"


# --------------------------------------------------------------------------
# 6. bounded Reflexion loop
# --------------------------------------------------------------------------
def test_reflexion_revises_once_then_approves():
    app, agents, _effects = make_app(verdicts=(REVISE, APPROVE))

    result = app.invoke(start_state(), thread())

    assert "__interrupt__" in result
    assert len(agents.plan_calls) == 2
    assert len(agents.review_calls) == 2
    assert result["revise_count"] == 1
    assert nodes_of(result).count("training_planner") == 2
    assert result["reviewer_concerns"] == []


def test_the_revised_plan_is_written_against_the_critique():
    app, agents, _effects = make_app(verdicts=(REVISE, APPROVE))

    app.invoke(start_state(), thread())

    assert agents.plan_calls[0][1] == ""
    assert agents.plan_calls[1][1] == REVISE.critique


def test_an_unsatisfiable_critic_exhausts_into_the_gate_carrying_concerns():
    """Honesty: the human decides on a plan the critic still dislikes."""
    app, agents, _effects = make_app(verdicts=(REVISE,))

    result = app.invoke(start_state(), thread())

    payload = interrupt_payload(result)
    assert len(agents.plan_calls) == MAX_REVISIONS + 1 == 2
    assert result["revise_count"] == 2
    assert result["reviewer_concerns"] == REVISE.concerns
    assert payload["reviewer_concerns"] == REVISE.concerns
    assert "contract_drafter" in nodes_of(result)
    assert verify_chain(result["audit_trail"])


# --------------------------------------------------------------------------
# 7. multi-event nodes and the hash chain
# --------------------------------------------------------------------------
def test_a_multi_event_node_chains_its_own_events():
    app, _agents, _effects = make_app(ticket_count=3)
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    trail = final["audit_trail"]
    assert nodes_of(final).count("it_provisioner") >= 3
    assert verify_chain(trail), "multi-event node broke the hash chain"


def test_no_two_audit_events_share_a_parent():
    """The exact shape of the old defect: siblings hanging off one prev_hash."""
    app, _agents, _effects = make_app(ticket_count=3)
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    parents = [event.prev_hash for event in final["audit_trail"]]
    assert len(parents) == len(set(parents))


def test_audit_chain_seeds_from_the_tail_of_the_existing_trail():
    first = AuditEvent(node="intake", summary="seed")
    events = audit_chain(
        {"audit_trail": [first]},
        [
            {"node": "it_provisioner", "summary": "a"},
            {"node": "it_provisioner", "summary": "b"},
        ],
    )

    assert len(events) == 2
    assert events[0].prev_hash == first.digest()
    assert events[1].prev_hash == events[0].digest()
    assert verify_chain([first, *events])


def test_the_tool_call_counter_tracks_the_provisioning_steps():
    app, _agents, _effects = make_app(ticket_count=3)
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    assert final["tool_calls"] == 3
    assert len(final["provision"].tickets) == 3


# --------------------------------------------------------------------------
# 8. named reasoning patterns (D1/D3 — a grepping grader must find them)
# --------------------------------------------------------------------------
def test_every_agent_labels_the_pattern_it_used():
    app, _agents, _effects = make_app()
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    patterns = {event.node: event.reasoning_pattern for event in final["audit_trail"]}
    assert patterns["profile_analyst"] == "extraction"
    assert patterns["training_planner"] == "plan-and-execute"
    assert patterns["plan_reviewer"] == "reflexion"
    assert patterns["it_provisioner"] == "react"
    assert patterns["hr_gate"] == "human-in-the-loop"


def test_deterministic_nodes_claim_no_reasoning_pattern():
    """Honest attribution: intake validates, it does not reason."""
    app, _agents, _effects = make_app()

    final = app.invoke(start_state(), thread())

    intake = next(e for e in final["audit_trail"] if e.node == "intake")
    assert intake.reasoning_pattern == ""


# --------------------------------------------------------------------------
# 9. construction
# --------------------------------------------------------------------------
def test_effects_are_optional_and_default_to_a_no_op():
    agents = Agents()
    app = build_graph(agents.as_deps(), InMemorySaver())
    config = thread()
    app.invoke(start_state(), config)

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    assert final["status"] == "completed"


def test_the_graph_compiles_without_a_checkpointer_for_diagrams():
    app = build_graph(Agents().as_deps())

    mermaid = app.get_graph().draw_mermaid()

    for node in ("intake", "hr_gate", "quarantine", "offboard", "notifier"):
        assert node in mermaid

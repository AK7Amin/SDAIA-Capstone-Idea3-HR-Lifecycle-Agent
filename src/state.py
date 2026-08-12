"""What one onboarding case carries between nodes, and the seam agents plug into.

Coordination in this system is centralized: nodes never call each other, they
only read and write :class:`CaseState`. Two shapes in here are load-bearing.

**The audit trail is the only accumulating channel.** It is annotated with
``operator.add``, so a node returns *the events it just emitted* and LangGraph
appends them. Every other field is last-write-wins, which is what makes the
"counters are written only by their owning node" rule enforceable: if two nodes
both returned ``extract_attempts`` the last one would silently win, and a
bounded loop with a miscounted bound is an unbounded loop.

**The raw resume never enters state.** Only ``masked_resume`` does — the PII
masking (both digit scripts) happens in the guardrails layer before the graph
is invoked, so nothing downstream — checkpoint rows, trace files, gate payloads
— can leak text the guard already cleaned.

:class:`AgentDeps` is the injection seam. The graph knows the *shape* of its
five agents and nothing about how they think: tests pass scripted closures,
production passes LLM-backed callables. That is why `build_graph` can be tested
without a network, and why an agent's prompt can change without touching the
orchestration.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Sequence, TypedDict

from src.schemas import (
    AuditEvent,
    CandidateProfile,
    ContractDraft,
    GateDecision,
    ProvisionResult,
    ReviewVerdict,
    TrainingPlan,
)

__all__ = [
    "AgentDeps",
    "AnalyzeProfile",
    "CaseState",
    "DraftContract",
    "PlanTraining",
    "ProvisionIT",
    "ReviewPlan",
]

# --------------------------------------------------------------------------
# agent signatures — the contract slice 6 implements and slice 11 wires
# --------------------------------------------------------------------------
#: ``(masked_resume, candidate_meta) -> CandidateProfile``
AnalyzeProfile = Callable[[str, dict[str, Any]], CandidateProfile]
#: ``(profile, critique) -> TrainingPlan`` — the critique is ``""`` on the first
#: pass and carries the reviewer's text on a revision, which is what makes the
#: second draft different from the first.
PlanTraining = Callable[[CandidateProfile, str], TrainingPlan]
#: ``(profile, plan) -> ReviewVerdict``
ReviewPlan = Callable[[CandidateProfile, TrainingPlan], ReviewVerdict]
#: ``(profile) -> ContractDraft`` — the draft's fields all come from the
#: profile; the training plan is deliberately not passed, because a contract
#: that quietly depended on it would be a contract nobody could review.
DraftContract = Callable[[CandidateProfile], ContractDraft]
#: ``(profile) -> ProvisionResult``
ProvisionIT = Callable[[CandidateProfile], ProvisionResult]


class CaseState(TypedDict, total=False):
    """The shared blackboard of one onboarding case.

    ``total=False`` because a case fills in fields as it advances: at intake
    there is no profile, at the gate there is no provisioning result, and a
    quarantined case never grows either. Nodes therefore read with ``.get()``
    and treat absence as "not reached yet", never as an error.
    """

    #: Candidate id, reused as the case id so artifacts land in `outbox/<id>/`.
    case_id: str
    #: Resume text AFTER PII masking — the raw text never enters the graph.
    masked_resume: str
    #: Untrusted intake payload, validated by the `intake` node.
    candidate_meta: dict[str, Any]

    profile: CandidateProfile
    #: Owned by `profile_analyst` — bounds the re-extraction loop.
    extract_attempts: int

    plan: TrainingPlan
    review: ReviewVerdict
    #: Owned by `plan_reviewer` — bounds the Reflexion revise loop.
    revise_count: int
    #: The critic's open objections; carried to the human when the loop
    #: exhausts without approval, so nobody signs off on a plan believing a
    #: reviewer blessed it.
    reviewer_concerns: list[str]

    #: State-only draft. Nothing binding reaches disk before the gate (M9).
    contract: ContractDraft
    gate: GateDecision
    provision: ProvisionResult

    #: Why a case was quarantined; also the routing signal out of `intake`.
    quarantine_reason: str
    #: Terminal status only ("completed" / "quarantined" / "offboarded").
    #: "awaiting_approval" is synthesized by the pipeline on `__interrupt__`.
    status: str
    #: Provisioning tool steps executed, for the budget and metrics views.
    tool_calls: int

    #: Append-only hash chain. The reducer is what makes it append-only.
    audit_trail: Annotated[list[AuditEvent], operator.add]


@dataclass
class AgentDeps:
    """The five agents and the effects port, injected into `build_graph`.

    Every agent is called **positionally**, in the order documented by the type
    aliases above, so a production wrapper is free to name its parameters
    whatever reads best and to accept extra keyword arguments with defaults.

    ``effects`` is duck-typed rather than a Protocol import so the graph never
    depends on the filesystem layer: it needs ``write_contract(case_id, draft,
    profile)``, ``write_welcome(case_id, profile)``, ``provision_tickets(
    case_id, result)`` and ``quarantine_case(case_id, reason)``. Leaving it
    ``None`` gets a no-op implementation, which is what diagram renders and
    routing tests want.
    """

    analyze_profile: AnalyzeProfile
    plan_training: PlanTraining
    review_plan: ReviewPlan
    draft_contract: DraftContract
    provision_it: ProvisionIT
    effects: Any = None

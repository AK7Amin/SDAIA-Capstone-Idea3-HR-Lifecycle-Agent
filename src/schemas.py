"""Typed contracts shared by every node of the HR lifecycle graph.

Coordination in this system is centralized: the LangGraph ``StateGraph`` is the
only coordinator and the agents never talk to each other directly — they read
and write the Pydantic models declared here. That makes this module the single
place where the vocabulary of the system is defined, so it is deliberately
dependency-free (stdlib + Pydantic only) and importable from any slice.

Two things are frozen here on purpose:

* the :class:`CaseStatus` literals, because the end-to-end test compares raw
  strings against them and that test is append-only; and
* the :class:`AuditEvent` digest format, because the audit chain is persisted
  through the checkpointer and re-verified in a different process — a hash that
  changed on the way to storage would prove nothing (critique M11).
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, ClassVar, Sequence

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AuditEvent",
    "CandidateProfile",
    "CaseStatus",
    "ContractDraft",
    "GateAction",
    "GateDecision",
    "ITTicket",
    "ProvisionResult",
    "ReviewAction",
    "ReviewVerdict",
    "TrainingPlan",
    "TrainingWeek",
    "verify_chain",
]


class CaseStatus(str, Enum):
    """Terminal or paused outcome of one onboarding case.

    Inherits from ``str`` so the pipeline can drop a member straight into a JSON
    payload or compare it to the plain literals the e2e test asserts on.
    """

    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    OFFBOARDED = "offboarded"


class ReviewAction(str, Enum):
    """What the Reflexion reviewer wants done with a training plan."""

    REVISE = "revise"
    APPROVE = "approve"


class GateAction(str, Enum):
    """What the human decided at the HR approval gate."""

    APPROVE = "approve"
    REJECT = "reject"


class CandidateProfile(BaseModel):
    """Structured view of a hired candidate, extracted from the resume text.

    Only ``candidate_id`` is mandatory at construction time: it comes from the
    intake payload and is always known, whereas every other field is produced by
    an extraction step that may legitimately come back empty. Emptiness is a
    routing signal (re-extract, bounded), not a crash — hence the permissive
    defaults plus :meth:`missing_fields` instead of hard Pydantic requirements.
    """

    #: Fields the graph treats as mandatory before it may proceed past extraction.
    REQUIRED: ClassVar[tuple[str, ...]] = ("name", "role", "start_date")

    candidate_id: str
    name: str = ""
    role: str = ""
    start_date: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_summary: str = ""

    def missing_fields(self) -> list[str]:
        """Return the required fields that are still blank, in declared order.

        Returns:
            The subset of :attr:`REQUIRED` whose value is empty or whitespace
            only. An empty list means the profile is complete enough to proceed.
        """
        # Whitespace counts as missing: an LLM answering "   " extracted nothing.
        return [
            field
            for field in self.REQUIRED
            if not str(getattr(self, field, "") or "").strip()
        ]


class TrainingWeek(BaseModel):
    """One week of the personalized onboarding plan."""

    week: int
    focus: str
    activities: list[str] = Field(default_factory=list)


class TrainingPlan(BaseModel):
    """Output of the Plan-and-Execute training planner.

    A plan with zero weeks is a silent failure dressed as success, so the
    minimum length is enforced by the contract rather than by each caller.
    """

    weeks: list[TrainingWeek] = Field(min_length=1)
    rationale: str = ""


class ReviewVerdict(BaseModel):
    """Reflexion critique of a training plan; drives the bounded revise loop."""

    action: ReviewAction
    critique: str = ""
    concerns: list[str] = Field(default_factory=list)


class ContractDraft(BaseModel):
    """Draft employment contract held in graph state only.

    Deliberately carries no file path: while the case waits at the human gate,
    nothing binding may exist on disk (governance ordering, critique M9). The
    rendering and the write happen after approval, inside the notifier node.
    """

    candidate_id: str
    role: str
    start_date: str
    salary_band: str = ""
    # Free-form template variables (synthetic data only); typed loosely because
    # the Jinja2 template owns their meaning, not this contract.
    body_fields: dict[str, Any] = Field(default_factory=dict)


class GateDecision(BaseModel):
    """Typed result of the human-in-the-loop approval gate.

    The gate resumes with a raw value handed in by a human, so it is validated
    into this model before it is allowed to influence routing.
    """

    decision: GateAction
    actor: str = "hr"
    decided_at: str = ""


class ITTicket(BaseModel):
    """One provisioning action performed by the ReAct IT agent."""

    ticket_id: str
    system: str
    action: str
    status: str


class ProvisionResult(BaseModel):
    """All provisioning tickets produced for a case."""

    tickets: list[ITTicket] = Field(default_factory=list)


class AuditEvent(BaseModel):
    """One immutable link in a case's hash-chained audit trail.

    Frozen for two reasons: an audit record that can be edited in place is not
    an audit record, and immutability makes the chain safe to pass around graph
    state where several nodes hold references to the same list.
    """

    model_config = ConfigDict(frozen=True)

    node: str
    summary: str = ""
    #: Named course pattern the emitting agent used (ReAct / Reflexion /
    #: Plan-and-Execute). Part of the digest so a label cannot be rewritten
    #: after the fact.
    reasoning_pattern: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    prev_hash: str = ""

    def digest(self) -> str:
        """Return the SHA-256 hex digest of this event's canonical form.

        The canonical form is ``json.dumps`` of a plain dict with sorted keys,
        ``ensure_ascii=False`` and no separator padding. Sorted keys make the
        bytes independent of field declaration order, and a plain dict keeps the
        digest identical after the event has been through the checkpointer's
        serializer — the round trip returns an equal model, not the same object.

        Returns:
            64-character hexadecimal SHA-256 digest.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_chain(events: Sequence[AuditEvent]) -> bool:
    """Check that an audit trail is an unbroken hash chain.

    The first event must open the chain (empty ``prev_hash``) and every later
    event must point at the digest of its immediate predecessor. This catches
    edited events, dropped events, and two events sharing one parent.

    Args:
        events: Audit events in emission order.

    Returns:
        True if the chain is intact (an empty trail is vacuously intact).
    """
    prev_hash = ""
    for event in events:
        if event.prev_hash != prev_hash:
            return False
        prev_hash = event.digest()
    return True

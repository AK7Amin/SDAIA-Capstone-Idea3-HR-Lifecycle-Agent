"""The coordinator: ten nodes, four decisions, two bounded loops, one gate.

The graph is the only thing in this system that decides what happens next. The
agents are pure functions of their inputs (injected via :class:`AgentDeps`), the
effects layer only writes what it is told to write, and every branch below is a
plain predicate over :class:`CaseState` — which is why the whole control flow is
testable offline.

Four decisions here exist because their opposites are the classic failures of
this architecture:

* **`hr_gate` calls `interrupt()` as its first statement.** LangGraph does not
  resume *inside* a node; on `Command(resume=...)` it re-runs the node from the
  top. Any line above the interrupt therefore executes once per resume attempt
  — a contract written there would be written twice, and written *before* the
  human ever answered. Nothing precedes the interrupt but a pure payload build.
* **Terminal states are real nodes.** `quarantine` and `offboard` emit audit
  events and set a typed status, instead of the graph quietly ending. A case
  that stops has to say why it stopped.
* **Multi-event nodes chain their own events** through :func:`audit_chain`.
  Hanging two sibling events off the same `prev_hash` looks fine to a naive
  "does every event have a parent?" check and breaks `verify_chain` — the
  single seam below is what stops that from ever being hand-rolled per node.
* **Counters live in one node each.** `extract_attempts` is written only by
  `profile_analyst`, `revise_count` only by `plan_reviewer`. Routers read them
  and never write, so a loop bound cannot drift.

`awaiting_approval` is deliberately absent: the graph writes terminal statuses
only, and the pipeline layer synthesizes the paused status when it sees
`__interrupt__` (critique M7).
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any, Mapping, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.schemas import (
    AuditEvent,
    CandidateProfile,
    CaseStatus,
    GateAction,
    GateDecision,
    ReviewAction,
)
from src.state import AgentDeps, CaseState

__all__ = [
    "MAX_EXTRACT_ATTEMPTS",
    "MAX_REVISIONS",
    "NODE_NAMES",
    "PATTERN_EXTRACTION",
    "PATTERN_HITL",
    "PATTERN_PLANNING",
    "PATTERN_REACT",
    "PATTERN_REFLEXION",
    "REQUIRED_META",
    "NullEffects",
    "audit_chain",
    "build_graph",
    "route_after_gate",
    "route_after_intake",
    "route_after_profile",
    "route_after_review",
]

# --------------------------------------------------------------------------
# bounds and vocabulary
# --------------------------------------------------------------------------
#: Total runs of `profile_analyst` allowed per case (the initial attempt plus
#: one re-extraction). Exhaustion routes to `quarantine`, never to a guess.
MAX_EXTRACT_ATTEMPTS = 2
#: Revision rounds the Reflexion critic may demand. Exhaustion routes forward
#: to the gate WITH the concerns attached — the human decides, not the loop.
MAX_REVISIONS = 1

#: Named course patterns, stamped into the audit trail by the node that used
#: them. A node doing deterministic work claims none: an empty
#: `reasoning_pattern` is the honest label for validation and file writing.
PATTERN_EXTRACTION = "extraction"
PATTERN_PLANNING = "plan-and-execute"
PATTERN_REFLEXION = "reflexion"
PATTERN_REACT = "react"
PATTERN_HITL = "human-in-the-loop"

#: Keys the intake payload must carry before any agent is allowed to see it.
#: Derived from the profile contract so the two cannot drift apart.
REQUIRED_META: tuple[str, ...] = ("candidate_id", *CandidateProfile.REQUIRED)

#: Keys `intake` drops from the payload. The graph runs on `masked_resume`
#: alone; an unmasked copy riding along inside the metadata would be persisted
#: into every checkpoint row of the case, which is exactly what the masking
#: step exists to prevent.
STRIPPED_META: tuple[str, ...] = ("resume_text", "resume")


class NullEffects:
    """Effects port that does nothing, used when the caller injects none.

    A null object rather than `if self.effects:` in four nodes: the governance
    story is easier to audit when every node calls the port unconditionally and
    the *wiring* decides whether anything reaches a disk.
    """

    def write_contract(self, case_id: str, draft: Any, profile: Any) -> None:
        return None

    def write_welcome(self, case_id: str, profile: Any) -> None:
        return None

    def provision_tickets(self, case_id: str, result: Any) -> None:
        return None

    def quarantine_case(self, case_id: str, reason: str) -> None:
        return None


# --------------------------------------------------------------------------
# audit chaining — the single seam
# --------------------------------------------------------------------------
def _digest_of(event: Any) -> str:
    """Digest of a trail entry, whether it is a model or a serialized dict.

    The trail crosses a checkpointer between supersteps; slice 1 proved the
    round trip returns an equal model, but the Postgres serializer is an
    allow-list and a dict coming back is a survivable outcome, not a crash.
    """
    if isinstance(event, AuditEvent):
        return event.digest()
    return AuditEvent.model_validate(event).digest()


def audit_chain(
    state: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> list[AuditEvent]:
    """Build a run of audit events, chained to the trail *and to each other*.

    Args:
        state: The current case state; only ``audit_trail`` is read.
        entries: Field dicts for :class:`~src.schemas.AuditEvent` **without**
            ``prev_hash`` — this function owns that field.

    Returns:
        The new events in emission order, ready to be returned from a node as
        ``{"audit_trail": events}``. Appending them to the trail keeps
        :func:`~src.schemas.verify_chain` true no matter how many a node emits.
    """
    trail = state.get("audit_trail") or []
    prev_hash = _digest_of(trail[-1]) if trail else ""
    events: list[AuditEvent] = []
    for entry in entries:
        event = AuditEvent(prev_hash=prev_hash, **entry)
        events.append(event)
        prev_hash = event.digest()  # the next sibling hangs off THIS one
    return events


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------
def _blank(value: Any) -> bool:
    """True when a value is missing or whitespace only (an LLM's empty hand)."""
    return not str(value or "").strip()


def intake(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Validate the untrusted hired-candidate payload before any agent runs.

    Deterministic gatekeeping only: a payload missing a required key is not
    something an LLM should be asked to repair, so the case is quarantined with
    the field names spelled out. It is also the boundary where unmasked resume
    text is dropped, so no later node can persist what the guard cleaned.
    """
    meta = {
        key: value
        for key, value in (state.get("candidate_meta") or {}).items()
        if key not in STRIPPED_META
    }
    missing = [key for key in REQUIRED_META if _blank(meta.get(key))]
    case_id = str(meta.get("candidate_id") or state.get("case_id") or "UNKNOWN")

    if missing:
        reason = f"invalid intake: missing required field(s): {', '.join(missing)}"
        summary = reason
    else:
        reason = ""
        summary = f"intake validated for {case_id}"

    return {
        "case_id": case_id,
        "candidate_meta": meta,
        "quarantine_reason": reason,
        "audit_trail": audit_chain(state, [{"node": "intake", "summary": summary}]),
    }


def profile_analyst(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Extract a :class:`CandidateProfile` from the masked resume.

    Owns ``extract_attempts``. An incomplete profile is a routing signal, not a
    failure: the router sends the case back here once, and only a second
    incomplete extraction quarantines it.
    """
    attempts = state.get("extract_attempts", 0) + 1
    profile = deps.analyze_profile(
        state.get("masked_resume", ""), dict(state.get("candidate_meta") or {})
    )
    missing = profile.missing_fields()

    if missing:
        summary = (
            f"extraction attempt {attempts}: incomplete profile, "
            f"missing {', '.join(missing)}"
        )
        # Written every incomplete attempt; read only if routing lands on
        # quarantine, and cleared below the moment extraction succeeds.
        reason = (
            f"profile extraction exhausted after {attempts} attempt(s); "
            f"still missing: {', '.join(missing)}"
        )
    else:
        summary = f"extraction attempt {attempts}: profile complete"
        reason = ""

    return {
        "profile": profile,
        "extract_attempts": attempts,
        "quarantine_reason": reason,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "profile_analyst",
                    "summary": summary,
                    "reasoning_pattern": PATTERN_EXTRACTION,
                }
            ],
        ),
    }


def training_planner(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Plan-and-Execute: turn the profile (and any critique) into a plan.

    The planner is handed the reviewer's critique *text*, not the concern list:
    the list is a structured summary for the human at the gate, while the prose
    is what actually tells a planner what to change.
    """
    review = state.get("review")
    critique = review.critique if review is not None else ""
    concerns = list(state.get("reviewer_concerns") or [])
    plan = deps.plan_training(state["profile"], critique)
    summary = f"planned {len(plan.weeks)} week(s)"
    if concerns:
        summary += f", addressing {len(concerns)} reviewer concern(s)"
    return {
        "plan": plan,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "training_planner",
                    "summary": summary,
                    "reasoning_pattern": PATTERN_PLANNING,
                }
            ],
        ),
    }


def plan_reviewer(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Reflexion: critique the plan and count the revisions demanded.

    Owns ``revise_count``, incremented only on a *revise* verdict — an approval
    must never consume a revision the critic never asked for.
    """
    verdict = deps.review_plan(state["profile"], state["plan"])
    revise_count = state.get("revise_count", 0)
    if verdict.action is ReviewAction.REVISE:
        revise_count += 1
    return {
        "review": verdict,
        # Overwritten each round: the concerns that matter are the open ones.
        "reviewer_concerns": list(verdict.concerns),
        "revise_count": revise_count,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "plan_reviewer",
                    "summary": (
                        f"verdict '{verdict.action.value}' after "
                        f"{revise_count} revision(s), "
                        f"{len(verdict.concerns)} open concern(s)"
                    ),
                    "reasoning_pattern": PATTERN_REFLEXION,
                }
            ],
        ),
    }


def contract_drafter(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Produce the contract draft **in state only** (governance ordering, M9).

    No effects call belongs in this node: while the case waits at the gate,
    nothing binding may exist on disk.
    """
    draft = deps.draft_contract(state["profile"])
    return {
        "contract": draft,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "contract_drafter",
                    "summary": (
                        f"contract drafted for {draft.candidate_id} "
                        f"({draft.role}); held in state until approval"
                    ),
                }
            ],
        ),
    }


def _gate_payload(state: CaseState) -> dict[str, Any]:
    """Everything the human needs in order to decide, as plain JSON types.

    Pure: it reads state and allocates a dict. It is safe above `interrupt()`
    precisely because it changes nothing — and it must be JSON-native because
    the payload is persisted with the checkpoint and read back by another
    process days later.
    """
    profile = state.get("profile")
    contract = state.get("contract")
    plan = state.get("plan")
    return {
        "question": f"Approve onboarding for {state.get('case_id', '')}?",
        "case_id": state.get("case_id", ""),
        "candidate": profile.model_dump(mode="json") if profile else {},
        "contract": contract.model_dump(mode="json") if contract else {},
        "training_weeks": (
            [week.model_dump(mode="json") for week in plan.weeks] if plan else []
        ),
        # Honesty: if the critic was overruled by the loop bound, the human
        # sees exactly what it was still unhappy about.
        "reviewer_concerns": list(state.get("reviewer_concerns") or []),
    }


def _coerce_gate(decision: Any) -> GateDecision:
    """Validate whatever the human resumed with into a :class:`GateDecision`.

    Raises:
        ValueError: if the value is not a readable decision. Refusing is the
            point — silently reading an unknown answer as "reject" would fake a
            human decision that nobody made, and as "approve" would be worse.
            The checkpoint is untouched, so a corrected resume still works.
    """
    if isinstance(decision, GateDecision):
        return decision
    if isinstance(decision, str):
        payload: dict[str, Any] = {"decision": decision}
    elif isinstance(decision, Mapping):
        payload = dict(decision)
    else:
        raise ValueError(
            f"unreadable gate decision {decision!r}: resume with "
            "'approve' or 'reject'"
        )
    try:
        return GateDecision.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError, kept as a ValueError
        raise ValueError(
            f"unreadable gate decision {decision!r}: resume with "
            "'approve' or 'reject'"
        ) from exc


def hr_gate(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Human-in-the-loop approval gate. Everything below the pause is post-hoc.

    The interrupt is the FIRST statement on purpose (critique M7): LangGraph
    re-executes this node from the top on every resume attempt, so anything
    above it would run again — and would have run before the human answered at
    all. Only the pure payload build precedes it.
    """
    decision = interrupt(_gate_payload(state))

    gate = _coerce_gate(decision)
    return {
        "gate": gate,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "hr_gate",
                    "summary": (
                        f"human decision '{gate.decision.value}' "
                        f"by {gate.actor}"
                    ),
                    "reasoning_pattern": PATTERN_HITL,
                }
            ],
        ),
    }


def it_provisioner(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """ReAct: run the provisioning tools and audit **every** step.

    The multi-event node of this graph: one event per tool step plus a closing
    summary, all chained to each other by :func:`audit_chain`.
    """
    result = deps.provision_it(state["profile"])
    case_id = state.get("case_id", "")
    deps.effects.provision_tickets(case_id, result)

    entries: list[dict[str, Any]] = [
        {
            "node": "it_provisioner",
            "summary": (
                f"tool step {index}: {ticket.action} on {ticket.system} "
                f"-> {ticket.status} ({ticket.ticket_id})"
            ),
            "reasoning_pattern": PATTERN_REACT,
        }
        for index, ticket in enumerate(result.tickets, start=1)
    ]
    entries.append(
        {
            "node": "it_provisioner",
            "summary": f"provisioning complete: {len(result.tickets)} ticket(s)",
            "reasoning_pattern": PATTERN_REACT,
        }
    )

    return {
        "provision": result,
        "tool_calls": state.get("tool_calls", 0) + len(result.tickets),
        "audit_trail": audit_chain(state, entries),
    }


def notifier(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Post-approval writes: the contract and the welcome pack, then done.

    This is the ONLY node that puts a binding document on disk, and it is
    unreachable without an approving gate decision. Summaries name the case,
    never a path — persisted artifacts stay free of machine-local paths.
    """
    case_id = state.get("case_id", "")
    profile = state["profile"]
    deps.effects.write_contract(case_id, state.get("contract"), profile)
    deps.effects.write_welcome(case_id, profile)
    return {
        "status": CaseStatus.COMPLETED.value,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "notifier",
                    "summary": f"contract document written for {case_id}",
                },
                {"node": "notifier", "summary": f"welcome pack written for {case_id}"},
            ],
        ),
    }


def quarantine(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Terminal: a case the system refuses to process, with the reason kept."""
    case_id = state.get("case_id", "")
    reason = state.get("quarantine_reason") or "quarantined without a recorded reason"
    deps.effects.quarantine_case(case_id, reason)
    return {
        "status": CaseStatus.QUARANTINED.value,
        "audit_trail": audit_chain(
            state, [{"node": "quarantine", "summary": f"quarantined: {reason}"}]
        ),
    }


def offboard(state: CaseState, deps: AgentDeps) -> dict[str, Any]:
    """Terminal: the human rejected the case at the gate. Nothing is written."""
    gate = state.get("gate")
    actor = gate.actor if gate is not None else "hr"
    return {
        "status": CaseStatus.OFFBOARDED.value,
        "audit_trail": audit_chain(
            state,
            [
                {
                    "node": "offboard",
                    "summary": (
                        f"offboarded: {actor} rejected the onboarding at the "
                        "human gate; no documents produced"
                    ),
                }
            ],
        ),
    }


#: The node roster, in the order a happy case visits it.
NODE_NAMES: tuple[str, ...] = (
    "intake",
    "profile_analyst",
    "training_planner",
    "plan_reviewer",
    "contract_drafter",
    "hr_gate",
    "it_provisioner",
    "notifier",
    "quarantine",
    "offboard",
)

_NODES = {
    "intake": intake,
    "profile_analyst": profile_analyst,
    "training_planner": training_planner,
    "plan_reviewer": plan_reviewer,
    "contract_drafter": contract_drafter,
    "hr_gate": hr_gate,
    "it_provisioner": it_provisioner,
    "notifier": notifier,
    "quarantine": quarantine,
    "offboard": offboard,
}


# --------------------------------------------------------------------------
# routing — pure predicates, never a write
# --------------------------------------------------------------------------
def route_after_intake(state: CaseState) -> str:
    """Invalid payload out, valid payload on to extraction."""
    return "quarantine" if state.get("quarantine_reason") else "profile_analyst"


def route_after_profile(state: CaseState) -> str:
    """Complete profile forward, incomplete one back — at most once."""
    profile = state.get("profile")
    if profile is not None and not profile.missing_fields():
        return "training_planner"
    if state.get("extract_attempts", 0) < MAX_EXTRACT_ATTEMPTS:
        return "profile_analyst"
    return "quarantine"


def route_after_review(state: CaseState) -> str:
    """Approve forward; revise back until the bound, then forward *with concerns*.

    Exhaustion does not quarantine: a plan the critic dislikes is a judgement
    call, and the concerns travel to the human at the gate.
    """
    review = state.get("review")
    if review is None or review.action is ReviewAction.APPROVE:
        return "contract_drafter"
    if state.get("revise_count", 0) <= MAX_REVISIONS:
        return "training_planner"
    return "contract_drafter"


def route_after_gate(state: CaseState) -> str:
    """Deny by default: only an explicit approval reaches provisioning."""
    gate = state.get("gate")
    if gate is not None and gate.decision is GateAction.APPROVE:
        return "it_provisioner"
    return "offboard"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build_graph(deps: AgentDeps, checkpointer: Any | None = None):
    """Compile the onboarding graph over injected agents.

    Args:
        deps: The five agent callables plus the effects port. A ``None`` effects
            port is replaced by :class:`NullEffects` on a *copy* — the caller's
            object is never mutated.
        checkpointer: Any LangGraph checkpointer. Required for the human gate
            (an interrupt with nowhere to persist cannot be resumed); omitting
            it is for diagram rendering and pure routing checks.

    Returns:
        The compiled graph, ready for ``invoke`` / ``Command(resume=...)``.
    """
    if deps.effects is None:
        deps = replace(deps, effects=NullEffects())

    graph = StateGraph(CaseState)
    for name, func in _NODES.items():
        graph.add_node(name, partial(func, deps=deps))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"profile_analyst": "profile_analyst", "quarantine": "quarantine"},
    )
    graph.add_conditional_edges(
        "profile_analyst",
        route_after_profile,
        {
            "training_planner": "training_planner",
            "profile_analyst": "profile_analyst",  # bounded re-extraction
            "quarantine": "quarantine",
        },
    )
    graph.add_edge("training_planner", "plan_reviewer")
    graph.add_conditional_edges(
        "plan_reviewer",
        route_after_review,
        {
            "training_planner": "training_planner",  # bounded Reflexion loop
            "contract_drafter": "contract_drafter",
        },
    )
    graph.add_edge("contract_drafter", "hr_gate")
    graph.add_conditional_edges(
        "hr_gate",
        route_after_gate,
        {"it_provisioner": "it_provisioner", "offboard": "offboard"},
    )
    graph.add_edge("it_provisioner", "notifier")
    graph.add_edge("notifier", END)
    graph.add_edge("quarantine", END)
    graph.add_edge("offboard", END)

    return graph.compile(checkpointer=checkpointer)

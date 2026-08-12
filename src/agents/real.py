"""The five concrete HR agents (slice 6).

`HRAgents` is the seam between the graph and the model: the graph nodes call
these methods, the methods return the typed contracts from `src.schemas`, and
nothing in between leaks an untyped dict into shared state. Each callable is
one named responsibility with one named reasoning pattern — profile extraction,
Plan-and-Execute planning, Reflexion review, template fill, and ReAct
provisioning — which is what makes this a multi-agent system rather than one
prompt wearing five hats.

The class takes `llm` and `registry` as constructor arguments and never
imports a provider: anything exposing ``invoke(prompt, node, case_id=None)``
works, so tests inject a scripted stub and production injects the failover
chain from `src.llm`.

Four rules are enforced here rather than trusted to the prompt:

* **Identity is not model output.** `candidate_id`, `role` and `start_date`
  come from the verified intake/profile every time. A model that renames the
  case, or promotes the candidate in the contract draft, is overwritten.
* **Bad output is an exception, not a half-filled contract.** Every parse ends
  in a Pydantic contract or in `AgentOutputError`; the graph decides whether
  that means re-extract or quarantine.
* **Never invent authority.** A cited policy id that does not exist in the
  handbook corpus is replaced by a visible marker and reported as a concern.
  Removal is recorded, never silent — an unaudited deletion is its own defect.
* **Honest attribution.** The provisioner's first policy lookup is imposed by
  the system, so it is recorded with `forced_first_call=True`; when the ReAct
  loop comes back empty the deterministic fallback sets `decision_source =
  "fallback"` and leaves that flag alone (the bug frozen in slice 4's tests).

No file IO happens anywhere in this module: the contract drafter produces state
only, and every document is written post-gate by the notifier (critique M9).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from pydantic import BaseModel, ValidationError

from src.agents import prompts
from src.agents.react import ReActResult, ReActStep, force_first_lookup, run_react
from src.guardrails import mask_pii, wrap_untrusted
from src.schemas import (
    CandidateProfile,
    ContractDraft,
    ITTicket,
    ProvisionResult,
    ReviewVerdict,
    TrainingPlan,
)
from src.tools import DEFAULT_POLICY_DIR, ToolRegistry

__all__ = [
    "AgentOutputError",
    "AgentRole",
    "CitationCheck",
    "HRAgents",
    "REASONING_PATTERNS",
    "ROLES",
    "UNVERIFIED_CITATION_MARK",
]


class AgentOutputError(ValueError):
    """A model reply could not be turned into the contract the agent owes.

    Subclasses `ValueError` for the same reason `ToolError` does: a caller
    guarding against bad input already catches it. Carries no raw model text —
    only a masked excerpt — because these messages end up in logs and traces.
    """


@dataclass(frozen=True)
class AgentRole:
    """One agent's identity card: what it is called, and how it reasons.

    `method` is the callable on `HRAgents`, `node` is the graph node that hosts
    it, and `pattern` is the course pattern it implements. Keeping the three
    together is what lets the graph emit a truthful `reasoning_pattern` into an
    audit event without hard-coding a second copy of the mapping.
    """

    method: str
    node: str
    pattern: str
    responsibility: str


#: The five specialised roles. Distinct responsibilities, distinct outputs.
ROLES: tuple[AgentRole, ...] = (
    AgentRole(
        method="analyze_profile",
        node="profile_analyst",
        pattern="extraction",
        responsibility="Extract typed candidate facts from the untrusted resume.",
    ),
    AgentRole(
        method="plan_training",
        node="training_planner",
        pattern="plan-and-execute",
        responsibility="Produce a typed multi-week onboarding plan.",
    ),
    AgentRole(
        method="review_plan",
        node="plan_reviewer",
        pattern="reflexion",
        responsibility="Critique the plan against the profile and the handbook.",
    ),
    AgentRole(
        method="draft_contract",
        node="contract_drafter",
        pattern="template-fill",
        responsibility="Fill the typed contract fields, in state only.",
    ),
    AgentRole(
        method="provision_it",
        node="it_provisioner",
        pattern="react",
        responsibility="Decide accounts and equipment through validated tools.",
    ),
)

#: Pattern per role, reachable by BOTH the callable name and the graph node —
#: the graph writes `AuditEvent(node=...)`, tests and notebook cells reach for
#: the method name, and neither should have to translate.
REASONING_PATTERNS: dict[str, str] = {
    key: role.pattern for role in ROLES for key in (role.method, role.node)
}

#: What replaces a citation the handbook cannot back. Visible on purpose: a
#: reader of the plan must see that something was removed.
UNVERIFIED_CITATION_MARK = "[unverified policy reference]"

_CITATION_RE = re.compile(r"\bPOL-\d+\b", re.IGNORECASE)
_POLICY_ID_RE = re.compile(r"^##\s*(POL-\d+)", re.MULTILINE)

#: Leading/trailing markdown fence, with or without a language tag.
_FENCE_RE = re.compile(r"\A\s*```[a-zA-Z0-9_+-]*[ \t]*\r?\n?|\r?\n?\s*```\s*\Z")

#: POL-004: nothing is created until manager AND HR approve, so a ticket this
#: agent raises is a request — calling it "created" would be a lie on paper.
TICKET_STATUS = "requested"

#: Roles POL-003 entitles to the developer machine. Substring match on a
#: lowercased title, so "Senior Data Engineer" and "ML Engineer" both land.
_DEVELOPER_ROLE_WORDS = (
    "engineer",
    "developer",
    "data",
    "software",
    "devops",
    "sre",
    "scientist",
    "architect",
    "programmer",
)

_DEVELOPER_LAPTOP = (
    "allocate developer laptop, 32 GB memory + external monitor (POL-003)"
)
_STANDARD_LAPTOP = "allocate standard laptop, 16 GB memory (POL-003)"


class CitationCheck(NamedTuple):
    """Result of validating the policy citations inside one object.

    `clean` is the same type as the input with unverifiable ids replaced;
    `removed` names them in first-seen order; `concerns` is the sentence a
    human approver reads. When nothing was removed, `clean` IS the input.
    """

    clean: Any
    concerns: tuple[str, ...]
    removed: tuple[str, ...]


def _excerpt(text: Any, limit: int = 160) -> str:
    """A short, PII-masked slice of model output, safe to put in an error."""
    flat = " ".join(str(text or "").split())
    if len(flat) > limit:
        flat = flat[:limit] + "…"
    return mask_pii(flat) or "(empty reply)"


class HRAgents:
    """The five HR agents, sharing one model client and one tool registry.

    Args:
        llm: Anything with ``invoke(prompt, node, case_id=None) -> str``.
        registry: The tool registry the reviewer and the provisioner dispatch
            through. Real dispatch, real execution log — the audit trail must
            record tool calls that actually happened.
        policy_dir: Where the handbook lives. Only the *known policy ids* are
            read from here (citation validation); the text itself is retrieved
            through the registry's tool so the read is logged. Defaults to the
            same directory `build_hr_registry` uses.
    """

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        policy_dir: Path | str | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._policy_dir = (
            Path(policy_dir) if policy_dir is not None else DEFAULT_POLICY_DIR
        )
        # Read once: the handbook does not change during a run, and a citation
        # check that re-reads the disk per string would be its own bottleneck.
        self.known_policy_ids: frozenset[str] = self._load_policy_ids(self._policy_dir)

    # ---------------------------------------------------------------- roles
    @staticmethod
    def describe_roles() -> tuple[AgentRole, ...]:
        """The role table — used by the graph, the dashboard and the notebook."""
        return ROLES

    @staticmethod
    def reasoning_pattern(name: str) -> str:
        """Pattern for a method or node name; `""` when this class owns neither.

        Empty rather than a guess: a node this module does not implement (the
        gate, the notifier) has no reasoning pattern to claim.
        """
        return REASONING_PATTERNS.get(name, "")

    # ------------------------------------------------------------ agent 1/5
    def analyze_profile(
        self,
        resume_text: str,
        candidate_meta: Mapping[str, Any] | None = None,
        attempt: int = 0,
    ) -> CandidateProfile:
        """Extract a `CandidateProfile` from untrusted resume text.

        Args:
            resume_text: Raw resume content. Wrapped as untrusted data before
                it reaches the model — it is written by the person being judged.
            candidate_meta: Verified intake facts. `candidate_id` is taken from
                here and never from the model.
            attempt: Extraction attempt number. From 1 onward the prompt says
                so explicitly; the graph bounds the retries.

        Returns:
            The extracted profile. Missing fields come back empty — that is a
            routing signal for the graph, not an error.

        Raises:
            AgentOutputError: The reply was not a JSON object matching the
                contract.
        """
        meta = dict(candidate_meta or {})
        known = {
            key: meta[key]
            for key in ("candidate_id", "name", "role", "start_date")
            if meta.get(key)
        }
        prompt = prompts.profile_prompt(
            resume_block=wrap_untrusted(str(resume_text or "")),
            meta_block=json.dumps(known, ensure_ascii=False, indent=2),
            retry=attempt > 0,
        )
        raw = self._llm.invoke(prompt, node="profile_analyst")
        return self._to_contract(
            raw, CandidateProfile, candidate_id=str(meta.get("candidate_id", ""))
        )

    # ------------------------------------------------------------ agent 2/5
    def plan_training(
        self, profile: CandidateProfile, critique: str = ""
    ) -> TrainingPlan:
        """Plan-and-Execute: design the whole arc, then emit it week by week.

        Args:
            profile: The candidate the plan is for. Re-wrapped as untrusted —
                every word of it was extracted from the resume.
            critique: Reviewer feedback. When present the call is a revision
                pass (the Reflexion loop), not a first draft.

        Returns:
            A typed plan whose citations are all backed by the handbook. If the
            model cited an id that does not exist, it is replaced by
            `UNVERIFIED_CITATION_MARK` and the removal is appended to the
            rationale so the receipt travels with the plan.

        Raises:
            AgentOutputError: Malformed JSON, or a plan with no weeks.
        """
        prompt = prompts.plan_prompt(
            profile_block=wrap_untrusted(self._as_json(profile)),
            known_ids=self._known_ids_text(),
            critique=critique,
        )
        raw = self._llm.invoke(prompt, node="training_planner")
        plan = self._to_contract(raw, TrainingPlan)

        check = self.validate_citations(plan)
        if not check.removed:
            return plan
        note = "Citation check: " + " ".join(check.concerns)
        rationale = f"{check.clean.rationale}\n\n{note}".strip()
        return check.clean.model_copy(update={"rationale": rationale})

    # ------------------------------------------------------------ agent 3/5
    def review_plan(
        self, profile: CandidateProfile, plan: TrainingPlan
    ) -> ReviewVerdict:
        """Reflexion: evaluate the plan against the profile and the handbook.

        The policy sections are fetched through the registry with
        `force_first_lookup`, so the reviewer's evidence is a REAL logged tool
        call. A model that merely writes "I consulted the handbook" leaves no
        row in `registry.execution_log`, and the difference is visible in the
        audit trail.

        Returns:
            The verdict, with citation concerns about the plan (and about the
            critique itself) merged into `concerns` — approval does not silence
            them; the human at the gate still sees them.

        Raises:
            AgentOutputError: The reply was not a verdict.
            ToolError: The handbook could not be read at all. A review with no
                evidence must fail loudly rather than pass on vibes.
        """
        query = (
            f"{profile.role} onboarding training security probation "
            f"buddy approval manager"
        )
        _, policy_text = force_first_lookup(
            self._registry, "hr_policy_lookup", {"query": query}
        )
        prompt = prompts.review_prompt(
            profile_block=self._as_json(profile),
            plan_block=self._as_json(plan),
            policy_block=policy_text,
            known_ids=self._known_ids_text(),
        )
        raw = self._llm.invoke(prompt, node="plan_reviewer")
        verdict = self._to_contract(raw, ReviewVerdict)

        plan_check = self.validate_citations(plan)
        verdict_check = self.validate_citations(verdict)
        verdict = verdict_check.clean
        merged = list(verdict.concerns) + list(plan_check.concerns)
        merged += list(verdict_check.concerns)
        # Order-preserving dedupe: the same concern raised twice is one concern.
        deduped = list(dict.fromkeys(concern for concern in merged if concern))
        if deduped == list(verdict.concerns):
            return verdict
        return verdict.model_copy(update={"concerns": deduped})

    # ------------------------------------------------------------ agent 4/5
    def draft_contract(self, profile: CandidateProfile) -> ContractDraft:
        """Fill the contract's typed fields — STATE ONLY, no document written.

        Governance ordering (critique M9): while the case waits at the human
        gate nothing binding may exist on disk, so this agent produces a draft
        object and the notifier renders it after approval.

        Identity fields are copied from the verified profile, never from the
        reply: a model is not allowed to change the role it is drafting for.

        Raises:
            AgentOutputError: The reply was not a JSON object.
        """
        raw = self._llm.invoke(
            prompts.contract_prompt(profile_block=self._as_json(profile)),
            node="contract_drafter",
        )
        draft = self._to_contract(
            raw,
            ContractDraft,
            candidate_id=profile.candidate_id,
            role=profile.role,
            start_date=profile.start_date,
        )
        check = self.validate_citations(draft)
        if not check.removed:
            return draft
        # The draft has no concerns field, so the receipt rides in the free-form
        # template variables where the notifier and the human can both see it.
        fields = dict(check.clean.body_fields)
        fields["citation_notes"] = list(check.concerns)
        return check.clean.model_copy(update={"body_fields": fields})

    # ------------------------------------------------------------ agent 5/5
    def provision_it(
        self, profile: CandidateProfile
    ) -> tuple[ProvisionResult, ReActResult]:
        """ReAct: reason about accounts and equipment, acting through real tools.

        The policy read is imposed before reasoning starts (POL-003 equipment by
        role, POL-004 dual approval) and recorded as step 0 with
        `forced_first_call=True` — the system chose it, not the model.

        Returns:
            `(result, react)`. When the loop exhausts itself or answers with
            something that is not a ticket list, `result` is a deterministic
            minimal provisioning set and `react.decision_source` becomes
            ``"fallback"``. `forced_first_call` is deliberately NOT touched
            there: two facts, two fields.
        """
        query = (
            f"{profile.role} equipment laptop hardware account mailbox "
            f"provisioning approval role"
        )
        call, policy_text = force_first_lookup(
            self._registry, "hr_policy_lookup", {"query": query}
        )
        task = prompts.provision_task(
            name=profile.name,
            role=profile.role,
            start_date=profile.start_date,
            policy_block=policy_text,
        )
        react = run_react(
            lambda prompt: self._llm.invoke(prompt, node="it_provisioner"),
            task,
            self._registry,
        )
        # The forced call belongs at the FRONT of the trace: it happened first.
        react.steps.insert(
            0,
            ReActStep(
                thought=prompts.FORCED_LOOKUP_THOUGHT,
                action=call.name,
                action_input=dict(call.arguments),
                observation=policy_text,
                call=call,
            ),
        )
        react.forced_first_call = True

        tickets = self._tickets_from_answer(react.final_answer, profile)
        if not tickets:
            # A provisioning run that provisions nothing is a silent failure;
            # substitute the policy baseline and label the substitution.
            tickets = self._fallback_tickets(profile)
            react.decision_source = "fallback"  # forced_first_call stays as-is

        result = ProvisionResult(tickets=tickets)
        check = self.validate_citations(result)
        if check.removed:
            # No concerns field on ProvisionResult — the ReAct trace is the
            # audit surface, so the receipt goes there.
            react.steps.append(
                ReActStep(
                    thought="Citation check on the proposed tickets.",
                    observation=" ".join(check.concerns),
                )
            )
            result = check.clean
        return result, react

    # ------------------------------------------------------------- citations
    def validate_citations(self, obj: Any) -> CitationCheck:
        """Replace every `POL-xxx` the handbook cannot back, and say which.

        Works on any Pydantic contract (or plain dict/list/str): the object is
        dumped, every string is scrubbed, and the result is validated back into
        the original type — so one rule covers plans, verdicts, drafts and
        tickets without five copies of it.

        Args:
            obj: The object whose text may carry policy citations.

        Returns:
            A `CitationCheck`. `clean` is the input itself when nothing was
            removed, so a caller can compare identity to skip a rebuild.
        """
        removed: list[str] = []

        def replace(match: re.Match[str]) -> str:
            citation = match.group(0)
            if citation.upper() in self.known_policy_ids:
                return citation
            if citation.upper() not in removed:
                removed.append(citation.upper())
            return UNVERIFIED_CITATION_MARK

        def scrub(node: Any) -> Any:
            if isinstance(node, str):
                return _CITATION_RE.sub(replace, node)
            if isinstance(node, list):
                return [scrub(item) for item in node]
            if isinstance(node, dict):
                return {key: scrub(value) for key, value in node.items()}
            return node

        is_model = isinstance(obj, BaseModel)
        payload = obj.model_dump(mode="json") if is_model else obj
        scrubbed = scrub(payload)
        if not removed:
            return CitationCheck(obj, (), ())

        concerns = tuple(
            f"Removed citation {citation}: the policy handbook has no section "
            f"{citation}, so that claim carries no authority."
            for citation in removed
        )
        clean = type(obj).model_validate(scrubbed) if is_model else scrubbed
        return CitationCheck(clean, concerns, tuple(removed))

    # -------------------------------------------------------------- parsing
    @staticmethod
    def _parse_json(raw: Any) -> dict:
        """Recover the outermost JSON object from a model reply.

        Models wrap JSON in markdown fences, in "Sure! Here you go:", or in a
        trailing offer to help. The fence is stripped, then the FIRST `{` opens
        a brace scan that ignores braces inside strings — so a value containing
        `}` does not truncate the object.

        Raises:
            AgentOutputError: No balanced JSON object, or one that is not an
                object (a bare array is not a contract).
        """
        text = _FENCE_RE.sub("", str(raw or ""))
        start = text.find("{")
        if start == -1:
            raise AgentOutputError(
                f"model reply contains no JSON object — got: {_excerpt(raw)}"
            )

        depth = 0
        in_string = False
        escaped = False
        blob = ""
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : index + 1]
                    break
        if not blob:
            raise AgentOutputError(
                f"model reply has an unbalanced JSON object — got: {_excerpt(raw)}"
            )

        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise AgentOutputError(
                f"model reply is not valid JSON ({exc.msg}) — got: {_excerpt(raw)}"
            ) from None
        if not isinstance(parsed, dict):
            raise AgentOutputError(
                f"model reply is a {type(parsed).__name__}, not a JSON object"
            )
        return parsed

    def _to_contract(self, raw: Any, model_cls: type[BaseModel], **fixed: Any):
        """Parse a reply into `model_cls`, overriding `fixed` fields verbatim.

        `fixed` is how identity survives the model: whatever the reply said
        about those keys is discarded before validation.
        """
        payload = self._parse_json(raw)
        payload.update(fixed)
        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()[:3]
            )
            raise AgentOutputError(
                f"{model_cls.__name__} contract violated — {details}"
            ) from None

    # ------------------------------------------------------------- internals
    @staticmethod
    def _load_policy_ids(policy_dir: Path) -> frozenset[str]:
        """Collect `## POL-xxx` ids from the handbook.

        A missing or empty directory yields an empty set, which makes every
        citation unverifiable and therefore loudly stripped — the safe
        direction: with no corpus, no claim can be backed.
        """
        if not policy_dir.is_dir():
            return frozenset()
        found: set[str] = set()
        for path in sorted(policy_dir.glob("*.md")):
            found.update(
                match.group(1).upper()
                for match in _POLICY_ID_RE.finditer(path.read_text(encoding="utf-8"))
            )
        return frozenset(found)

    def _known_ids_text(self) -> str:
        return ", ".join(sorted(self.known_policy_ids)) or "(no policy ids on file)"

    @staticmethod
    def _as_json(model: BaseModel) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def _tickets_from_answer(
        self, answer: str | None, profile: CandidateProfile
    ) -> list[ITTicket]:
        """Turn a ReAct final answer into tickets; `[]` when it is not one.

        Returning empty rather than raising is deliberate: an unusable answer is
        a routing decision (fall back), not an exception the graph must catch.
        """
        if not answer:
            return []
        try:
            payload = self._parse_json(answer)
        except AgentOutputError:
            return []
        rows = payload.get("tickets")
        if not isinstance(rows, list):
            return []

        tickets: list[ITTicket] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            system = str(row.get("system", "")).strip()
            action = str(row.get("action", "")).strip()
            if not system or not action:
                continue
            tickets.append(
                self._ticket(profile, len(tickets) + 1, system, action)
            )
        return tickets

    def _fallback_tickets(self, profile: CandidateProfile) -> list[ITTicket]:
        """The policy baseline: an identity request plus role-correct hardware.

        Deterministic on purpose — a fallback that varies run to run cannot be
        reviewed. Both entries stay `requested`: POL-004 forbids creating
        anything before the manager and HR have both approved.
        """
        laptop = (
            _DEVELOPER_LAPTOP
            if self._is_developer_role(profile.role)
            else _STANDARD_LAPTOP
        )
        return [
            self._ticket(
                profile,
                1,
                "email",
                "create mailbox and directory identity — needs hiring manager "
                "AND HR approval before creation (POL-004)",
            ),
            self._ticket(profile, 2, "hardware", laptop),
        ]

    @staticmethod
    def _ticket(
        profile: CandidateProfile, index: int, system: str, action: str
    ) -> ITTicket:
        return ITTicket(
            ticket_id=f"{profile.candidate_id}-IT-{index:02d}",
            system=system,
            action=action,
            status=TICKET_STATUS,
        )

    @staticmethod
    def _is_developer_role(role: str) -> bool:
        """POL-003: engineering and data roles get the developer machine."""
        lowered = str(role or "").lower()
        return any(word in lowered for word in _DEVELOPER_ROLE_WORDS)


# --------------------------------------------------------------------------
# Optional live probe — never part of the test suite.
# --------------------------------------------------------------------------
def _live_probe() -> None:  # pragma: no cover - manual, network-dependent
    """One real extraction call, to check the model handles English JSON at
    temperature 0 (the low-risk assumption recorded in the PRD).

    Runs only when a key is configured, makes exactly ONE call, and swallows
    every failure: a probe that breaks the module is worse than no probe.
    """
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001 - the probe never fails the module
        pass

    if not os.getenv("LLM_API_KEY"):
        print("live probe SKIPPED: no LLM_API_KEY configured (.env not set up)")
        return

    resume = (
        "Layla Al-Otaibi — Backend Developer.\n"
        "4 years with Python, FastAPI and PostgreSQL at a fintech.\n"
        "Available from 2026-10-01."
    )
    try:
        from src.llm import LLMClient
        from src.tools import build_hr_registry

        agents = HRAgents(LLMClient(), build_hr_registry())
        profile = agents.analyze_profile(
            resume, {"candidate_id": "PROBE-001"}, attempt=0
        )
        print("live probe OK:", profile.model_dump_json())
    except Exception as exc:  # noqa: BLE001 - report, never raise
        print(f"live probe FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    _live_probe()

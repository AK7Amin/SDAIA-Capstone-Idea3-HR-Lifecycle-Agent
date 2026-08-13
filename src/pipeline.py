"""One case, end to end: guards in, graph run, evidence out.

The graph is the coordinator, but a graph on its own is not a *run*. This
module is the layer that turns an intake file into something a grader can
check afterwards, and it owns the four responsibilities the graph deliberately
refuses:

**The guards run here, before the first node.** Size, then injection, then PII
masking — in that order, because regex work on attacker-controlled text is the
denial-of-service surface and the size check has to happen before any pattern
is compiled against it. By the time `candidate_meta` and `masked_resume` reach
the graph, the raw resume has already been dropped: nothing downstream — no
checkpoint row, no gate payload, no trace file — can leak text the guard
cleaned.

**The paused status is synthesized here** (critique M7). `hr_gate` calls
`interrupt()` and the graph writes terminal statuses only, so "awaiting
approval" is the pipeline's reading of `__interrupt__` — and it stays in the
*return value*. No audit event is minted to go with it: a pause is a fact about
the run, not a link in the case's hash chain, and forging one would put a
system's convenience into an audit record a human is meant to trust.

**Every invocation gets its own thread id.** The previous project derived the
id from the case id, so re-running a case appended a second run to the same
trace file; the verifier now calls that shape "merged runs". Two runs of one
case file are two threads and two files, always.

**Resuming updates the evidence.** A trace frozen at the pause is a false
record of a case that finished, so `resume_case` rewrites the file with the
whole trail. It finds the evidence folder through `candidate_meta`, where
`process_case` recorded it: the resume call is given a thread id and a graph
and nothing else, and the alternative — a machine-wide registry file — would
put one process's paths in another process's way. The key is prefixed with an
underscore, carries no personal data, and never reaches a rendered artifact.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from langgraph.types import Command

from src import llm as llm_module
from src.checkpointing import make_sqlite_saver
from src.graph import AgentDeps, build_graph
from src.guardrails import (
    DEFAULT_MAX_CALLS,
    DEFAULT_SIZE_LIMIT,
    BudgetGuard,
    InputTooLarge,
    enforce_input_size,
    find_pii,
    mask_pii,
    sanitize_resume,
    scan_text,
)
from src.observability import (
    BLOCK_INJECTION,
    BLOCK_PII,
    BLOCK_SIZE,
    DASHBOARD_FILENAME,
    METRICS_FILENAME,
    TRACES_DIRNAME,
    observe_case_latency,
    record_case,
    record_guardrail_block,
    record_node,
    render_dashboard,
    write_metrics_snapshot,
    write_trace,
)
from src.schemas import (
    AuditEvent,
    CandidateProfile,
    CaseStatus,
    ContractDraft,
    ITTicket,
    ProvisionResult,
    ReviewAction,
    ReviewVerdict,
    TrainingPlan,
    TrainingWeek,
)

__all__ = [
    "BUDGET_ENV_VAR",
    "DEFAULT_INTAKE_DIRNAME",
    "PROJECT_ROOT",
    "REPORTS_DIRNAME",
    "REPORTS_ROOT_KEY",
    "STUB_ENV_FLAG",
    "GuardResult",
    "ProductionWiring",
    "build_graph_with_stubs",
    "build_production_graph",
    "build_production_wiring",
    "default_reports_dir",
    "guard_resume",
    "load_case",
    "new_thread_id",
    "process_case",
    "resume_case",
    "stub_deps",
    "write_run_summary",
]

#: Repository root, resolved from this file so the CLI works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Evidence folder name, relative to the root a run is anchored at.
REPORTS_DIRNAME = "reports"
#: Where `main.py run` looks for hired-candidate files by default.
DEFAULT_INTAKE_DIRNAME = "sample_candidates"

#: Per-case model-call allowance; `.env.example` ships it.
BUDGET_ENV_VAR = "MAX_LLM_CALLS_PER_CASE"

#: Test/demo hook honoured by :func:`build_production_wiring`: with it set, the
#: CLI runs the deterministic stub agents instead of a provider. Documented on
#: purpose — an undocumented switch that disables the real model is a trap.
STUB_ENV_FLAG = "HR_AGENT_STUBS"

#: Where `process_case` records this run's evidence root, so `resume_case` can
#: find it days later with nothing but a thread id. Config, not case data.
REPORTS_ROOT_KEY = "_reports_root"

#: Keys never handed to the graph: the raw resume enters as `masked_resume` or
#: not at all. `intake` strips them too — this is the belt to that's braces.
_RAW_RESUME_KEYS = ("resume_text", "resume")

#: Characters allowed in the case-id part of a thread id. A thread id names a
#: file in `reports/traces/`, and the case id originates in an untrusted intake
#: file, so it is transliterated here rather than trusted.
_SAFE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


# --------------------------------------------------------------------------
# guards — everything that happens before the first node
# --------------------------------------------------------------------------
class GuardResult(NamedTuple):
    """What the input guards did to one resume, and what they found.

    `text` is what the graph is allowed to see. `injection_flagged` and `rule`
    report the *detection*, which is deliberately independent of whether the
    offending lines were removed — the attack demo runs the same scan with
    sanitising switched off, and a demo that reported "nothing found" there
    would be showing the wrong lesson.
    """

    text: str
    injection_flagged: bool
    removed_lines: tuple[str, ...]
    rule: str | None
    pii_labels: tuple[str, ...]


def guard_resume(
    text: str, *, size_limit: int = DEFAULT_SIZE_LIMIT, sanitize: bool = True
) -> GuardResult:
    """Screen untrusted resume text and return what the agents may read.

    Args:
        text: The resume exactly as it arrived in the intake file.
        size_limit: Characters the guards are willing to scan at all.
        sanitize: When False, detect but do not remove or mask — the
            `--no-guardrails` comparison path of the attack demo. Nothing is
            counted as blocked in that mode, because nothing was.

    Returns:
        A :class:`GuardResult`.

    Raises:
        InputTooLarge: The input exceeds `size_limit`. Counted as a size block
            on the way out, because a refusal nobody can see is not a control.
    """
    try:
        enforce_input_size(text, limit=size_limit)
    except InputTooLarge:
        record_guardrail_block(BLOCK_SIZE)
        raise

    verdict = scan_text(text, size_limit=size_limit)

    if sanitize:
        cleaned = sanitize_resume(text, size_limit=size_limit)
        body, removed = cleaned.clean_text, cleaned.removed_lines
        if verdict.blocked:
            record_guardrail_block(BLOCK_INJECTION)
    else:
        body, removed = text, ()

    labels = tuple(match.label for match in find_pii(body))
    if sanitize and labels:
        body = mask_pii(body)
        # One block per case, not one per match: the counter answers "how many
        # inputs carried PII", and a resume with six phone numbers is one input.
        record_guardrail_block(BLOCK_PII)

    return GuardResult(body, bool(verdict.blocked), tuple(removed), verdict.pattern, labels)


# --------------------------------------------------------------------------
# stub agents — deterministic, offline, and defined HERE
# --------------------------------------------------------------------------
# Production imports must never reach into `tests/`, so these are a sibling of
# the test doubles rather than the same objects: the e2e drives this graph, and
# a shared definition would let a test edit change what production runs.
_STUB_SKILL_WORDS = (
    "Python", "SQL", "Spark", "Airflow", "Kubernetes", "Terraform", "FastAPI",
    "Postgres", "Docker", "AWS", "Delta Lake", "ETL", "Kafka",
)


def _first_sentence(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text or "").split())
    head = flat.split(". ")[0]
    return head[:limit]


def _stub_skills(resume: str) -> list[str]:
    lowered = str(resume or "").lower()
    seen: list[str] = []
    for word in _STUB_SKILL_WORDS:
        if word.lower() in lowered and word not in seen:
            seen.append(word)
    return seen[:5]


def stub_deps(effects: Any = None) -> AgentDeps:
    """Deterministic stand-ins for the five agents — no model, no network.

    Every value is derived from the verified intake metadata and the masked
    resume, so a stubbed run exercises the real control flow (including one
    Reflexion revision, which is what puts `plan_reviewer` in the audit trail)
    while producing byte-identical output for identical input.

    Args:
        effects: Effects port for the graph; ``None`` gets the no-op port.

    Returns:
        An :class:`~src.state.AgentDeps` ready for
        :func:`~src.graph.build_graph`.
    """
    # One revision per case, counted per candidate so a graph reused across
    # cases does not carry the previous case's verdict into this one.
    reviews: dict[str, int] = {}

    def analyze_profile(masked_resume: str, candidate_meta: Mapping[str, Any]):
        meta = dict(candidate_meta or {})
        return CandidateProfile(
            candidate_id=str(meta.get("candidate_id") or ""),
            name=str(meta.get("name") or ""),
            role=str(meta.get("role") or ""),
            start_date=str(meta.get("start_date") or ""),
            skills=_stub_skills(masked_resume),
            experience_summary=_first_sentence(masked_resume),
        )

    def plan_training(profile: CandidateProfile, critique: str) -> TrainingPlan:
        weeks = [
            TrainingWeek(
                week=1,
                focus="Orientation and mandatory security training (POL-002)",
                activities=[
                    "Complete information security and data-handling training",
                    "Meet the onboarding buddy (POL-005)",
                ],
            ),
            TrainingWeek(
                week=2,
                focus=f"{profile.role or 'Role'} tooling and codebase tour",
                activities=[f"Shadow a {skill} task" for skill in profile.skills[:2]]
                or ["Shadow a first delivery task"],
            ),
        ]
        if critique:
            weeks.append(
                TrainingWeek(
                    week=3,
                    focus="Revision requested by the reviewer",
                    activities=[_first_sentence(critique, 120)],
                )
            )
        return TrainingPlan(
            weeks=weeks,
            rationale=(
                f"Deterministic stub plan for {profile.role or 'the role'}"
                + (" (revised after review)" if critique else "")
            ),
        )

    def review_plan(profile: CandidateProfile, plan: TrainingPlan) -> ReviewVerdict:
        key = profile.candidate_id or "-"
        round_number = reviews.get(key, 0)
        reviews[key] = round_number + 1
        if round_number == 0:
            return ReviewVerdict(
                action=ReviewAction.REVISE,
                critique="Week 2 does not say who signs off the first delivery task.",
                concerns=["sign-off owner for the first task is unnamed"],
            )
        return ReviewVerdict(
            action=ReviewAction.APPROVE,
            critique="Security training, buddy check-in and role tooling are covered.",
        )

    def draft_contract(profile: CandidateProfile) -> ContractDraft:
        return ContractDraft(
            candidate_id=profile.candidate_id,
            role=profile.role,
            start_date=profile.start_date,
            salary_band="B3",
            body_fields={
                "reporting_line": "Hiring manager",
                "work_mode": "on-site",
                "probation_days": 90,
            },
        )

    def provision_it(profile: CandidateProfile) -> ProvisionResult:
        return ProvisionResult(
            tickets=[
                ITTicket(
                    ticket_id=f"{profile.candidate_id}-IT-01",
                    system="email",
                    action=(
                        "create mailbox and directory identity — needs hiring "
                        "manager AND HR approval before creation (POL-004)"
                    ),
                    status="requested",
                ),
                ITTicket(
                    ticket_id=f"{profile.candidate_id}-IT-02",
                    system="hardware",
                    action="allocate laptop and monitor by role (POL-003)",
                    status="requested",
                ),
            ]
        )

    return AgentDeps(
        analyze_profile=analyze_profile,
        plan_training=plan_training,
        review_plan=review_plan,
        draft_contract=draft_contract,
        provision_it=provision_it,
        effects=effects,
    )


def build_graph_with_stubs(effects: Any, checkpoint_db: Path | str):
    """Compile the onboarding graph over stub agents and a sqlite checkpointer.

    The offline entry point: the end-to-end test and the notebook drive the
    real graph, the real guards, the real effects layer and a real checkpointer
    through it, with only the five model-backed agents replaced.

    Args:
        effects: Effects port (usually :class:`~src.effects.FileEffects`).
        checkpoint_db: Sqlite file for the checkpointer. Building a fresh graph
            on the same file is what gives a resume its new-process semantics.

    Returns:
        The compiled graph.
    """
    return build_graph(stub_deps(effects), make_sqlite_saver(checkpoint_db))


# --------------------------------------------------------------------------
# production wiring
# --------------------------------------------------------------------------
@dataclass
class ProductionWiring:
    """The compiled graph plus the objects a run wants to report on.

    `react_runs` is why this exists rather than a bare graph: the provisioning
    agent returns `(ProvisionResult, ReActResult)` and the graph only takes the
    first half, so the reasoning trace — how many tool steps ran, whether the
    verdict came from the model or from the deterministic fallback, whether the
    first call was imposed by policy — would otherwise be dropped on the floor.
    It is reported by the CLI and read by the notebook; it is deliberately NOT
    appended to the audit chain, because the chain records what the graph did.
    """

    graph: Any
    deps: AgentDeps
    agents: Any = None
    registry: Any = None
    llm: Any = None
    react_runs: list = field(default_factory=list)
    stubbed: bool = False

    def meter_snapshot(self) -> dict | None:
        """Usage numbers for the metrics artifact, or None on a stubbed run."""
        meter = getattr(self.llm, "meter", None)
        return meter.snapshot() if meter is not None else None


def _stubs_requested() -> bool:
    return os.getenv(STUB_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def build_production_wiring(
    checkpointer: Any, effects: Any = None, *, force_stubs: bool | None = None
) -> ProductionWiring:
    """Wire the five LLM-backed agents into the graph.

    Args:
        checkpointer: A saver from :mod:`src.checkpointing`.
        effects: Effects port; ``None`` means nothing reaches a disk.
        force_stubs: Override the :data:`STUB_ENV_FLAG` environment hook.

    Returns:
        A :class:`ProductionWiring`.

    Raises:
        MissingKeyError: No provider credential is configured. Raised at
            construction so an accidental real-agent path fails loudly instead
            of looking healthy until the first call.
    """
    stubbed = _stubs_requested() if force_stubs is None else bool(force_stubs)
    if stubbed:
        deps = stub_deps(effects)
        return ProductionWiring(
            graph=build_graph(deps, checkpointer), deps=deps, stubbed=True
        )

    # Imported here, not at module import: `src.llm` raises on a missing key
    # and `build_hr_registry` reads the handbook, and neither should happen
    # just because something imported the offline half of this module.
    from src.agents.real import HRAgents
    from src.llm import LLMClient
    from src.tools import build_hr_registry

    llm = LLMClient()
    registry = build_hr_registry()
    agents = HRAgents(llm, registry)
    react_runs: list = []

    def provision_it(profile):
        """Adapt `(ProvisionResult, ReActResult)` to the graph's contract."""
        result, react = agents.provision_it(profile)
        react_runs.append(react)
        return result

    deps = AgentDeps(
        # `analyze_profile` takes an extra `attempt` with a default, so it fits
        # the positional contract as it stands. The retry bound lives in the
        # graph and the graph passes no attempt number, so the default holds.
        analyze_profile=agents.analyze_profile,
        plan_training=agents.plan_training,
        review_plan=agents.review_plan,
        draft_contract=agents.draft_contract,
        provision_it=provision_it,
        effects=effects,
    )
    return ProductionWiring(
        graph=build_graph(deps, checkpointer),
        deps=deps,
        agents=agents,
        registry=registry,
        llm=llm,
        react_runs=react_runs,
    )


def build_production_graph(checkpointer: Any, effects: Any = None):
    """The production graph alone, for callers with nothing to report on."""
    return build_production_wiring(checkpointer, effects).graph


# --------------------------------------------------------------------------
# ids, paths and per-request state
# --------------------------------------------------------------------------
def _slug(case_id: str) -> str:
    """Make an untrusted case id safe to embed in a file name."""
    text = "".join(
        char if char in _SAFE_ID_CHARS else "_" for char in str(case_id or "").strip()
    ).strip("._-")
    return text[:48] or "case"


def new_thread_id(case_id: str) -> str:
    """A fresh checkpointer thread id: timestamp, entropy, then the case.

    The random tail is not decoration. Two runs of one case inside the same
    second are exactly the situation that produced a merged trace file last
    time, and a timestamp alone does not separate them.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"run{stamp}-{uuid.uuid4().hex[:6]}-{_slug(case_id)}"


def default_reports_dir(case_file: Path | str) -> Path:
    """`<the intake folder's parent>/reports` — evidence beside the run's data.

    An intake file lives in `<root>/intake/CAND-001.json` (or in
    `<root>/sample_candidates/...`), so the run's root is the grandparent and
    the evidence folder is its sibling. That keeps a run anchored to the data
    it processed instead of to whatever directory the operator happened to be
    standing in.
    """
    return Path(case_file).parent.parent / REPORTS_DIRNAME


def _max_calls_per_case() -> int:
    raw = os.getenv(BUDGET_ENV_VAR, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CALLS
    return value if value >= 1 else DEFAULT_MAX_CALLS


def _install_request_state(case_id: str) -> BudgetGuard:
    """Give this case its own budget and cost attribution.

    Written straight to the context variables in :mod:`src.llm`: the public
    setters hang off an `LLMClient` instance, and this layer must install the
    per-case state on stubbed and offline runs too, where constructing a client
    would raise for want of a key. `reset_request_state` — the public
    counterpart — clears both in the caller's `finally`.
    """
    guard = BudgetGuard(_max_calls_per_case())
    llm_module._ACTIVE_BUDGET.set(guard)
    llm_module._ACTIVE_CASE.set(str(case_id or "-"))
    return guard


# --------------------------------------------------------------------------
# reading state back out of the graph
# --------------------------------------------------------------------------
def _events(state: Mapping[str, Any]) -> list[AuditEvent]:
    """The audit trail as models, whatever shape the checkpointer returned."""
    trail = state.get("audit_trail") or []
    return [
        event if isinstance(event, AuditEvent) else AuditEvent.model_validate(event)
        for event in trail
    ]


def _status_text(value: Any) -> str:
    """Read a status as its wire literal, never as `CaseStatus.COMPLETED`."""
    return str(getattr(value, "value", value) or "")


def _node_runs(events: Sequence[AuditEvent]) -> list[str]:
    """Node *visits*, not events: a node emitting three events ran once.

    Consecutive events from one node belong to one visit, which is exactly how
    the graph emits them — and it keeps a bounded loop visible in the counter
    (two visits to `profile_analyst` still count as two).
    """
    runs: list[str] = []
    for event in events:
        if not runs or runs[-1] != event.node:
            runs.append(event.node)
    return runs


def _reports_root_of(values: Mapping[str, Any]) -> Path:
    """Where this case's run recorded its evidence folder, or the repo default."""
    meta = values.get("candidate_meta") or {}
    recorded = str(meta.get(REPORTS_ROOT_KEY) or "").strip()
    return Path(recorded) if recorded else PROJECT_ROOT / REPORTS_DIRNAME


def write_run_summary(
    reports_dir: Path | str,
    thread_ids: Sequence[str],
    meter_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Rewrite the metrics snapshot and the dashboard for a whole run.

    Called once per case by :func:`process_case` and once more by a batch
    runner with every thread it produced — a snapshot naming only the last case
    of a four-case run is exactly the stale artifact the thread-id list exists
    to expose (M12).

    The order is load-bearing: the snapshot reads the counters (so they must
    already be recorded), and the dashboard reads the snapshot and the trace
    files back off the disk. Rendering the page from live objects instead would
    produce something that looked healthy next to stale files.

    Args:
        reports_dir: Evidence root.
        thread_ids: Every thread this run produced.
        meter_snapshot: :meth:`src.llm.UsageMeter.snapshot` output, if any.

    Returns:
        Path of the rendered dashboard.
    """
    reports_dir = Path(reports_dir)
    write_metrics_snapshot(reports_dir, meter_snapshot, thread_ids)
    return render_dashboard(
        reports_dir / METRICS_FILENAME,
        reports_dir / TRACES_DIRNAME,
        reports_dir / DASHBOARD_FILENAME,
    )


def _write_evidence(
    reports_dir: Path,
    thread_id: str,
    case_id: str,
    events: Sequence[AuditEvent],
    meter_snapshot: Mapping[str, Any] | None,
) -> Path:
    """This case's trace, then the run artifacts that summarise it."""
    trace_path = write_trace(Path(reports_dir), thread_id, case_id, events)
    write_run_summary(reports_dir, [thread_id], meter_snapshot)
    return trace_path


# --------------------------------------------------------------------------
# the two public verbs
# --------------------------------------------------------------------------
def load_case(case_file: Path | str) -> dict:
    """Read one hired-candidate intake file.

    Raises:
        ValueError: The file is not a JSON object. An array or a bare string is
            not an intake payload, and guessing what was meant is how untrusted
            input becomes untrusted behaviour.
    """
    path = Path(case_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path.name} is a {type(payload).__name__}, not a candidate object"
        )
    return payload


def process_case(
    case_file: Path | str,
    graph: Any,
    reports_dir: Path | str | None = None,
    meter_snapshot: Mapping[str, Any] | None = None,
) -> dict:
    """Run one intake file up to its first stop, and write the evidence.

    Args:
        case_file: Path to the hired-candidate JSON payload.
        graph: A compiled graph with a checkpointer.
        reports_dir: Evidence root; defaults to :func:`default_reports_dir`.
        meter_snapshot: Usage numbers for the metrics artifact (a production
            run passes `ProductionWiring.meter_snapshot()`; a stubbed one has
            nothing to report).

    Returns:
        ``{"status", "thread_id", "case_id", "audit_trail", "masked_resume",
        "injection_flagged", "removed_lines", "pii_labels", "tool_calls",
        "reports_dir", "trace_file"}``. ``status`` is ``"awaiting_approval"``
        when the case stopped at the human gate — synthesized here, never
        written into the audit chain.

    Raises:
        InputTooLarge: The resume exceeded the guarded size. Nothing is written
            and no case outcome is counted: the run never started.
    """
    path = Path(case_file)
    payload = load_case(path)
    case_id = str(payload.get("candidate_id") or path.stem)
    resume_text = ""
    for key in _RAW_RESUME_KEYS:
        if payload.get(key):
            resume_text = str(payload[key])
            break

    started = time.perf_counter()
    _install_request_state(case_id)
    try:
        guard = guard_resume(resume_text)

        reports = Path(reports_dir) if reports_dir else default_reports_dir(path)
        thread_id = new_thread_id(case_id)
        meta = {
            key: value
            for key, value in payload.items()
            if key not in _RAW_RESUME_KEYS
        }
        meta[REPORTS_ROOT_KEY] = str(reports)

        result = graph.invoke(
            {
                "case_id": case_id,
                "candidate_meta": meta,
                "masked_resume": guard.text,
            },
            {"configurable": {"thread_id": thread_id}},
        )
    finally:
        llm_module.reset_request_state()

    paused = "__interrupt__" in result
    status = (
        CaseStatus.AWAITING_APPROVAL.value
        if paused
        else _status_text(result.get("status"))
    )
    events = _events(result)

    record_case(status)
    for node in _node_runs(events):
        record_node(node)
    observe_case_latency((time.perf_counter() - started) * 1000)

    trace_path = _write_evidence(
        reports, thread_id, str(result.get("case_id") or case_id), events, meter_snapshot
    )

    return {
        "status": status,
        "thread_id": thread_id,
        "case_id": str(result.get("case_id") or case_id),
        "audit_trail": events,
        "masked_resume": guard.text,
        "injection_flagged": guard.injection_flagged,
        "injection_rule": guard.rule,
        "removed_lines": list(guard.removed_lines),
        "pii_labels": list(guard.pii_labels),
        "tool_calls": int(result.get("tool_calls") or 0),
        "reports_dir": reports,
        "trace_file": trace_path,
    }


def resume_case(
    thread_id: str,
    decision: Any,
    graph: Any,
    reports_dir: Path | str | None = None,
    meter_snapshot: Mapping[str, Any] | None = None,
) -> dict:
    """Answer the human gate of a paused case and finish the run.

    Args:
        thread_id: The thread `process_case` reported.
        decision: ``"approve"``, ``"reject"``, or a `GateDecision`-shaped
            mapping. Anything unreadable is refused by the gate rather than
            guessed — the checkpoint is untouched, so a corrected resume works.
        graph: A compiled graph over the SAME checkpointer. A freshly built one
            is the point: that is what proves the pause survived the process.
        reports_dir: Evidence root; defaults to the one the original run
            recorded in the case's metadata.
        meter_snapshot: Usage numbers for the metrics artifact.

    Returns:
        ``{"status", "thread_id", "case_id", "audit_trail", "tool_calls",
        "reports_dir", "trace_file"}`` — the trail is the WHOLE run, and the
        trace file on disk is rewritten to match it.

    Raises:
        ValueError: The thread is unknown, or the decision is unreadable.
    """
    config = {"configurable": {"thread_id": str(thread_id)}}
    before = dict(graph.get_state(config).values or {})
    if not before:
        raise ValueError(
            f"unknown thread id {thread_id!r}: there is no paused case to resume "
            "(check the checkpointer — a sqlite run cannot be resumed against "
            "Postgres, or the other way round)"
        )

    case_id = str(before.get("case_id") or "")
    reports = Path(reports_dir) if reports_dir else _reports_root_of(before)
    already_seen = len(before.get("audit_trail") or [])

    started = time.perf_counter()
    _install_request_state(case_id)
    try:
        final = graph.invoke(Command(resume={"decision": decision}), config)
    finally:
        llm_module.reset_request_state()

    status = (
        CaseStatus.AWAITING_APPROVAL.value
        if "__interrupt__" in final
        else _status_text(final.get("status"))
    )
    events = _events(final)

    record_case(status)
    # Only the events this resume produced: the pre-gate visits were counted
    # when the case first ran, and counting them twice would report a graph
    # that walked its own trail again.
    for node in _node_runs(events[already_seen:]):
        record_node(node)
    observe_case_latency((time.perf_counter() - started) * 1000)

    trace_path = _write_evidence(
        reports, str(thread_id), str(final.get("case_id") or case_id), events,
        meter_snapshot,
    )

    return {
        "status": status,
        "thread_id": str(thread_id),
        "case_id": str(final.get("case_id") or case_id),
        "audit_trail": events,
        "tool_calls": int(final.get("tool_calls") or 0),
        "reports_dir": reports,
        "trace_file": trace_path,
    }

"""Slice 6 — the concrete LLM-backed agents and their prompts.

Written RED before `src/agents/real.py` exists. Every test here is OFFLINE: the
model is a `StubLLM` replaying scripted replies, while the tool registry, the
policy handbook and the ReAct loop are the REAL modules from slices 3-5. So a
green run proves the parsing, the citation rule and the tool dispatch — not a
mock's return value.

Five properties are locked here:

1. **Contract in, contract out.** Every agent turns one model reply into a
   typed Pydantic contract, or raises `AgentOutputError`. Garbage never becomes
   a half-filled contract that flows silently into the graph.
2. **Untrusted content stays fenced.** The resume — and anything derived from
   it — reaches the model inside the guardrails wrapper, marked DATA.
3. **The reviewer's policy read is a REAL dispatch.** `review_plan` grows the
   registry's execution log and the retrieved text lands in the prompt; a model
   narrating "I checked the handbook" would fail these assertions.
4. **Never invent authority.** A citation to a policy id that is not in the
   corpus is stripped and reported as a concern.
5. **Honest attribution.** When provisioning falls back to a deterministic
   result the verdict says `decision_source="fallback"`, and the separate
   `forced_first_call` flag survives untouched — the bug slice 4 froze.
"""
import urllib.request
from pathlib import Path

import pytest

from src.agents import prompts
from src.agents.real import (
    REASONING_PATTERNS,
    UNVERIFIED_CITATION_MARK,
    AgentOutputError,
    HRAgents,
)
from src.guardrails import UNTRUSTED_BEGIN, UNTRUSTED_END
from src.schemas import (
    CandidateProfile,
    ContractDraft,
    ProvisionResult,
    ReviewAction,
    ReviewVerdict,
    TrainingPlan,
    TrainingWeek,
)
from src.tools import build_hr_registry

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"


# --------------------------------------------------------------------------
# helpers — fake model, real registry, real handbook
# --------------------------------------------------------------------------
class StubLLM:
    """Replays scripted replies and records every call it was asked to make.

    `calls` is the whole point: the prompt-shape tests read the text the agent
    actually built, and the metering tests read the `node` label it tagged.
    """

    def __init__(self, *replies, repeat=False):
        self.queue = list(replies)
        self.repeat = repeat
        self.calls: list[dict] = []

    def invoke(self, prompt, node="-", case_id=None):
        self.calls.append({"prompt": prompt, "node": node, "case_id": case_id})
        if not self.queue:
            if self.repeat and self.calls:
                return self.last_reply
            raise AssertionError(
                f"model invoked {len(self.calls)} times but replies ran out"
            )
        if self.repeat and len(self.queue) == 1:
            self.last_reply = self.queue[0]
            return self.last_reply
        self.last_reply = self.queue.pop(0)
        return self.last_reply

    @property
    def prompts(self) -> list[str]:
        return [call["prompt"] for call in self.calls]

    @property
    def last_prompt(self) -> str:
        return self.calls[-1]["prompt"]


def make_agents(*replies, repeat=False, registry=None):
    """`HRAgents` wired to a stub model and the real handbook registry."""
    llm = StubLLM(*replies, repeat=repeat)
    registry = registry if registry is not None else build_hr_registry(POLICY_DIR)
    return HRAgents(llm, registry, policy_dir=POLICY_DIR), llm, registry


def a_profile(**overrides) -> CandidateProfile:
    base = dict(
        candidate_id="CAND-001",
        name="Sara Alqahtani",
        role="Data Engineer",
        start_date="2026-09-01",
        skills=["Spark", "Airflow"],
        experience_summary="5 years of ETL pipelines.",
    )
    base.update(overrides)
    return CandidateProfile(**base)


def a_plan(rationale="Ramp-up plan.") -> TrainingPlan:
    return TrainingPlan(
        weeks=[
            TrainingWeek(week=1, focus="Security onboarding", activities=["training"]),
            TrainingWeek(week=2, focus="Codebase tour", activities=["pairing"]),
        ],
        rationale=rationale,
    )


PROFILE_REPLY = """{
  "name": "Sara Alqahtani",
  "role": "Data Engineer",
  "start_date": "2026-09-01",
  "skills": ["Spark", "Airflow"],
  "experience_summary": "Five years of ETL pipeline work."
}"""

PLAN_REPLY = """{
  "weeks": [
    {"week": 1, "focus": "Security training", "activities": ["POL-002 course"]},
    {"week": 2, "focus": "Data platform", "activities": ["shadow a pipeline"]}
  ],
  "rationale": "Security first, then the stack."
}"""

REVIEW_REPLY = """{
  "action": "revise",
  "critique": "Week 1 omits the mandatory security training.",
  "concerns": ["no buddy check-in"]
}"""

CONTRACT_REPLY = """{
  "salary_band": "B3",
  "body_fields": {"probation_days": 90, "manager": "Head of Data"}
}"""

PROVISION_FINAL = """Thought: The policy excerpt tells me what this role needs.
Final Answer: {"tickets": [
  {"system": "email", "action": "create mailbox"},
  {"system": "hardware", "action": "allocate developer laptop 32 GB (POL-003)"}
]}"""


# --------------------------------------------------------------------------
# analyze_profile — extraction over untrusted text
# --------------------------------------------------------------------------
def test_analyze_profile_parses_a_well_formed_reply_into_the_contract():
    agents, _, _ = make_agents(PROFILE_REPLY)
    profile = agents.analyze_profile("Resume text.", {"candidate_id": "CAND-001"})
    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Sara Alqahtani"
    assert profile.role == "Data Engineer"
    assert profile.skills == ["Spark", "Airflow"]
    assert profile.missing_fields() == []


def test_analyze_profile_takes_the_candidate_id_from_intake_not_from_the_model():
    """The model may rewrite prose; it may never rewrite the case identity."""
    agents, _, _ = make_agents('{"candidate_id": "ATTACKER-9", "name": "X"}')
    profile = agents.analyze_profile("Resume.", {"candidate_id": "CAND-001"})
    assert profile.candidate_id == "CAND-001"


def test_analyze_profile_accepts_code_fenced_json():
    agents, _, _ = make_agents("```json\n" + PROFILE_REPLY + "\n```")
    assert agents.analyze_profile("Resume.", {"candidate_id": "C1"}).role == (
        "Data Engineer"
    )


def test_analyze_profile_raises_agent_output_error_on_garbage():
    agents, _, _ = make_agents("I am afraid I cannot help with that.")
    with pytest.raises(AgentOutputError):
        agents.analyze_profile("Resume.", {"candidate_id": "C1"})


def test_analyze_profile_empty_object_yields_missing_fields_not_a_crash():
    """Emptiness is a routing signal (re-extract), never an exception."""
    agents, _, _ = make_agents("{}")
    profile = agents.analyze_profile("Resume.", {"candidate_id": "C1"})
    assert profile.missing_fields() == ["name", "role", "start_date"]


def test_analyze_profile_wraps_the_resume_in_the_untrusted_fence():
    agents, llm, _ = make_agents(PROFILE_REPLY)
    agents.analyze_profile("Ten years of Kafka.", {"candidate_id": "C1"})
    prompt = llm.last_prompt
    assert UNTRUSTED_BEGIN in prompt and UNTRUSTED_END in prompt
    assert "Ten years of Kafka." in prompt
    # The resume must sit INSIDE the fence, not before it.
    assert prompt.index(UNTRUSTED_BEGIN) < prompt.index("Ten years of Kafka.")
    assert prompt.index("Ten years of Kafka.") < prompt.index(UNTRUSTED_END)


def test_analyze_profile_prompt_says_the_resume_is_data_not_instructions():
    agents, llm, _ = make_agents(PROFILE_REPLY)
    agents.analyze_profile("Resume.", {"candidate_id": "C1"})
    lowered = llm.last_prompt.lower()
    assert "data" in lowered and "not instructions" in lowered


def test_analyze_profile_adds_no_retry_hint_on_the_first_attempt():
    agents, llm, _ = make_agents(PROFILE_REPLY)
    agents.analyze_profile("Resume.", {"candidate_id": "C1"}, attempt=0)
    assert prompts.RETRY_HINT not in llm.last_prompt


def test_analyze_profile_adds_a_retry_hint_on_later_attempts():
    agents, llm, _ = make_agents(PROFILE_REPLY)
    agents.analyze_profile("Resume.", {"candidate_id": "C1"}, attempt=1)
    assert prompts.RETRY_HINT in llm.last_prompt


def test_analyze_profile_meters_under_its_own_node_name():
    agents, llm, _ = make_agents(PROFILE_REPLY)
    agents.analyze_profile("Resume.", {"candidate_id": "C1"})
    assert llm.calls[-1]["node"] == "profile_analyst"


# --------------------------------------------------------------------------
# plan_training — Plan-and-Execute
# --------------------------------------------------------------------------
def test_plan_training_parses_a_typed_multi_week_plan():
    agents, _, _ = make_agents(PLAN_REPLY)
    plan = agents.plan_training(a_profile())
    assert isinstance(plan, TrainingPlan)
    assert [week.week for week in plan.weeks] == [1, 2]
    assert plan.weeks[0].activities == ["POL-002 course"]


def test_plan_training_rejects_a_plan_with_no_weeks():
    """A zero-week plan is a silent failure dressed as success."""
    agents, _, _ = make_agents('{"weeks": [], "rationale": "nothing to do"}')
    with pytest.raises(AgentOutputError):
        agents.plan_training(a_profile())


def test_plan_training_raises_on_malformed_json():
    agents, _, _ = make_agents("Here is a nice plan for the new hire!")
    with pytest.raises(AgentOutputError):
        agents.plan_training(a_profile())


def test_plan_training_fences_the_profile_as_untrusted_derived_content():
    """The profile was extracted FROM the resume, so it is untrusted too."""
    agents, llm, _ = make_agents(PLAN_REPLY)
    agents.plan_training(a_profile())
    assert UNTRUSTED_BEGIN in llm.last_prompt
    assert UNTRUSTED_END in llm.last_prompt


def test_plan_training_injects_the_reviewer_critique_only_when_revising():
    agents, llm, _ = make_agents(PLAN_REPLY, PLAN_REPLY)
    agents.plan_training(a_profile())
    first = llm.prompts[0]
    agents.plan_training(a_profile(), critique="Week 1 omits security training.")
    second = llm.prompts[1]
    assert "Week 1 omits security training." in second
    assert prompts.REVISION_HEADER in second
    assert prompts.REVISION_HEADER not in first


def test_plan_training_strips_a_citation_the_handbook_does_not_contain():
    agents, _, _ = make_agents(
        '{"weeks": [{"week": 1, "focus": "Read POL-999", "activities": []}],'
        ' "rationale": "As required by POL-999 and POL-002."}'
    )
    plan = agents.plan_training(a_profile())
    # The invented id no longer stands anywhere as a rule the plan relies on.
    assert "POL-999" not in plan.weeks[0].focus
    assert UNVERIFIED_CITATION_MARK in plan.weeks[0].focus
    assert "POL-002" in plan.rationale
    # ...but the removal is on the record, not silent.
    assert "removed" in plan.rationale.lower() and "POL-999" in plan.rationale


def test_plan_training_meters_under_its_own_node_name():
    agents, llm, _ = make_agents(PLAN_REPLY)
    agents.plan_training(a_profile())
    assert llm.calls[-1]["node"] == "training_planner"


# --------------------------------------------------------------------------
# review_plan — Reflexion, with a REAL policy lookup
# --------------------------------------------------------------------------
def test_review_plan_parses_a_verdict():
    agents, _, _ = make_agents(REVIEW_REPLY)
    verdict = agents.review_plan(a_profile(), a_plan())
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.action is ReviewAction.REVISE
    assert "security training" in verdict.critique


def test_review_plan_parses_an_approval():
    agents, _, _ = make_agents('{"action": "approve", "critique": "Solid."}')
    assert agents.review_plan(a_profile(), a_plan()).action is ReviewAction.APPROVE


def test_review_plan_raises_on_malformed_json():
    agents, _, _ = make_agents("Looks good to me.")
    with pytest.raises(AgentOutputError):
        agents.review_plan(a_profile(), a_plan())


def test_review_plan_dispatches_the_policy_tool_for_real():
    """A narrated 'I consulted the handbook' cannot grow the execution log."""
    agents, _, registry = make_agents(REVIEW_REPLY)
    before = len(registry.execution_log)
    agents.review_plan(a_profile(), a_plan())
    assert len(registry.execution_log) == before + 1
    entry = registry.execution_log[-1]
    assert entry.name == "hr_policy_lookup"
    assert entry.ok is True


def test_review_plan_feeds_the_retrieved_policy_text_into_the_prompt():
    agents, llm, _ = make_agents(REVIEW_REPLY)
    agents.review_plan(a_profile(), a_plan())
    # Text that exists only in the handbook file, never in our templates.
    assert "probation" in llm.last_prompt.lower() or "POL-00" in llm.last_prompt


def test_review_plan_reports_an_invented_citation_as_a_concern():
    agents, _, _ = make_agents('{"action": "approve", "critique": "ok"}')
    verdict = agents.review_plan(a_profile(), a_plan(rationale="Per POL-999."))
    assert any("POL-999" in concern for concern in verdict.concerns)


def test_review_plan_meters_under_its_own_node_name():
    agents, llm, _ = make_agents(REVIEW_REPLY)
    agents.review_plan(a_profile(), a_plan())
    assert llm.calls[-1]["node"] == "plan_reviewer"


# --------------------------------------------------------------------------
# draft_contract — typed fields only, no file IO (M9)
# --------------------------------------------------------------------------
def test_draft_contract_fills_typed_fields():
    agents, _, _ = make_agents(CONTRACT_REPLY)
    draft = agents.draft_contract(a_profile())
    assert isinstance(draft, ContractDraft)
    assert draft.salary_band == "B3"
    assert draft.body_fields["probation_days"] == 90


def test_draft_contract_takes_identity_fields_from_the_profile():
    agents, _, _ = make_agents(
        '{"candidate_id": "X", "role": "CEO", "start_date": "1999-01-01",'
        ' "salary_band": "A1"}'
    )
    draft = agents.draft_contract(a_profile())
    assert draft.candidate_id == "CAND-001"
    assert draft.role == "Data Engineer"
    assert draft.start_date == "2026-09-01"


def test_draft_contract_writes_nothing_to_disk(tmp_path, monkeypatch):
    """Governance ordering M9: nothing binding exists while the case is open."""
    monkeypatch.chdir(tmp_path)
    agents, _, _ = make_agents(CONTRACT_REPLY)
    agents.draft_contract(a_profile())
    assert list(tmp_path.iterdir()) == []


def test_draft_contract_records_a_stripped_citation_in_its_notes():
    agents, _, _ = make_agents(
        '{"salary_band": "B3", "body_fields": {"probation": "per POL-777"}}'
    )
    draft = agents.draft_contract(a_profile())
    assert "POL-777" not in draft.body_fields["probation"]
    assert any("POL-777" in note for note in draft.body_fields["citation_notes"])


def test_draft_contract_raises_on_malformed_json():
    agents, _, _ = make_agents("The contract is ready.")
    with pytest.raises(AgentOutputError):
        agents.draft_contract(a_profile())


# --------------------------------------------------------------------------
# provision_it — ReAct with a forced policy read
# --------------------------------------------------------------------------
def test_provision_it_turns_the_final_answer_into_tickets():
    agents, _, _ = make_agents(PROVISION_FINAL)
    result, react = agents.provision_it(a_profile())
    assert isinstance(result, ProvisionResult)
    assert [ticket.system for ticket in result.tickets] == ["email", "hardware"]
    assert react.decision_source == "model"


def test_provision_it_assigns_deterministic_ticket_ids():
    agents, _, _ = make_agents(PROVISION_FINAL)
    result, _ = agents.provision_it(a_profile())
    assert [t.ticket_id for t in result.tickets] == ["CAND-001-IT-01", "CAND-001-IT-02"]


def test_provision_it_forces_the_policy_read_before_reasoning():
    agents, _, registry = make_agents(PROVISION_FINAL)
    _, react = agents.provision_it(a_profile())
    assert react.forced_first_call is True
    assert react.steps[0].action == "hr_policy_lookup"
    assert any(entry.name == "hr_policy_lookup" for entry in registry.execution_log)


def test_provision_it_shows_the_policy_text_to_the_model():
    agents, llm, _ = make_agents(PROVISION_FINAL)
    agents.provision_it(a_profile())
    # Wording that exists only in the handbook file, never in our templates —
    # so this asserts the RETRIEVED text reached the model, not our own prose.
    assert "hiring manager" in llm.prompts[0]


def test_provision_it_falls_back_when_the_loop_is_exhausted():
    tool_forever = (
        "Thought: I need the handbook.\n"
        "Action: hr_policy_lookup\n"
        'Action Input: {"query": "equipment"}'
    )
    agents, _, _ = make_agents(tool_forever, repeat=True)
    result, react = agents.provision_it(a_profile())
    assert react.exhausted is True
    assert react.decision_source == "fallback"
    assert result.tickets, "a fallback that provisions nothing is not a fallback"


def test_provision_it_fallback_keeps_the_forced_call_flag_honest():
    """The frozen slice-4 bug: `decision_source` must not overwrite the flag."""
    agents, _, _ = make_agents("Thought: hmm.", repeat=True)
    _, react = agents.provision_it(a_profile())
    assert react.decision_source == "fallback"
    assert react.forced_first_call is True


def test_provision_it_falls_back_when_the_final_answer_is_prose():
    agents, _, _ = make_agents("Final Answer: I have provisioned everything.")
    result, react = agents.provision_it(a_profile())
    assert react.decision_source == "fallback"
    assert result.tickets


def test_provision_it_falls_back_when_the_model_provisions_nothing():
    agents, _, _ = make_agents('Final Answer: {"tickets": []}')
    result, react = agents.provision_it(a_profile())
    assert react.decision_source == "fallback"
    assert result.tickets


def test_provision_it_fallback_follows_the_equipment_policy_by_role():
    engineer, _, _ = make_agents("Thought: silence.", repeat=True)
    dev_result, _ = engineer.provision_it(a_profile(role="Data Engineer"))
    finance, _, _ = make_agents("Thought: silence.", repeat=True)
    std_result, _ = finance.provision_it(
        a_profile(candidate_id="CAND-007", role="Finance Officer")
    )
    dev_text = dev_result.model_dump_json()
    std_text = std_result.model_dump_json()
    assert "32 GB" in dev_text and "32 GB" not in std_text
    assert "16 GB" in std_text


def test_provision_it_meters_under_its_own_node_name():
    agents, llm, _ = make_agents(PROVISION_FINAL)
    agents.provision_it(a_profile())
    assert llm.calls[-1]["node"] == "it_provisioner"


def test_provision_it_strips_an_invented_citation_from_ticket_text():
    agents, _, _ = make_agents(
        'Final Answer: {"tickets": [{"system": "vpn", "action": "grant per POL-999"}]}'
    )
    result, react = agents.provision_it(a_profile())
    assert "POL-999" not in result.model_dump_json()
    # The receipt survives in the trace rather than vanishing silently.
    assert any("POL-999" in step.observation for step in react.steps)


# --------------------------------------------------------------------------
# citation validation — never invent authority
# --------------------------------------------------------------------------
def test_validate_citations_strips_unknown_ids_and_keeps_real_ones():
    agents, _, _ = make_agents()
    plan = a_plan(rationale="Follow POL-002 and POL-999 in week one.")
    check = agents.validate_citations(plan)
    assert "POL-999" not in check.clean.model_dump_json()
    assert "POL-002" in check.clean.model_dump_json()
    assert check.removed == ("POL-999",)
    assert any("POL-999" in concern for concern in check.concerns)


def test_validate_citations_leaves_a_clean_object_untouched():
    agents, _, _ = make_agents()
    plan = a_plan(rationale="Follow POL-001 and POL-005.")
    check = agents.validate_citations(plan)
    assert check.clean == plan
    assert check.concerns == () and check.removed == ()


def test_validate_citations_works_on_any_contract():
    agents, _, _ = make_agents()
    verdict = ReviewVerdict(
        action=ReviewAction.APPROVE, critique="Cites POL-404.", concerns=[]
    )
    check = agents.validate_citations(verdict)
    assert "POL-404" not in check.clean.critique
    assert check.removed == ("POL-404",)


def test_known_policy_ids_come_from_the_handbook_corpus():
    agents, _, _ = make_agents()
    assert {"POL-001", "POL-002", "POL-003", "POL-004", "POL-005"} <= (
        agents.known_policy_ids
    )


# --------------------------------------------------------------------------
# JSON extraction helper
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here is the JSON:\n{"a": 1}\nLet me know if you need changes.',
        '{"a": 1}\n\nI hope that helps.',
    ],
)
def test_parse_json_recovers_the_object_from_common_model_wrappers(raw):
    agents, _, _ = make_agents()
    assert agents._parse_json(raw) == {"a": 1}


def test_parse_json_keeps_nested_objects_and_braces_inside_strings():
    agents, _, _ = make_agents()
    raw = '{"outer": {"inner": [1, 2]}, "text": "a } brace in a string"}'
    parsed = agents._parse_json(raw)
    assert parsed["outer"]["inner"] == [1, 2]
    assert parsed["text"].endswith("string")


@pytest.mark.parametrize("raw", ["", "no json here", "{unbalanced: ", "[1, 2, 3]"])
def test_parse_json_raises_on_garbage(raw):
    agents, _, _ = make_agents()
    with pytest.raises(AgentOutputError):
        agents._parse_json(raw)


# --------------------------------------------------------------------------
# roles and reasoning patterns (rubric D1/D3)
# --------------------------------------------------------------------------
def test_every_agent_declares_a_named_reasoning_pattern():
    expected = {
        "analyze_profile": "extraction",
        "plan_training": "plan-and-execute",
        "review_plan": "reflexion",
        "draft_contract": "template-fill",
        "provision_it": "react",
    }
    for method, pattern in expected.items():
        assert REASONING_PATTERNS[method] == pattern


def test_reasoning_pattern_is_reachable_by_graph_node_name():
    agents, _, _ = make_agents()
    assert agents.reasoning_pattern("training_planner") == "plan-and-execute"
    assert agents.reasoning_pattern("it_provisioner") == "react"
    assert agents.reasoning_pattern("plan_reviewer") == "reflexion"


def test_unknown_role_has_no_invented_pattern():
    agents, _, _ = make_agents()
    assert agents.reasoning_pattern("notifier") == ""


def test_roles_are_five_distinct_named_responsibilities():
    agents, _, _ = make_agents()
    roles = agents.describe_roles()
    assert len(roles) == 5
    assert len({role.node for role in roles}) == 5
    assert len({role.method for role in roles}) == 5
    assert all(role.responsibility and role.pattern for role in roles)


def test_every_declared_role_is_a_real_callable():
    agents, _, _ = make_agents()
    for role in agents.describe_roles():
        assert callable(getattr(agents, role.method))


# --------------------------------------------------------------------------
# offline guarantee
# --------------------------------------------------------------------------
def test_no_agent_touches_the_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("an agent opened a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    agents, _, _ = make_agents(PROFILE_REPLY, PLAN_REPLY, REVIEW_REPLY, PROVISION_FINAL)
    profile = agents.analyze_profile("Resume.", {"candidate_id": "CAND-001"})
    plan = agents.plan_training(profile)
    agents.review_plan(profile, plan)
    agents.provision_it(profile)

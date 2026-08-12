"""Slice 4 — the ReAct loop.

Written RED before `src/agents/react.py` exists. Offline by design: the model
is a closure replaying scripted replies, and the registry is the REAL
`ToolRegistry` from slice 3 holding a stub tool — so every observation in these
tests is an actual dispatch output, not a mock's return value.

Four properties are locked here because each one has burned a previous run:

1. **Earliest match wins.** A reply carrying both `Action:` and `Final Answer:`
   must be resolved by position in the text, not by which regex we happened to
   check first.
2. **Bounded.** The loop stops at `max_steps` and says so (`exhausted`).
3. **Scratchpad.** Every later prompt carries the earlier observations —
   short-term memory is what makes the second step better than the first.
4. **`forced_first_call` is independent of `decision_source`.** In the previous
   project the audit label was read off `decision_source`; downstream fallback
   code overwrote that field and the trace then credited the model with a tool
   call the system had forced. The regression test at the bottom of this file
   is that bug, frozen.
"""
import pytest

from src.agents.react import (
    ReActResult,
    ReActStep,
    build_react_prompt,
    force_first_lookup,
    run_react,
)
from src.tools import Tool, ToolCall, ToolError, ToolRegistry

TASK = "Provision a laptop for the new hire."


# --------------------------------------------------------------------------
# helpers — fake model, real registry
# --------------------------------------------------------------------------
def scripted_llm(*replies):
    """Closure replaying `replies` in order; returns `(llm_call, prompts)`.

    `prompts` is the live list of every prompt the loop built, which is how the
    scratchpad and last-step-nudge tests inspect the loop from outside.
    """
    remaining = list(replies)
    prompts: list[str] = []

    def llm_call(prompt: str) -> str:
        prompts.append(prompt)
        if not remaining:
            raise AssertionError(
                f"llm_call invoked {len(prompts)} times but only "
                f"{len(replies)} replies were scripted"
            )
        return remaining.pop(0)

    return llm_call, prompts


def looping_llm(reply):
    """Closure that returns the same reply forever — for the bound test."""
    prompts: list[str] = []

    def llm_call(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) > 50:  # a broken bound must fail fast, not hang
            raise AssertionError("loop is unbounded")
        return reply

    return llm_call, prompts


def make_registry():
    """Real registry, stub tools; `seen` records what actually executed."""
    seen: list[str] = []

    def lookup(query):
        """Look up an onboarding fact in the handbook."""
        seen.append(query)
        return f"handbook says: {query}"

    def boom(topic):
        """A tool that always refuses — feeds the error-observation test."""
        raise ToolError(f"no handbook entry for {topic!r}")

    registry = ToolRegistry(
        [
            Tool(name="lookup", description="Look up an onboarding fact.", run=lookup),
            Tool(name="boom", description="Always fails.", run=boom),
        ]
    )
    return registry, seen


def act(tool: str, payload: str) -> str:
    return f'Thought: I need {tool}.\nAction: {tool}\nAction Input: {payload}'


# --------------------------------------------------------------------------
# result shape
# --------------------------------------------------------------------------
def test_result_defaults_are_model_attributed_and_unforced():
    result = ReActResult(final_answer=None, steps=[])
    assert result.decision_source == "model"
    assert result.forced_first_call is False
    assert result.tool_calls == 0


def test_tool_calls_counts_only_steps_that_acted():
    steps = [
        ReActStep(thought="t1", action="lookup", action_input="{}", observation="o"),
        ReActStep(thought="t2", action=None, action_input=None, observation=""),
    ]
    assert ReActResult(final_answer="done", steps=steps).tool_calls == 1


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------
def test_prompt_carries_tools_task_and_format_contract():
    registry, _ = make_registry()
    prompt = build_react_prompt(TASK, registry)

    assert "lookup(query: string)" in prompt  # registry.describe() is embedded
    assert TASK in prompt
    assert "Thought:" in prompt
    assert "Action Input:" in prompt
    assert "Final Answer:" in prompt


def test_last_step_prompt_nudges_for_the_final_answer():
    registry, _ = make_registry()
    normal = build_react_prompt(TASK, registry, "", last_step=False)
    final = build_react_prompt(TASK, registry, "", last_step=True)

    assert "Final Answer now" in final
    assert "Final Answer now" not in normal


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def test_full_cycle_two_tools_then_final_answer():
    registry, seen = make_registry()
    llm, _ = scripted_llm(
        act("lookup", '{"query": "laptop policy"}'),
        act("lookup", '{"query": "shipping window"}'),
        "Thought: I have enough.\nFinal Answer: Laptop ordered, arrives in 5 days.",
    )

    result = run_react(llm, TASK, registry, max_steps=4)

    assert result.final_answer == "Laptop ordered, arrives in 5 days."
    assert result.exhausted is False
    assert result.tool_calls == 2
    assert seen == ["laptop policy", "shipping window"]
    # observations are REAL dispatch outputs, not invented text
    assert result.steps[0].observation == "handbook says: laptop policy"
    assert result.steps[1].observation == "handbook says: shipping window"
    assert isinstance(result.steps[0].call, ToolCall)
    assert result.steps[0].call.arguments == {"query": "laptop policy"}
    # and the registry logged exactly those two dispatches
    assert len(registry.execution_log) == result.tool_calls == 2
    assert [row.ok for row in registry.execution_log] == [True, True]


def test_scratchpad_feeds_the_previous_observation_into_the_next_prompt():
    registry, _ = make_registry()
    llm, prompts = scripted_llm(
        act("lookup", '{"query": "laptop policy"}'),
        "Final Answer: done",
    )

    run_react(llm, TASK, registry, max_steps=4)

    assert len(prompts) == 2
    assert "handbook says: laptop policy" not in prompts[0]
    assert "handbook says: laptop policy" in prompts[1]
    assert "laptop policy" in prompts[1]  # the action input is remembered too


def test_loop_is_bounded_when_the_model_never_answers():
    registry, seen = make_registry()
    llm, prompts = looping_llm(act("lookup", '{"query": "again"}'))

    result = run_react(llm, TASK, registry, max_steps=3)

    assert result.exhausted is True
    assert result.final_answer is None
    assert result.tool_calls == 3
    assert len(prompts) == 3
    assert len(seen) == 3


# --------------------------------------------------------------------------
# earliest match wins
# --------------------------------------------------------------------------
def test_action_before_final_answer_executes_the_tool_and_continues():
    registry, seen = make_registry()
    llm, _ = scripted_llm(
        'Thought: check first.\nAction: lookup\nAction Input: {"query": "vpn"}\n'
        "Final Answer: premature guess",
        "Final Answer: vpn access granted",
    )

    result = run_react(llm, TASK, registry, max_steps=4)

    assert seen == ["vpn"]  # the tool ran
    assert result.tool_calls == 1
    assert result.final_answer == "vpn access granted"  # not the premature one
    assert result.steps[0].observation == "handbook says: vpn"


def test_final_answer_before_action_ends_without_executing():
    registry, seen = make_registry()
    llm, _ = scripted_llm(
        "Thought: I already know this.\nFinal Answer: badge issued\n"
        'Action: lookup\nAction Input: {"query": "badge"}'
    )

    result = run_react(llm, TASK, registry, max_steps=4)

    assert seen == []  # nothing dispatched
    assert registry.execution_log == []
    assert result.tool_calls == 0
    assert result.exhausted is False
    assert result.final_answer == "badge issued"
    # greedy capture must not smuggle the action block into the answer
    assert "Action" not in result.final_answer


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------
def test_tool_error_becomes_an_observation_and_the_loop_survives():
    registry, _ = make_registry()
    llm, _ = scripted_llm(
        act("boom", '{"topic": "parking"}'),
        "Final Answer: recovered without the tool",
    )

    result = run_react(llm, TASK, registry, max_steps=4)

    assert result.steps[0].observation.startswith("tool error:")
    assert "parking" in result.steps[0].observation
    assert result.final_answer == "recovered without the tool"
    assert result.exhausted is False
    assert registry.execution_log[-1].ok is False  # the failure is audited


def test_unregistered_tool_is_refused_as_an_observation():
    registry, _ = make_registry()
    llm, _ = scripted_llm(
        act("wire_transfer", '{"amount": "9000"}'),
        "Final Answer: refused, out of role",
    )

    result = run_react(llm, TASK, registry, max_steps=4)

    assert result.steps[0].observation.startswith("tool error:")
    assert "REFUSED" in result.steps[0].observation
    assert result.steps[0].call is None
    assert result.final_answer == "refused, out of role"


def test_malformed_reply_exhausts_without_crashing():
    registry, _ = make_registry()
    llm, _ = scripted_llm("I think we should probably just email somebody.")

    result = run_react(llm, TASK, registry, max_steps=4)

    assert result.exhausted is True
    assert result.final_answer is None
    assert result.tool_calls == 0


# --------------------------------------------------------------------------
# honesty — the branch that lied last time
# --------------------------------------------------------------------------
def test_force_first_lookup_dispatches_through_the_registry():
    registry, seen = make_registry()

    call, output = force_first_lookup(registry, "lookup", {"query": "id badge"})

    assert isinstance(call, ToolCall)
    assert call.name == "lookup"
    assert call.arguments == {"query": "id badge"}
    assert output == "handbook says: id badge"
    assert seen == ["id badge"]
    assert len(registry.execution_log) == 1


def test_forced_first_call_survives_a_decision_source_fallback():
    """Regression: the audit flag must NOT be derived from decision_source.

    Simulates the real caller — policy forces the first lookup, the loop runs
    afterwards, and then downstream fallback code overwrites `decision_source`.
    Reading honesty off `decision_source` (the old bug) would now report the
    forced call as a model choice.
    """
    registry, _ = make_registry()
    call, output = force_first_lookup(registry, "lookup", {"query": "policy"})
    llm, _ = scripted_llm("Final Answer: provisioned")

    result = run_react(llm, TASK, registry, max_steps=2)
    result.steps.insert(
        0,
        ReActStep(
            thought="Policy requires a handbook read before any provisioning.",
            action="lookup",
            action_input={"query": "policy"},
            observation=output,
            call=call,
        ),
    )
    result.forced_first_call = True

    assert result.decision_source == "model"
    assert result.forced_first_call is True

    # downstream fallback rewrites the verdict label...
    result.decision_source = "fallback"

    # ...and the honesty flag is untouched.
    assert result.decision_source == "fallback"
    assert result.forced_first_call is True
    assert result.steps[0].call is call
    assert result.tool_calls == 1  # the forced step still counts as a tool call


def test_forced_first_call_stays_false_for_a_purely_model_driven_run():
    registry, _ = make_registry()
    llm, _ = scripted_llm(
        act("lookup", '{"query": "email account"}'),
        "Final Answer: account created",
    )

    result = run_react(llm, TASK, registry, max_steps=3)

    assert result.forced_first_call is False
    assert result.decision_source == "model"


def test_force_first_lookup_raises_on_an_unregistered_tool():
    registry, _ = make_registry()

    with pytest.raises(ToolError):
        force_first_lookup(registry, "wire_transfer", {"amount": "9000"})

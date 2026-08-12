"""ReAct loop (slice 4).

ReAct = Reason + Act: the model alternates a short *Thought* with either one
*Action* (a validated tool call) or its *Final Answer*, and every tool output
is fed back as an *Observation* before the next turn. The accumulated
Thought/Action/Observation text is the agent's short-term memory — it is
rebuilt into every subsequent prompt, which is why step two can be smarter
than step one.

Three decisions in here exist because their opposites have already cost a run:

* **Earliest match wins.** Models cheerfully emit both `Action:` and
  `Final Answer:` in one reply. Resolving that by "whichever regex we check
  first" makes behaviour depend on source-code order; resolving it by POSITION
  in the text makes it depend on what the model actually wrote first.
* **The last allowed step says so.** Without an explicit "give your Final
  Answer now" instruction the model spends the final turn on another tool call
  and the run ends empty-handed.
* **`forced_first_call` is a separate field from `decision_source`.** See
  `ReActResult`.

The loop is deliberately transport-free: `llm_call` is any `str -> str`
callable, so tests inject a closure and production injects the provider chain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.tools import ToolCall, ToolError, ToolRegistry

__all__ = [
    "ReActReply",
    "ReActResult",
    "ReActStep",
    "build_react_prompt",
    "force_first_lookup",
    "parse_react_reply",
    "run_react",
]

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
# Headers are matched at line starts (re.M) so prose that merely mentions
# "action" cannot hijack the parse. Note that `Action[ \t]*:` never matches an
# "Action Input:" line — the colon lands on "I" — so the two headers stay
# distinct without extra bookkeeping.
_THOUGHT_RE = re.compile(
    r"^[ \t]*Thought[ \t]*:[ \t]*(?P<thought>.*?)"
    r"(?=^[ \t]*(?:Action|Observation|Final[ \t]*Answer)[ \t]*:|\Z)",
    re.M | re.S | re.I,
)
_ACTION_RE = re.compile(
    r"^[ \t]*Action[ \t]*:[ \t]*(?P<tool>.+?)[ \t]*$",
    re.M | re.I,
)
_ACTION_INPUT_RE = re.compile(
    r"^[ \t]*Action[ \t]+Input[ \t]*:[ \t]*(?P<value>.*?)"
    r"(?=^[ \t]*(?:Thought|Action|Observation|Final[ \t]*Answer)[ \t]*:|\Z)",
    re.M | re.S | re.I,
)
# Greedy on purpose: a final answer may span paragraphs. Anything the model
# appended after it (a stray Action block) is trimmed by `_ANSWER_CUT_RE`.
_FINAL_RE = re.compile(
    r"^[ \t]*Final[ \t]*Answer[ \t]*:[ \t]*(?P<answer>.*)",
    re.M | re.S | re.I,
)
_ANSWER_CUT_RE = re.compile(
    r"^[ \t]*(?:Action[ \t]+Input|Action)[ \t]*:",
    re.M | re.I,
)


@dataclass(frozen=True)
class ReActReply:
    """One parsed model turn: at most one action, at most one final answer."""

    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    final_answer: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when the reply honoured neither half of the format contract."""
        return self.action is None and self.final_answer is None


def parse_react_reply(text: str) -> ReActReply:
    """Parse one model reply; **the earliest header in the text wins**.

    A reply containing both `Action:` and `Final Answer:` resolves to whichever
    appears first, in either order — the model's own sequencing decides, not
    ours. The final answer is captured greedily and then trimmed at any
    trailing action block so no `Action:` residue leaks into the answer.
    """
    text = text or ""
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group("thought").strip() if thought_match else ""

    action_match = _ACTION_RE.search(text)
    final_match = _FINAL_RE.search(text)

    if final_match is not None and (
        action_match is None or final_match.start() < action_match.start()
    ):
        answer = final_match.group("answer")
        cut = _ANSWER_CUT_RE.search(answer)
        if cut is not None:
            answer = answer[: cut.start()]
        return ReActReply(thought=thought, final_answer=answer.strip())

    if action_match is not None:
        tool = action_match.group("tool").strip()
        input_match = _ACTION_INPUT_RE.search(text, action_match.end())
        raw_input = input_match.group("value").strip() if input_match else None
        return ReActReply(thought=thought, action=tool, action_input=raw_input)

    return ReActReply(thought=thought)


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------
@dataclass
class ReActStep:
    """One Thought → Action → Observation cycle.

    `call` is the validated `ToolCall` that actually reached the registry, or
    `None` when the proposal was refused before dispatch (unknown tool, bad
    arguments) — the distinction matters in the audit trail.
    """

    thought: str = ""
    action: str | None = None
    action_input: Any = None
    observation: str = ""
    call: ToolCall | None = None


@dataclass
class ReActResult:
    """The outcome of a bounded ReAct run.

    `decision_source` labels *how the verdict was reached* — ``"model"`` when
    the model answered, ``"policy_enforced"`` when a rule overrode it,
    ``"fallback"`` when downstream code had to substitute a default.

    `forced_first_call` is a SEPARATE field BY DESIGN, and must never be
    derived from `decision_source`. Downstream fallback code overwrites
    `decision_source` after the loop returns; in the previous project the audit
    label was read off that field, so once it flipped to ``"fallback"`` a tool
    call the SYSTEM had forced was reported as a model choice. Two facts, two
    fields: one records who chose the answer, the other records whether the
    first tool call was imposed by policy.
    """

    final_answer: str | None
    steps: list[ReActStep] = field(default_factory=list)
    exhausted: bool = False
    decision_source: str = "model"
    forced_first_call: bool = False

    @property
    def tool_calls(self) -> int:
        """How many steps proposed an action (forced calls included)."""
        return sum(1 for step in self.steps if step.action is not None)


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------
_INSTRUCTIONS = """\
You are an onboarding agent that reasons with the ReAct pattern (Reason +
Act). Work in single steps: one short thought, then EITHER one tool action OR
your final answer. Never invent a tool that is not listed below, and never
invent an observation — wait for the real one."""

_FORMAT = """\
Reply with exactly one of these two blocks and nothing else.

To use a tool:
Thought: <why you need the tool>
Action: <one tool name from the list above>
Action Input: <a JSON object of that tool's arguments>

To finish:
Thought: <why you can answer now>
Final Answer: <the answer>"""

_LAST_STEP_NUDGE = """\
This is your LAST allowed step: give your Final Answer now, no more tools.
Answer from the observations you already have, and say plainly what is still
unverified."""


def build_react_prompt(
    task: str,
    registry: ToolRegistry,
    scratchpad: str = "",
    *,
    last_step: bool = False,
) -> str:
    """Instructions + tool catalogue + task + scratchpad (short-term memory).

    On the last allowed step the format contract is replaced by an explicit
    "Final Answer now" instruction — otherwise models spend the final turn on
    another tool call and the loop ends with nothing to show.
    """
    sections = [
        _INSTRUCTIONS,
        registry.describe(),
        _FORMAT,
        f"Task: {task}",
        "Steps so far:\n" + (scratchpad.strip() or "(none yet)"),
    ]
    if last_step:
        sections.append(_LAST_STEP_NUDGE)
    return "\n\n".join(sections)


def _render_scratchpad(steps: list[ReActStep]) -> str:
    """Replay the run so far in the same wire format the model writes."""
    lines: list[str] = []
    for step in steps:
        if step.thought:
            lines.append(f"Thought: {step.thought}")
        if step.action is not None:
            lines.append(f"Action: {step.action}")
            lines.append(f"Action Input: {step.action_input}")
        lines.append(f"Observation: {step.observation}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def run_react(
    llm_call: Callable[[str], str],
    task: str,
    registry: ToolRegistry,
    max_steps: int = 4,
) -> ReActResult:
    """Run a bounded ReAct loop and return the trace alongside the answer.

    `llm_call` is any `str -> str` callable (closure in tests, provider chain
    in production). Tool failures do not abort the run: a `ToolError` becomes a
    `tool error: ...` observation and the model gets a chance to recover, which
    is the whole point of feeding observations back.

    Stops when the model answers, when a reply honours neither half of the
    format contract (`exhausted=True`), or when `max_steps` is spent
    (`exhausted=True`).
    """
    steps: list[ReActStep] = []

    for index in range(max_steps):
        prompt = build_react_prompt(
            task,
            registry,
            _render_scratchpad(steps),
            last_step=(index == max_steps - 1),
        )
        reply = parse_react_reply(llm_call(prompt))

        if reply.final_answer is not None:
            return ReActResult(final_answer=reply.final_answer, steps=steps)

        if reply.action is None:
            # Neither header present: nothing to execute and nothing to report.
            # Record the dead turn so the trace shows why the run stopped.
            steps.append(
                ReActStep(
                    thought=reply.thought,
                    observation=(
                        "parse error: the reply contained neither an Action nor "
                        "a Final Answer"
                    ),
                )
            )
            return ReActResult(final_answer=None, steps=steps, exhausted=True)

        call: ToolCall | None = None
        try:
            call = registry.parse_call(reply.action, reply.action_input)
            observation = registry.dispatch(call).output
        except ToolError as exc:
            # `call` stays None when the refusal happened during parsing (an
            # unregistered tool never became a real call); it survives when a
            # validated call failed inside the tool body.
            observation = f"tool error: {exc}"

        steps.append(
            ReActStep(
                thought=reply.thought,
                action=reply.action,
                action_input=reply.action_input,
                observation=observation,
                call=call,
            )
        )

    return ReActResult(final_answer=None, steps=steps, exhausted=True)


def force_first_lookup(
    registry: ToolRegistry,
    tool_name: str,
    arguments: Any,
) -> tuple[ToolCall, str]:
    """Execute a policy-mandated tool call and return `(call, output)`.

    For callers that must read a rule BEFORE reasoning starts (the model is not
    consulted about whether to do it). The caller inserts the returned step at
    the front of the trace and sets `ReActResult.forced_first_call = True` —
    the flag exists precisely so this system-imposed call is never later
    credited to the model.

    `ToolError` propagates: a policy read that could not happen must not be
    silently downgraded to an observation the model may ignore.
    """
    call = registry.parse_call(tool_name, arguments)
    return call, registry.dispatch(call).output

"""MCP-style tool registry (slice 3).

MCP = Model Context Protocol. This module borrows MCP's *shape* without
running a server: every tool declares an `inputSchema` (JSON Schema), the model
sees that schema in the prompt, and the registry validates a proposed call
against it **before** any tool code runs. Dispatch forwards arguments by NAME,
never positionally — a schema whose property order differs from the callable's
signature must still land on the right parameters.

Two design decisions worth stating plainly:

* **No vector database for policy lookup.** The handbook is a handful of
  markdown sections; keyword overlap over `## POL-` sections is honest and
  auditable at this scale. Pretending otherwise would be theatre.
* **No `eval` in the date tool.** Expressions are parsed with a regex and
  computed with `datetime`/`timedelta`, so `__import__('os')` is a parse error
  rather than a code path.

Role boundary: `build_hr_registry` deliberately contains no finance tool. A
request for one is refused by `validate`, and the refusal is appended to
`execution_log` as a failed `ToolResult` so the audit trail records the attempt.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_hr_registry",
]


class ToolError(ValueError):
    """Raised for any refused or failed tool interaction.

    Subclasses `ValueError` so a caller that only guards against bad input
    still catches it.
    """


# JSON Schema type name -> accepted Python types.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _type_matches(value: Any, json_type: str | None) -> bool:
    if not json_type:
        return True
    accepted = _JSON_TYPES.get(json_type)
    if accepted is None:  # unknown type name: do not invent a rule
        return True
    # `bool` is a subclass of `int` in Python; numeric schemas must not accept it.
    if json_type in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, accepted)


@dataclass(frozen=True)
class ToolCall:
    """A proposed call: a tool name plus named arguments."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """One row of the execution log — success or refusal, both recorded."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    output: str
    latency_ms: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe row for traces and the notebook evidence cells."""
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "ok": self.ok,
            "output": self.output,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Tool:
    """A named callable plus the schema the model is shown."""

    name: str
    description: str
    run: Callable[..., Any]
    input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = _infer_schema(self.run)

    def descriptor(self) -> dict[str, Any]:
        """MCP `tools/list` entry: `{name, description, inputSchema}`."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": json.loads(json.dumps(self.input_schema)),
        }

    # -- schema helpers -------------------------------------------------
    @property
    def properties(self) -> dict[str, Any]:
        return dict(self.input_schema.get("properties", {}))

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []))

    def arg_type(self, name: str) -> str | None:
        return self.properties.get(name, {}).get("type")

    def signature_text(self) -> str:
        """`name(arg: type, ...)` — the line the model reads in the prompt."""
        args = ", ".join(
            f"{arg}: {spec.get('type', 'any')}" for arg, spec in self.properties.items()
        )
        return f"{self.name}({args})"


def _infer_schema(run: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema from a callable's signature.

    Every parameter is declared as a string (the model speaks text); a
    parameter without a default is required. `*args`/`**kwargs` are skipped —
    this registry only calls by name.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in inspect.signature(run).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param.name] = {"type": "string"}
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class ToolRegistry:
    """Validated, logged, name-based dispatch over a fixed set of tools."""

    def __init__(self, tools: Iterable[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ToolError(f"duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool
        self.execution_log: list[ToolResult] = []

    # -- introspection --------------------------------------------------
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def list_tools(self) -> list[dict[str, Any]]:
        """MCP `tools/list` payload for every registered tool."""
        return [tool.descriptor() for tool in self._tools.values()]

    def describe(self) -> str:
        """Prompt-ready catalogue: one line per tool, argument names and types."""
        if not self._tools:
            return "No tools are available."
        lines = ["Available tools:"]
        lines += [
            f"- {tool.signature_text()} — {tool.description}"
            for tool in self._tools.values()
        ]
        return "\n".join(lines)

    def refuse_reason(self, name: str) -> str:
        """Audit-ready sentence explaining why `name` is not callable here."""
        if name in self._tools:
            return f"Tool {name!r} is registered in this registry; no refusal applies."
        available = ", ".join(self._tools) or "none"
        return (
            f"REFUSED: tool {name!r} is not registered in this registry. "
            f"Role boundary — this agent holds no such capability, so the call "
            f"was rejected before execution. Registered tools: {available}."
        )

    # -- parse / validate / dispatch -------------------------------------
    def parse_call(self, name: str, raw_input: Any) -> ToolCall:
        """Turn raw model output into a `ToolCall` with named arguments.

        Accepted forms: a mapping, a JSON object string, or — when the tool has
        exactly one required argument — a bare value. A JSON array is refused:
        positional input has no place in a name-based registry.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(self.refuse_reason(name))

        if raw_input is None:
            return ToolCall(name, {})
        if isinstance(raw_input, Mapping):
            return ToolCall(name, dict(raw_input))

        text = raw_input if isinstance(raw_input, str) else repr(raw_input)
        stripped = text.strip()
        if not stripped:
            raise ToolError(
                f"tool {name!r}: empty input; expected a JSON object of arguments "
                f"{tool.required or list(tool.properties)}"
            )

        parsed: Any = None
        parsed_ok = False
        try:
            parsed = json.loads(stripped)
            parsed_ok = True
        except json.JSONDecodeError:
            pass

        if parsed_ok and isinstance(parsed, dict):
            return ToolCall(name, parsed)
        if parsed_ok and isinstance(parsed, list):
            raise ToolError(
                f"tool {name!r}: positional input is not supported; "
                f"pass a JSON object naming the arguments {list(tool.properties)}"
            )

        # Bare value: only unambiguous when exactly one argument is required.
        if len(tool.required) != 1:
            raise ToolError(
                f"tool {name!r}: cannot map a bare value onto "
                f"{len(tool.required)} required arguments {tool.required}; "
                f"pass a JSON object"
            )
        arg = tool.required[0]
        value: Any = stripped
        if parsed_ok and _type_matches(parsed, tool.arg_type(arg)):
            value = parsed
        return ToolCall(name, {arg: value})

    def validate(self, call: ToolCall) -> Tool:
        """Reject unknown tools and bad arguments BEFORE any tool code runs."""
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(self.refuse_reason(call.name))

        if not isinstance(call.arguments, Mapping):
            raise ToolError(
                f"tool {call.name!r}: arguments must be a mapping of names to values"
            )

        properties = tool.properties
        unknown = sorted(set(call.arguments) - set(properties))
        if unknown:
            raise ToolError(
                f"tool {call.name!r}: undeclared argument(s) {unknown}; "
                f"schema declares {sorted(properties)}"
            )

        missing = [arg for arg in tool.required if arg not in call.arguments]
        if missing:
            raise ToolError(
                f"tool {call.name!r}: missing required argument(s) {missing}"
            )

        for arg, value in call.arguments.items():
            expected = properties[arg].get("type")
            if not _type_matches(value, expected):
                raise ToolError(
                    f"tool {call.name!r}: argument {arg!r} expects {expected}, "
                    f"got {type(value).__name__}"
                )
        return tool

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Validate, run by keyword, and log the outcome either way."""
        started = time.perf_counter()
        try:
            tool = self.validate(call)
        except ToolError as exc:
            # A refusal is evidence too — record it, then re-raise.
            self._log(call, ok=False, output=str(exc), started=started)
            raise

        try:
            # BY NAME, always. Schema order never dictates parameter order.
            output = tool.run(**dict(call.arguments))
        except Exception as exc:  # noqa: BLE001 — tool bodies may raise anything
            message = f"{type(exc).__name__}: {exc}"
            self._log(call, ok=False, output=message, started=started)
            if isinstance(exc, ToolError):
                raise
            raise ToolError(f"tool {call.name!r} failed — {message}") from exc

        return self._log(call, ok=True, output=str(output), started=started)

    def run(self, name: str, raw_input: Any = None) -> ToolResult:
        """`parse_call` + `dispatch`, logging a parse refusal as well."""
        started = time.perf_counter()
        try:
            call = self.parse_call(name, raw_input)
        except ToolError as exc:
            self._log(ToolCall(name, {}), ok=False, output=str(exc), started=started)
            raise
        return self.dispatch(call)

    # -- internals -------------------------------------------------------
    def _log(
        self, call: ToolCall, *, ok: bool, output: str, started: float
    ) -> ToolResult:
        result = ToolResult(
            name=call.name,
            arguments=dict(call.arguments),
            ok=ok,
            output=output,
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
        )
        self.execution_log.append(result)
        return result


# ---------------------------------------------------------------------------
# HR tools
# ---------------------------------------------------------------------------
DEFAULT_POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"

_SECTION_SPLIT = re.compile(r"(?m)^(?=## POL-)")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from how in is it of on or that the to what
    when where which who why with""".split()
)

# `<date> ± <n> <unit>` — the only shape the calculator accepts.
_OFFSET = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s*(?P<op>[+-])\s*(?P<n>\d+)\s*"
    r"(?P<unit>day|days|week|weeks)$",
    re.IGNORECASE,
)
# `<date> - <date>` — how many days between two dates.
_ISO_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DIFF = re.compile(r"^(?P<a>\d{4}-\d{2}-\d{2})\s*-\s*(?P<b>\d{4}-\d{2}-\d{2})$")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


def _load_sections(policy_dir: Path) -> list[str]:
    """Read `## POL-` sections from every markdown file in `policy_dir`."""
    if not policy_dir.is_dir():
        raise ToolError(
            f"policy directory not found: {policy_dir.name!r} — "
            f"no handbook is available to search"
        )
    sections: list[str] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in _SECTION_SPLIT.split(text):
            chunk = chunk.strip()
            if chunk.startswith("## POL-"):
                sections.append(chunk[3:].strip())  # drop the "## " marker
    if not sections:
        raise ToolError(
            f"no '## POL-' sections found in {policy_dir.name!r}; "
            f"the handbook is empty or malformed"
        )
    return sections


def _parse_iso(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"not a valid calendar date: {value!r}") from exc


def build_hr_registry(policy_dir: Path | str | None = None) -> ToolRegistry:
    """The HR agent's registry — policy lookup and date arithmetic.

    Deliberately excluded: any finance tool (role boundary), and the IT ticket
    stub (provisioning is a side effect and lives in the effects layer).
    """
    root = Path(policy_dir) if policy_dir is not None else DEFAULT_POLICY_DIR

    def hr_policy_lookup(query):
        """Search the onboarding handbook and return the 2 closest sections."""
        sections = _load_sections(root)
        wanted = _tokens(str(query))
        scored = [
            (len(wanted & _tokens(section)), -index, section)
            for index, section in enumerate(sections)
        ]
        scored.sort(reverse=True)
        top = scored[:2]
        body = "\n\n".join(section for _, _, section in top)
        if not top or top[0][0] == 0:
            # Honest attribution: never dress a zero-overlap result as a match.
            return (
                "No strong match for this query in the handbook. "
                "Closest sections by document order:\n\n" + body
            )
        return body

    def date_calculator(expression):
        """Add or subtract days/weeks, or count the days between two dates."""
        text = str(expression).strip()

        diff = _DIFF.match(text)
        if diff:
            days = (_parse_iso(diff["a"]) - _parse_iso(diff["b"])).days
            return f"{days} days"

        # A bare ISO date is a degenerate valid expression (date + 0 days).
        # Both committed live transcripts showed the model asking exactly
        # this and getting an error back — a tolerant tool beats a failed
        # dispatch, and the calendar is still validated (2026-02-30 raises).
        if _ISO_ONLY.match(text):
            return _parse_iso(text).isoformat()

        offset = _OFFSET.match(text)
        if not offset:
            raise ToolError(
                f"unsupported date expression: {text!r}. Supported forms: "
                f"'YYYY-MM-DD + N days', 'YYYY-MM-DD - N weeks', "
                f"'YYYY-MM-DD - YYYY-MM-DD'. Expressions are parsed, never "
                f"evaluated as code."
            )
        amount = int(offset["n"])
        if offset["unit"].lower().startswith("week"):
            amount *= 7
        if offset["op"] == "-":
            amount = -amount
        return (_parse_iso(offset["date"]) + timedelta(days=amount)).isoformat()

    return ToolRegistry(
        [
            Tool(
                name="hr_policy_lookup",
                description=(
                    "Look up onboarding policy sections by keyword. Returns the "
                    "two closest '## POL-' sections from the handbook."
                ),
                run=hr_policy_lookup,
            ),
            Tool(
                name="date_calculator",
                description=(
                    "Date arithmetic such as '2026-09-01 + 90 days' or the number "
                    "of days between two ISO dates."
                ),
                run=date_calculator,
            ),
        ]
    )

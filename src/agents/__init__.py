"""Agent reasoning loops.

`react` holds the pattern-level machinery (slice 4); the concrete LLM-backed
agents live alongside it and import from here rather than re-implementing the
loop.
"""

from src.agents.react import (
    ReActResult,
    ReActStep,
    build_react_prompt,
    force_first_lookup,
    parse_react_reply,
    run_react,
)

__all__ = [
    "ReActResult",
    "ReActStep",
    "build_react_prompt",
    "force_first_lookup",
    "parse_react_reply",
    "run_react",
]

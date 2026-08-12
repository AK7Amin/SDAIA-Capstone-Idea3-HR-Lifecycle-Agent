"""The orchestration layer: one door to the graph and its vocabulary.

Everything downstream (pipeline, CLI, service, notebook) imports from here
rather than from `build`, so the compiled graph, the injection seam, the loop
bounds and the audit-chaining helper all stay findable in one grep.
"""

from __future__ import annotations

from src.graph.build import (
    MAX_EXTRACT_ATTEMPTS,
    MAX_REVISIONS,
    NODE_NAMES,
    PATTERN_EXTRACTION,
    PATTERN_HITL,
    PATTERN_PLANNING,
    PATTERN_REACT,
    PATTERN_REFLEXION,
    REQUIRED_META,
    NullEffects,
    audit_chain,
    build_graph,
    route_after_gate,
    route_after_intake,
    route_after_profile,
    route_after_review,
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
    "AgentDeps",
    "CaseState",
    "NullEffects",
    "audit_chain",
    "build_graph",
    "route_after_gate",
    "route_after_intake",
    "route_after_profile",
    "route_after_review",
]

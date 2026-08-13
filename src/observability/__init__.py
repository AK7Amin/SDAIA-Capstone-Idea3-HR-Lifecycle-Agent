"""Observability: what the run counts, what it writes down, and who checks it.

Three files, one import:

- `tracing`   — Prometheus counters, per-case trace files, metrics snapshot
- `verifier`  — independent re-verification of those traces (`verify-traces`)
- `dashboard` — one self-contained HTML page rendered from those files

The split is not cosmetic. The writer and the verifier are deliberately
separate programs sharing exactly one thing — :func:`src.schemas.verify_chain`,
the single implementation of the chain rule — so that verifying a trace is not
the same code path as producing it. Evidence that is only ever checked by the
tool that wrote it is not evidence.
"""

from __future__ import annotations

from .dashboard import DASHBOARD_FILENAME
from .dashboard import render as render_dashboard
from .tracing import (
    BLOCK_BUDGET,
    BLOCK_INJECTION,
    BLOCK_PII,
    BLOCK_SIZE,
    BLOCK_TOOL_REFUSAL,
    EVENT_FIELDS,
    METRICS_FILENAME,
    REACT_DIRNAME,
    TRACES_DIRNAME,
    counter_snapshot,
    metrics_registry,
    metrics_text,
    observe_case_latency,
    record_case,
    record_guardrail_block,
    record_llm_failover,
    record_node,
    reset_metrics,
    write_metrics_snapshot,
    write_react_transcript,
    write_trace,
)
from .verifier import EXIT_NO_TRACES, EXIT_OK, EXIT_PROBLEMS, verify_all, verify_trace_file

__all__ = [
    "BLOCK_BUDGET",
    "BLOCK_INJECTION",
    "BLOCK_PII",
    "BLOCK_SIZE",
    "BLOCK_TOOL_REFUSAL",
    "DASHBOARD_FILENAME",
    "EVENT_FIELDS",
    "EXIT_NO_TRACES",
    "EXIT_OK",
    "EXIT_PROBLEMS",
    "METRICS_FILENAME",
    "REACT_DIRNAME",
    "TRACES_DIRNAME",
    "counter_snapshot",
    "metrics_registry",
    "metrics_text",
    "observe_case_latency",
    "record_case",
    "record_guardrail_block",
    "record_llm_failover",
    "record_node",
    "render_dashboard",
    "reset_metrics",
    "verify_all",
    "verify_trace_file",
    "write_metrics_snapshot",
    "write_react_transcript",
    "write_trace",
]

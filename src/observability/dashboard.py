"""A single self-contained HTML page summarising one run.

Three constraints shape this module.

**It renders from files, never from live state.** The only inputs are the
metrics snapshot and the traces folder, both written by the run itself. A
dashboard that read in-process counters would look perfect while the artifacts
next to it were stale — the exact defect the snapshot's thread-id list exists to
expose. The signature is the guarantee: paths in, path out.

**It re-verifies rather than repeats.** The chain column comes from
:func:`src.observability.verifier.verify_trace_file`, so a tampered trace is
reported as BROKEN on the page even though the file itself claims to be intact.

**It is one file with nothing to fetch.** No stylesheet, no font, no script,
no image host: the page opens from a zip, from a graded submission folder, or
from a machine with no network. Everything read from disk is HTML-escaped —
trace summaries derive from candidate-supplied text, which is untrusted by
definition.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.observability.tracing import TRACES_DIRNAME, write_text_lf
from src.observability.verifier import verify_trace_file

__all__ = ["DASHBOARD_FILENAME", "TRACES_DIRNAME", "render"]

#: Where a run is expected to put the page, relative to `reports/`.
DASHBOARD_FILENAME = "dashboard.html"

#: Inline, because an external stylesheet would be an external asset.
STYLE = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 2rem; color: #1b1f23;
       background: #ffffff; }
h1 { font-size: 1.5rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #d0d7de;
     padding-bottom: 0.3rem; }
p.meta { color: #57606a; margin-top: 0; }
table { border-collapse: collapse; margin-top: 0.6rem; width: 100%;
        max-width: 60rem; }
th, td { border: 1px solid #d0d7de; padding: 0.35rem 0.6rem; text-align: left;
         font-size: 0.9rem; vertical-align: top; }
th { background: #f6f8fa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #0a7d33; font-weight: 600; }
.broken { color: #b42318; font-weight: 600; }
.empty { color: #57606a; font-style: italic; }
"""


@dataclass(frozen=True)
class _Raw:
    """Markup this module built itself.

    A typed wrapper rather than "escape unless it looks like markup": trace
    summaries come from candidate-supplied text, and any rule that sniffs a
    cell's *content* to decide whether to escape it is an injection waiting for
    the right resume.
    """

    html: str


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _num(value: Any) -> str:
    """Render a metric number without scientific notation or float noise."""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    return _esc(value)


def _cell(value: Any) -> str:
    if isinstance(value, _Raw):
        return f"<td>{value.html}</td>"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<td class="num">{_num(value)}</td>'
    return f"<td>{_esc(value)}</td>"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]], empty: str) -> str:
    body = [
        "<tr>" + "".join(_cell(cell) for cell in row) + "</tr>" for row in rows
    ]
    if not body:
        return f'<p class="empty">{_esc(empty)}</p>'
    head = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _usage_rows(buckets: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for key in sorted(buckets):
        stats = buckets[key] or {}
        rows.append(
            [
                key,
                stats.get("calls", 0),
                stats.get("tokens", 0),
                stats.get("latency_ms", 0),
                stats.get("ref_cost_usd", 0.0),
            ]
        )
    return rows


def _counter_rows(counters: Mapping[str, Any], name: str) -> list[list[Any]]:
    return [
        [label, value] for label, value in sorted((counters.get(name) or {}).items())
    ]


def _relative_link(target: Path, page: Path) -> str:
    """Link a trace from the page without ever writing a machine-local path."""
    try:
        return os.path.relpath(target, page.parent).replace(os.sep, "/")
    except ValueError:  # different drives on Windows — the name still identifies it
        return target.name


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def render(
    metrics_snapshot_path: Path | str,
    traces_dir: Path | str,
    out_path: Path | str,
) -> Path:
    """Render the run dashboard as one self-contained HTML file.

    Args:
        metrics_snapshot_path: `reports/metrics-snapshot.json` of this run.
        traces_dir: `reports/traces/` of this run.
        out_path: Where to write the page.

    Returns:
        Path of the written page. A missing snapshot or traces folder renders an
        honest empty page rather than raising: a run that produced nothing must
        still be visible as a run that produced nothing.
    """
    snapshot_path = Path(metrics_snapshot_path)
    folder = Path(traces_dir)
    page = Path(out_path)

    snapshot = _load_json(snapshot_path)
    usage = snapshot.get("usage") or {}
    counters = snapshot.get("counters") or {}
    thread_ids = snapshot.get("thread_ids") or []

    trace_rows: list[list[Any]] = []
    event_sections: list[str] = []
    for trace_file in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        ok, problems = verify_trace_file(trace_file)
        payload = _load_json(trace_file)
        events = payload.get("events") or []
        status = _Raw(
            '<span class="ok">OK</span>' if ok else '<span class="broken">BROKEN</span>'
        )
        trace_rows.append(
            [
                _relative_link(trace_file, page),
                payload.get("thread_id", trace_file.stem),
                payload.get("case_id", "-"),
                len(events),
                status,
                "; ".join(problems) if problems else "-",
            ]
        )
        event_sections.append(
            f"<h3>{_esc(payload.get('thread_id', trace_file.stem))}</h3>"
            + _table(
                ["#", "node", "pattern", "latency ms", "cost usd", "summary"],
                [
                    [
                        index,
                        event.get("node", "-"),
                        event.get("reasoning_pattern", "") or "-",
                        event.get("latency_ms", 0),
                        event.get("cost_usd", 0.0),
                        event.get("summary", ""),
                    ]
                    for index, event in enumerate(events)
                ],
                "this trace has no events",
            )
        )

    totals = _table(
        ["metric", "value"],
        [
            ["threads this run", len(thread_ids)],
            ["total tokens", usage.get("total_tokens", 0)],
            ["total latency ms", usage.get("total_latency_ms", 0)],
            ["total reference cost usd", usage.get("total_ref_cost_usd", 0.0)],
            ["cases observed", (counters.get("case_latency_ms") or {}).get("count", 0)],
        ],
        "no metrics snapshot found",
    )

    sections = [
        "<h1>HR onboarding agent - run dashboard</h1>",
        f'<p class="meta">generated {_esc(snapshot.get("generated_at", "-"))} '
        f"| threads: {_esc(', '.join(map(str, thread_ids)) or '-')}</p>",
        "<h2>Run totals</h2>",
        totals,
        "<h2>Per node</h2>",
        _table(
            ["node", "calls", "tokens", "latency ms", "cost usd"],
            _usage_rows(usage.get("per_node") or {}),
            "no per-node usage recorded",
        ),
        "<h2>Per case</h2>",
        _table(
            ["case", "calls", "tokens", "latency ms", "cost usd"],
            _usage_rows(usage.get("per_case") or {}),
            "no per-case usage recorded",
        ),
        "<h2>Per provider</h2>",
        _table(
            ["provider", "calls", "tokens", "latency ms", "cost usd"],
            _usage_rows(usage.get("per_provider") or {}),
            "no provider usage recorded",
        ),
        "<h2>Guardrail blocks</h2>",
        _table(
            ["kind", "blocks"],
            _counter_rows(counters, "guardrail_blocks_total"),
            "no guardrail blocks recorded",
        ),
        "<h2>Case outcomes</h2>",
        _table(
            ["status", "cases"],
            _counter_rows(counters, "cases_processed_total"),
            "no cases counted",
        ),
        "<h2>Node runs</h2>",
        _table(
            ["node", "runs"],
            _counter_rows(counters, "node_runs_total"),
            "no node runs counted",
        ),
        "<h2>LLM failovers</h2>",
        _table(
            ["provider", "failovers"],
            _counter_rows(counters, "llm_failovers_total"),
            "no provider failover recorded",
        ),
        "<h2>Traces (re-verified while rendering)</h2>",
        _table(
            ["file", "thread", "case", "events", "chain", "problems"],
            trace_rows,
            "no trace files found",
        ),
        "<h2>Audit events</h2>",
        *event_sections,
    ]

    document = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>HR onboarding agent - run dashboard</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>\n"
    )
    return write_text_lf(page, document)

"""Independent re-verification of persisted traces — the `verify-traces` engine.

This module answers one question about a file somebody hands you: *was this
trace produced by an unbroken run, or was it edited afterwards?* Four rules
make that answer worth something.

* **It never trusts the file's own verdict.** The stored ``chain_intact`` flag
  is treated as a claim to be checked, not as an answer; a disagreement between
  the claim and the recomputed chain is itself reported, because a file that
  lies about its own integrity is worse than one that admits it is broken.
* **It re-derives every event's digest from the event's content.** Editing a
  summary and leaving the hashes in place is the cheap forgery, and it does not
  disturb the chain links at all — only the per-event check catches it.
* **It owns no chain rule of its own.** The verdict comes from
  :func:`src.schemas.verify_chain`, reached through the module attribute so the
  substitution is observable in a test. Two implementations of one rule mean
  one of them is wrong and nobody knows which; that is why there is no hashing
  code anywhere below, and a test enforces its absence.
* **It reports two run-shape defects the chain cannot see.** A duplicated event
  and two runs concatenated into one thread both produce a perfectly valid
  chain — and both make the trace a false record of what happened.

Output is deliberately plain ASCII: this runs on a cp1256 Windows console, and
an audit tool that dies rendering a candidate's name is an audit tool nobody
runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from src import schemas  # module, not the function — see the module docstring
from src.observability.tracing import EVENT_FIELDS, TRACES_DIRNAME
from src.schemas import AuditEvent

__all__ = [
    "EXIT_NO_TRACES",
    "EXIT_OK",
    "EXIT_PROBLEMS",
    "TRACES_DIRNAME",
    "verify_all",
    "verify_trace_file",
]

#: Exit codes. Distinct on purpose: "everything passed" and "there was nothing
#: to check" are opposite facts, and a CI job that cannot tell them apart will
#: eventually pass an empty evidence folder.
EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_NO_TRACES = 2

#: Node whose second appearance means two runs were merged into one trace.
RUN_OPENING_NODE = "intake"


def _ascii(text: str) -> str:
    """Console-safe rendering; trace content may hold any script."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _events_from(rows: list[Any]) -> tuple[list[AuditEvent], list[str]]:
    """Rebuild audit events from the persisted rows, reporting bad rows."""
    events: list[AuditEvent] = []
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"event index {index} is not an object")
            continue
        try:
            events.append(
                AuditEvent.model_validate(
                    {field: row[field] for field in EVENT_FIELDS if field in row}
                )
            )
        except Exception as exc:  # noqa: BLE001 — any validation failure is a defect
            problems.append(f"event index {index} is not a valid audit event: {exc}")
    return events, problems


def _first_broken_index(events: list[AuditEvent]) -> int:
    """Index of the earliest event the shared chain rule rejects.

    Found by asking that same rule about growing prefixes, so this stays a
    *locator* and never becomes a second implementation of the rule itself.
    """
    for length in range(1, len(events) + 1):
        if not schemas.verify_chain(events[:length]):
            return length - 1
    return -1


def verify_trace_file(path: Path | str) -> tuple[bool, list[str]]:
    """Re-verify one persisted trace file from scratch.

    Args:
        path: Path to a `reports/traces/<thread_id>.json` file.

    Returns:
        ``(ok, problems)`` — ``ok`` is True only when ``problems`` is empty.
        Every problem is a single plain-language line naming the event index it
        refers to.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return False, [f"unreadable trace file: {type(exc).__name__}: {exc}"]

    if not isinstance(payload, dict):
        return False, ["trace file is not a JSON object"]

    rows = payload.get("events")
    if not isinstance(rows, list) or not rows:
        return False, ["trace file carries no events"]

    events, problems = _events_from(rows)
    if problems:
        return False, problems  # a partial trail cannot be chain-checked

    # ---- the chain itself (borrowed rule, not a local one) ----
    chain_ok = bool(schemas.verify_chain(events))
    if not chain_ok:
        index = _first_broken_index(events)
        where = (
            f" (node '{_ascii(events[index].node)}')" if 0 <= index < len(events) else ""
        )
        problems.append(f"chain breaks at event index {index}{where}")

    # ---- per-event forgery check: content vs stored digest ----
    seen_digests: dict[str, int] = {}
    for index, (row, event) in enumerate(zip(rows, events)):
        recomputed = event.digest()
        stored = row.get("digest")
        if not stored:
            problems.append(f"event index {index} has no stored digest")
        elif stored != recomputed:
            problems.append(
                f"event index {index} digest does not match its content "
                f"(node '{_ascii(event.node)}') - the event was edited after signing"
            )
        first = seen_digests.setdefault(recomputed, index)
        if first != index:
            problems.append(
                f"duplicate event at index {index} (identical to index {first})"
            )

    # ---- run-shape checks the chain cannot see ----
    openings = [i for i, event in enumerate(events) if event.node == RUN_OPENING_NODE]
    if len(openings) > 1:
        problems.append(
            f"merged runs: {len(openings)} '{RUN_OPENING_NODE}' events in one trace "
            f"(indexes {openings})"
        )

    # ---- the file's own claims ----
    claimed = payload.get("chain_intact")
    if claimed is None:
        problems.append("trace file does not state chain_intact")
    elif bool(claimed) != chain_ok:
        problems.append(
            f"stored chain_intact={bool(claimed)} disagrees with the recomputed "
            f"chain ({chain_ok})"
        )

    thread_id = payload.get("thread_id")
    if not thread_id:
        problems.append("trace file does not state its thread_id")
    elif str(thread_id) != path.stem:
        problems.append(
            f"thread_id '{_ascii(thread_id)}' does not match the file name stem "
            f"'{_ascii(path.stem)}'"
        )

    return not problems, problems


def verify_all(traces_dir: Path | str, stream: TextIO | None = None) -> int:
    """Verify every trace in a folder and print one line per file.

    Args:
        traces_dir: Folder of `<thread_id>.json` traces.
        stream: Where to print; defaults to stdout at call time.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_PROBLEMS` (at least one bad file) or
        :data:`EXIT_NO_TRACES` (nothing to verify — never silently a pass).
    """
    out = sys.stdout if stream is None else stream
    folder = Path(traces_dir)

    if not folder.is_dir():
        print(_ascii(f"no traces directory: {folder.name}"), file=out)
        return EXIT_NO_TRACES

    files = sorted(folder.glob("*.json"))
    if not files:
        print(_ascii(f"no trace files found in {folder.name}"), file=out)
        return EXIT_NO_TRACES

    bad = 0
    for path in files:
        ok, problems = verify_trace_file(path)
        bad += 0 if ok else 1
        detail = "chain verified" if ok else f"{len(problems)} problem(s)"
        print(_ascii(f"{'OK ' if ok else 'BAD'}  {path.name:<48}  {detail}"), file=out)
        for problem in problems:
            print(_ascii(f"       - {problem}"), file=out)

    print(
        _ascii(
            f"checked {len(files)} trace file(s): {len(files) - bad} ok, "
            f"{bad} with problems"
        ),
        file=out,
    )
    return EXIT_OK if bad == 0 else EXIT_PROBLEMS

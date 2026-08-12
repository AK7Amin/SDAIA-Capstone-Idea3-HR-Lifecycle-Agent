"""The only layer in this system that is allowed to touch a disk.

Every other module — agents, guardrails, the graph — computes and returns
values. Concentrating the side effects here is what makes three of the
project's governance promises checkable rather than aspirational:

* **Nothing binding exists while the case waits at the human gate (M9).**
  The graph never writes; it calls this port, and only from `notifier`,
  `it_provisioner` and `quarantine`. Constructing :class:`FileEffects` creates
  no directory and no file, so an empty ``outbox`` at the gate is a property of
  the code, not of luck.
* **A replayed node writes nothing twice (C2-m11).** LangGraph re-runs a node
  from the top on ``Command(resume=...)``. The ticket ledger therefore dedupes
  on ``(case_id, ticket_id)`` with a read before every write, and inserts only
  — never ``INSERT OR REPLACE``. It is append-only history: a row already
  written is never rewritten, so the ledger can be read as an audit record.
* **No machine-local path reaches an artifact.** Every writer returns a path
  *relative to the effects root*, in posix form. The previous project leaked an
  absolute ``C:\\Users\\<name>\\...`` path into a public repo through exactly
  this seam.

Two smaller decisions worth naming:

``StrictUndefined`` is on. A contract notice that renders an empty name is
worse than one that fails to render, because the blank one gets sent. Identity
fields (``profile.name``, ``contract.role``, ``contract.start_date``) are read
straight out of the model dump so a missing one raises; genuinely optional,
model-authored values (``body_fields``, ``skills``, citation notes) are
normalised here first, because absent optional data is not an error.

``safe_case_id`` guards every path built from a ``case_id``, and the id
originates in an untrusted intake file. ``..\\evil`` is refused rather than
sanitised: silently rewriting an attacker's id would produce an artifact filed
under a case nobody can trace.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

__all__ = [
    "CONTRACT_TEMPLATE",
    "FOOTER",
    "LEDGER_PATH",
    "OUTBOX_DIR",
    "PROBATION_DAYS",
    "QUARANTINE_DIR",
    "TEMPLATES_DIR",
    "WELCOME_TEMPLATE",
    "FileEffects",
    "UnsafeCaseId",
    "safe_case_id",
    "template_environment",
]

#: Repository-level template directory. Resolved from this file so the port
#: works whatever the process's working directory is — the effects *root* is a
#: runtime argument, but the templates ship with the code.
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

CONTRACT_TEMPLATE = "contract_notice.md.j2"
WELCOME_TEMPLATE = "welcome.md.j2"

#: Layout under the effects root. The e2e freezes ``outbox/<id>/contract.md``
#: and ``outbox/<id>/welcome.md``; these names are that contract.
OUTBOX_DIR = "outbox"
QUARANTINE_DIR = "quarantine"
LEDGER_PATH = ("state", "it_tickets.sqlite")

#: POL-001 of the (synthetic) handbook. Used when the drafting agent did not
#: state a probation length — the number then comes from the policy, and the
#: notice says so, rather than being credited to the model.
PROBATION_DAYS = 90

#: Stamped on every generated document. These files look like real HR paperwork
#: and must never be mistaken for it.
FOOTER = "Synthetic training artifact — SDAIA Academy capstone."

#: Characters that turn an id into a path. ``:`` is included for Windows drive
#: and alternate-data-stream syntax, ``..`` for the classic traversal.
_FORBIDDEN_TOKENS = ("/", "\\", ":", "\0", "..")

#: Windows device names: opening ``NUL`` or ``CON`` writes to a device, not a
#: file, so the artifact would silently vanish.
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)

_CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS tickets (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    ticket_id  TEXT NOT NULL,
    system     TEXT NOT NULL,
    action     TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""
# INSERT, never INSERT OR REPLACE: this table is history. Idempotency lives in
# the SELECT above the insert, not in a conflict clause that would overwrite a
# row already committed to the audit record.
_INSERT_TICKET = (
    "INSERT INTO tickets (case_id, ticket_id, system, action, status, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)
_SELECT_KNOWN = "SELECT ticket_id FROM tickets WHERE case_id = ?"
_SELECT_CASE = "SELECT * FROM tickets WHERE case_id = ? ORDER BY seq"


class UnsafeCaseId(ValueError):
    """A ``case_id`` that must never be turned into a filesystem path.

    Subclasses :class:`ValueError` so callers that only know the built-in still
    catch it, while a targeted ``except UnsafeCaseId`` can tell a hostile id
    apart from an ordinary bad argument.
    """


def safe_case_id(case_id: Any) -> str:
    """Validate a case id that is about to become part of a path.

    Args:
        case_id: The id as it arrived — ultimately from an untrusted intake
            file.

    Returns:
        The id with surrounding whitespace stripped, unchanged otherwise.

    Raises:
        UnsafeCaseId: The id is empty, contains a path separator, a ``..``
            segment, a NUL or control character, or names a Windows device.
    """
    text = str(case_id or "").strip()
    if not text:
        raise UnsafeCaseId(
            "empty case id: refusing to write an artifact nobody can trace back"
        )
    for token in _FORBIDDEN_TOKENS:
        if token in text:
            raise UnsafeCaseId(
                f"case id {case_id!r} contains {token!r}: refusing path traversal"
            )
    if text in {".", ".."}:
        raise UnsafeCaseId(f"case id {case_id!r} is a directory reference")
    if any(character < " " or character == "\x7f" for character in text):
        raise UnsafeCaseId(f"case id {case_id!r} contains a control character")
    if text.split(".")[0].upper() in _RESERVED_NAMES:
        raise UnsafeCaseId(f"case id {case_id!r} is a reserved device name")
    return text


def template_environment(templates_dir: Path | str) -> Environment:
    """Build the Jinja2 environment these documents are rendered with.

    Args:
        templates_dir: Directory holding the ``.md.j2`` files.

    Returns:
        An environment with ``autoescape`` off — the output is Markdown, and
        escaping would turn ``&`` into ``&amp;`` inside a contract — and
        ``StrictUndefined`` on, so a missing field raises instead of rendering
        an empty string into a document a human acts on.
    """
    return Environment(
        loader=FileSystemLoader(str(templates_dir), encoding="utf-8"),
        autoescape=False,
        undefined=StrictUndefined,
        # A line holding nothing but a block tag disappears entirely, so the
        # templates read as documents with loops in them rather than as
        # whitespace puzzles solved with `{%-` markers.
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _fields(obj: Any) -> dict[str, Any]:
    """Read a Pydantic model or a plain mapping as a dict of JSON-safe values.

    Both shapes really occur: nodes hand over models, but state that has been
    through the Postgres checkpointer's allow-list serializer can come back as
    a dict (the same tolerance ``src/graph/build.py`` builds into its digest
    helper). Refusing the dict here would make a resumed case unrenderable.
    """
    if obj is None:
        return {}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"expected a model or mapping, got {type(obj).__name__}")


def _cell(value: Any) -> str:
    """Render one model-authored value as a single Markdown table cell.

    The values come from ``body_fields``, which the drafting agent fills
    freely. A list, a newline or a stray ``|`` would break the table silently,
    so flattening happens here rather than in the template — the template stays
    a document, not a program.
    """
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, bool) or value is None:
        text = "yes" if value is True else ("no" if value is False else "")
    else:
        text = str(value)
    return " ".join(text.split()).replace("|", "\\|")


def _label(key: str) -> str:
    """Turn a ``body_fields`` key into a human table label."""
    return key.replace("_", " ").strip().capitalize() or key


def _as_list(value: Any) -> list[Any]:
    """Coerce a possibly-absent, possibly-scalar model value into a list."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _tickets_of(result: Any) -> list[dict[str, Any]]:
    """Extract the ticket dicts from a :class:`~src.schemas.ProvisionResult`."""
    if result is None:
        return []
    tickets: Any = getattr(result, "tickets", None)
    if tickets is None and isinstance(result, Mapping):
        tickets = result.get("tickets")
    if not tickets:
        return []
    if not isinstance(tickets, Iterable):
        raise TypeError("provisioning result carries a non-iterable 'tickets'")
    return [_fields(ticket) for ticket in tickets]


class FileEffects:
    """Filesystem implementation of the effects port the graph calls.

    Args:
        root: Directory everything is written under. Nothing is created until a
            node actually asks for a write — the constructor is pure.
        templates_dir: Override for the shipped template directory. Exists for
            tests and for a future packaged deployment; production passes
            nothing.

    The four writer methods mirror ``src.graph.build.NullEffects`` exactly
    (name and parameter names), which is what lets the graph be tested against
    the null port and run against this one. :meth:`it_tickets` is the extra
    reader the end-to-end test uses to prove provisioning really happened.
    """

    def __init__(self, root: Path | str, templates_dir: Path | str | None = None):
        self.root = Path(root)
        self.templates_dir = (
            Path(templates_dir) if templates_dir is not None else TEMPLATES_DIR
        )
        # Built on first render: constructing this object must be free of any
        # filesystem access, including a template lookup.
        self._environment: Environment | None = None

    # ------------------------------------------------------------------ base
    @property
    def environment(self) -> Environment:
        """The Jinja2 environment, created on first use."""
        if self._environment is None:
            self._environment = template_environment(self.templates_dir)
        return self._environment

    def _render(self, template_name: str, context: Mapping[str, Any]) -> str:
        return self.environment.get_template(template_name).render(**context)

    def _write(self, path: Path, text: str) -> str:
        """Write UTF-8 text with LF endings and return the root-relative path.

        Windows would translate ``\\n`` to ``\\r\\n`` by default, which makes a
        document's bytes differ between the machine that generated the evidence
        and the machine that grades it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path.relative_to(self.root).as_posix()

    def _case_dir(self, case_id: str) -> Path:
        return self.root / OUTBOX_DIR / case_id

    def _ledger(self) -> Path:
        return self.root.joinpath(*LEDGER_PATH)

    # --------------------------------------------------------------- writers
    def write_contract(self, case_id: str, draft: Any, profile: Any) -> str:
        """Render the contract notice for an approved case.

        Called only from ``notifier``, i.e. only after the human gate — the
        draft lives in graph state until then (M9).

        Args:
            case_id: Case identifier; validated by :func:`safe_case_id`.
            draft: The :class:`~src.schemas.ContractDraft` (or its dict form).
            profile: The :class:`~src.schemas.CandidateProfile`; supplies the
                candidate's name, which the draft deliberately does not carry.

        Returns:
            ``outbox/<case_id>/contract.md``, relative to the root.

        Raises:
            UnsafeCaseId: The case id could escape the root.
            ValueError: No draft was supplied. A blank contract is worse than
                no contract, because a blank one still gets sent.
            jinja2.UndefinedError: An identity field is missing.
        """
        safe_id = safe_case_id(case_id)
        if draft is None:
            raise ValueError(
                f"no contract draft for case {safe_id}: refusing to render a "
                "blank employment notice"
            )
        contract = _fields(draft)
        body = dict(contract.get("body_fields") or {})
        # Both are pulled out of the free-form block: the notes get their own
        # disclosure section, and probation gets its own policy-cited line.
        notes = body.pop("citation_notes", None) or []
        probation = body.pop("probation_days", None) or PROBATION_DAYS

        text = self._render(
            CONTRACT_TEMPLATE,
            {
                "case_id": safe_id,
                "profile": _fields(profile),
                "contract": contract,
                "body_fields": [
                    (_label(key), _cell(value)) for key, value in body.items()
                ],
                "citation_notes": [_cell(note) for note in _as_list(notes)],
                "probation_days": _cell(probation),
                "footer": FOOTER,
            },
        )
        return self._write(self._case_dir(safe_id) / "contract.md", text)

    def write_welcome(self, case_id: str, profile: Any) -> str:
        """Render the welcome pack for an approved case.

        Args:
            case_id: Case identifier; validated by :func:`safe_case_id`.
            profile: The :class:`~src.schemas.CandidateProfile`.

        Returns:
            ``outbox/<case_id>/welcome.md``, relative to the root.

        Raises:
            UnsafeCaseId: The case id could escape the root.
            jinja2.UndefinedError: An identity field is missing.
        """
        safe_id = safe_case_id(case_id)
        fields = _fields(profile)
        text = self._render(
            WELCOME_TEMPLATE,
            {
                "case_id": safe_id,
                "profile": fields,
                # Optional by contract: an extraction that found no skills is a
                # valid profile, so this one is defaulted rather than strict.
                "skills": [_cell(skill) for skill in _as_list(fields.get("skills"))],
                "footer": FOOTER,
            },
        )
        return self._write(self._case_dir(safe_id) / "welcome.md", text)

    def provision_tickets(self, case_id: str, result: Any) -> None:
        """Append this case's IT tickets to the append-only ledger.

        Idempotent by ``(case_id, ticket_id)``: a replayed ``it_provisioner``
        node re-submits the identical result and must insert nothing. A
        genuinely new provisioning for the same case still accumulates, because
        the ledger is history — every row that was ever true stays true.

        Args:
            case_id: Case identifier; validated by :func:`safe_case_id`.
            result: A :class:`~src.schemas.ProvisionResult` (or its dict form).

        Raises:
            UnsafeCaseId: The case id could escape the root.
        """
        safe_id = safe_case_id(case_id)
        tickets = _tickets_of(result)
        ledger = self._ledger()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        connection = sqlite3.connect(ledger)
        try:
            connection.execute(_CREATE_LEDGER)
            known = {
                row[0] for row in connection.execute(_SELECT_KNOWN, (safe_id,))
            }
            rows = []
            for ticket in tickets:
                ticket_id = str(ticket.get("ticket_id") or "")
                if ticket_id in known:
                    continue
                # Guards against a duplicate inside a single result too, not
                # just against a replay of a previous call.
                known.add(ticket_id)
                rows.append(
                    (
                        safe_id,
                        ticket_id,
                        str(ticket.get("system") or ""),
                        str(ticket.get("action") or ""),
                        str(ticket.get("status") or ""),
                        created_at,
                    )
                )
            if rows:
                connection.executemany(_INSERT_TICKET, rows)
            connection.commit()
        finally:
            connection.close()

    def quarantine_case(self, case_id: str, reason: str) -> str:
        """Record why a case was refused, next to nothing else.

        Args:
            case_id: Case identifier; validated by :func:`safe_case_id`.
            reason: The quarantine reason, copied verbatim.

        Returns:
            ``quarantine/<case_id>.txt``, relative to the root.

        Raises:
            UnsafeCaseId: The case id could escape the root.
        """
        safe_id = safe_case_id(case_id)
        text = (
            f"Case {safe_id} was quarantined and did not proceed to onboarding.\n"
            "\n"
            f"Reason: {reason}\n"
            "\n"
            "No contract, welcome pack, or IT provisioning was produced for this\n"
            "case.\n"
            "\n"
            f"{FOOTER}\n"
        )
        return self._write(self.root / QUARANTINE_DIR / f"{safe_id}.txt", text)

    # ---------------------------------------------------------------- reader
    def it_tickets(self, case_id: str) -> list[dict[str, Any]]:
        """Return this case's ledger rows, oldest first.

        A pure reader: a case that was never provisioned returns ``[]`` without
        creating the ledger file, so calling it cannot forge the evidence that
        provisioning happened.

        Args:
            case_id: Case identifier; validated by :func:`safe_case_id`.

        Returns:
            One dict per row, with all ledger columns including ``seq`` and
            ``created_at``.

        Raises:
            UnsafeCaseId: The case id could escape the root.
        """
        safe_id = safe_case_id(case_id)
        ledger = self._ledger()
        if not ledger.exists():
            return []
        connection = sqlite3.connect(ledger)
        try:
            connection.row_factory = sqlite3.Row
            try:
                rows = connection.execute(_SELECT_CASE, (safe_id,)).fetchall()
            except sqlite3.OperationalError:
                # Ledger file exists but nothing was ever provisioned into it.
                return []
            return [dict(row) for row in rows]
        finally:
            connection.close()

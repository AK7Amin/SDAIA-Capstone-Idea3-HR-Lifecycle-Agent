"""Slice 9 — effects: the only layer allowed to touch a disk.

Written RED before the implementation. Zero network, zero Docker, zero keys.

What this file pins down, and why each one is a real failure mode:

* **Governance ordering (M9).** Constructing the effects port must write
  nothing. The e2e asserts an empty `outbox` while the case waits at the human
  gate, so a module that creates its directories at construction time turns
  that guarantee into a lie without failing a single unit test.
* **Node replay (C2-m11).** LangGraph re-runs a node from the top on resume.
  Provisioning the same tickets twice must leave the ledger unchanged, and the
  ledger must still be append-only history — never a row rewritten in place.
* **Relative paths.** The previous project leaked `C:\\Users\\<name>\\...` into
  a public repo through persisted artifacts. Every path this layer returns is
  relative to the effects root, in posix form.
* **Path traversal.** `case_id` arrives from an untrusted intake file and is
  used to build a path. `..\\evil` must be refused, not written.
* **Strict templates.** A contract notice silently rendering an empty name is
  worse than one that fails: `StrictUndefined` makes a missing identity field
  loud.
"""
import inspect
import sqlite3

import pytest
from jinja2 import StrictUndefined
from jinja2.exceptions import UndefinedError

from src.effects import (
    CONTRACT_TEMPLATE,
    PROBATION_DAYS,
    TEMPLATES_DIR,
    WELCOME_TEMPLATE,
    FileEffects,
    UnsafeCaseId,
    safe_case_id,
    template_environment,
)
from src.schemas import CandidateProfile, ContractDraft, ITTicket, ProvisionResult

CASE = "CAND-001"

FOOTER = "Synthetic training artifact — SDAIA Academy capstone."


# --------------------------------------------------------------------------
# Fixtures: one complete, realistic (synthetic) case.
# --------------------------------------------------------------------------
@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id=CASE,
        name="Sara Alqahtani",
        role="Data Engineer",
        start_date="2026-09-01",
        skills=["Spark", "Airflow", "SQL"],
        experience_summary="5 years building ETL pipelines.",
    )


@pytest.fixture
def draft() -> ContractDraft:
    return ContractDraft(
        candidate_id=CASE,
        role="Data Engineer",
        start_date="2026-09-01",
        salary_band="B3",
        body_fields={"manager": "Head of Data", "work_mode": "onsite"},
    )


@pytest.fixture
def provision() -> ProvisionResult:
    return ProvisionResult(
        tickets=[
            ITTicket(
                ticket_id="IT-1001",
                system="identity",
                action="create_account",
                status="done",
            ),
            ITTicket(
                ticket_id="IT-1002",
                system="hardware",
                action="allocate_laptop",
                status="done",
            ),
        ]
    )


@pytest.fixture
def effects(tmp_path) -> FileEffects:
    return FileEffects(tmp_path)


# --------------------------------------------------------------------------
# Governance: nothing happens until the graph asks for it.
# --------------------------------------------------------------------------
def test_constructing_the_port_touches_no_disk(tmp_path):
    """Importing and constructing must not create a single file (M9)."""
    port = FileEffects(tmp_path)
    assert list(tmp_path.iterdir()) == [], "effects port wrote at construction time"
    assert port.root == tmp_path


def test_reading_tickets_before_any_provisioning_creates_no_ledger(effects, tmp_path):
    """The reader is a reader: an unprovisioned case is [], not a new file."""
    assert effects.it_tickets(CASE) == []
    assert list(tmp_path.iterdir()) == [], "reader created the ledger file"


# --------------------------------------------------------------------------
# Contract notice
# --------------------------------------------------------------------------
def test_contract_renders_identity_fields(effects, tmp_path, draft, profile):
    effects.write_contract(CASE, draft, profile)
    text = (tmp_path / "outbox" / CASE / "contract.md").read_text(encoding="utf-8")
    assert "Sara Alqahtani" in text
    assert "Data Engineer" in text
    assert "2026-09-01" in text
    assert "B3" in text


def test_contract_returns_a_relative_posix_path(effects, tmp_path, draft, profile):
    """No machine-local absolute path may reach a persisted artifact."""
    path = effects.write_contract(CASE, draft, profile)
    assert path == f"outbox/{CASE}/contract.md"
    assert "\\" not in path
    assert (tmp_path / path).exists()


def test_contract_renders_free_form_body_fields(effects, tmp_path, draft, profile):
    effects.write_contract(CASE, draft, profile)
    text = (tmp_path / "outbox" / CASE / "contract.md").read_text(encoding="utf-8")
    assert "Head of Data" in text
    assert "onsite" in text


def test_body_field_rows_land_one_per_table_line(effects, tmp_path, draft, profile):
    """Whitespace control in the template is load-bearing: a collapsed loop
    glues every row onto one line and the Markdown table stops being a table."""
    effects.write_contract(CASE, draft, profile)
    lines = (
        (tmp_path / "outbox" / CASE / "contract.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "| Manager | Head of Data |" in lines
    assert "| Work mode | onsite |" in lines


def test_model_authored_values_cannot_break_the_table(effects, tmp_path, profile):
    """`body_fields` is free-form model output; a stray pipe or newline in it
    must be flattened, not allowed to reshape a contract document."""
    messy = ContractDraft(
        candidate_id=CASE,
        role="Data Engineer",
        start_date="2026-09-01",
        body_fields={"note": "a | b\nsecond line", "perks": ["laptop", "monitor"]},
    )
    effects.write_contract(CASE, messy, profile)
    lines = (
        (tmp_path / "outbox" / CASE / "contract.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "| Note | a \\| b second line |" in lines
    assert "| Perks | laptop, monitor |" in lines


def test_contract_cites_the_probation_policy(effects, tmp_path, draft, profile):
    """POL-001 is the policy that sets probation; the notice must say so."""
    effects.write_contract(CASE, draft, profile)
    text = (tmp_path / "outbox" / CASE / "contract.md").read_text(encoding="utf-8")
    assert "POL-001" in text
    assert str(PROBATION_DAYS) in text


def test_contract_prints_citation_notes_when_the_validator_left_any(
    effects, tmp_path, profile
):
    """A stripped fake citation is disclosed, never silently dropped."""
    flagged = ContractDraft(
        candidate_id=CASE,
        role="Data Engineer",
        start_date="2026-09-01",
        body_fields={"citation_notes": ["removed unverifiable citation POL-777"]},
    )
    effects.write_contract(CASE, flagged, profile)
    text = (tmp_path / "outbox" / CASE / "contract.md").read_text(encoding="utf-8")
    assert "POL-777" in text


def test_contract_omits_the_notes_section_when_there_is_nothing_to_disclose(
    effects, tmp_path, draft, profile
):
    effects.write_contract(CASE, draft, profile)
    text = (tmp_path / "outbox" / CASE / "contract.md").read_text(encoding="utf-8")
    assert "Review notes" not in text


def test_contract_without_a_draft_is_refused(effects, tmp_path, profile):
    """Rendering a blank contract is worse than not rendering one."""
    with pytest.raises(ValueError):
        effects.write_contract(CASE, None, profile)
    assert not (tmp_path / "outbox").exists()


def test_missing_identity_field_raises_instead_of_rendering_a_blank(
    effects, tmp_path, draft
):
    """StrictUndefined bites: a profile with no name cannot produce a notice."""
    with pytest.raises(UndefinedError):
        effects.write_contract(CASE, draft, {"candidate_id": CASE, "role": "x"})
    assert not (tmp_path / "outbox" / CASE / "contract.md").exists()


def test_template_environment_is_strict_and_unescaped():
    """Markdown, not HTML: autoescaping would turn `&` into `&amp;`."""
    env = template_environment(TEMPLATES_DIR)
    assert env.undefined is StrictUndefined
    assert env.autoescape is False


# --------------------------------------------------------------------------
# Welcome pack
# --------------------------------------------------------------------------
def test_welcome_renders_role_start_date_and_week_one_pointers(
    effects, tmp_path, profile
):
    effects.write_welcome(CASE, profile)
    text = (tmp_path / "outbox" / CASE / "welcome.md").read_text(encoding="utf-8")
    assert "Sara Alqahtani" in text
    assert "Data Engineer" in text
    assert "2026-09-01" in text
    # Week-1 pointers are policy-backed, not invented: security training and
    # the onboarding buddy.
    assert "POL-002" in text
    assert "POL-005" in text


def test_welcome_returns_a_relative_posix_path(effects, tmp_path, profile):
    path = effects.write_welcome(CASE, profile)
    assert path == f"outbox/{CASE}/welcome.md"
    assert (tmp_path / path).exists()


def test_welcome_lists_extracted_skills_one_per_line(effects, tmp_path, profile):
    effects.write_welcome(CASE, profile)
    lines = (
        (tmp_path / "outbox" / CASE / "welcome.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "- Spark" in lines
    assert "- Airflow" in lines


def test_welcome_drops_the_skills_section_when_extraction_found_none(
    effects, tmp_path
):
    """An empty skill list is a valid profile, not a rendering failure."""
    bare = CandidateProfile(
        candidate_id=CASE, name="Omar Nasser", role="Analyst", start_date="2026-10-01"
    )
    effects.write_welcome(CASE, bare)
    text = (tmp_path / "outbox" / CASE / "welcome.md").read_text(encoding="utf-8")
    assert "already know you bring" not in text
    assert "Omar Nasser" in text


def test_welcome_missing_identity_field_raises(effects, tmp_path):
    with pytest.raises(UndefinedError):
        effects.write_welcome(CASE, {"candidate_id": CASE, "name": "Sara"})
    assert not (tmp_path / "outbox" / CASE / "welcome.md").exists()


def test_both_documents_declare_themselves_synthetic(
    effects, tmp_path, draft, profile
):
    """A grader (or a stranger) must never mistake these for real HR papers."""
    effects.write_contract(CASE, draft, profile)
    effects.write_welcome(CASE, profile)
    for name in ("contract.md", "welcome.md"):
        text = (tmp_path / "outbox" / CASE / name).read_text(encoding="utf-8")
        assert FOOTER in text, f"{name} is missing the synthetic-artifact footer"


# --------------------------------------------------------------------------
# Encoding discipline
# --------------------------------------------------------------------------
def test_documents_are_utf8_with_lf_newlines(effects, tmp_path, draft, profile):
    """Windows would write CRLF by default; artifacts stay LF everywhere."""
    effects.write_contract(CASE, draft, profile)
    effects.write_welcome(CASE, profile)
    effects.quarantine_case("CAND-999", "unreadable intake")
    for path in (
        tmp_path / "outbox" / CASE / "contract.md",
        tmp_path / "outbox" / CASE / "welcome.md",
        tmp_path / "quarantine" / "CAND-999.txt",
    ):
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} was written with CRLF"
        raw.decode("utf-8")  # loud on any non-utf-8 byte


def test_non_ascii_content_survives_the_round_trip(effects, tmp_path, draft):
    """An Arabic name is data, not an encoding accident."""
    arabic = CandidateProfile(
        candidate_id=CASE,
        name="سارة القحطاني",
        role="Data Engineer",
        start_date="2026-09-01",
    )
    effects.write_contract(CASE, draft, arabic)
    text = (tmp_path / "outbox" / CASE / "contract.md").read_text(encoding="utf-8")
    assert "سارة القحطاني" in text


# --------------------------------------------------------------------------
# IT ticket ledger
# --------------------------------------------------------------------------
def test_provisioning_writes_one_row_per_ticket(effects, tmp_path, provision):
    effects.provision_tickets(CASE, provision)
    ledger = tmp_path / "state" / "it_tickets.sqlite"
    assert ledger.exists()
    conn = sqlite3.connect(ledger)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 2
    finally:
        conn.close()


def test_reader_returns_rows_as_dicts(effects, provision):
    effects.provision_tickets(CASE, provision)
    rows = effects.it_tickets(CASE)
    assert len(rows) == 2
    assert all(isinstance(row, dict) for row in rows)
    first = rows[0]
    assert first["case_id"] == CASE
    assert first["ticket_id"] == "IT-1001"
    assert first["system"] == "identity"
    assert first["action"] == "create_account"
    assert first["status"] == "done"
    assert first["created_at"]


def test_replaying_the_node_inserts_nothing(effects, provision):
    """The double-invoke test: LangGraph re-runs a node on resume (C2-m11)."""
    effects.provision_tickets(CASE, provision)
    before = effects.it_tickets(CASE)
    effects.provision_tickets(CASE, provision)
    after = effects.it_tickets(CASE)
    assert len(after) == len(before) == 2
    assert [row["seq"] for row in after] == [row["seq"] for row in before]


def test_the_ledger_is_insert_only_history(effects, provision):
    """A genuinely different provisioning accumulates; nothing is rewritten."""
    effects.provision_tickets(CASE, provision)
    first_pass = effects.it_tickets(CASE)
    effects.provision_tickets(
        CASE,
        ProvisionResult(
            tickets=[
                ITTicket(
                    ticket_id="IT-2001",
                    system="identity",
                    action="reset_password",
                    status="done",
                )
            ]
        ),
    )
    rows = effects.it_tickets(CASE)
    assert len(rows) == 3
    # The original rows are untouched — no INSERT OR REPLACE anywhere.
    assert rows[: len(first_pass)] == first_pass
    assert rows[-1]["ticket_id"] == "IT-2001"


def test_duplicate_ticket_ids_inside_one_result_collapse(effects, provision):
    """Dedupe is by (case_id, ticket_id), including within a single call."""
    ticket = ITTicket(
        ticket_id="IT-3001", system="identity", action="create_account", status="done"
    )
    effects.provision_tickets(CASE, ProvisionResult(tickets=[ticket, ticket]))
    assert len(effects.it_tickets(CASE)) == 1


def test_the_ledger_is_scoped_per_case(effects, provision):
    effects.provision_tickets(CASE, provision)
    effects.provision_tickets(
        "CAND-002",
        ProvisionResult(
            tickets=[
                ITTicket(
                    ticket_id="IT-1001",  # same id, different case: not a replay
                    system="identity",
                    action="create_account",
                    status="done",
                )
            ]
        ),
    )
    assert len(effects.it_tickets(CASE)) == 2
    assert len(effects.it_tickets("CAND-002")) == 1


def test_an_empty_provisioning_result_is_survivable(effects):
    effects.provision_tickets(CASE, ProvisionResult(tickets=[]))
    assert effects.it_tickets(CASE) == []


# --------------------------------------------------------------------------
# Quarantine
# --------------------------------------------------------------------------
def test_quarantine_writes_the_reason(effects, tmp_path):
    path = effects.quarantine_case(CASE, "invalid intake: missing required field(s)")
    assert path == f"quarantine/{CASE}.txt"
    text = (tmp_path / path).read_text(encoding="utf-8")
    assert "invalid intake: missing required field(s)" in text
    assert CASE in text


# --------------------------------------------------------------------------
# Path traversal — `case_id` comes from an untrusted intake file.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        "..\\evil",
        "../evil",
        "a/b",
        "a\\b",
        "",
        "   ",
        "..",
        ".",
        "C:\\Windows",
        "a\x00b",
        "line\nbreak",
        "NUL",
        "con.txt",
    ],
)
def test_safe_case_id_refuses_dangerous_ids(bad):
    with pytest.raises(UnsafeCaseId):
        safe_case_id(bad)


@pytest.mark.parametrize("good", ["CAND-001", "cand_002", "Case.2026", " CAND-3 "])
def test_safe_case_id_accepts_ordinary_ids(good):
    assert safe_case_id(good) == good.strip()


def test_unsafe_case_id_is_a_value_error():
    """Callers that only know `ValueError` still catch it."""
    assert issubclass(UnsafeCaseId, ValueError)


def test_every_writer_refuses_a_traversing_case_id(tmp_path, draft, profile, provision):
    port = FileEffects(tmp_path / "root")
    for call in (
        lambda: port.write_contract("..\\evil", draft, profile),
        lambda: port.write_welcome("..\\evil", profile),
        lambda: port.provision_tickets("..\\evil", provision),
        lambda: port.quarantine_case("..\\evil", "because"),
        lambda: port.it_tickets("..\\evil"),
    ):
        with pytest.raises(UnsafeCaseId):
            call()
    assert not (tmp_path / "root").exists(), "a refused case id still created files"
    assert list(tmp_path.iterdir()) == [], "a refused case id escaped the root"


# --------------------------------------------------------------------------
# Frozen contracts — the e2e and the graph port, asserted verbatim.
# --------------------------------------------------------------------------
def test_matches_the_frozen_e2e_contract(tmp_path, draft, profile, provision):
    """`FileEffects(root)`, `it_tickets(...)`, `outbox/<id>/{contract,welcome}.md`.

    tests/test_integration_e2e.py is append-only; this meta-test is where slice
    9 proves it honours that file without editing it.
    """
    port = FileEffects(tmp_path)  # single positional root
    assert port.write_contract(CASE, draft, profile) == f"outbox/{CASE}/contract.md"
    assert port.write_welcome(CASE, profile) == f"outbox/{CASE}/welcome.md"
    assert (tmp_path / "outbox" / CASE / "contract.md").exists()
    assert (tmp_path / "outbox" / CASE / "welcome.md").exists()
    port.provision_tickets(CASE, provision)
    tickets = port.it_tickets(CASE)  # e2e calls this positionally
    assert isinstance(tickets, list) and tickets


def test_satisfies_the_graph_effects_port():
    """Duck-typed against src/graph/build.py's NullEffects — same names, same args."""
    from src.graph.build import NullEffects

    for name in ("write_contract", "write_welcome", "provision_tickets",
                 "quarantine_case"):
        port_sig = inspect.signature(getattr(NullEffects, name))
        file_sig = inspect.signature(getattr(FileEffects, name))
        assert list(port_sig.parameters) == list(file_sig.parameters), (
            f"{name} drifted from the port the graph calls"
        )


def test_template_files_exist_where_the_module_says_they_do():
    assert (TEMPLATES_DIR / CONTRACT_TEMPLATE).is_file()
    assert (TEMPLATES_DIR / WELCOME_TEMPLATE).is_file()

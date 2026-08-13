"""Slice 8 — persistence: the checkpointer this project's HITL story rests on.

Written RED before `src/checkpointing.py` exists. Split in two on purpose:

* **Default run (offline).** Serializer behaviour, the sqlite fallback, and the
  fail-fast contract. Zero network, zero Docker, zero keys — a dead Postgres is
  simulated by patching the connect seam, never by dialling a closed port.
* **`@pytest.mark.docker` run.** The claims that only a real database can
  settle: a case paused on one connection and resumed on a *fresh* one, a case
  resumed by a **separate operating-system process**, and a pool-backed saver
  serving the same cycle from several threads. Excluded by `addopts`; the
  evidence run is `pytest -m docker -rs`, where a SKIP is a failed proof.

Three properties are locked here because each one is a promise the project
makes out loud:

1. **The serializer is an allow-list of our own contracts.** Every model that
   enters graph state comes back as itself; a type we never declared comes back
   as inert data. This is the cause-level fix for the "Deserializing
   unregistered type" warning — not a filter on the message.
2. **A dead Postgres fails loudly, never quietly.** No automatic downgrade to
   sqlite (critique M10): a case that silently changed its durability story
   mid-flight is worse than a case that stops. The sqlite saver exists, but only
   a caller that names it gets it.
3. **The audit chain survives the database.** The digest is computed over the
   model's canonical form, so a chain written by one process and verified by
   another must still verify — that is the whole point of hash-chaining it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest
from langgraph.types import Command
from pydantic import BaseModel

import src.checkpointing as checkpointing
from src.checkpointing import (
    ALLOWLIST_PARAM,
    CHECKPOINTED_TYPES,
    DEFAULT_DSN,
    EXIT_POSTGRES_UNAVAILABLE,
    PostgresUnavailable,
    dsn_from_env,
    make_pool_saver,
    make_postgres_saver_cm,
    make_sqlite_saver,
    open_postgres_pool,
    redacted_dsn,
    strict_serializer,
)
from src.graph import build_graph
from src.schemas import (
    AuditEvent,
    CandidateProfile,
    CaseStatus,
    ContractDraft,
    GateAction,
    GateDecision,
    ITTicket,
    ProvisionResult,
    ReviewAction,
    ReviewVerdict,
    TrainingPlan,
    TrainingWeek,
    verify_chain,
)

# The stub harness of slice 7, reused verbatim: the graph under test here is the
# real one, and the agents/effects around it are the same offline doubles.
from tests.test_graph_paths import Agents, SpyEffects, start_state

HELPER_SCRIPT = Path(__file__).resolve().parent / "helpers" / "resume_case.py"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def thread_config() -> dict:
    """A fresh thread id per test — checkpoint rows are never reused."""
    return {"configurable": {"thread_id": f"slice8-{uuid.uuid4()}"}}


def stub_app(saver):
    """Compile the real graph over stub agents and the given checkpointer."""
    agents = Agents()
    effects = SpyEffects()
    return build_graph(agents.as_deps(effects), saver), effects


def nodes_of(state) -> list[str]:
    return [event.node for event in state["audit_trail"]]


#: One populated instance per checkpointed contract. Populated, not default —
#: a round trip that only carries empty strings proves nothing about fields.
SAMPLES: dict[type, object] = {
    CandidateProfile: CandidateProfile(
        candidate_id="CAND-001",
        name="Sara Alqahtani",
        role="Data Engineer",
        start_date="2026-09-01",
        skills=["Spark", "Airflow"],
        experience_summary="5 years of ETL pipelines.",
    ),
    TrainingWeek: TrainingWeek(week=2, focus="Airflow", activities=["Ship a DAG"]),
    TrainingPlan: TrainingPlan(
        weeks=[TrainingWeek(week=1, focus="Platform tour", activities=["Runbooks"])],
        rationale="ramp on the platform first",
    ),
    ReviewVerdict: ReviewVerdict(
        action=ReviewAction.REVISE,
        critique="Week 1 has no Spark work.",
        concerns=["no Spark in week 1"],
    ),
    ContractDraft: ContractDraft(
        candidate_id="CAND-001",
        role="Data Engineer",
        start_date="2026-09-01",
        salary_band="B3",
        body_fields={"probation_months": 3},
    ),
    GateDecision: GateDecision(
        decision=GateAction.APPROVE, actor="hr", decided_at="2026-09-01T09:00:00"
    ),
    ITTicket: ITTicket(
        ticket_id="IT-1", system="email", action="create_account", status="done"
    ),
    ProvisionResult: ProvisionResult(
        tickets=[
            ITTicket(
                ticket_id="IT-1", system="email", action="create_account", status="done"
            )
        ]
    ),
    AuditEvent: AuditEvent(
        node="it_provisioner",
        summary="tool step 1",
        reasoning_pattern="react",
        cost_usd=0.0012,
        latency_ms=430,
        prev_hash="a" * 64,
    ),
    CaseStatus: CaseStatus.AWAITING_APPROVAL,
    ReviewAction: ReviewAction.APPROVE,
    GateAction: GateAction.REJECT,
}


class ForeignModel(BaseModel):
    """A Pydantic model this project never declared — the attacker's shape."""

    payload: str = "attacker"
    amount: int = 9


# --------------------------------------------------------------------------
# 1. the allow-list serializer
# --------------------------------------------------------------------------
def test_the_installed_langgraph_exposes_an_allow_list_parameter():
    """Canary: if this ever goes None, the strict serializer silently weakened."""
    assert ALLOWLIST_PARAM, (
        "no msgpack allow-list parameter found on JsonPlusSerializer — "
        "the strict serializer would degrade to the permissive default"
    )


@pytest.mark.parametrize("declared", list(CHECKPOINTED_TYPES), ids=lambda t: t.__name__)
def test_every_checkpointed_type_has_a_sample(declared):
    """Guard on the guard: a new contract must not slip past the round trip."""
    assert declared in SAMPLES


@pytest.mark.parametrize("declared", list(CHECKPOINTED_TYPES), ids=lambda t: t.__name__)
def test_the_strict_serializer_round_trips_every_state_model(declared):
    serde = strict_serializer()
    original = SAMPLES[declared]

    restored = serde.loads_typed(serde.dumps_typed(original))

    assert type(restored) is type(original)
    assert restored == original


def test_the_allow_list_covers_every_contract_in_the_schema_module():
    """Drift guard: a model added to `src.schemas` must be declared here too."""
    import enum
    import inspect

    import src.schemas as schemas

    exported = [getattr(schemas, name) for name in schemas.__all__]
    contracts = {
        obj
        for obj in exported
        if inspect.isclass(obj) and issubclass(obj, (BaseModel, enum.Enum))
    }

    assert contracts <= set(CHECKPOINTED_TYPES), (
        "undeclared contract(s): "
        f"{sorted(c.__name__ for c in contracts - set(CHECKPOINTED_TYPES))}"
    )


def test_a_foreign_pydantic_type_does_not_come_back_as_that_type():
    """The point of the allow-list: unknown constructors are not invoked."""
    serde = strict_serializer()
    smuggled = ForeignModel()

    restored = serde.loads_typed(serde.dumps_typed(smuggled))

    assert not isinstance(restored, ForeignModel)
    # Inert data is the acceptable outcome; a reconstructed object is not.
    assert restored in (None, {"payload": "attacker", "amount": 9})


def test_a_nested_foreign_type_is_neutralised_without_losing_our_own():
    serde = strict_serializer()
    mixed = {"ours": SAMPLES[CandidateProfile], "theirs": ForeignModel()}

    restored = serde.loads_typed(serde.dumps_typed(mixed))

    assert isinstance(restored["ours"], CandidateProfile)
    assert not isinstance(restored["theirs"], ForeignModel)


def test_the_audit_digest_survives_the_checkpoint_serializer():
    """M11: a hash that changed on the way to storage would prove nothing."""
    serde = strict_serializer()
    first = AuditEvent(node="intake", summary="validated")
    second = AuditEvent(node="profile_analyst", prev_hash=first.digest())

    restored = serde.loads_typed(serde.dumps_typed([first, second]))

    assert [event.digest() for event in restored] == [first.digest(), second.digest()]
    assert verify_chain(restored)


# --------------------------------------------------------------------------
# 2. fail fast — never a silent downgrade (M10)
# --------------------------------------------------------------------------
class DeadConnection:
    """Stand-in for `psycopg.Connection` with the database switched off."""

    #: Every `connect` call, so a test can inspect the parameters that were used.
    calls: list[tuple[tuple, dict]] = []

    @classmethod
    def connect(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        raise checkpointing.OperationalError("connection refused")


@pytest.fixture
def dead_postgres(monkeypatch):
    """Patch the connect seam so a dead database needs no dead database."""
    DeadConnection.calls = []
    monkeypatch.setattr(checkpointing, "Connection", DeadConnection)
    return DeadConnection


def test_an_unreachable_postgres_raises_postgres_unavailable(dead_postgres):
    with pytest.raises(PostgresUnavailable):
        with make_postgres_saver_cm(DEFAULT_DSN):
            pass  # pragma: no cover - the context body must never run


def test_the_failure_message_tells_the_operator_what_to_do(dead_postgres):
    with pytest.raises(PostgresUnavailable) as excinfo:
        with make_postgres_saver_cm(DEFAULT_DSN):
            pass  # pragma: no cover

    message = str(excinfo.value)
    assert "docker start idea3-pg" in message
    assert "localhost:5433" in message
    assert "connection refused" in message, "the root cause was swallowed"


def test_the_failure_message_never_prints_the_password(dead_postgres):
    with pytest.raises(PostgresUnavailable) as excinfo:
        with make_postgres_saver_cm("postgresql://postgres:s3cr3t@localhost:5433/db"):
            pass  # pragma: no cover

    assert "s3cr3t" not in str(excinfo.value)


def test_a_dead_postgres_never_falls_back_to_sqlite(dead_postgres, monkeypatch):
    """M10: durability must not change itself mid-flight."""

    def forbidden(*args, **kwargs):
        raise AssertionError("silent sqlite fallback")

    monkeypatch.setattr(checkpointing, "make_sqlite_saver", forbidden)

    with pytest.raises(PostgresUnavailable):
        with make_postgres_saver_cm(DEFAULT_DSN):
            pass  # pragma: no cover


def test_the_pool_factory_fails_fast_too(monkeypatch):
    class DeadPool:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def wait(self, timeout=None):
            raise checkpointing.PoolTimeout("pool initialization incomplete")

        def close(self):
            self.closed = True

    monkeypatch.setattr(checkpointing, "ConnectionPool", DeadPool)

    with pytest.raises(PostgresUnavailable) as excinfo:
        open_postgres_pool(DEFAULT_DSN)

    assert "docker start idea3-pg" in str(excinfo.value)


def test_the_connection_is_bounded_by_a_connect_timeout(dead_postgres):
    """Fail-fast needs a deadline; libpq has none by default.

    Measured during this slice: connecting to a black-holed local port with no
    ``connect_timeout`` blocked for over 60 seconds, which would turn "exit 2"
    into a hang and would freeze the docker-marked evidence run instead of
    skipping it. Asserted on the parameters rather than on the wall clock — a
    timing test would have to dial a dead address, which the default suite is
    not allowed to do.
    """
    with pytest.raises(PostgresUnavailable):
        with make_postgres_saver_cm(DEFAULT_DSN):
            pass  # pragma: no cover

    _args, kwargs = dead_postgres.calls[0]
    assert 1 <= kwargs["connect_timeout"] <= 30
    assert kwargs["autocommit"] is True
    assert kwargs["prepare_threshold"] == 0


def test_the_pool_bounds_its_connections_too(monkeypatch):
    """`pool.wait()` alone leaves a worker thread stuck inside libpq."""
    seen: dict = {}

    class RecordingPool:
        def __init__(self, conninfo, **kwargs):
            seen.update(kwargs)

        def wait(self, timeout=None):
            raise checkpointing.PoolTimeout("pool initialization incomplete")

        def close(self):
            return None

    monkeypatch.setattr(checkpointing, "ConnectionPool", RecordingPool)

    with pytest.raises(PostgresUnavailable):
        open_postgres_pool(DEFAULT_DSN)

    assert 1 <= seen["kwargs"]["connect_timeout"] <= 30
    assert seen["kwargs"]["autocommit"] is True


def test_the_exit_code_for_an_unreachable_database_is_two():
    """The CLI contract slice 11 imports rather than re-invents."""
    assert EXIT_POSTGRES_UNAVAILABLE == 2


# --------------------------------------------------------------------------
# 3. the DSN comes from the environment (compose points it at postgres:5432)
# --------------------------------------------------------------------------
def test_dsn_from_env_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://postgres:pw@postgres:5432/hr_agent")

    assert dsn_from_env() == "postgresql://postgres:pw@postgres:5432/hr_agent"


def test_dsn_from_env_falls_back_to_the_documented_dev_database(monkeypatch):
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    assert dsn_from_env() == DEFAULT_DSN
    assert "5433" in DEFAULT_DSN


def test_a_blank_environment_value_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "   ")

    assert dsn_from_env() == DEFAULT_DSN


def test_redacted_dsn_hides_the_password_and_keeps_the_address():
    assert redacted_dsn(DEFAULT_DSN) == "postgresql://postgres:***@localhost:5433/hr_agent"
    assert redacted_dsn("postgresql://postgres@localhost:5433/hr_agent") == (
        "postgresql://postgres@localhost:5433/hr_agent"
    )


# --------------------------------------------------------------------------
# 4. the sqlite fallback — explicit, never automatic
# --------------------------------------------------------------------------
def test_the_sqlite_fallback_pauses_and_resumes_in_process(tmp_path):
    saver = make_sqlite_saver(tmp_path / "cases.sqlite")
    app, effects = stub_app(saver)
    config = thread_config()

    paused = app.invoke(start_state(), config)
    assert "__interrupt__" in paused
    assert effects.order == [], "an effect fired while the case was still paused"

    final = app.invoke(Command(resume={"decision": "approve"}), config)

    assert final["status"] == CaseStatus.COMPLETED.value
    assert isinstance(final["profile"], CandidateProfile)
    assert verify_chain(final["audit_trail"])


def test_the_sqlite_fallback_resumes_over_a_fresh_connection(tmp_path):
    """Same file, different connection: the state lives in the file, not the RAM."""
    path = tmp_path / "cases.sqlite"
    config = thread_config()

    app, _effects = stub_app(make_sqlite_saver(path))
    app.invoke(start_state(), config)

    reopened, effects = stub_app(make_sqlite_saver(path))
    final = reopened.invoke(Command(resume={"decision": "approve"}), config)

    assert final["status"] == CaseStatus.COMPLETED.value
    assert "hr_gate" in nodes_of(final)
    assert effects.order == ["provision_tickets", "write_contract", "write_welcome"]


def test_the_sqlite_connection_is_usable_from_another_thread(tmp_path):
    """`check_same_thread=False`, proven by using it from a second thread."""
    saver = make_sqlite_saver(tmp_path / "cases.sqlite")
    app, _effects = stub_app(saver)
    config = thread_config()
    app.invoke(start_state(), config)

    seen: list = []

    def read_from_worker():
        try:
            seen.append(saver.get_tuple(config))
        except BaseException as exc:  # noqa: BLE001 - the failure IS the result
            seen.append(exc)

    worker = threading.Thread(target=read_from_worker)
    worker.start()
    worker.join(timeout=30)

    assert seen, "the worker thread never finished"
    assert not isinstance(seen[0], BaseException), seen[0]
    assert seen[0] is not None


def test_the_sqlite_saver_uses_the_strict_serializer(tmp_path):
    saver = make_sqlite_saver(tmp_path / "cases.sqlite")

    restored = saver.serde.loads_typed(saver.serde.dumps_typed(ForeignModel()))

    assert not isinstance(restored, ForeignModel)


# --------------------------------------------------------------------------
# 5. Docker Postgres — excluded by default (`pytest -m docker -rs` to run)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dsn() -> str:
    """The dev DSN, or a loud skip naming the address that did not answer."""
    value = dsn_from_env()
    try:
        with make_postgres_saver_cm(value):
            pass
    except PostgresUnavailable as exc:
        pytest.skip(f"postgres not reachable at {redacted_dsn(value)} — {exc}")
    return value


@pytest.mark.docker
def test_a_case_paused_on_one_connection_resumes_on_a_fresh_one(dsn):
    """The spike's finding, now a standing test over the real graph."""
    config = thread_config()

    with make_postgres_saver_cm(dsn) as saver:
        app, effects = stub_app(saver)
        paused = app.invoke(start_state(), config)
        assert "__interrupt__" in paused
        assert effects.order == []

    with make_postgres_saver_cm(dsn) as fresh_saver:
        reopened, fresh_effects = stub_app(fresh_saver)
        final = reopened.invoke(Command(resume={"decision": "approve"}), config)

    assert final["status"] == CaseStatus.COMPLETED.value
    assert final["gate"].decision is GateAction.APPROVE
    # Pre-interrupt state came back as our own types, not as loose dicts.
    assert isinstance(final["profile"], CandidateProfile)
    assert isinstance(final["contract"], ContractDraft)
    assert final["profile"].name == "Sara Alqahtani"
    assert verify_chain(final["audit_trail"]), "the chain broke crossing the database"
    assert fresh_effects.order == [
        "provision_tickets",
        "write_contract",
        "write_welcome",
    ]


@pytest.mark.docker
def test_a_separate_process_resumes_the_paused_case(dsn):
    """D5's headline claim: the pause outlives the process that created it."""
    config = thread_config()
    with make_postgres_saver_cm(dsn) as saver:
        app, _effects = stub_app(saver)
        assert "__interrupt__" in app.invoke(start_state(), config)

    child_env = dict(
        os.environ,
        POSTGRES_DSN=dsn,
        PYTHONIOENCODING="utf-8",
        CASE_THREAD_ID=config["configurable"]["thread_id"],
    )
    child = subprocess.run(
        [sys.executable, "-X", "utf8", str(HELPER_SCRIPT)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert child.returncode == 0, child.stderr
    reported = dict(
        line.split("=", 1) for line in child.stdout.split() if "=" in line
    )
    assert reported["STATUS"] == CaseStatus.COMPLETED.value
    assert int(reported["PID"]) != os.getpid(), "the resume ran in this process"
    assert reported["CHAIN_OK"] == "True"

    # And this process can read what the other one wrote.
    with make_postgres_saver_cm(dsn) as saver:
        app, _effects = stub_app(saver)
        snapshot = app.get_state(config)
    assert snapshot.values["status"] == CaseStatus.COMPLETED.value
    assert snapshot.next == ()


@pytest.mark.docker
def test_a_pool_backed_saver_serves_the_same_pause_and_resume(dsn):
    """The service shape (critique B2): one long-lived pool, many requests."""
    pool = open_postgres_pool(dsn)
    try:
        saver = make_pool_saver(pool)
        config = thread_config()

        # Two separately compiled apps over ONE saver — two "requests".
        first_request, _effects = stub_app(saver)
        assert "__interrupt__" in first_request.invoke(start_state(), config)

        second_request, effects = stub_app(saver)
        final = second_request.invoke(Command(resume={"decision": "approve"}), config)

        assert final["status"] == CaseStatus.COMPLETED.value
        assert verify_chain(final["audit_trail"])
        assert effects.order == ["provision_tickets", "write_contract", "write_welcome"]
    finally:
        pool.close()


@pytest.mark.docker
def test_the_pool_serves_concurrent_cases_from_several_threads(dsn):
    """A single psycopg connection is not thread-safe; a pool is."""
    pool = open_postgres_pool(dsn)
    try:
        saver = make_pool_saver(pool)
        configs = [thread_config() for _ in range(3)]
        results: dict[str, object] = {}

        def run_case(config):
            app, _effects = stub_app(saver)
            try:
                paused = app.invoke(start_state(), config)
                app.invoke(Command(resume={"decision": "approve"}), config)
                final = app.get_state(config).values
                results[config["configurable"]["thread_id"]] = (
                    "__interrupt__" in paused,
                    final.get("status"),
                )
            except BaseException as exc:  # noqa: BLE001 - reported, not hidden
                results[config["configurable"]["thread_id"]] = exc

        workers = [threading.Thread(target=run_case, args=(c,)) for c in configs]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=300)

        assert len(results) == len(configs)
        for thread_id, outcome in results.items():
            assert not isinstance(outcome, BaseException), f"{thread_id}: {outcome}"
            assert outcome == (True, CaseStatus.COMPLETED.value), thread_id
    finally:
        pool.close()

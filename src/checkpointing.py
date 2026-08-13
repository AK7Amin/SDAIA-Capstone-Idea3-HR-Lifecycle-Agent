"""Where a paused case lives while nobody is looking.

This project's central claim is that a case can stop at the human gate on a
Tuesday and be resumed by a different process on a Thursday. That claim is only
as good as the checkpointer underneath it, so three decisions are made here and
nowhere else.

**Postgres is the checkpointer; sqlite is a fallback you must name.** When the
database is unreachable this module raises :class:`PostgresUnavailable` and the
caller exits with :data:`EXIT_POSTGRES_UNAVAILABLE`. It never quietly opens a
sqlite file instead (critique M10): a system that downgrades its own durability
mid-flight tells the operator everything is fine while paused cases silently
stop being recoverable. :func:`make_sqlite_saver` exists, is fully supported,
and is reached only by a caller that asked for it by name.

**Two construction shapes, because two callers have different lifetimes.**
The CLI runs one case and exits, so it takes :func:`make_postgres_saver_cm` and
holds it in a ``with`` block that closes the connection on the way out. A
service handles overlapping requests on many threads, and a single
``psycopg.Connection`` is *not* thread-safe — so it opens one
:func:`open_postgres_pool` for the lifetime of the application and wraps it with
:func:`make_pool_saver` (critique B2). Same saver class, same rows, different
ownership of the socket.

**The serializer is an allow-list of this project's own contracts.**
:func:`strict_serializer` names every model and enum that can enter graph state;
anything else comes back as inert data instead of being reconstructed by
importing whatever module the checkpoint row names. Beyond the security
argument, this is the cause-level fix for langgraph's "Deserializing
unregistered type" warning — the warning is emitted precisely because the
default allow-list is "everything", so declaring the types removes it rather
than muting it.

The DSN is read from ``POSTGRES_DSN`` so the same image runs against the dev
container on ``localhost:5433`` and against the compose service on
``postgres:5432`` with no code change.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from inspect import signature
from os import environ
from pathlib import Path
from typing import Iterator, Mapping

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# `Connection` and `ConnectionPool` are referenced through this module (never
# re-imported inside functions) so a test can replace them with a dead double;
# `OperationalError` is re-exported for the same reason — the tests must be able
# to raise exactly what the driver raises without importing psycopg themselves.
from psycopg import Connection, Error as PsycopgError, OperationalError  # noqa: F401
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

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
)

__all__ = [
    "ALLOWLIST_PARAM",
    "CHECKPOINTED_TYPES",
    "CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_DSN",
    "DSN_ENV_VAR",
    "EXIT_POSTGRES_UNAVAILABLE",
    "PostgresUnavailable",
    "dsn_from_env",
    "make_pool_saver",
    "make_postgres_saver_cm",
    "make_sqlite_saver",
    "open_postgres_pool",
    "redacted_dsn",
    "strict_serializer",
]

# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
#: Environment variable holding the connection string. Compose overrides it
#: with the in-network address; nothing in the code hard-codes a host.
DSN_ENV_VAR = "POSTGRES_DSN"

#: The documented local dev database (`docker run --name idea3-pg`). Same value
#: as `.env.example`; a throwaway container password, not a secret.
DEFAULT_DSN = "postgresql://postgres:capstone@localhost:5433/hr_agent"

#: Seconds to wait for a connection before declaring Postgres unreachable.
#:
#: Load-bearing, not decoration. libpq has **no** connect timeout by default, and
#: a host that drops packets instead of refusing them (a stopped container on
#: Windows, a firewall, a wrong address) leaves `connect()` blocked for minutes —
#: measured here at over 60 seconds against a black-holed local port. That turns
#: "fail fast, exit 2" into "hang forever", and it would hang the docker-marked
#: evidence run instead of skipping it. An integer because libpq parses this
#: parameter as whole seconds (and clamps anything below 2 up to 2).
CONNECT_TIMEOUT_SECONDS = 10

#: Process exit code when the database cannot be reached. Defined here so the
#: CLI and the child-process helpers agree on one number.
EXIT_POSTGRES_UNAVAILABLE = 2

#: Command that fixes the common case, quoted verbatim in the error message.
_RESTART_HINT = "docker start idea3-pg"


class PostgresUnavailable(RuntimeError):
    """Raised when the checkpoint database cannot be reached.

    A `RuntimeError` rather than a custom hierarchy because there is exactly one
    reasonable response to it — tell the operator which address failed and stop.
    """


# --------------------------------------------------------------------------
# the allow-list serializer
# --------------------------------------------------------------------------
#: Every type this project can put into graph state. The graph writes Pydantic
#: models (`profile`, `plan`, `contract`, `gate`, `provision`, `audit_trail`)
#: and the enums nested inside them; anything not on this list has no business
#: being reconstructed out of a checkpoint row.
CHECKPOINTED_TYPES: tuple[type, ...] = (
    CandidateProfile,
    TrainingWeek,
    TrainingPlan,
    ReviewVerdict,
    ContractDraft,
    GateDecision,
    ITTicket,
    ProvisionResult,
    AuditEvent,
    CaseStatus,
    ReviewAction,
    GateAction,
)

#: Constructor keyword the installed langgraph uses for the msgpack allow-list.
#: Resolved by inspection rather than assumed: the parameter was introduced
#: recently, and a hard-coded name that silently disappears would leave the
#: serializer permissive while every test still passed. Empty string means the
#: installed version has no allow-list at all — the canary test in
#: `tests/test_checkpoint_resume.py` fails loudly if that ever happens.
_ALLOWLIST_PARAM_CANDIDATES = (
    "allowed_msgpack_modules",
    "allowed_modules",
    "allowed_types",
)


def _find_allowlist_param() -> str:
    parameters = signature(JsonPlusSerializer.__init__).parameters
    for name in _ALLOWLIST_PARAM_CANDIDATES:
        if name in parameters:
            return name
    return ""


ALLOWLIST_PARAM: str = _find_allowlist_param()


def strict_serializer() -> JsonPlusSerializer:
    """Return the checkpoint serializer restricted to :data:`CHECKPOINTED_TYPES`.

    Every saver built by this module uses it, so a case is stored the same way
    whichever backend holds it — and a checkpoint row that names a class this
    project never declared is decoded as plain data instead of being turned back
    into an object by importing that class.

    Returns:
        A :class:`JsonPlusSerializer` whose msgpack allow-list is exactly this
        project's contracts (plus langgraph's own built-in safe types, which the
        library always permits so `Interrupt` and `Command` still work).
    """
    if not ALLOWLIST_PARAM:
        # No allow-list in this version: still return a working serializer, and
        # let the canary test be the thing that reports the weakening.
        return JsonPlusSerializer()
    return JsonPlusSerializer(**{ALLOWLIST_PARAM: list(CHECKPOINTED_TYPES)})


# --------------------------------------------------------------------------
# DSN handling
# --------------------------------------------------------------------------
_PASSWORD_IN_URI = re.compile(r"(?P<head>://[^:/?#@]+):(?P<password>[^@]*)@")


def dsn_from_env(env: Mapping[str, str] | None = None) -> str:
    """Return the checkpoint DSN from the environment, or the dev default.

    Args:
        env: Mapping to read instead of `os.environ` (tests, child processes).

    Returns:
        The value of ``POSTGRES_DSN`` when it holds something other than
        whitespace, else :data:`DEFAULT_DSN`. A blank variable counts as unset:
        an empty string in a compose file is a forgotten value, not a request to
        connect to nowhere.
    """
    source = environ if env is None else env
    return (source.get(DSN_ENV_VAR) or "").strip() or DEFAULT_DSN


def redacted_dsn(dsn: str) -> str:
    """Return `dsn` with any password replaced by ``***``.

    Used everywhere a connection string reaches a human: error messages, skip
    reasons, evidence logs. The address has to stay readable — "which database
    did it try?" is the first question — but the credential must not travel with
    it into a committed log file.
    """
    return _PASSWORD_IN_URI.sub(r"\g<head>:***@", dsn or "")


def _scrub(text: str, dsn: str) -> str:
    """Remove the DSN password from arbitrary text (driver messages included)."""
    cleaned = redacted_dsn(text)
    match = _PASSWORD_IN_URI.search(dsn or "")
    password = match.group("password") if match else ""
    if password:
        cleaned = cleaned.replace(password, "***")
    return cleaned


def _unreachable(dsn: str, cause: BaseException) -> PostgresUnavailable:
    """Build the one error this module raises, with the fix spelled out."""
    message = (
        f"cannot reach the checkpoint database at {redacted_dsn(dsn)}: "
        f"{_scrub(str(cause).strip(), dsn)} | start it with: {_RESTART_HINT} "
        f"(first run: docker run -d --name idea3-pg -e POSTGRES_PASSWORD=... "
        f"-e POSTGRES_DB=hr_agent -p 5433:5432 postgres:16-alpine), or point "
        f"{DSN_ENV_VAR} at a reachable server. Refusing to continue: this "
        f"system never downgrades to sqlite on its own — a paused case must "
        f"not change durability behind the operator's back."
    )
    return PostgresUnavailable(message)


# --------------------------------------------------------------------------
# Postgres — one set of connection parameters, two ownership shapes
# --------------------------------------------------------------------------
def _connection_kwargs(connect_timeout: int = CONNECT_TIMEOUT_SECONDS) -> dict:
    """Connection parameters shared by the CLI shape and the service shape.

    None of these is stylistic: ``autocommit`` because the saver manages its own
    transactions, ``prepare_threshold=0`` because pooler-fronted servers choke
    on prepared statements, ``dict_row`` because the saver indexes result rows
    by name, and ``connect_timeout`` because without it an unreachable host
    hangs instead of failing (see :data:`CONNECT_TIMEOUT_SECONDS`).

    Keeping them in one place is what makes "the pool behaves like the CLI"
    true rather than aspirational.
    """
    return {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
        "connect_timeout": int(connect_timeout),
    }


# --------------------------------------------------------------------------
# CLI shape — one case, one connection, closed on exit
# --------------------------------------------------------------------------
@contextmanager
def make_postgres_saver_cm(
    dsn: str | None = None,
    *,
    setup: bool = True,
    connect_timeout: int = CONNECT_TIMEOUT_SECONDS,
) -> Iterator[PostgresSaver]:
    """Yield a `PostgresSaver` on a private connection, closed on exit.

    Mirrors `PostgresSaver.from_conn_string` — same context-manager shape, same
    connection parameters plus a bounded ``connect_timeout`` — but constructs
    the saver itself, because `from_conn_string` accepts no ``serde`` and this
    project insists on the allow-list serializer.

    Args:
        dsn: Connection string; defaults to :func:`dsn_from_env`.
        setup: Run `saver.setup()` before yielding. Idempotent (proven by the
            spike), so it stays on by default and a first run needs no ritual.
        connect_timeout: Seconds before an unreachable server is given up on.

    Yields:
        A ready `PostgresSaver`. The connection is closed when the block ends,
        which is why a *service* must not use this shape — see
        :func:`open_postgres_pool`.

    Raises:
        PostgresUnavailable: if the database cannot be reached. Callers exit
            with :data:`EXIT_POSTGRES_UNAVAILABLE`.
    """
    target = dsn or dsn_from_env()
    try:
        connection = Connection.connect(target, **_connection_kwargs(connect_timeout))
    except (PsycopgError, OSError) as exc:
        raise _unreachable(target, exc) from exc

    try:
        saver = PostgresSaver(connection, serde=strict_serializer())
        if setup:
            saver.setup()
        yield saver
    finally:
        connection.close()


# --------------------------------------------------------------------------
# Service shape — one pool, many threads, application lifetime
# --------------------------------------------------------------------------
def open_postgres_pool(
    dsn: str | None = None,
    *,
    min_size: int = 1,
    max_size: int = 8,
    connect_timeout: int = CONNECT_TIMEOUT_SECONDS,
) -> ConnectionPool:
    """Open a connection pool that outlives every individual request.

    A `psycopg.Connection` is not thread-safe, so a FastAPI process must not
    share one across handlers; a pool hands each thread its own connection and
    takes it back afterwards. The pool therefore belongs to the *application
    lifespan* — opened at start-up, closed at shutdown — never to a request.

    `pool.wait()` is what makes the failure immediate: without it the pool
    reports success and retries in a background thread, so a wrong DSN would
    surface much later as a timeout inside an unrelated request. It is not
    sufficient on its own, though — the per-connection ``connect_timeout`` has
    to bound the worker thread too, or `close()` waits on a thread still stuck
    inside libpq.

    Args:
        dsn: Connection string; defaults to :func:`dsn_from_env`.
        min_size: Connections opened eagerly (and required by ``wait``).
        max_size: Ceiling on concurrent connections.
        connect_timeout: Seconds to wait for the pool to fill before failing.

    Returns:
        An open, warmed-up `ConnectionPool`. The caller owns it and must
        `close()` it.

    Raises:
        PostgresUnavailable: if the pool cannot be filled in time.
    """
    target = dsn or dsn_from_env()
    pool: ConnectionPool | None = None
    try:
        pool = ConnectionPool(
            target,
            kwargs=_connection_kwargs(connect_timeout),
            min_size=min_size,
            max_size=max_size,
            open=True,
            timeout=connect_timeout,
        )
        pool.wait(timeout=connect_timeout)
    except (PoolTimeout, PsycopgError, OSError) as exc:
        if pool is not None:
            try:
                pool.close()
            except Exception:  # noqa: BLE001 - shutdown noise must not mask the cause
                pass
        raise _unreachable(target, exc) from exc
    return pool


def make_pool_saver(pool: ConnectionPool, *, setup: bool = True) -> PostgresSaver:
    """Wrap a long-lived pool in a `PostgresSaver` (the service shape).

    The saver is safe to share across threads *because* the pool is: every
    operation borrows a connection and returns it. One saver per application is
    the intended shape — compiling several graphs over it is fine and is exactly
    what per-request isolation looks like.

    Args:
        pool: An open pool from :func:`open_postgres_pool`. Its lifetime must
            cover every request that touches this saver; closing it mid-flight
            breaks in-flight cases.
        setup: Create the checkpoint tables once at start-up. Idempotent.

    Returns:
        A `PostgresSaver` using the allow-list serializer.
    """
    saver = PostgresSaver(pool, serde=strict_serializer())
    if setup:
        saver.setup()
    return saver


# --------------------------------------------------------------------------
# sqlite — the explicit fallback (never selected automatically)
# --------------------------------------------------------------------------
def make_sqlite_saver(path: str | Path, *, setup: bool = True) -> SqliteSaver:
    """Build the file-backed fallback checkpointer, on request only.

    Nothing in this module calls it: a caller reaches it by passing
    ``--checkpointer sqlite`` (critique M10). It is a real fallback — same
    graph, same serializer, resumable across processes on one machine — just not
    the deployment story, and never a silent substitute for a database that
    failed to answer.

    ``check_same_thread=False`` because the checkpointer is touched from
    whichever thread runs the graph, and sqlite3's default would reject that
    with a `ProgrammingError` the first time a worker thread saved a step.

    Args:
        path: Database file. Parent directories are created.
        setup: Create the checkpoint tables now instead of on first use.

    Returns:
        A `SqliteSaver` using the same allow-list serializer as Postgres, so a
        case serialized under one backend means the same thing under the other.
    """
    target = Path(path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), check_same_thread=False)
    saver = SqliteSaver(connection, serde=strict_serializer())
    if setup:
        saver.setup()
    return saver

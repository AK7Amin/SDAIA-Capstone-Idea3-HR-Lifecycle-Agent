"""The HTTP service: the same pipeline, exposed to callers who are not the CLI.

Run it with ``uvicorn src.app:app`` (compose does exactly that). Five endpoints:
``POST /process``, ``POST /resume``, ``GET /healthz``, ``GET /metrics``, and the
schema FastAPI generates for them.

Four decisions here are load-bearing, and none of them is FastAPI boilerplate.

**The checkpointer is a pool, opened once in the lifespan.** A
``psycopg.Connection`` is not thread-safe and this process answers overlapping
requests on a threadpool, so the CLI's context-manager shape would corrupt state
under the second concurrent case (critique B2). One
:func:`~src.checkpointing.open_postgres_pool`, one
:func:`~src.checkpointing.make_pool_saver`, one compiled graph, for the lifetime
of the application. And if the database is unreachable at start-up the exception
is allowed to escape: uvicorn prints it and the process exits. A service that
starts anyway would accept cases it cannot persist, which is precisely the
"paused case quietly stops being recoverable" failure this project refuses.

**Both write endpoints are closed until their own token is configured.** An
unset variable answers 503 "closed", never 200 — a deployment that forgot the
secret must be visibly shut, not silently open. `/resume` has a *second* token,
because submitting a candidate and approving the contract that binds the company
are different powers, and one leaked key should not buy both.

**A rate limit is not the budget guard.** `MAX_LLM_CALLS_PER_CASE` bounds what
one case may spend; the fixed window here bounds how many cases a caller may
start, which is the only control that exists once the caller is a socket. It is
in-process on purpose: this is one container, and a shared-Redis limiter would
add an outage mode to buy accuracy nobody is grading.

**Per-request state is installed and cleared per request.** The case id and the
budget guard live in `contextvars` (see :mod:`src.llm`), the pipeline installs
them per invocation, and this layer clears them in a ``finally`` so no request
inherits the attribution of the one before it on the same worker thread.

Nothing sensitive travels back out: a rejected credential is never echoed, the
guarded resume text stays on the server, and an unexpected failure answers with
a fixed message while the log keeps a scrubbed one line.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from src.checkpointing import make_pool_saver, open_postgres_pool, redacted_dsn
from src.effects import FileEffects
from src.guardrails import InputTooLarge
from src.llm import redact_secrets, reset_request_state
from src.observability import metrics_text
from src.pipeline import (
    PROJECT_ROOT,
    REPORTS_DIRNAME,
    STUB_ENV_FLAG,
    build_production_wiring,
    process_case,
    resume_case,
)

__all__ = [
    "API_TOKEN_ENV",
    "API_TOKEN_HEADER",
    "APPROVAL_TOKEN_ENV",
    "APPROVAL_TOKEN_HEADER",
    "INTAKE_SPOOL_DIRNAME",
    "RATE_LIMIT_MAX",
    "RATE_LIMIT_MAX_ENV",
    "RATE_LIMIT_WINDOW_ENV",
    "RATE_LIMIT_WINDOW_SECONDS",
    "ROOT_ENV_VAR",
    "ProcessRequest",
    "ResumeRequest",
    "app",
    "create_app",
    "lifespan",
    "reset_rate_limits",
]

_LOG = logging.getLogger("hr_agent.service")

# --------------------------------------------------------------------------
# configuration — every knob reads an environment variable; literals are defaults
# --------------------------------------------------------------------------
#: Token for `POST /process`, and the header that carries it.
API_TOKEN_ENV = "API_TOKEN"
API_TOKEN_HEADER = "X-Api-Token"

#: Token for `POST /resume` — the privileged one. Separate variable, separate
#: header: approving a contract is not the same permission as submitting a CV.
APPROVAL_TOKEN_ENV = "APPROVAL_API_TOKEN"
APPROVAL_TOKEN_HEADER = "X-Approval-Token"

#: Where the service anchors its data: the intake spool, `outbox/`, `reports/`,
#: `state/`. Compose mounts a volume and points this at it; unset means the repo
#: root, which is what makes a bare `uvicorn src.app:app` behave like the CLI.
ROOT_ENV_VAR = "HR_AGENT_ROOT"

#: Sub-folder of the root where an accepted request is written before it runs.
#: The same layout the CLI reads, so a case submitted over HTTP can be re-run,
#: inspected or replayed from the command line afterwards.
INTAKE_SPOOL_DIRNAME = "intake"

#: Fixed-window rate limit for the write endpoints: requests per window, per
#: client, per endpoint. These are the *defaults*; the environment overrides
#: them, and both are read at call time so a deployment can retune without a
#: rebuild.
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_ENV = "RATE_LIMIT_MAX"
RATE_LIMIT_WINDOW_ENV = "RATE_LIMIT_WINDOW_SECONDS"

#: Distinct clients tracked before the limiter prunes idle buckets. A dictionary
#: keyed by peer address is a memory-growth surface if it is never swept.
_MAX_TRACKED_BUCKETS = 1024


# --------------------------------------------------------------------------
# rate limiting — fixed window, in this process, behind one lock
# --------------------------------------------------------------------------
_RATE_BUCKETS: dict[str, deque[float]] = {}
_RATE_LOCK = threading.Lock()


def reset_rate_limits() -> None:
    """Forget every window. For tests, and for an operator restarting the clock."""
    with _RATE_LOCK:
        _RATE_BUCKETS.clear()


def _bucket_of(request: Request) -> str:
    """Name the window a request falls into: this endpoint, this peer.

    The token is deliberately *not* part of the key — it would put a credential
    in a process-global dictionary to buy nothing, since one deployment token is
    shared by every caller anyway.
    """
    client = request.client.host if request.client else "-"
    return f"{request.url.path}|{client}"


def _prune_locked(now: float, window: float) -> None:
    """Drop buckets with nothing left in the window (called under the lock)."""
    if len(_RATE_BUCKETS) <= _MAX_TRACKED_BUCKETS:
        return
    for key in [
        key
        for key, hits in _RATE_BUCKETS.items()
        if not hits or now - hits[-1] >= window
    ]:
        _RATE_BUCKETS.pop(key, None)


def _positive(raw: str | None, default, cast):
    """Read a positive number from the environment, or keep the default.

    A malformed value falls back rather than raising: a typo in a compose file
    must not take an endpoint down, and the fallback direction is the safe one —
    the limit stays on at its documented setting instead of vanishing.
    """
    try:
        value = cast(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _window_settings() -> tuple[int, float]:
    """The window in force right now: environment first, module default next."""
    return (
        _positive(os.getenv(RATE_LIMIT_MAX_ENV), int(RATE_LIMIT_MAX), int),
        _positive(
            os.getenv(RATE_LIMIT_WINDOW_ENV), float(RATE_LIMIT_WINDOW_SECONDS), float
        ),
    )


def _enforce_rate_limit(bucket: str) -> None:
    """Count one request against its window, or refuse it with 429.

    Raises:
        HTTPException: 429 with a ``Retry-After`` header saying when the oldest
            hit in the window expires — a limiter that refuses without saying
            when to come back just produces a retry storm.
    """
    limit, window = _window_settings()
    now = time.monotonic()
    with _RATE_LOCK:
        hits = _RATE_BUCKETS.setdefault(bucket, deque())
        while hits and now - hits[0] >= window:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = max(1, int(window - (now - hits[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded: too many requests in this window",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        _prune_locked(now, window)


# --------------------------------------------------------------------------
# authentication — closed by default
# --------------------------------------------------------------------------
def _require_token(header_value: str | None, env_var: str, what: str) -> None:
    """Check one request credential against one environment variable.

    Args:
        header_value: Whatever the caller sent in the header, or ``None``.
        env_var: Name of the variable holding the expected value.
        what: Human name of the capability, for the "closed" message.

    Raises:
        HTTPException: 503 when the variable is unset — the endpoint is *closed*,
            which is a different fact from "you are not allowed" and must not be
            reported as 401, or an operator will hunt for a bad key that does not
            exist. 401 when the credential is absent or wrong. Neither message
            ever contains the expected value or the supplied one.
    """
    expected = (os.getenv(env_var) or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"{what} is closed: set {env_var} to open this endpoint",
        )
    supplied = (header_value or "").strip()
    # Bytes, not str: `compare_digest` rejects non-ASCII strings outright, and a
    # caller sending one would get a 500 instead of the 401 they earned.
    if not supplied or not secrets.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid or missing credentials")


def require_process_token(
    token: str | None = Header(default=None, alias=API_TOKEN_HEADER),
) -> None:
    """Dependency guarding `POST /process`."""
    _require_token(token, API_TOKEN_ENV, "case processing")


def require_approval_token(
    token: str | None = Header(default=None, alias=APPROVAL_TOKEN_HEADER),
) -> None:
    """Dependency guarding `POST /resume` — the privileged verb."""
    _require_token(token, APPROVAL_TOKEN_ENV, "gate approval")


def enforce_rate_limit(request: Request) -> None:
    """Dependency applying the fixed window. Declared *after* the token check,
    so an unauthenticated flood cannot spend the window of a caller who has one.
    """
    _enforce_rate_limit(_bucket_of(request))


# --------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------
class ProcessRequest(BaseModel):
    """One hired-candidate payload, exactly as the CLI reads it from a file."""

    case: dict[str, Any] = Field(
        ..., description="Hired-candidate JSON: candidate_id, name, role, "
        "start_date, resume_text. Untrusted — the guards run before the graph."
    )


class ResumeRequest(BaseModel):
    """A human's answer to the approval gate of one paused case."""

    thread_id: str = Field(..., description="Thread id reported by /process.")
    decision: str | dict[str, Any] = Field(
        ..., description='"approve", "reject", or a GateDecision-shaped object.'
    )


# --------------------------------------------------------------------------
# the intake spool
# --------------------------------------------------------------------------
def _spool_case(spool_dir: Path, case: Mapping[str, Any]) -> Path:
    """Persist an accepted payload as an intake file, and return its path.

    The name comes from this request, never from the payload: a candidate id
    arrives over the network, and a file name is the one place where an
    untrusted string turns into a path. Two requests therefore always get two
    files — including two submissions of the same candidate, which must stay two
    runs with two traces rather than one overwritten record.
    """
    spool_dir.mkdir(parents=True, exist_ok=True)
    path = spool_dir / f"case-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(dict(case), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


# --------------------------------------------------------------------------
# application lifetime
# --------------------------------------------------------------------------
def _root_from_env() -> Path:
    return Path((os.getenv(ROOT_ENV_VAR) or "").strip() or PROJECT_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the pool, wire the graph once, and close the pool on the way out.

    Raises:
        PostgresUnavailable: The checkpoint database could not be reached. It is
            deliberately not caught: start-up fails, uvicorn logs the reason and
            the fix, and the process exits instead of serving requests it cannot
            checkpoint.
    """
    root = _root_from_env()
    pool = open_postgres_pool()
    try:
        wiring = build_production_wiring(make_pool_saver(pool), FileEffects(root))
    except BaseException:
        pool.close()
        raise

    app.state.root = root
    app.state.spool_dir = root / INTAKE_SPOOL_DIRNAME
    app.state.reports_dir = root / REPORTS_DIRNAME
    app.state.pool = pool
    app.state.wiring = wiring
    if wiring.stubbed:
        # WARNING, not INFO, and it is not a style choice: uvicorn configures
        # its own loggers and leaves the root one alone, so an INFO line from
        # this module goes nowhere while a warning still reaches stderr through
        # logging's last-resort handler. "This deployment calls no model" is
        # exactly the fact an operator must not be able to miss.
        _LOG.warning(
            "service ready with STUB agents (%s is set): no model will be "
            "called and every result is deterministic",
            STUB_ENV_FLAG,
        )
    else:
        _LOG.info("service ready: live agents, checkpointer=postgres pool")
    try:
        yield
    finally:
        app.state.wiring = None
        pool.close()
        _LOG.info("service stopped: checkpoint pool closed")


# --------------------------------------------------------------------------
# error surface
# --------------------------------------------------------------------------
def _scrubbed(exc: BaseException) -> str:
    """An exception's text with provider keys and DSN passwords removed."""
    return redacted_dsn(redact_secrets(str(exc)))


async def _unexpected_failure(request: Request, exc: Exception) -> JSONResponse:
    """Answer any unhandled error with a fixed message.

    The client gets nothing to work with, and the log gets one scrubbed line —
    no traceback, because the last line of a traceback is the exception message
    again and this handler exists precisely because that message is untrusted.
    """
    _LOG.error(
        "unhandled failure on %s: %s: %s",
        request.url.path,
        type(exc).__name__,
        _scrubbed(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# --------------------------------------------------------------------------
# the application
# --------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build the service. No I/O happens here — that is the lifespan's job."""
    app = FastAPI(
        title="HR Onboarding Agent",
        version="1.0",
        summary=(
            "Multi-agent onboarding with a human approval gate that survives "
            "restarts."
        ),
        lifespan=lifespan,
    )
    app.add_exception_handler(Exception, _unexpected_failure)

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness for compose and for the load balancer. No token, no data."""
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        """Prometheus exposition of the live registry.

        Ungated on purpose: it carries counters and latencies, never a case, a
        name or a resume — and a scraper that needs a secret is a scraper nobody
        wires up. Deployment keeps it on the internal network (compose does).
        """
        return PlainTextResponse(metrics_text(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/process",
        dependencies=[Depends(require_process_token), Depends(enforce_rate_limit)],
    )
    def process(payload: ProcessRequest, request: Request) -> dict:
        """Run one candidate up to the human gate.

        The response says what the guards did — but never what they read: the
        masked resume and the removed lines stay on the server, and the caller
        gets counts and labels. Every request gets its own thread id (the
        pipeline mints it), so two submissions of one candidate are two runs.
        """
        state = request.app.state
        case_file = _spool_case(state.spool_dir, payload.case)
        try:
            result = process_case(
                case_file,
                state.wiring.graph,
                reports_dir=state.reports_dir,
                meter_snapshot=state.wiring.meter_snapshot(),
            )
        except InputTooLarge:
            raise HTTPException(
                status_code=413, detail="resume too large for the input guard"
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="the payload is not a hired-candidate object",
            )
        finally:
            reset_request_state()

        return {
            "status": result["status"],
            "thread_id": result["thread_id"],
            "case_id": result["case_id"],
            "guard_flags": {
                "injection_flagged": bool(result["injection_flagged"]),
                "injection_rule": result["injection_rule"],
                "removed_lines": len(result["removed_lines"]),
                "pii_labels": list(result["pii_labels"]),
            },
        }

    @app.post(
        "/resume",
        dependencies=[Depends(require_approval_token), Depends(enforce_rate_limit)],
    )
    def resume(payload: ResumeRequest, request: Request) -> dict:
        """Answer the gate of a paused case — possibly days and restarts later.

        The evidence root is not a parameter: `process_case` recorded it in the
        case's own state, so a resume needs the thread id and nothing else, and
        one deployment's paths never end up steering another's.
        """
        state = request.app.state
        try:
            result = resume_case(
                payload.thread_id,
                payload.decision,
                state.wiring.graph,
                meter_snapshot=state.wiring.meter_snapshot(),
            )
        except ValueError:
            # One message for both causes on purpose: "unknown thread" and
            # "unreadable decision" are the same fact to the caller, and telling
            # them apart would turn this endpoint into a thread-id oracle.
            raise HTTPException(
                status_code=400,
                detail=(
                    "cannot resume: unknown thread id, or a decision that is "
                    "neither approve nor reject"
                ),
            )
        finally:
            reset_request_state()

        return {
            "status": result["status"],
            "thread_id": result["thread_id"],
            "case_id": result["case_id"],
            "audit_events": len(result["audit_trail"]),
        }

    return app


#: The ASGI application uvicorn and compose import (`uvicorn src.app:app`).
app = create_app()

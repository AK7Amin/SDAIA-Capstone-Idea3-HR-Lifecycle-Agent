"""Command line for the HR onboarding agent: run, resume, attack, verify.

Five verbs, one process each:

* ``run`` — walk an intake folder, stop every case at the human gate, and
  print the command that unpauses it.
* ``resume`` — answer that gate days later, from a different process.
* ``attack`` — replay the hostile resume through the input guards, with a
  ``--no-guardrails`` comparison so the difference is a demonstration rather
  than a claim.
* ``verify-traces`` — re-verify the evidence with the independent verifier and
  exit non-zero if anything fails.
* ``demo-failover`` — show the provider chain stepping past a spent key.

Two decisions here are not stylistic. **Postgres down is a hard stop**: the
checkpointer never silently downgrades to sqlite, because a run that quietly
changes its durability tells the operator everything is fine while paused cases
stop being recoverable — so the process exits ``2`` with the fix quoted
(critique M10), and sqlite is reached only by ``--checkpointer sqlite``.
**Everything printed is transliterated to ASCII**: trace summaries and
candidate names carry whatever script the intake file used, and a CLI that dies
rendering a name on a cp1256 console is a CLI nobody runs.

`.env` is loaded at import, from the working directory first and then from the
repository — which is what makes "copy `.env.example` to `.env` and run" true.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src import checkpointing
from src.effects import FileEffects
from src.guardrails import InputTooLarge
from src.llm import MissingKeyError
from src.observability import TRACES_DIRNAME, record_llm_failover, verify_all
from src.pipeline import (
    DEFAULT_INTAKE_DIRNAME,
    PROJECT_ROOT,
    REPORTS_DIRNAME,
    build_production_wiring,
    guard_resume,
    load_case,
    process_case,
    resume_case,
    write_run_summary,
)

__all__ = ["build_parser", "load_environment", "main", "open_checkpointer"]

#: Exit codes owned by this file. `2` is not free: `src.checkpointing` already
#: means "the database is unreachable" by it, so nothing else may claim it.
EXIT_OK = 0
EXIT_FAILED = 1

#: The hostile fixture the attack demo replays by default.
ATTACK_CASE = "02_injected_resume.json"

#: Default checkpoint file when the operator asked for sqlite by name.
SQLITE_RELATIVE_PATH = ("state", "checkpoints.sqlite")

#: How much of a guarded resume the attack demo prints back.
_PREVIEW_CHARS = 600


def load_environment() -> None:
    """Load `.env` into the process: working directory first, then the repo.

    Neither call overrides a variable that is already set, so a compose file or
    a shell export always wins over a file on disk — configuration closest to
    the operator has the last word. A missing `python-dotenv` is not fatal:
    every setting has an environment fallback, and a CLI that refuses to start
    because a convenience library is absent is worse than one that reads fewer
    files.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - dependency is in requirements.txt
        return
    nearest = find_dotenv(usecwd=True)
    if nearest:
        load_dotenv(nearest)
    load_dotenv(PROJECT_ROOT / ".env")


load_environment()


# --------------------------------------------------------------------------
# console
# --------------------------------------------------------------------------
def say(text: str = "") -> None:
    """Print one line, transliterated so no console encoding can kill the run."""
    print(str(text).encode("ascii", "replace").decode("ascii"))


def _detail(text: str) -> None:
    say(f"    {text}")


# --------------------------------------------------------------------------
# checkpointer selection
# --------------------------------------------------------------------------
@contextmanager
def open_checkpointer(kind: str, sqlite_path: Path) -> Iterator[object]:
    """Yield the checkpointer the operator asked for, and close it afterwards.

    Args:
        kind: ``"postgres"`` (the deployment story) or ``"sqlite"`` (the
            documented fallback, never selected on this module's initiative).
        sqlite_path: Database file for the sqlite shape.

    Yields:
        A LangGraph checkpointer.

    Raises:
        PostgresUnavailable: The database could not be reached. Handled once,
            in :func:`main`.
    """
    if kind == "sqlite":
        saver = checkpointing.make_sqlite_saver(sqlite_path)
        try:
            yield saver
        finally:
            connection = getattr(saver, "conn", None)
            if connection is not None:
                try:
                    connection.close()
                except Exception:  # noqa: BLE001 - shutdown noise, never the story
                    pass
        return

    with checkpointing.make_postgres_saver_cm() as saver:
        yield saver


def _sqlite_path(args: argparse.Namespace, root: Path) -> Path:
    if getattr(args, "sqlite_path", None):
        return Path(args.sqlite_path)
    return root.joinpath(*SQLITE_RELATIVE_PATH)


def _describe_backend(kind: str) -> str:
    if kind == "sqlite":
        return "checkpointer: sqlite (explicit fallback)"
    return (
        "checkpointer: postgres at "
        f"{checkpointing.redacted_dsn(checkpointing.dsn_from_env())}"
    )


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def _report_guards(result: dict) -> None:
    """Say what the guards did, so a blocked attack is visible in the log."""
    if result["injection_flagged"]:
        _detail(
            f"guard: prompt injection detected ({result['injection_rule']}), "
            f"{len(result['removed_lines'])} line(s) removed"
        )
        for line in result["removed_lines"]:
            _detail(f"  removed: {line}")
    if result["pii_labels"]:
        _detail(f"guard: PII masked ({', '.join(sorted(set(result['pii_labels'])))})")


def _report_reasoning(wiring, before: int) -> None:
    """Say what the ReAct provisioner did — the graph only kept its result."""
    for react in wiring.react_runs[before:]:
        _detail(
            f"react: {react.tool_calls} tool step(s), "
            f"decision_source={react.decision_source}, "
            f"forced_first_call={react.forced_first_call}"
        )


def cmd_run(args: argparse.Namespace) -> int:
    """Process every intake file in a folder, pausing each at the gate."""
    intake = Path(args.intake) if args.intake else PROJECT_ROOT / DEFAULT_INTAKE_DIRNAME
    root = Path(args.root) if args.root else intake.parent
    reports = Path(args.reports) if args.reports else root / REPORTS_DIRNAME

    cases = sorted(intake.glob("*.json"))
    if not cases:
        say(f"no intake files found in {intake.name}/ - nothing to do")
        return EXIT_FAILED

    say(f"intake: {intake.name}/ ({len(cases)} case(s))")
    say(_describe_backend(args.checkpointer))
    say("")

    refused = 0
    threads: list[str] = []
    with open_checkpointer(args.checkpointer, _sqlite_path(args, root)) as saver:
        wiring = build_production_wiring(saver, FileEffects(root))
        if wiring.stubbed:
            say("agents: deterministic stubs (HR_AGENT_STUBS is set)")
        for case_file in cases:
            seen_react = len(wiring.react_runs)
            try:
                result = process_case(
                    case_file,
                    wiring.graph,
                    reports_dir=reports,
                    meter_snapshot=wiring.meter_snapshot(),
                )
            except (InputTooLarge, ValueError) as exc:
                refused += 1
                say(f"{case_file.name}  REFUSED  {type(exc).__name__}: {exc}")
                continue

            threads.append(result["thread_id"])
            say(
                f"{result['case_id']}  {result['status']}  "
                f"thread={result['thread_id']}"
            )
            _report_guards(result)
            _report_reasoning(wiring, seen_react)
            if result["status"] == "awaiting_approval":
                _detail(
                    "resume with: python main.py resume "
                    f"{result['thread_id']} approve"
                )

        # `process_case` names one thread per file it writes; the batch closes
        # by naming them all, so the snapshot describes THIS run and not just
        # the last case of it.
        write_run_summary(reports, threads, wiring.meter_snapshot())

    say("")
    verdict = verify_all(reports / TRACES_DIRNAME)
    if refused:
        say(f"{refused} case(s) refused by the input guards")
    return EXIT_OK if verdict == 0 and not refused else EXIT_FAILED


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------
def cmd_resume(args: argparse.Namespace) -> int:
    """Answer the human gate of one paused case."""
    root = Path(args.root) if args.root else PROJECT_ROOT
    say(_describe_backend(args.checkpointer))

    with open_checkpointer(args.checkpointer, _sqlite_path(args, root)) as saver:
        wiring = build_production_wiring(saver, FileEffects(root))
        seen_react = len(wiring.react_runs)
        result = resume_case(
            args.thread_id,
            args.decision,
            wiring.graph,
            reports_dir=Path(args.reports) if args.reports else None,
            meter_snapshot=wiring.meter_snapshot(),
        )
        say(
            f"{result['case_id']}  {result['status']}  "
            f"thread={result['thread_id']}"
        )
        _report_reasoning(wiring, seen_react)

    case_dir = root / "outbox" / result["case_id"]
    documents = sorted(path.name for path in case_dir.glob("*")) if case_dir.is_dir() else []
    _detail(f"documents: {', '.join(documents) if documents else '(none - not approved)'}")
    _detail(f"trace: {Path(result['trace_file']).name}")
    return EXIT_OK


# --------------------------------------------------------------------------
# attack demo
# --------------------------------------------------------------------------
def cmd_attack(args: argparse.Namespace) -> int:
    """Replay the hostile resume through the guards — or deliberately past them."""
    case_file = (
        Path(args.case)
        if args.case
        else PROJECT_ROOT / DEFAULT_INTAKE_DIRNAME / ATTACK_CASE
    )
    payload = load_case(case_file)
    resume_text = str(payload.get("resume_text") or "")
    guarded = not args.no_guardrails

    say(f"attack demo on {case_file.name} (candidate {payload.get('candidate_id', '?')})")
    say(f"guardrails: {'ON' if guarded else 'OFF (--no-guardrails)'}")
    say("")

    result = guard_resume(resume_text, sanitize=guarded)

    if result.injection_flagged:
        say(f"injection detected by rule: {result.rule}")
    else:
        say("injection detected: no")

    if not guarded:
        say("no line was removed - the payload below is what the agent would read")
    elif result.removed_lines:
        say(f"removed {len(result.removed_lines)} line(s):")
        for line in result.removed_lines:
            say(f"  - {line}")
    else:
        say("nothing to remove")

    if result.pii_labels:
        labels = ", ".join(sorted(set(result.pii_labels)))
        say(f"PII found: {labels} ({'masked' if guarded else 'left in place'})")
    else:
        say("PII found: none")

    say("")
    say("text handed to the agents:")
    preview = result.text.strip()
    say(preview[:_PREVIEW_CHARS] + ("..." if len(preview) > _PREVIEW_CHARS else ""))
    return EXIT_OK


# --------------------------------------------------------------------------
# verify-traces
# --------------------------------------------------------------------------
def cmd_verify_traces(args: argparse.Namespace) -> int:
    """Re-verify persisted traces with the independent verifier."""
    reports = Path(args.reports) if args.reports else PROJECT_ROOT / REPORTS_DIRNAME
    return verify_all(reports / TRACES_DIRNAME)


# --------------------------------------------------------------------------
# failover demo
# --------------------------------------------------------------------------
#: A two-provider chain with credentials that exist only inside this demo.
_DEMO_ENV = {
    "LLM_BASE_URL": "https://provider-one.invalid/v1",
    "LLM_MODEL": "demo-model-one",
    "LLM_API_KEY": "demo-key-provider-one-0001",
    "LLM_API_KEY_FALLBACK": "",
    "LLM_PROVIDER_NAME": "provider-one",
    "LLM_BASE_URL_2": "https://provider-two.invalid/v1",
    "LLM_MODEL_2": "demo-model-two",
    "LLM_API_KEY_2": "demo-key-provider-two-0002",
    "LLM_PROVIDER_NAME_2": "provider-two",
}

_DEMO_PROMPT = "Reply with the single word: ok."


@contextmanager
def _temporary_env(values: dict) -> Iterator[None]:
    """Set environment variables for the duration of a block, then put them back."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update({key: value for key, value in values.items()})
    try:
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _simulated_failover() -> int:
    """Show the chain stepping past a spent key without touching a socket."""
    from src.llm import LLMClient, ProviderError

    class SimulatedChain(LLMClient):
        """The provider chain with a script where its transport should be.

        Subclassing the one seam that speaks HTTP keeps the demo honest: the
        failover logic, the retry rule and the usage meter under test here are
        the production ones, and only the socket is missing.
        """

        def _post(self, key, prompt, base_url=None, model=None):
            if base_url == self.providers[0].base_url:
                raise ProviderError("simulated 402: this key is out of credit", 402)
            return ("ok (simulated answer from the second provider)", 24, 12)

    with _temporary_env(_DEMO_ENV):
        client = SimulatedChain()
        chain = " -> ".join(provider.name for provider in client.providers)
        say(f"simulated provider chain: {chain}")
        answer = client.invoke(_DEMO_PROMPT, node="demo", case_id="DEMO-001")
        say("provider-one refused with a simulated 402 (out of credit)")
        say(f"served by: {client.active_provider}")
        say(f"answer: {answer}")
        if client.active_provider != client.providers[0].name:
            record_llm_failover(client.providers[0].name)
        usage = client.meter.snapshot()
        say(f"usage recorded per provider: {usage['per_provider']}")
    return EXIT_OK


def _live_failover() -> int:
    """The same demo against real endpoints, with a deliberately dead first key.

    A dead *URL* would not prove anything here — an unreachable host raises
    without a status code, and this chain fails over on spent credentials
    (401/402/403/429), not on network errors. So the first provider keeps its
    real address and is handed an invalid key, which is the failure the chain
    actually exists to survive.
    """
    from src.llm import LLMClient

    if not (os.getenv("LLM_API_KEY_2") and os.getenv("LLM_BASE_URL_2")):
        say(
            "live failover needs a second provider in .env "
            "(LLM_BASE_URL_2 and LLM_API_KEY_2)"
        )
        return EXIT_FAILED

    with _temporary_env(
        {
            "LLM_API_KEY": "invalid-key-for-the-failover-demo",
            "LLM_API_KEY_FALLBACK": "",
        }
    ):
        client = LLMClient()
        say(f"live provider chain: {' -> '.join(p.name for p in client.providers)}")
        say("provider one is holding a deliberately invalid key")
        answer = client.invoke(_DEMO_PROMPT, node="demo", case_id="DEMO-001")
        say(f"served by: {client.active_provider}")
        say(f"answer: {answer.strip()[:200]}")
        if client.active_provider != client.providers[0].name:
            record_llm_failover(client.providers[0].name)
    return EXIT_OK


def cmd_demo_failover(args: argparse.Namespace) -> int:
    """Provider failover, simulated by default and live on request."""
    return _live_failover() if args.live else _simulated_failover()


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def _add_checkpointer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpointer",
        choices=("postgres", "sqlite"),
        default="postgres",
        help=(
            "where paused cases live. postgres is the deployment story; sqlite "
            "is a documented fallback and is never selected automatically"
        ),
    )
    parser.add_argument(
        "--sqlite-path",
        default=None,
        help="checkpoint file for --checkpointer sqlite",
    )
    parser.add_argument(
        "--reports", default=None, help="evidence folder (default: <root>/reports)"
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, as one object so tests can drive it without a shell."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Employee onboarding and lifecycle agent (SDAIA capstone).",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="process an intake folder up to the human gate")
    run.add_argument(
        "--intake",
        default=None,
        help=f"folder of hired-candidate JSON files (default: {DEFAULT_INTAKE_DIRNAME}/)",
    )
    run.add_argument(
        "--root",
        default=None,
        help="where outbox/, reports/ and state/ are written (default: the intake folder's parent)",
    )
    _add_checkpointer_options(run)
    run.set_defaults(handler=cmd_run)

    resume = sub.add_parser("resume", help="answer the human gate of a paused case")
    resume.add_argument("thread_id", help="thread id printed by `run`")
    resume.add_argument("decision", choices=("approve", "reject"))
    resume.add_argument(
        "--root", default=None, help="where outbox/ is written (default: the repo)"
    )
    _add_checkpointer_options(resume)
    resume.set_defaults(handler=cmd_resume)

    attack = sub.add_parser(
        "attack", help="replay the hostile resume through the input guards"
    )
    attack.add_argument(
        "--case", default=None, help=f"intake file to replay (default: {ATTACK_CASE})"
    )
    attack.add_argument(
        "--no-guardrails",
        action="store_true",
        help="detect but do not sanitise - shows what the agent would have read",
    )
    attack.set_defaults(handler=cmd_attack)

    verify = sub.add_parser(
        "verify-traces", help="re-verify persisted traces (exit non-zero on a problem)"
    )
    verify.add_argument(
        "--reports", default=None, help="evidence folder (default: reports/)"
    )
    verify.set_defaults(handler=cmd_verify_traces)

    failover = sub.add_parser(
        "demo-failover", help="show the provider chain stepping past a spent key"
    )
    failover.add_argument(
        "--live",
        action="store_true",
        help="call the real providers with an invalid first key (needs .env)",
    )
    failover.set_defaults(handler=cmd_demo_failover)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run one command.

    Returns:
        ``0`` on success, :data:`EXIT_FAILED` on a refused or failed run, and
        :data:`src.checkpointing.EXIT_POSTGRES_UNAVAILABLE` when the checkpoint
        database could not be reached — one number, defined in one place, so
        the CLI, the child-process helpers and the compose healthcheck agree.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_FAILED

    try:
        return handler(args)
    except checkpointing.PostgresUnavailable as exc:
        say(f"FATAL: {exc}")
        return checkpointing.EXIT_POSTGRES_UNAVAILABLE
    except MissingKeyError as exc:
        # Setup mistakes get a sentence, not a stack trace: the two ways out
        # are a credential or the documented stub agents, and both are named.
        say(f"FATAL: {exc}")
        say(
            "Fix it by putting a key in .env (see .env.example), or run the "
            "deterministic stub agents with HR_AGENT_STUBS=1 for an offline demo."
        )
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())

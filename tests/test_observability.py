"""Slice 10 — observability: traces, an independent verifier, and a dashboard.

Written RED before the implementation. Zero network, zero Docker, zero keys.

What this file is really defending (critique M16, and a defect inherited from
the previous capstone): *evidence that cannot be falsified by accident*.

- A trace file states whether its own hash chain is intact. A writer that
  computes that flag with its own private hashing would happily certify a
  broken chain, so the flag must come from :func:`src.schemas.verify_chain` —
  the same function the end-to-end test calls. Two tests pin that by spying on
  the shared function, and one more forbids hashing code inside the verifier.
- A verifier that trusts the stored `chain_intact` verifies nothing. Every
  negative control below hands it a file that *claims* to be healthy.
- Artifacts must be produced by the run that reports them: the snapshot names
  the thread ids of this run, and the dashboard is rendered from files alone,
  never from live in-process counters.
"""
import inspect
import json
import re
from pathlib import Path

import pytest

from src import schemas
from src.observability import dashboard, tracing, verifier
from src.observability import (
    EXIT_NO_TRACES,
    EXIT_OK,
    EXIT_PROBLEMS,
    METRICS_FILENAME,
    TRACES_DIRNAME,
    counter_snapshot,
    metrics_text,
    observe_case_latency,
    record_case,
    record_guardrail_block,
    record_llm_failover,
    record_node,
    reset_metrics,
    verify_all,
    verify_trace_file,
    write_metrics_snapshot,
    write_trace,
)
from src.schemas import AuditEvent

THREAD_ID = "CAND-001-20260813T0300-abcdef"
CASE_ID = "CAND-001"

# One realistic full-path trail. The provisioning summary carries the honesty
# flag verbatim: the persisted trace must never launder a system-imposed step
# into a model decision (project rule "honest attribution").
TRAIL_ENTRIES = [
    {"node": "intake", "summary": "payload validated", "reasoning_pattern": ""},
    {
        "node": "profile_analyst",
        "summary": "profile extracted: Sara Alqahtani / Data Engineer",
        "reasoning_pattern": "extraction",
        "cost_usd": 0.000012,
        "latency_ms": 420,
    },
    {
        "node": "training_planner",
        "summary": "4-week plan drafted",
        "reasoning_pattern": "plan-and-execute",
        "cost_usd": 0.000031,
        "latency_ms": 810,
    },
    {
        "node": "plan_reviewer",
        "summary": "verdict approve after 1 revision(s)",
        "reasoning_pattern": "reflexion",
        "cost_usd": 0.000009,
        "latency_ms": 260,
    },
    {
        "node": "it_provisioner",
        "summary": "tool step 0: policy lookup (forced_first_call=True)",
        "reasoning_pattern": "react",
        "latency_ms": 15,
    },
    {"node": "notifier", "summary": "contract document written for CAND-001"},
]

METER_SNAPSHOT = {
    "total_tokens": 1234,
    "total_latency_ms": 1505,
    "total_ref_cost_usd": 0.000052,
    "per_node": {
        "profile_analyst": {
            "calls": 1,
            "tokens": 400,
            "latency_ms": 420,
            "ref_cost_usd": 0.000012,
        },
        "training_planner": {
            "calls": 2,
            "tokens": 700,
            "latency_ms": 810,
            "ref_cost_usd": 0.000031,
        },
    },
    "per_case": {
        CASE_ID: {
            "calls": 3,
            "tokens": 1100,
            "latency_ms": 1490,
            "ref_cost_usd": 0.000043,
        },
        "CAND-009": {
            "calls": 1,
            "tokens": 134,
            "latency_ms": 15,
            "ref_cost_usd": 0.000009,
        },
    },
    "per_provider": {
        "api.mistral.ai": {
            "calls": 4,
            "tokens": 1234,
            "latency_ms": 1505,
            "ref_cost_usd": 0.000052,
        }
    },
}


# --------------------------------------------------------------------------
# helpers — build data, never verify it (verification has exactly one home)
# --------------------------------------------------------------------------
def make_trail(entries=None) -> list[AuditEvent]:
    """Chain plain field dicts into a valid audit trail."""
    events: list[AuditEvent] = []
    prev_hash = ""
    for entry in entries if entries is not None else TRAIL_ENTRIES:
        event = AuditEvent(prev_hash=prev_hash, **entry)
        events.append(event)
        prev_hash = event.digest()
    return events


def read_trace(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_trace(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def flip_hex(value: str) -> str:
    """Tamper with exactly one byte of a hex digest."""
    head = "0" if value[0] != "0" else "1"
    return head + value[1:]


@pytest.fixture(autouse=True)
def isolated_metrics():
    """Prometheus collectors are process-global; every test starts at zero."""
    reset_metrics()
    yield
    reset_metrics()


@pytest.fixture
def reports(tmp_path) -> Path:
    return tmp_path / "reports"


@pytest.fixture
def healthy_trace(reports) -> Path:
    return write_trace(reports, THREAD_ID, CASE_ID, make_trail())


# --------------------------------------------------------------------------
# trace writer
# --------------------------------------------------------------------------

def test_write_trace_lands_at_the_path_the_e2e_expects(reports):
    path = write_trace(reports, THREAD_ID, CASE_ID, make_trail())
    assert path == reports / TRACES_DIRNAME / f"{THREAD_ID}.json"
    assert path.exists()


def test_write_trace_records_ids_and_an_intact_chain(healthy_trace):
    trace = read_trace(healthy_trace)
    assert trace["thread_id"] == THREAD_ID
    assert trace["case_id"] == CASE_ID
    assert trace["chain_intact"] is True
    assert [e["node"] for e in trace["events"]] == [
        e["node"] for e in TRAIL_ENTRIES
    ]


def test_write_trace_keeps_every_evidence_field_including_the_pattern(healthy_trace):
    events = read_trace(healthy_trace)["events"]
    patterns = [e["reasoning_pattern"] for e in events]
    assert "reflexion" in patterns and "react" in patterns
    for event in events:
        for field in (
            "node",
            "summary",
            "reasoning_pattern",
            "cost_usd",
            "latency_ms",
            "prev_hash",
            "digest",
        ):
            assert field in event, f"trace event lost the {field} field"


def test_write_trace_preserves_the_honesty_flag_verbatim(healthy_trace):
    # A system-imposed first tool call must still read as system-imposed after
    # the trip through the trace file.
    text = healthy_trace.read_text(encoding="utf-8")
    assert "forced_first_call=True" in text


def test_write_trace_stores_the_digest_of_each_event(healthy_trace):
    events = read_trace(healthy_trace)["events"]
    for stored, event in zip(events, make_trail()):
        assert stored["digest"] == event.digest()


def test_write_trace_accepts_events_that_came_back_as_dicts(reports):
    # The checkpointer may return the trail as plain dicts (slice 7's note).
    trail = [e.model_dump(mode="json") for e in make_trail()]
    path = write_trace(reports, THREAD_ID, CASE_ID, trail)
    assert read_trace(path)["chain_intact"] is True


def test_write_trace_does_not_certify_a_broken_chain(reports):
    trail = make_trail()
    broken = list(trail[:2]) + [trail[3]]  # one event dropped mid-chain
    path = write_trace(reports, THREAD_ID, CASE_ID, broken)
    assert read_trace(path)["chain_intact"] is False


def test_write_trace_writes_utf8_with_lf_endings(healthy_trace):
    raw = healthy_trace.read_bytes()
    assert b"\r\n" not in raw
    raw.decode("utf-8")  # utf-8 by construction, not by console default


def test_write_trace_leaks_no_machine_local_path(healthy_trace):
    assert not re.search(r"[A-Za-z]:[\\/]", healthy_trace.read_text(encoding="utf-8"))


def test_write_trace_refuses_a_thread_id_that_escapes_the_traces_dir(reports):
    with pytest.raises(ValueError):
        write_trace(reports, "../../etc/passwd", CASE_ID, make_trail())


def test_write_trace_computes_chain_intact_with_the_shared_function(
    reports, monkeypatch
):
    seen = []

    def spy(events):
        seen.append(list(events))
        return False  # the writer must report what this function says

    monkeypatch.setattr("src.schemas.verify_chain", spy)
    path = write_trace(reports, THREAD_ID, CASE_ID, make_trail())
    assert seen, "write_trace did not call src.schemas.verify_chain"
    assert read_trace(path)["chain_intact"] is False


# --------------------------------------------------------------------------
# metrics snapshot
# --------------------------------------------------------------------------

def test_metrics_snapshot_lands_at_the_path_the_e2e_expects(reports):
    path = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    assert path == reports / METRICS_FILENAME
    assert path.exists()


def test_metrics_snapshot_names_the_threads_this_run_produced(reports):
    # Stale-artifact detection: the e2e greps the raw text for its thread id.
    path = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID, "T-2"])
    assert THREAD_ID in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["thread_ids"] == [
        THREAD_ID,
        "T-2",
    ]


def test_metrics_snapshot_embeds_the_usage_meter(reports):
    path = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    usage = json.loads(path.read_text(encoding="utf-8"))["usage"]
    assert usage["total_tokens"] == 1234
    assert "training_planner" in usage["per_node"]
    assert CASE_ID in usage["per_case"]


def test_metrics_snapshot_embeds_the_prometheus_counters(reports):
    record_case("completed")
    record_guardrail_block("injection")
    path = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    counters = json.loads(path.read_text(encoding="utf-8"))["counters"]
    assert counters["cases_processed_total"]["completed"] == 1.0
    assert counters["guardrail_blocks_total"]["injection"] == 1.0


def test_metrics_snapshot_writes_utf8_lf_without_local_paths(reports):
    path = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert not re.search(r"[A-Za-z]:[\\/]", raw.decode("utf-8"))


# --------------------------------------------------------------------------
# counters
# --------------------------------------------------------------------------

def test_counters_increment_under_their_labels():
    record_case("completed")
    record_case("quarantined")
    record_case("completed")
    record_node("hr_gate")
    record_guardrail_block("pii")
    record_llm_failover("api.mistral.ai")

    snap = counter_snapshot()
    assert snap["cases_processed_total"] == {"completed": 2.0, "quarantined": 1.0}
    assert snap["node_runs_total"]["hr_gate"] == 1.0
    assert snap["guardrail_blocks_total"]["pii"] == 1.0
    assert snap["llm_failovers_total"]["api.mistral.ai"] == 1.0


def test_case_latency_histogram_counts_and_sums():
    observe_case_latency(120)
    observe_case_latency(880)
    latency = counter_snapshot()["case_latency_ms"]
    assert latency["count"] == 2.0
    assert latency["sum"] == 1000.0


def test_metrics_text_is_prometheus_exposition():
    record_case("completed")
    text = metrics_text()
    for name in (
        "cases_processed_total",
        "node_runs_total",
        "guardrail_blocks_total",
        "llm_failovers_total",
        "case_latency_ms",
    ):
        assert name in text
    assert 'cases_processed_total{status="completed"} 1.0' in text


def test_reset_metrics_returns_every_counter_to_zero():
    record_case("completed")
    reset_metrics()
    assert counter_snapshot()["cases_processed_total"] == {}


# --------------------------------------------------------------------------
# verifier — negative controls (M16). Every bad file below CLAIMS to be healthy.
# --------------------------------------------------------------------------

def test_verifier_passes_a_healthy_trace(healthy_trace):
    ok, problems = verify_trace_file(healthy_trace)
    assert ok is True
    assert problems == []


def test_verifier_catches_one_tampered_byte_in_a_stored_digest(healthy_trace):
    payload = read_trace(healthy_trace)
    payload["events"][2]["digest"] = flip_hex(payload["events"][2]["digest"])
    rewrite_trace(healthy_trace, payload)

    ok, problems = verify_trace_file(healthy_trace)
    assert ok is False
    assert any("2" in p and "digest" in p for p in problems), problems


def test_verifier_catches_an_edited_event_body(healthy_trace):
    # The classic forgery: rewrite what an agent "said" and leave the hashes.
    payload = read_trace(healthy_trace)
    payload["events"][1]["summary"] = "profile extracted: someone else entirely"
    rewrite_trace(healthy_trace, payload)

    ok, problems = verify_trace_file(healthy_trace)
    assert ok is False
    assert any("1" in p and "digest" in p for p in problems), problems


def test_verifier_catches_a_broken_chain_and_names_the_index(healthy_trace):
    payload = read_trace(healthy_trace)
    payload["events"][3]["prev_hash"] = flip_hex(payload["events"][3]["prev_hash"])
    payload["events"][3]["digest"] = flip_hex(payload["events"][3]["digest"])
    rewrite_trace(healthy_trace, payload)

    ok, problems = verify_trace_file(healthy_trace)
    assert ok is False
    assert any("chain" in p and "3" in p for p in problems), problems


def test_verifier_catches_two_runs_merged_into_one_trace(reports):
    # Two intakes in one thread means two runs were concatenated — the exact
    # shape the e2e's `count("intake") == 1` assertion exists to forbid.
    doubled = TRAIL_ENTRIES + TRAIL_ENTRIES
    path = write_trace(reports, THREAD_ID, CASE_ID, make_trail(doubled))

    ok, problems = verify_trace_file(path)
    assert ok is False
    assert any("merged" in p.lower() for p in problems), problems


def test_verifier_catches_a_duplicated_event(reports):
    trail = make_trail()
    path = write_trace(reports, THREAD_ID, CASE_ID, [*trail, trail[-1]])

    ok, problems = verify_trace_file(path)
    assert ok is False
    assert any("duplicate" in p.lower() for p in problems), problems


def test_verifier_never_trusts_the_stored_chain_intact_flag(healthy_trace):
    payload = read_trace(healthy_trace)
    del payload["events"][2]  # break the chain...
    payload["chain_intact"] = True  # ...and claim it is fine
    rewrite_trace(healthy_trace, payload)

    ok, problems = verify_trace_file(healthy_trace)
    assert ok is False
    assert any("chain_intact" in p for p in problems), problems


def test_verifier_flags_a_trace_renamed_to_another_thread(healthy_trace):
    copy = healthy_trace.with_name("some-other-thread.json")
    copy.write_text(healthy_trace.read_text(encoding="utf-8"), encoding="utf-8")

    ok, problems = verify_trace_file(copy)
    assert ok is False
    assert any("thread_id" in p for p in problems), problems


def test_verifier_reports_malformed_json_instead_of_raising(reports):
    path = reports / TRACES_DIRNAME / "broken.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    ok, problems = verify_trace_file(path)
    assert ok is False
    assert problems


def test_verifier_reports_a_trace_with_no_events(reports):
    path = reports / TRACES_DIRNAME / "empty.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"thread_id": "empty", "case_id": "x", "chain_intact": True}),
        encoding="utf-8",
    )

    ok, problems = verify_trace_file(path)
    assert ok is False
    assert problems


# --------------------------------------------------------------------------
# verifier — single implementation of the chain rule
# --------------------------------------------------------------------------

def test_verifier_calls_the_shared_verify_chain(healthy_trace, monkeypatch):
    seen = []
    real = schemas.verify_chain

    def spy(events):
        seen.append(list(events))
        return real(events)

    monkeypatch.setattr("src.schemas.verify_chain", spy)
    ok, _ = verify_trace_file(healthy_trace)
    assert ok is True
    assert seen, "the verifier did not call src.schemas.verify_chain"


def test_verifier_verdict_follows_the_shared_verify_chain(healthy_trace, monkeypatch):
    # If the shared rule says "broken", a healthy-looking file must fail —
    # proof the verdict is not computed twice in two places.
    monkeypatch.setattr("src.schemas.verify_chain", lambda events: False)
    ok, problems = verify_trace_file(healthy_trace)
    assert ok is False
    assert any("chain" in p for p in problems), problems


def test_verifier_module_contains_no_hashing_of_its_own():
    source = inspect.getsource(verifier)
    assert "hashlib" not in source
    assert "sha256" not in source


# --------------------------------------------------------------------------
# verify_all — the CLI engine
# --------------------------------------------------------------------------

def test_verify_all_exits_zero_on_a_healthy_directory(reports, capsys):
    write_trace(reports, THREAD_ID, CASE_ID, make_trail())
    write_trace(reports, "CAND-002-thread", "CAND-002", make_trail())

    code = verify_all(reports / TRACES_DIRNAME)
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert f"{THREAD_ID}.json" in out and "CAND-002-thread.json" in out


def test_verify_all_exits_nonzero_when_any_file_is_bad(reports, capsys):
    write_trace(reports, THREAD_ID, CASE_ID, make_trail())
    bad = write_trace(reports, "CAND-002-thread", "CAND-002", make_trail())
    payload = read_trace(bad)
    payload["events"][0]["summary"] = "rewritten after the fact"
    rewrite_trace(bad, payload)

    code = verify_all(reports / TRACES_DIRNAME)
    out = capsys.readouterr().out
    assert code == EXIT_PROBLEMS
    assert "CAND-002-thread.json" in out
    assert code != EXIT_OK


def test_verify_all_exits_nonzero_on_an_empty_directory(tmp_path, capsys):
    empty = tmp_path / "traces"
    empty.mkdir()

    code = verify_all(empty)
    out = capsys.readouterr().out
    assert code == EXIT_NO_TRACES != EXIT_OK
    assert "no trace" in out.lower()


def test_verify_all_exits_nonzero_when_the_directory_is_missing(tmp_path, capsys):
    code = verify_all(tmp_path / "nope")
    assert code != EXIT_OK
    assert capsys.readouterr().out.strip()


def test_verify_all_prints_one_ascii_line_per_file(reports, capsys):
    # Windows consoles are cp1256 here: a non-ascii node name in a trace must
    # not turn the verifier's report into a UnicodeEncodeError.
    write_trace(
        reports,
        THREAD_ID,
        CASE_ID,
        make_trail([{"node": "intake", "summary": "تم التحقق"}]),
    )
    verify_all(reports / TRACES_DIRNAME)
    out = capsys.readouterr().out
    out.encode("ascii")  # raises if the report smuggled non-ascii through
    assert len([line for line in out.splitlines() if ".json" in line]) == 1


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------

@pytest.fixture
def rendered(reports) -> Path:
    record_case("completed")
    record_node("hr_gate")
    record_guardrail_block("injection")
    record_llm_failover("api.mistral.ai")
    observe_case_latency(1505)
    write_trace(reports, THREAD_ID, CASE_ID, make_trail())
    snapshot = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    return dashboard.render(
        snapshot, reports / TRACES_DIRNAME, reports / "dashboard.html"
    )


def test_dashboard_render_returns_the_written_html(reports, rendered):
    assert rendered == reports / "dashboard.html"
    assert rendered.exists()
    assert rendered.read_text(encoding="utf-8").lstrip().startswith("<")


def test_dashboard_shows_per_node_and_per_case_rows(rendered):
    html = rendered.read_text(encoding="utf-8")
    assert "training_planner" in html and "profile_analyst" in html
    assert CASE_ID in html and "CAND-009" in html
    assert "1234" in html  # total tokens, straight from the meter snapshot


def test_dashboard_shows_guardrail_counters(rendered):
    html = rendered.read_text(encoding="utf-8")
    assert "guardrail" in html.lower()
    assert "injection" in html


def test_dashboard_lists_every_trace_with_its_chain_status(reports, rendered):
    html = rendered.read_text(encoding="utf-8")
    assert THREAD_ID in html
    assert "OK" in html


def test_dashboard_marks_a_broken_trace_as_broken(reports):
    trail = make_trail()
    write_trace(reports, "broken-thread", "CAND-003", [trail[0], trail[2]])
    snapshot = write_metrics_snapshot(reports, METER_SNAPSHOT, ["broken-thread"])
    out = dashboard.render(
        snapshot, reports / TRACES_DIRNAME, reports / "dashboard.html"
    )
    assert "BROKEN" in out.read_text(encoding="utf-8")


def test_dashboard_embeds_no_external_asset(rendered):
    html = rendered.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()


def test_dashboard_escapes_trace_content(reports):
    write_trace(
        reports,
        THREAD_ID,
        CASE_ID,
        make_trail([{"node": "intake", "summary": "<script>alert(1)</script>"}]),
    )
    snapshot = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    html = dashboard.render(
        snapshot, reports / TRACES_DIRNAME, reports / "dashboard.html"
    ).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_dashboard_renders_from_files_only(reports, rendered):
    # Structural proof: the signature accepts paths and nothing live.
    assert list(inspect.signature(dashboard.render).parameters) == [
        "metrics_snapshot_path",
        "traces_dir",
        "out_path",
    ]
    # Behavioural proof: with every in-process counter wiped, a re-render off
    # the same files still shows this run's numbers.
    reset_metrics()
    again = dashboard.render(
        reports / METRICS_FILENAME,
        reports / TRACES_DIRNAME,
        reports / "dashboard2.html",
    )
    assert "injection" in again.read_text(encoding="utf-8")


def test_dashboard_writes_utf8_lf_without_local_paths(rendered):
    raw = rendered.read_bytes()
    assert b"\r\n" not in raw
    assert not re.search(r"[A-Za-z]:[\\/]", raw.decode("utf-8"))


def test_dashboard_survives_a_missing_traces_directory(reports):
    snapshot = write_metrics_snapshot(reports, METER_SNAPSHOT, [THREAD_ID])
    out = dashboard.render(snapshot, reports / "no-traces", reports / "d.html")
    assert out.exists()


# --------------------------------------------------------------------------
# package surface
# --------------------------------------------------------------------------

def test_package_exports_the_run_and_verify_surface():
    import src.observability as obs

    for name in (
        "write_trace",
        "write_metrics_snapshot",
        "verify_trace_file",
        "verify_all",
        "render_dashboard",
        "record_case",
        "record_node",
        "record_guardrail_block",
        "record_llm_failover",
        "observe_case_latency",
        "metrics_text",
        "reset_metrics",
    ):
        assert hasattr(obs, name), f"src.observability does not export {name}"
    assert obs.render_dashboard is dashboard.render
    assert tracing.TRACES_DIRNAME == "traces"


class TestReactTranscriptIsPersisted:
    """The README promised traces carrying tool arguments and results; the
    trace events are one-line summaries. Rather than soften the claim, the
    ReAct transcript itself is now written next to the trace — D1's evidence."""

    def test_writes_steps_with_arguments_and_observations(self, tmp_path):
        import json

        from src.observability import write_react_transcript

        class _Step:
            def __init__(self, thought, action, action_input, observation):
                self.thought, self.action = thought, action
                self.action_input, self.observation = action_input, observation

        class _Run:
            decision_source = "model"
            forced_first_call = True
            steps = [
                _Step("I must read the handbook first", "hr_policy_lookup",
                      {"query": "equipment by role"}, "POL-003 - Equipment allocation..."),
                _Step("Now the start date", "date_calculator",
                      {"expression": "2026-09-01 + 90 days"}, "2026-11-30"),
            ]

        path = write_react_transcript(tmp_path, "thread-1", "CASE-1", [_Run()])
        assert path.parent.name == "react"   # NOT traces/ — one file type per dir
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["thread_id"] == "thread-1" and doc["case_id"] == "CASE-1"
        run = doc["runs"][0]
        assert run["decision_source"] == "model"
        assert run["forced_first_call"] is True        # honest attribution travels
        step = run["steps"][0]
        assert step["action"] == "hr_policy_lookup"
        assert step["action_input"] == {"query": "equipment by role"}
        assert "POL-003" in step["observation"]
        assert step["thought"]

    def test_no_file_when_there_was_no_react_run(self, tmp_path):
        from src.observability import write_react_transcript

        assert write_react_transcript(tmp_path, "t", "c", []) is None

"""Slice 2 — LLM provider chain: failover, secret redaction, usage meter.

Determinism policy (PRD C3-m15): every test here is offline. `_post` is the
only seam that touches the network and it is monkeypatched in every test that
reaches it, so no socket is ever opened and no key is ever needed.

Hermetic by construction: the autouse fixture deletes every `LLM_*` variable
before a client is built, so a developer's real `.env` can never change what
these tests assert (and can never be spent by them).
"""
from __future__ import annotations

import json
import os
import threading

import pytest

from src.llm import (
    LLMClient,
    MissingKeyError,
    Provider,
    ProviderError,
    UsageMeter,
    redact_secrets,
    register_secret,
    reset_request_state,
)

P1 = "https://p1.example/v1/chat/completions"
P2 = "https://p2.example/v1/chat/completions"


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """No provider variable from the shell or `.env` reaches a test."""
    for name in [k for k in os.environ if k.startswith("LLM_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Exponential backoff is asserted, never waited on."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def clean_request_state():
    """Context vars live in the *thread*, so they outlive a test unless reset."""
    reset_request_state()
    yield
    reset_request_state()


class FakePost:
    """Scripted stand-in for `_post`; records the (endpoint, key) attempt order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, key, prompt, base_url=None, model=None):
        self.calls.append((base_url, key))
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, 10, 5


def two_provider_client(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", P1)
    monkeypatch.setenv("LLM_MODEL", "model-1")
    monkeypatch.setenv("LLM_API_KEY", "key-one-aaaaaaaaaaaa")
    monkeypatch.setenv("LLM_API_KEY_FALLBACK", "key-two-bbbbbbbbbbbb")
    monkeypatch.setenv("LLM_BASE_URL_2", P2)
    monkeypatch.setenv("LLM_MODEL_2", "model-2")
    monkeypatch.setenv("LLM_API_KEY_2", "key-three-cccccccccccc")
    return LLMClient()


# ---------------------------------------------------------------- chain build


def test_single_provider_chain_when_only_primary_configured(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", P1)
    monkeypatch.setenv("LLM_API_KEY", "key-one-aaaaaaaaaaaa")
    client = LLMClient()

    assert len(client.providers) == 1
    assert client.providers[0].name == "p1.example"  # default label = host
    assert client.providers[0].live_keys() == ("key-one-aaaaaaaaaaaa",)


def test_second_provider_built_from_env_with_own_model_and_label(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_NAME_2", "gemini")
    client = two_provider_client(monkeypatch)

    assert [p.name for p in client.providers] == ["p1.example", "gemini"]
    assert [p.model for p in client.providers] == ["model-1", "model-2"]
    assert client.providers[1].live_keys() == ("key-three-cccccccccccc",)


def test_second_provider_ignored_when_its_key_is_missing(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", P1)
    monkeypatch.setenv("LLM_API_KEY", "key-one-aaaaaaaaaaaa")
    monkeypatch.setenv("LLM_BASE_URL_2", P2)  # url without key = not usable
    client = LLMClient()

    assert len(client.providers) == 1


def test_missing_key_raises_at_construction(monkeypatch):
    """An accidental real-agent path in a test must fail loudly, not silently."""
    monkeypatch.setenv("LLM_BASE_URL", P1)

    with pytest.raises(MissingKeyError):
        LLMClient()


def test_provider_live_keys_drops_empty_slots():
    provider = Provider(name="p", base_url=P1, model="m", keys=("a", "", None))

    assert provider.live_keys() == ("a",)


def test_endpoint_normalisation_accepts_bare_base_url():
    """`.env.example` ships the full path; a bare /v1 root must still work."""
    assert LLMClient._endpoint("https://x.example/v1") == (
        "https://x.example/v1/chat/completions"
    )
    assert LLMClient._endpoint(P1) == P1


# ------------------------------------------------------------------- failover


def test_quota_exhausted_provider_fails_over_to_next_provider(monkeypatch):
    client = two_provider_client(monkeypatch)
    rate_limited = [ProviderError("rate limited", 429) for _ in range(6)]
    fake = FakePost(*rate_limited, "served by two")
    monkeypatch.setattr(client, "_post", fake)

    content = client.invoke("hello", node="profile_analyst", case_id="CAND-001")

    assert content == "served by two"
    # 3 backoff attempts per key, keys in configured order, then provider 2.
    assert fake.calls == (
        [(P1, "key-one-aaaaaaaaaaaa")] * 3
        + [(P1, "key-two-bbbbbbbbbbbb")] * 3
        + [(P2, "key-three-cccccccccccc")]
    )
    assert client.active_provider == "p2.example"
    # The meter names who actually served — that is the failover evidence.
    assert list(client.meter.per_provider) == ["p2.example"]
    assert client.meter.per_provider["p2.example"]["calls"] == 1


def test_retry_on_429_succeeds_on_same_key_without_failing_over(monkeypatch):
    client = two_provider_client(monkeypatch)
    fake = FakePost(ProviderError("rate limited", 429), "recovered")
    monkeypatch.setattr(client, "_post", fake)

    assert client.invoke("hello", node="planner") == "recovered"
    assert fake.calls == [(P1, "key-one-aaaaaaaaaaaa")] * 2
    assert client.active_provider == "p1.example"


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_every_failover_status_moves_to_the_next_key(monkeypatch, status):
    """401 included: a dead key is a dead key, not a dead end."""
    client = two_provider_client(monkeypatch)
    attempts = 3 if status == 429 else 1  # only 429 is worth retrying
    fake = FakePost(*[ProviderError("nope", status)] * attempts, "second key ok")
    monkeypatch.setattr(client, "_post", fake)

    assert client.invoke("hello", node="planner") == "second key ok"
    assert fake.calls[-1] == (P1, "key-two-bbbbbbbbbbbb")
    assert len(fake.calls) == attempts + 1


def test_server_error_raises_immediately_without_burning_remaining_keys(monkeypatch):
    client = two_provider_client(monkeypatch)
    fake = FakePost(ProviderError("upstream exploded", 500), "never reached")
    monkeypatch.setattr(client, "_post", fake)

    with pytest.raises(RuntimeError, match="upstream exploded"):
        client.invoke("hello", node="planner")

    assert len(fake.calls) == 1, "a real outage must not burn the other keys"
    assert client.meter.total_tokens == 0, "a failed call must not be metered"


def test_all_providers_exhausted_error_carries_no_key_material(monkeypatch):
    client = two_provider_client(monkeypatch)
    leaky = ProviderError("quota gone for key-three-cccccccccccc", 402)
    fake = FakePost(*[leaky] * 8)
    monkeypatch.setattr(client, "_post", fake)

    with pytest.raises(RuntimeError) as excinfo:
        client.invoke("hello", node="planner")

    message = str(excinfo.value)
    for key in ("key-one-aaaaaaaaaaaa", "key-two-bbbbbbbbbbbb", "key-three-cccccccccccc"):
        assert key not in message
    assert "***REDACTED***" in message
    assert len(fake.calls) == 3  # 402 is not retried: one attempt per key


def test_failover_classification_covers_quota_and_dead_keys_only():
    assert LLMClient._should_failover(ProviderError("rate", 429)) is True
    assert LLMClient._should_failover(ProviderError("dead key", 401)) is True
    assert LLMClient._should_failover(ProviderError("boom", 500)) is False


# ------------------------------------------------------- transport (no socket)


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, *args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def capture_urlopen(monkeypatch, payload):
    """Replace the socket-opening call itself; nothing reaches the network."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_post_sends_a_deterministic_payload_and_parses_usage(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "1234")
    client = two_provider_client(monkeypatch)
    captured = capture_urlopen(
        monkeypatch,
        {
            "choices": [{"message": {"content": "hello there"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )

    result = client._post("key-one-aaaaaaaaaaaa", "prompt", P1, "model-1")

    assert result == ("hello there", 7, 3)
    assert captured["url"] == P1
    assert captured["body"]["temperature"] == 0  # determinism lever
    assert captured["body"]["max_tokens"] == 1234
    assert captured["body"]["model"] == "model-1"
    assert captured["timeout"] == 90


def test_post_raises_classified_error_when_a_200_body_has_no_choices(monkeypatch):
    """Providers really answer 200 + `{"error": ...}` when rate limiting."""
    client = two_provider_client(monkeypatch)
    capture_urlopen(
        monkeypatch,
        {"error": {"message": "service tier capacity exceeded", "code": "429"}},
    )

    with pytest.raises(ProviderError) as excinfo:
        client._post("key-one-aaaaaaaaaaaa", "prompt", P1, "model-1")

    assert excinfo.value.status_code == 429  # numeric string coerced
    assert LLMClient._should_failover(excinfo.value) is True


def test_post_redacts_a_key_echoed_back_inside_an_error_body(monkeypatch):
    client = two_provider_client(monkeypatch)
    capture_urlopen(
        monkeypatch,
        {"error": {"message": "key-one-aaaaaaaaaaaa is revoked", "code": 401}},
    )

    with pytest.raises(ProviderError) as excinfo:
        client._post("key-one-aaaaaaaaaaaa", "prompt", P1, "model-1")

    assert "key-one-aaaaaaaaaaaa" not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


# ---------------------------------------------------------------------- meter


def test_meter_accumulates_per_node_case_and_provider(monkeypatch):
    client = two_provider_client(monkeypatch)
    monkeypatch.setattr(client, "_post", FakePost())

    client.invoke("a", node="profile_analyst", case_id="CAND-001")
    client.invoke("b", node="profile_analyst", case_id="CAND-002")
    client.invoke("c", node="training_planner", case_id="CAND-001")

    meter = client.meter
    assert meter.total_tokens == 45  # 3 calls x (10 prompt + 5 completion)
    assert meter.per_node["profile_analyst"]["calls"] == 2
    assert meter.per_node["training_planner"]["tokens"] == 15
    assert meter.per_case["CAND-001"]["calls"] == 2
    assert meter.per_case["CAND-002"]["tokens"] == 15
    assert meter.per_provider["p1.example"]["calls"] == 3
    assert meter.total_ref_cost_usd > 0, "reference pricing must be non-zero"


def test_meter_snapshot_is_json_serialisable(monkeypatch):
    client = two_provider_client(monkeypatch)
    monkeypatch.setattr(client, "_post", FakePost())
    client.invoke("a", node="notifier", case_id="CAND-001")

    payload = json.loads(json.dumps(client.meter.snapshot()))

    assert payload["per_node"]["notifier"]["calls"] == 1
    assert payload["per_case"]["CAND-001"]["tokens"] == 15
    assert payload["per_provider"]["p1.example"]["calls"] == 1
    assert payload["total_latency_ms"] >= 0


def test_meter_falls_back_to_the_context_case_id(monkeypatch):
    client = two_provider_client(monkeypatch)
    monkeypatch.setattr(client, "_post", FakePost())
    client.active_case_id = "CAND-007"

    client.invoke("a", node="notifier")  # no explicit case_id

    assert client.meter.per_case["CAND-007"]["calls"] == 1


def test_usage_meter_record_is_independent_of_the_client():
    meter = UsageMeter()

    meter.record(node="n", prompt_tokens=100, completion_tokens=50, latency_ms=12,
                 case_id="C", provider="p")

    assert meter.total_tokens == 150
    assert meter.per_provider["p"]["latency_ms"] == 12


# ------------------------------------------------------------------ redaction


def test_registered_secret_never_appears_in_repr_or_str(monkeypatch):
    client = two_provider_client(monkeypatch)

    for text in (repr(client), str(client)):
        for key in ("key-one-aaaaaaaaaaaa", "key-two-bbbbbbbbbbbb",
                    "key-three-cccccccccccc"):
            assert key not in text


def test_redact_secrets_scrubs_a_runtime_registered_value():
    register_secret("mistral-abcdef123456")

    scrubbed = redact_secrets("Authorization: Bearer mistral-abcdef123456 failed")

    assert "mistral-abcdef123456" not in scrubbed
    assert "***REDACTED***" in scrubbed


def test_redact_secrets_ignores_short_noise():
    register_secret("short")  # too short to be a key; scrubbing it would eat logs

    assert redact_secrets("a short sentence") == "a short sentence"


# -------------------------------------------------- per-request state (C2-m14)


def test_budget_guard_is_charged_before_the_call(monkeypatch):
    class Guard:
        def __init__(self):
            self.charges = 0

        def charge(self):
            self.charges += 1

    client = two_provider_client(monkeypatch)
    fake = FakePost()
    monkeypatch.setattr(client, "_post", fake)
    guard = Guard()
    client.budget = guard

    client.invoke("a", node="notifier")

    assert guard.charges == 1


def test_budget_refusal_stops_the_call(monkeypatch):
    class Broke:
        def charge(self):
            raise RuntimeError("budget exceeded")

    client = two_provider_client(monkeypatch)
    fake = FakePost()
    monkeypatch.setattr(client, "_post", fake)
    client.budget = Broke()

    with pytest.raises(RuntimeError, match="budget exceeded"):
        client.invoke("a", node="notifier")

    assert fake.calls == [], "a refused budget must not reach the provider"


def test_case_id_and_budget_are_isolated_per_thread(monkeypatch):
    """FastAPI runs sync handlers in a threadpool sharing ONE client (C2-m14)."""
    client = two_provider_client(monkeypatch)
    barrier = threading.Barrier(2)
    seen: dict[str, tuple[str, object]] = {}

    def worker(case_id, guard):
        client.active_case_id = case_id
        client.budget = guard
        barrier.wait()  # both threads have written before either reads
        seen[case_id] = (client.active_case_id, client.budget)

    guards = {"CAND-A": object(), "CAND-B": object()}
    threads = [
        threading.Thread(target=worker, args=(case_id, guard))
        for case_id, guard in guards.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["CAND-A"] == ("CAND-A", guards["CAND-A"])
    assert seen["CAND-B"] == ("CAND-B", guards["CAND-B"])
    assert client.active_case_id == "-", "worker state must not leak to the caller"

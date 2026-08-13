"""LLM layer — a chain of OpenAI-compatible providers, a usage meter, and
central secret redaction.

Locked decisions (carried over from the previous capstone, where each one was
paid for with a live failure):

- Fail over on 402 (out of credit), **403** (total-limit reached — the one that
  used to stall the run), 429 (rate limit, seen live) and **401** (a dead key is
  a dead key; retrying it is pointless and moving on burns nothing, because the
  next credential is a different account entirely).
- The chain steps over the whole *provider*, not just the next key: a free quota
  runs out per provider, so a second key on the same host buys nothing.
- Any other status (500 and friends) is a real outage: raise immediately instead
  of spending the remaining keys on it.
- A key must never appear in a repr, a log line, or an exception message.
  Provider keys share no common prefix (Mistral's do not look like OpenRouter's),
  so redaction works off a registry populated at construction time.
- Reference pricing is applied even when the model is free, so the cost report
  shows real cost engineering instead of a column of zeros.

Per-request state (case id, budget guard) lives in `contextvars`, never on the
instance: the FastAPI service shares one client across a threadpool, and
instance attributes let two concurrent cases swap budgets and cost attribution.

HTTP goes through stdlib `urllib` on purpose — no client SDK to pin, and the
whole request is auditable in one screen.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field

#: Per-request state: see the module docstring. A new thread starts with an
#: empty context, so a threadpool handler cannot read another request's case.
_ACTIVE_CASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_case", default="-"
)
_ACTIVE_BUDGET: contextvars.ContextVar[object] = contextvars.ContextVar(
    "active_budget", default=None
)

#: Values registered at runtime are scrubbed verbatim from any later output.
_KNOWN_SECRETS: set[str] = set()

#: Shorter than this is not a credential — scrubbing it would eat real log text.
_MIN_SECRET_LEN = 12

REDACTION_MARK = "***REDACTED***"

#: Reference prices in USD per token (gpt-4o-mini style), for the cost report.
REF_PRICE_PROMPT = 0.15 / 1_000_000
REF_PRICE_COMPLETION = 0.60 / 1_000_000

DEFAULT_BASE_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "mistral-medium-latest"
CHAT_PATH = "/chat/completions"

#: Statuses that mean "this credential is spent or invalid — try the next one".
FAILOVER_STATUSES = (401, 402, 403, 429)
#: Only a rate limit is worth waiting out on the same key.
RETRY_STATUS = 429


def register_secret(value: str | None) -> None:
    """Register a secret so every later log, trace or error text is scrubbed."""
    if value and len(value) >= _MIN_SECRET_LEN:
        _KNOWN_SECRETS.add(value)


def redact_secrets(text) -> str:
    """Remove every registered key from a string before it is shown anywhere."""
    out = str(text)
    for secret in _KNOWN_SECRETS:
        out = out.replace(secret, REDACTION_MARK)
    return out


def reset_request_state() -> None:
    """Clear per-request state (used per HTTP request and between tests)."""
    _ACTIVE_CASE.set("-")
    _ACTIVE_BUDGET.set(None)


def _host_of(url: str) -> str:
    """Short provider label taken from its endpoint, for metrics and logs."""
    return url.split("//")[-1].split("/")[0] or "unknown"


def _status_code(value) -> int | None:
    """Providers report the status as an int, a numeric string, or a slug."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


class ProviderError(RuntimeError):
    """A provider failure carrying its status code — including HTTP 200 bodies."""

    def __init__(self, message: str, status_code=None):
        super().__init__(message)
        self.status_code = _status_code(status_code)


class MissingKeyError(RuntimeError):
    """No credential configured at all.

    Raised at construction, not at first call: a test or notebook that reaches
    the real agent path by accident must fail loudly and immediately rather
    than look healthy until something tries to talk to a provider.
    """


@dataclass(frozen=True)
class Provider:
    """One endpoint + model + its keys, in the order they should be tried."""

    name: str
    base_url: str
    model: str
    keys: tuple

    def live_keys(self) -> tuple:
        return tuple(k for k in self.keys if k)


@dataclass
class UsageMeter:
    """Tokens, latency and reference cost per node, per case and per provider.

    `per_provider` is the failover evidence: it names who actually served.
    """

    total_tokens: int = 0
    total_latency_ms: int = 0
    total_ref_cost_usd: float = 0.0
    per_node: dict = field(default_factory=dict)
    per_case: dict = field(default_factory=dict)
    per_provider: dict = field(default_factory=dict)

    def record(
        self,
        node: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        case_id: str = "-",
        provider: str = "-",
    ) -> float:
        tokens = prompt_tokens + completion_tokens
        ref_cost = (
            prompt_tokens * REF_PRICE_PROMPT + completion_tokens * REF_PRICE_COMPLETION
        )
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.total_ref_cost_usd += ref_cost
        for bucket, key in (
            (self.per_node, node),
            (self.per_case, case_id),
            (self.per_provider, provider),
        ):
            slot = bucket.setdefault(
                key, {"calls": 0, "tokens": 0, "latency_ms": 0, "ref_cost_usd": 0.0}
            )
            slot["calls"] += 1
            slot["tokens"] += tokens
            slot["latency_ms"] += latency_ms
            slot["ref_cost_usd"] += ref_cost
        return ref_cost

    def snapshot(self) -> dict:
        """JSON-serialisable view for the metrics artifact and the dashboard."""
        return {
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_ref_cost_usd": round(self.total_ref_cost_usd, 6),
            "per_node": self.per_node,
            "per_case": self.per_case,
            "per_provider": self.per_provider,
        }


class LLMClient:
    """Calls an OpenAI-compatible chat endpoint through a chain of providers."""

    def __init__(
        self,
        api_key: str | None = None,
        fallback_key: str | None = None,
        model: str | None = None,
        meter: UsageMeter | None = None,
    ):
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self._fallback_key = fallback_key or os.getenv("LLM_API_KEY_FALLBACK", "")
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        register_secret(self._api_key)
        register_secret(self._fallback_key)
        self.providers = self._build_providers()
        if not any(p.live_keys() for p in self.providers):
            raise MissingKeyError(
                "No LLM key configured — set LLM_API_KEY in .env "
                "(tests must stub the LLM instead of reaching this path)."
            )
        #: Name of the provider that served the last successful call.
        self.active_provider = self.providers[0].name
        self.meter = meter or UsageMeter()

    # ------------------------------------------------------------- chain setup

    @staticmethod
    def _endpoint(base_url: str) -> str:
        """Accept both `.../v1` and the full `.../v1/chat/completions` form."""
        url = base_url.rstrip("/")
        return url if url.endswith(CHAT_PATH) else url + CHAT_PATH

    def _build_providers(self) -> list:
        """Built once, at construction — the chain is fixed for the process."""
        chain = [
            Provider(
                name=os.getenv("LLM_PROVIDER_NAME", "") or _host_of(self.base_url),
                base_url=self._endpoint(self.base_url),
                model=self.model,
                keys=(self._api_key, self._fallback_key),
            )
        ]
        second_url = os.getenv("LLM_BASE_URL_2", "")
        second_key = os.getenv("LLM_API_KEY_2", "")
        if second_url and second_key:
            register_secret(second_key)
            chain.append(
                Provider(
                    name=os.getenv("LLM_PROVIDER_NAME_2", "") or _host_of(second_url),
                    base_url=self._endpoint(second_url),
                    model=os.getenv("LLM_MODEL_2", "") or self.model,
                    keys=(second_key,),
                )
            )
        return chain

    # ------------------------------------------------------ per-request state

    @property
    def active_case_id(self) -> str:
        return _ACTIVE_CASE.get()

    @active_case_id.setter
    def active_case_id(self, value) -> None:
        _ACTIVE_CASE.set(value or "-")

    @property
    def budget(self):
        return _ACTIVE_BUDGET.get()

    @budget.setter
    def budget(self, value) -> None:
        _ACTIVE_BUDGET.set(value)

    def __repr__(self) -> str:  # a key never leaks through a repr
        return (
            f"LLMClient(model={self.model!r}, "
            f"providers={[p.name for p in self.providers]!r}, "
            f"key_set={bool(self._api_key)})"
        )

    __str__ = __repr__

    # -------------------------------------------------------------- transport

    def _post(
        self,
        key: str,
        prompt: str,
        base_url: str | None = None,
        model: str | None = None,
    ) -> tuple:
        """One HTTP call to an OpenAI-compatible endpoint. The only seam that
        touches the network — tests monkeypatch exactly this method."""
        body = json.dumps(
            {
                "model": model or self.model,
                "messages": [{"role": "user", "content": prompt}],
                # Reasoning models spend hundreds of tokens before any content,
                # so a low ceiling returns an EMPTY answer (diagnosed live).
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
                # temperature=0: deterministic sampling, the first reliability
                # lever in the day-5 slides.
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base_url or self._endpoint(self.base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
        # Providers do answer HTTP 200 with an error body (no "choices"),
        # typically when rate limited upstream. Classify it so failover works.
        if "choices" not in payload:
            error = payload.get("error") or {}
            raise ProviderError(
                redact_secrets(error.get("message") or payload)[:200],
                status_code=error.get("code") or payload.get("code"),
            )
        content = payload["choices"][0]["message"]["content"] or ""
        usage = payload.get("usage") or {}
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    # ----------------------------------------------------------- failover rules

    @staticmethod
    def _status_of(exc: Exception):
        """`urllib.error.HTTPError` carries `.code`; ProviderError `.status_code`."""
        return _status_code(getattr(exc, "status_code", getattr(exc, "code", None)))

    @classmethod
    def _should_failover(cls, exc: Exception) -> bool:
        # Quota/auth statuses rotate; and a provider that is DOWN has no HTTP
        # status at all (connection refused, DNS death, timeout) — the chain
        # must move on exactly as for a dead key. Found live: the failover
        # demo pointed provider 1 at an unroutable port and the request died.
        # An HTTP 500 (provider UP but erroring) still raises immediately:
        # HTTPError carries a status and is checked before the URLError base.
        status = cls._status_of(exc)
        if status is not None:
            return status in FAILOVER_STATUSES
        import urllib.error

        return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))

    def _post_with_retry(
        self,
        key: str,
        prompt: str,
        base_url: str | None = None,
        model: str | None = None,
        attempts: int = 3,
    ) -> tuple:
        """Exponential backoff on 429 only — a rate limit clears with time,
        an invalid key never does."""
        delay = 2.0
        for attempt in range(attempts):
            try:
                return self._post(key, prompt, base_url, model)
            except Exception as exc:  # noqa: BLE001 — classified below
                if self._status_of(exc) != RETRY_STATUS or attempt == attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ public

    def invoke(self, prompt: str, node: str = "-", case_id: str | None = None) -> str:
        """Timed call with usage recorded per node, case and serving provider.

        The budget guard is charged first: a runaway case must stop spending
        rather than discover the overrun in the invoice.
        """
        guard = self.budget
        if guard is not None:
            guard.charge()
        started = time.perf_counter()
        # Attempt order: every key of every provider, in configuration order.
        attempts = [(p, key) for p in self.providers for key in p.live_keys()]
        content, prompt_tokens, completion_tokens = "", 0, 0
        for index, (provider, key) in enumerate(attempts):
            try:
                content, prompt_tokens, completion_tokens = self._post_with_retry(
                    key, prompt, provider.base_url, provider.model
                )
                self.active_provider = provider.name
                break
            except Exception as exc:  # noqa: BLE001 — classified by status
                if self._should_failover(exc) and index < len(attempts) - 1:
                    continue  # spent or invalid credential → next in the chain
                # `from None` keeps the un-redacted original out of the chained
                # traceback; the message itself is scrubbed.
                raise RuntimeError(redact_secrets(exc)) from None
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.meter.record(
            node=node,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            case_id=case_id or self.active_case_id,
            provider=self.active_provider,
        )
        return content

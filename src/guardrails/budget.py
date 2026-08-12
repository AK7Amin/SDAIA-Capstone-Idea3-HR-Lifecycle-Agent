"""Resource guards: how many model/tool calls a case may spend, and how much
untrusted text the system will look at in the first place.

Both failures are *refusals*, not crashes: the caller catches them, audits the
refusal and routes the case to quarantine. They are separate exception types
because they mean different things to an operator — a budget overrun means an
agent is looping, an oversized input means someone sent us 5 MB of "resume".
"""
from __future__ import annotations

__all__ = [
    "BudgetExceeded",
    "BudgetGuard",
    "DEFAULT_MAX_CALLS",
    "DEFAULT_SIZE_LIMIT",
    "InputTooLarge",
    "enforce_input_size",
]

DEFAULT_MAX_CALLS = 12
DEFAULT_SIZE_LIMIT = 20_000


class BudgetExceeded(RuntimeError):
    """A case tried to spend more calls than its allowance."""


class InputTooLarge(ValueError):
    """Untrusted input exceeded the size the guards are willing to scan."""


class BudgetGuard:
    """Counts chargeable calls for ONE case and refuses past the allowance.

    Not thread-safe by design: one guard belongs to one case (the pipeline
    builds a fresh guard per request), so there is no shared counter to race on.
    """

    def __init__(self, max_calls: int = DEFAULT_MAX_CALLS) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        self._max_calls = int(max_calls)
        self._calls = 0

    @property
    def max_calls(self) -> int:
        return self._max_calls

    @property
    def calls(self) -> int:
        """Calls actually granted so far — a refused charge is never counted."""
        return self._calls

    @property
    def remaining(self) -> int:
        return self._max_calls - self._calls

    def charge(self, amount: int = 1) -> int:
        """Grant `amount` calls, or refuse the whole request.

        Checked before incrementing, so a refusal leaves the counter exactly
        where it was: the audit trail shows what was spent, not what was tried.
        """
        if amount < 1:
            raise ValueError(f"amount must be >= 1, got {amount}")
        if self._calls + amount > self._max_calls:
            # Never interpolate the payload into the message — messages get logged.
            raise BudgetExceeded(
                f"call budget exhausted: {self._calls} of {self._max_calls} used, "
                f"{amount} more requested"
            )
        self._calls += amount
        return self._calls

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BudgetGuard(calls={self._calls}, max_calls={self._max_calls})"


def enforce_input_size(text: str, limit: int = DEFAULT_SIZE_LIMIT) -> str:
    """Return `text` unchanged, or refuse it for being too large.

    Called FIRST by every scanner: regex work on attacker-controlled text is
    the denial-of-service surface, so the length check has to happen before a
    single pattern is compiled against it.
    """
    if len(text) > limit:
        raise InputTooLarge(f"input of {len(text)} chars exceeds limit of {limit}")
    return text

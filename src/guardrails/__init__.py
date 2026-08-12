"""Guardrails: what the system refuses to read, obey, spend, or print.

Three boundaries, one import:

- `input_guard`  — untrusted resume text in (prompt-injection screening)
- `output_guard` — anything on its way out (PII masking, both digit scripts)
- `budget`       — how much a single case may spend, and how much text the
                   guards will scan at all

Nodes import from here, never from a submodule, so the guarded surface of the
system is one grep away.
"""
from __future__ import annotations

from .budget import (
    DEFAULT_MAX_CALLS,
    DEFAULT_SIZE_LIMIT,
    BudgetExceeded,
    BudgetGuard,
    InputTooLarge,
    enforce_input_size,
)
from .input_guard import (
    RULE_NAMES,
    UNTRUSTED_BEGIN,
    UNTRUSTED_END,
    InjectionVerdict,
    SanitizeResult,
    sanitize_resume,
    scan_text,
    wrap_untrusted,
)
from .output_guard import PII_LABELS, PIIMatch, find_pii, mask_pii, normalize_digits

__all__ = [
    "DEFAULT_MAX_CALLS",
    "DEFAULT_SIZE_LIMIT",
    "PII_LABELS",
    "RULE_NAMES",
    "UNTRUSTED_BEGIN",
    "UNTRUSTED_END",
    "BudgetExceeded",
    "BudgetGuard",
    "InjectionVerdict",
    "InputTooLarge",
    "PIIMatch",
    "SanitizeResult",
    "enforce_input_size",
    "find_pii",
    "mask_pii",
    "normalize_digits",
    "sanitize_resume",
    "scan_text",
    "wrap_untrusted",
]

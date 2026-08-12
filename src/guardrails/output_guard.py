"""Output guard — PII masking for anything that leaves the system.

Carried lesson from the previous project: the masking patterns there started
with literal ASCII digits, so a national ID written in Arabic-Indic numerals
(١٠٢٣٤٥٦٧٨٩) walked straight through into the logs. Nothing about the code
looked wrong — the tests only ever fed it ASCII.

The fix is structural, not another pattern: detection runs on a digit-
normalised COPY where every Arabic-Indic (U+0660..U+0669) and Extended
Arabic-Indic (U+06F0..U+06F9) digit has been translated to ASCII. The
translation is strictly 1:1, so offsets in the copy are offsets in the
original, and masking is applied to the ORIGINAL at those exact spans.

The opposite failure matters too: over-masking. Salaries, start dates, ticket
counts and version numbers are the fields the extraction agents exist to read,
so every pattern here carries digit boundaries and a shape, never "a run of
digits".
"""
from __future__ import annotations

import re
from typing import NamedTuple

__all__ = [
    "PII_LABELS",
    "PIIMatch",
    "find_pii",
    "mask_pii",
    "normalize_digits",
]

# --------------------------------------------------------------------------
# Digit normalisation (length-preserving by construction)
# --------------------------------------------------------------------------

_DIGIT_TABLE: dict[int, str] = {
    **{0x0660 + n: str(n) for n in range(10)},  # Arabic-Indic ٠..٩
    **{0x06F0 + n: str(n) for n in range(10)},  # Extended Arabic-Indic ۰..۹
}


def normalize_digits(text: str) -> str:
    """Map both Arabic digit scripts onto ASCII, one character for one.

    Every replacement is a single character, so `len()` and every index are
    preserved — that is what lets spans found here be applied to the original.
    """
    return text.translate(_DIGIT_TABLE)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

# Order = tie-break for matches that start at the same offset (longest first,
# then this order). EMAIL comes first because its local part can itself look
# like a phone number.
_PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")),
    # Saudi IBAN: SA + 2 check digits + 20 digits, written without separators.
    ("IBAN", re.compile(r"(?<![A-Za-z0-9])[Ss][Aa]\d{22}(?![A-Za-z0-9])")),
    # Saudi mobile: local 05XXXXXXXX, or international with +966 / 00966 / 966.
    (
        "PHONE",
        re.compile(r"(?<![\d+])(?:\+9665\d{8}|009665\d{8}|9665\d{8}|05\d{8})(?!\d)"),
    ),
    # National ID / Iqama: exactly 10 digits starting with 1 or 2, standing
    # alone. Digit boundaries keep amounts and dates out.
    ("NATIONAL_ID", re.compile(r"(?<!\d)[12]\d{9}(?!\d)")),
)

PII_LABELS: tuple[str, ...] = tuple(label for label, _ in _PII_RULES)


class PIIMatch(NamedTuple):
    """One accepted span in the ORIGINAL text. `label` is the mask token."""

    label: str
    start: int
    end: int

    @property
    def mask(self) -> str:
        return f"[{self.label}]"


def find_pii(text: str) -> tuple[PIIMatch, ...]:
    """Locate PII spans, earliest-longest first, guaranteed non-overlapping.

    Overlaps are real: an IBAN contains digit runs that look like a phone
    number, an email address can contain a whole mobile number. Resolving them
    here — rather than in `mask_pii` — keeps the masking loop a plain rewrite.
    """
    probe = normalize_digits(text)
    candidates = sorted(
        (m.start(), -(m.end() - m.start()), order, label, m.end())
        for order, (label, pattern) in enumerate(_PII_RULES)
        for m in pattern.finditer(probe)
    )

    accepted: list[PIIMatch] = []
    last_end = 0
    for start, _neg_len, _order, label, end in candidates:
        if start >= last_end:  # earliest-longest wins; the loser is dropped
            accepted.append(PIIMatch(label, start, end))
            last_end = end
    return tuple(accepted)


def mask_pii(text: str) -> str:
    """Replace every detected PII span with its label; touch nothing else.

    Text with no PII is returned as the very same object — masking must never
    be a silent rewrite of clean content.
    """
    matches = find_pii(text)
    if not matches:
        return text

    chunks: list[str] = []
    cursor = 0
    for match in matches:
        chunks.append(text[cursor:match.start])
        chunks.append(match.mask)
        cursor = match.end
    chunks.append(text[cursor:])
    return "".join(chunks)

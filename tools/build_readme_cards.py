"""Render README terminal cards FROM the committed logs — never by hand.

The dashboard screenshot went stale once (regenerated from an old capture)
and the terminal SVGs carried thread ids from a superseded run. Hand-built
evidence images rot the moment a capture is redone; these are parsed from
`reports/logs/` at build time, so an image that disagrees with its log
cannot exist.

Usage: python tools/build_readme_cards.py   (after every capture)
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
GREEN, RED, YELLOW, DIM, FG, CYAN = (
    "#3fb950", "#f85149", "#d29922", "#8b949e", "#c9d1d9", "#58a6ff",
)


def _svg(title: str, lines: list[tuple[str, str]], out: Path, width: int = 880) -> None:
    lh, pad_top, pad_x = 21, 64, 22
    height = pad_top + lh * len(lines) + 24
    rows = []
    for i, (text, color) in enumerate(lines):
        y = pad_top + lh * (i + 1) - 6
        rows.append(
            f'<text x="{pad_x}" y="{y}" fill="{color}" '
            f'font-family="Consolas,Menlo,monospace" font-size="13.5">'
            f"{html.escape(text)}</text>"
        )
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <rect width="{width}" height="{height}" rx="10" fill="#0d1117"/>\n'
        f'  <rect width="{width}" height="36" rx="10" fill="#161b22"/>\n'
        f'  <rect y="26" width="{width}" height="10" fill="#161b22"/>\n'
        f'  <circle cx="20" cy="18" r="6" fill="#ff5f57"/>\n'
        f'  <circle cx="40" cy="18" r="6" fill="#febc2e"/>\n'
        f'  <circle cx="60" cy="18" r="6" fill="#28c840"/>\n'
        f'  <text x="80" y="23" fill="#8b949e" '
        f'font-family="Consolas,Menlo,monospace" font-size="12">{html.escape(title)}</text>\n'
        + "\n".join(rows)
        + "\n</svg>",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{out.name}: {len(lines)} lines")


def _color_of(line: str) -> str:
    if line.startswith("OK ") or "chain verified" in line or ": 5 ok" in line:
        return GREEN
    if "removed:" in line or "injection detected" in line or "quarantined" in line:
        return RED
    if "PII masked" in line:
        return CYAN
    if "awaiting_approval" in line:
        return YELLOW
    if line.startswith(("intake:", "checkpointer:")):
        return FG
    return FG


def live_run_card() -> None:
    log = (ROOT / "reports/logs/01-live-run.log").read_text(encoding="utf-8")
    keep: list[tuple[str, str]] = [("$ python main.py run", DIM)]
    for raw in log.splitlines():
        line = raw.rstrip()
        # drop the per-case resume hints — noise at card scale
        if "resume with:" in line:
            continue
        keep.append((line[:96], _color_of(line)))
    _svg(
        "python main.py run — real capture (reports/logs/01-live-run.log)",
        keep,
        ROOT / "docs/images/live-run.svg",
    )


def restart_card() -> None:
    log = (ROOT / "reports/logs/05-compose-up.log").read_text(encoding="utf-8")
    m = re.search(r"== 7\) RESTART SURVIVAL.*?(?=\Z|== \d)", log, re.DOTALL)
    lines: list[tuple[str, str]] = []
    for raw in m.group(0).splitlines():
        line = raw.rstrip()
        if not line:
            continue
        color = FG
        if line.startswith("=="):
            color = DIM
        elif '"completed"' in line or line.startswith("contract.md"):
            color = GREEN
        elif '"awaiting_approval"' in line:
            color = YELLOW
        lines.append((line[:96], color))
        if line.startswith("app "):
            lines.append(("# the process is gone; the state is not", DIM))
    _svg(
        "restart survival — real capture (reports/logs/05-compose-up.log §7)",
        lines,
        ROOT / "docs/images/restart-survival.svg",
    )


if __name__ == "__main__":
    live_run_card()
    restart_card()

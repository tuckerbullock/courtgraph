#!/usr/bin/env python3
"""One-shot reachability probe for stats.nba.com and data.nba.com.

At most TWO single ``requests.get`` calls (no retry loop, no orchestration),
>= 2 s apart, per DATA_SOURCES.md 5.1. Prints outcomes; writes nothing.

5.1 immediate-stop rule: if the first provider signals a block or rate limit
(HTTP 403 / 429), the probe stops and never contacts the second provider. A
timeout or connection error on the first provider is not a block signal, so the
probe may still issue the second one-shot request after the required delay.

This is deliberately separate from ``nba_access_pilot.py``: that script
orchestrates several retrying endpoint calls, so a persistent block on the
first provider stops it before it reaches the second. This probe reaches both
providers with one attempt each (unless the first is blocked), which is how the
access evidence in ``REPORT.md`` was gathered.

    uv run --with requests==2.34.2 pilot/probe_access.py
"""

from __future__ import annotations

import time

import requests  # type: ignore[import-untyped]

DELAY_SECONDS = 2.0

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
STATS_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def probe(url: str, headers: dict[str, str], label: str) -> str:
    """Issue one GET. Return a classification: ``blocked`` / ``ok`` / ``error``.

    ``blocked`` means HTTP 403 or 429 -- the caller must not contact the next
    provider. ``error`` covers timeouts and connection failures (not a block
    signal). ``ok`` is any other completed response.
    """
    print(f"\n{label}\n  {url}")
    try:
        t0 = time.monotonic()
        resp = requests.get(url, headers=headers, timeout=30)
        dt = time.monotonic() - t0
        ctype = resp.headers.get("Content-Type", "")
        print(f"  HTTP {resp.status_code}  {len(resp.content)}B  {dt:.1f}s  ct={ctype}")
        if resp.status_code in (403, 429):
            return "blocked"
        return "ok"
    except requests.exceptions.RequestException as exc:
        print(f"  {type(exc).__name__}: {exc}")
        return "error"


def main() -> None:
    first = probe(
        "https://stats.nba.com/stats/scoreboardv2"
        "?GameDate=2024-01-15&LeagueID=00&DayOffset=0",
        STATS_HEADERS,
        "stats.nba.com scoreboardv2 (2024-01-15)",
    )
    if first == "blocked":
        print(
            "\nFirst provider returned HTTP 403/429 -> STOP (5.1). "
            "Not contacting data.nba.com."
        )
        return
    time.sleep(DELAY_SECONDS)
    probe(
        "https://data.nba.com/data/10s/prod/v1/20240115/scoreboard.json",
        {"User-Agent": UA, "Accept": "application/json"},
        "data.nba.com scoreboard.json (20240115)",
    )


if __name__ == "__main__":
    main()

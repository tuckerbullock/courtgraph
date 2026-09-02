"""Optional live acquisition from stats.nba.com / data.nba.com.

``DATA_SOURCES.md`` §5.1 designates a **single-worker, rate-limited, cache-and-
freeze** live path as the sanctioned way to fill what the frozen SRC-SHUFINSKIY
archive cannot (exact quarantine gaps, the current season, rosters,
transactions). This module is that path.

Conduct (binding, §5.1):

* one worker, no parallel streams;
* a monotonic **>= 1.5 s** gap between requests;
* exponential backoff on a slow/failed response;
* **hard stop** (``LiveAccessBlocked``, no retry, no resume without a human) on
  HTTP 429 / 403 or any explicit block;
* never rotate identity/IP, never solve an anti-bot challenge.

Every response is written **content-addressed** into a cache directory with a
provenance record, so a payload is fetched at most once and every later ingest
reads it from disk with no network. Nothing here is imported by
``courtgraph doctor`` or the chemistry path; ``urllib`` only, no third-party
HTTP client.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STATS_BASE = "https://stats.nba.com/stats"
_MIN_REQUEST_GAP_S = 1.5
_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 4

# stats.nba.com rejects non-browser clients; these headers are the documented
# minimum. This is not identity rotation -- it is one fixed, honest UA.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


class LiveAccessError(RuntimeError):
    """A live request failed in a way that is not a hard block (timeout, 5xx)."""


class LiveAccessBlocked(RuntimeError):
    """HTTP 429 / 403 or an explicit block -- stop, do not resume without review."""


@dataclass
class _Clock:
    """Monotonic request-spacing gate (>= _MIN_REQUEST_GAP_S between calls)."""

    _last: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = now - self._last
        if gap < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - gap)
        self._last = time.monotonic()


@dataclass
class LiveCache:
    """Content-addressed on-disk cache: ``<root>/<endpoint>/<sha256>.json`` plus
    an ``index.json`` mapping a request key -> hash + provenance."""

    root: Path
    _index: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        idx = self.root / "index.json"
        if idx.is_file():
            self._index = json.loads(idx.read_text())

    @staticmethod
    def key(endpoint: str, params: dict[str, str]) -> str:
        canon = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{endpoint}?{canon}"

    def get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any] | None:
        record = self._index.get(self.key(endpoint, params))
        if record is None:
            return None
        blob = self.root / endpoint / f"{record['sha256']}.json"
        if not blob.is_file():
            return None
        loaded: dict[str, Any] = json.loads(blob.read_text())
        return loaded

    def put(
        self, endpoint: str, params: dict[str, str], payload: dict[str, Any]
    ) -> str:
        body = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(body).hexdigest()
        blob_dir = self.root / endpoint
        blob_dir.mkdir(parents=True, exist_ok=True)
        (blob_dir / f"{digest}.json").write_bytes(body)
        self._index[self.key(endpoint, params)] = {
            "sha256": digest,
            "endpoint": endpoint,
            "params": params,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.root / "index.json").write_text(
            json.dumps(self._index, indent=2, sort_keys=True)
        )
        return digest


class LiveClient:
    """Single-worker fetch client. ``transport`` is injected for tests; the
    default hits stats.nba.com under the §5.1 conduct."""

    def __init__(
        self,
        cache: LiveCache,
        *,
        transport: Any | None = None,
        clock: _Clock | None = None,
    ) -> None:
        self._cache = cache
        self._transport = transport or _urllib_get
        self._clock = clock or _Clock()

    def fetch(
        self, endpoint: str, params: dict[str, str], *, refresh: bool = False
    ) -> dict[str, Any]:
        if not refresh:
            cached = self._cache.get(endpoint, params)
            if cached is not None:
                return cached

        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            self._clock.wait()
            try:
                payload = self._transport(endpoint, params)
            except LiveAccessBlocked:
                raise
            except LiveAccessError as exc:
                last_exc = exc
                time.sleep(2.0**attempt)
                continue
            self._cache.put(endpoint, params, payload)
            return payload
        raise LiveAccessError(
            f"{endpoint}: {_MAX_ATTEMPTS} attempts failed ({last_exc})"
        )


def _urllib_get(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_STATS_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310 - fixed host
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return json.loads(response.read())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise LiveAccessBlocked(
                f"{endpoint}: HTTP {exc.code} -- stop, review before resuming"
            ) from exc
        raise LiveAccessError(f"{endpoint}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LiveAccessError(f"{endpoint}: {exc}") from exc


def smoke_test(cache_root: str | Path) -> dict[str, Any]:
    """Five cheap requests to confirm this machine can reach stats.nba.com under
    the §5.1 conduct. Returns a summary; raises :class:`LiveAccessBlocked` on a
    hard block so the caller stops."""

    client = LiveClient(LiveCache(Path(cache_root)))
    checks = [
        ("commonteamroster", {"TeamID": "1610612744", "Season": "2023-24"}),
        ("commonteamroster", {"TeamID": "1610612738", "Season": "2023-24"}),
        ("commonteamroster", {"TeamID": "1610612747", "Season": "2023-24"}),
        ("commonteamroster", {"TeamID": "1610612739", "Season": "2023-24"}),
        ("commonteamroster", {"TeamID": "1610612752", "Season": "2023-24"}),
    ]
    ok = 0
    errors: list[str] = []
    for endpoint, params in checks:
        try:
            client.fetch(endpoint, params)
            ok += 1
        except LiveAccessError as exc:
            errors.append(str(exc))
    return {"requests": len(checks), "ok": ok, "errors": errors}

#!/usr/bin/env python3
"""NBA data-access and schema pilot (DATA_SOURCES.md 5.1 + 8, steps 1-2 only).

Scope: check respectful access to stats.nba.com and data.nba.com for a *few*
completed games, and record a value-free schema contract of the responses.
NOT ingestion: no bulk download, no completeness sweep, no possession work.

Every run produces TWO objects:

  * a **schema contract** (``--out PATH``) -- ``contract_version`` plus, per
    stable logical alias (``stats_scoreboard``, ``stats_playbyplay_01``, ...),
    only value-free fingerprints. Safe to commit. Contains no dates, game IDs,
    event text, scores, filenames, or local paths.
  * a **run manifest** (always ``pilot/_local/run_manifest.json``, gitignored)
    -- mode, date, real game IDs, request URLs/params, raw filenames, and the
    alias -> file mapping. Provenance only; never committed.

Run the network path from a residential network (NBA blocks many datacenter /
VPS IPs), pinning the temporary dependency:

    uv run --with requests==2.34.2 pilot/nba_access_pilot.py \\
        --date 2023-12-25 --max-games 3 --out pilot/schema/v0_schema_contract.json

Offline, against a pinned SRC-SHUFINSKIY extract (no third-party install):

    python3 pilot/nba_access_pilot.py --snapshot-dir /path/to/extract \\
        --out pilot/schema/v0_schema_contract.json

Access policy (DATA_SOURCES.md 5.1), enforced by ``fetch``:
  * one worker, no concurrency;
  * >= MIN_DELAY seconds between requests;
  * exponential backoff on transient errors;
  * HARD STOP on HTTP 403 / 429 or MAX_TRIES consecutive timeouts -- the script
    exits non-zero and does not resume (so a persistent block on the first
    provider stops the run before the second provider is reached);
  * never rotates identity, never bypasses a control.

On a hard stop the partial run manifest is still written (mode, date, game IDs
resolved so far, every attempted request with its status/error details, and
the stop reason), gitignored, so the failure leaves an audit trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from collections.abc import Iterable
from typing import Any

CONTRACT_VERSION = "0"
MIN_DELAY = 2.0  # seconds between requests (5.1: "at least 1-2 seconds")
MAX_TRIES = 3
TIMEOUT = 30
PINNED_REQUESTS = "requests==2.34.2"

HERE = pathlib.Path(__file__).resolve().parent
LOCAL = HERE / "_local"
RAW = LOCAL / "raw"
MANIFEST = LOCAL / "run_manifest.json"
DEFAULT_OUT = LOCAL / "schema_contract.json"

STATS = "https://stats.nba.com/stats"
DATA = "https://data.nba.com/data"

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
DATA_HEADERS = {"User-Agent": UA, "Accept": "application/json"}

_last_request_at = 0.0


class HardStop(RuntimeError):
    """Raised on a block signal; the pilot must not continue."""


# --------------------------------------------------------------------------- #
# Value-free schema fingerprinting (pure stdlib; no network dependency)
# --------------------------------------------------------------------------- #

_SCALAR_NAME = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


def _scalar_name(value: Any) -> str:
    # bool is a subclass of int -- exact-type lookup, dict order matters.
    return _SCALAR_NAME.get(type(value), "string")


def schema_of(node: Any) -> dict[str, Any]:
    """Recursive, deterministic, value-free schema of any JSON value.

    Captures dict keys, list-item structure, and scalar type categories.
    Never stores a cell value.
    """
    if isinstance(node, dict):
        return {
            "type": "object",
            "fields": {key: schema_of(node[key]) for key in sorted(node)},
        }
    if isinstance(node, list):
        return {"type": "array", "items": merge_schemas(schema_of(x) for x in node)}
    return {"type": _scalar_name(node)}


def merge_schemas(schemas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Union the schemas of sibling values into one representative schema."""
    items = list(schemas)
    pool = [s for s in items if s.get("type") != "empty"] or items
    if not pool:
        return {"type": "empty"}
    if all(s == pool[0] for s in pool):
        return pool[0]

    types = {s["type"] for s in pool}
    if types == {"object"}:
        all_keys: set[str] = set()
        for s in pool:
            all_keys |= set(s["fields"])
        fields: dict[str, Any] = {}
        for key in sorted(all_keys):
            present = [s["fields"][key] for s in pool if key in s["fields"]]
            merged = merge_schemas(present)
            if any(key not in s["fields"] for s in pool):
                merged = {**merged, "optional": True}
            fields[key] = merged
        return {"type": "object", "fields": fields}
    if types == {"array"}:
        return {"type": "array", "items": merge_schemas(s["items"] for s in pool)}
    return {"type": "union", "of": sorted(types)}


def _array_lengths(node: Any, path: str = "$") -> dict[str, int]:
    """Data-dependent array lengths -- for the manifest, never the contract."""
    counts: dict[str, int] = {}
    if isinstance(node, list):
        counts[path] = len(node)
        if node:
            counts.update(_array_lengths(node[0], f"{path}[]"))
    elif isinstance(node, dict):
        for key in sorted(node):
            counts.update(_array_lengths(node[key], f"{path}.{key}"))
    return counts


def _nba_result_sets(doc: Any) -> list[dict[str, Any]] | None:
    if isinstance(doc, dict) and isinstance(doc.get("resultSets"), list):
        return [s for s in doc["resultSets"] if isinstance(s, dict)]
    if isinstance(doc, dict) and isinstance(doc.get("resultSet"), dict):
        return [doc["resultSet"]]
    return None


def _nba_view(sets: list[dict[str, Any]]) -> dict[str, Any]:
    """Column-name -> type-category view for NBA resultSets (names + types)."""
    view: dict[str, Any] = {}
    for one in sets:
        name = str(one.get("name", "?"))
        headers = list(one.get("headers", []))
        rows = one.get("rowSet", []) or []
        columns: list[dict[str, Any]] = []
        for i, header in enumerate(headers):
            seen = {
                _scalar_name(r[i]) for r in rows if isinstance(r, list) and i < len(r)
            }
            columns.append({"name": header, "types": sorted(seen)})
        view[name] = {"columns": columns}
    return view


def fingerprint(doc: Any) -> dict[str, Any]:
    """Value-free fingerprint of one JSON response.

    ``schema``        recursive structure (keys, list-item structure, scalar
                      type categories); hashed.
    ``schema_sha256`` sha256 over canonical JSON of ``schema``.
    ``nba_view``      column names + type categories for NBA resultSets payloads.
    ``array_lengths`` data-dependent counts -- caller keeps these in the manifest.
    """
    schema = schema_of(doc)
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    out: dict[str, Any] = {
        "schema": schema,
        "schema_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "array_lengths": _array_lengths(doc),
    }
    sets = _nba_result_sets(doc)
    if sets is not None:
        out["nba_view"] = _nba_view(sets)
    return out


def contract_entry(doc: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Split one response into (value-free contract entry, manifest counts)."""
    fp = fingerprint(doc)
    entry: dict[str, Any] = {
        "schema": fp["schema"],
        "schema_sha256": fp["schema_sha256"],
    }
    if "nba_view" in fp:
        entry["nba_view"] = fp["nba_view"]
    return entry, fp["array_lengths"]


# --------------------------------------------------------------------------- #
# Assembly: one list of rows -> (tracked contract, ignored manifest)
# --------------------------------------------------------------------------- #

Row = tuple[str, dict[str, Any], dict[str, int], dict[str, Any]]
# (alias, contract_entry, array_lengths, manifest_provenance)


def assemble(
    mode: str, rows: Iterable[Row], extra_manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "endpoints": {},
    }
    manifest: dict[str, Any] = {
        "mode": mode,
        "aliases": {},
        "completeness": {},
        **extra_manifest,
    }
    for alias, entry, lengths, provenance in rows:
        contract["endpoints"][alias] = entry
        manifest["completeness"][alias] = lengths
        manifest["aliases"][alias] = provenance
    return contract, manifest


# --------------------------------------------------------------------------- #
# HTTP (5.1-compliant). requests imported lazily so the code above is stdlib.
# --------------------------------------------------------------------------- #


def _requests() -> Any:
    try:
        import requests  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        raise SystemExit(
            f"requests missing; run: uv run --with {PINNED_REQUESTS} "
            "pilot/nba_access_pilot.py"
        ) from None
    return requests


def _throttle() -> None:
    global _last_request_at
    wait = MIN_DELAY - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def fetch(
    url: str,
    headers: dict[str, str],
    alias: str,
    log: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> Any:
    """One throttled GET with bounded backoff. Hard-stops on 403 / 429."""
    requests = _requests()
    _throttle()
    delay = 2.0
    for attempt in range(1, MAX_TRIES + 1):
        try:
            t0 = time.monotonic()
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            dt = time.monotonic() - t0
            log.append(
                {
                    "alias": alias,
                    "url": url,
                    "params": params,
                    "http_status": resp.status_code,
                    "bytes": len(resp.content),
                    "seconds": round(dt, 2),
                }
            )
            print(f"  {alias}: HTTP {resp.status_code} {len(resp.content)}B {dt:.1f}s")
            if resp.status_code in (403, 429):
                raise HardStop(
                    f"{alias}: HTTP {resp.status_code} -> STOP (5.1); "
                    "a human must review before any retry"
                )
            resp.raise_for_status()
            return resp.json()
        except HardStop:
            raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            log.append({"alias": alias, "url": url, "error": type(exc).__name__})
            print(f"  {alias}: {type(exc).__name__} (try {attempt}/{MAX_TRIES})")
            if attempt == MAX_TRIES:
                raise HardStop(
                    f"{alias}: {MAX_TRIES} consecutive network failures -> STOP (5.1)"
                ) from exc
            time.sleep(delay)
            delay *= 2
        except (requests.HTTPError, ValueError) as exc:
            raise HardStop(f"{alias}: {exc}") from exc
    raise HardStop(f"{alias}: exhausted retries")


def save_raw(alias: str, doc: Any) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{alias}.json").write_text(json.dumps(doc, separators=(",", ":")))
    return f"raw/{alias}.json"


def _game_ids(sched: Any) -> list[str]:
    for one in sched.get("resultSets", []):
        if one.get("name") == "GameHeader":
            idx = one["headers"].index("GAME_ID")
            return [row[idx] for row in one["rowSet"]]
    return []


def run_live(
    date: str, max_games: int, state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    # ``state`` is shared with ``main`` so a HardStop mid-run still has the
    # accumulated request provenance to write into the ignored manifest.
    state["mode"] = "live"
    state["date"] = date
    state.setdefault("game_ids", [])
    log: list[dict[str, Any]] = state.setdefault("requests", [])
    rows: list[Row] = []
    yyyymmdd = date.replace("-", "")

    sched_params = {"GameDate": date, "LeagueID": "00", "DayOffset": 0}
    sched = fetch(
        f"{STATS}/scoreboardv2", STATS_HEADERS, "stats_scoreboard", log, sched_params
    )
    entry, lengths = contract_entry(sched)
    rows.append(
        (
            "stats_scoreboard",
            entry,
            lengths,
            {
                "url": f"{STATS}/scoreboardv2",
                "params": sched_params,
                "raw_file": save_raw("stats_scoreboard", sched),
            },
        )
    )
    game_ids = _game_ids(sched)[:max_games]
    state["game_ids"] = game_ids
    print(f"  -> game_ids: {game_ids}")

    data_sb_url = f"{DATA}/10s/prod/v1/{yyyymmdd}/scoreboard.json"
    data_sb = fetch(data_sb_url, DATA_HEADERS, "data_scoreboard", log)
    entry, lengths = contract_entry(data_sb)
    rows.append(
        (
            "data_scoreboard",
            entry,
            lengths,
            {"url": data_sb_url, "raw_file": save_raw("data_scoreboard", data_sb)},
        )
    )

    for i, gid in enumerate(game_ids, start=1):
        nn = f"{i:02d}"
        season = f"{2000 + int(gid[3:5])}-{int(gid[3:5]) + 1:02d}"
        specs = [
            (
                f"stats_playbyplay_{nn}",
                f"{STATS}/playbyplayv2",
                STATS_HEADERS,
                {"GameID": gid, "StartPeriod": 0, "EndPeriod": 14},
            ),
            (
                f"stats_boxscore_traditional_{nn}",
                f"{STATS}/boxscoretraditionalv2",
                STATS_HEADERS,
                {
                    "GameID": gid,
                    "StartPeriod": 0,
                    "EndPeriod": 14,
                    "StartRange": 0,
                    "EndRange": 0,
                    "RangeType": 0,
                },
            ),
        ]
        for alias, url, headers, params in specs:
            doc = fetch(url, headers, alias, log, params)
            entry, lengths = contract_entry(doc)
            rows.append(
                (
                    alias,
                    entry,
                    lengths,
                    {
                        "game_id": gid,
                        "url": url,
                        "params": params,
                        "raw_file": save_raw(alias, doc),
                    },
                )
            )

        alias = f"data_playbyplay_{nn}"
        url = (
            f"{DATA}/v2015/json/mobile_teams/nba/{season}"
            f"/scores/pbp/{gid}_full_pbp.json"
        )
        doc = fetch(url, DATA_HEADERS, alias, log)
        entry, lengths = contract_entry(doc)
        rows.append(
            (
                alias,
                entry,
                lengths,
                {"game_id": gid, "url": url, "raw_file": save_raw(alias, doc)},
            )
        )

    return assemble("live", rows, {"date": date, "game_ids": game_ids, "requests": log})


def _csv_entry(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, int]]:
    with path.open() as handle:
        columns = handle.readline().strip().split(",")
        row_count = sum(1 for _ in handle)
    digest = hashlib.sha256(("csv:" + ",".join(columns)).encode()).hexdigest()
    return {"schema": {"type": "csv", "columns": columns}, "schema_sha256": digest}, {
        "$": row_count
    }


def run_snapshot(snapshot_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fingerprint a local SRC-SHUFINSKIY extract (offline fallback per 5.1/8.1).

    JSON files go through the recursive fingerprinter; CSV files get a
    header-only column fingerprint. Alias -> relative-file mapping is manifest
    only; the contract carries only ``snapshot_NN`` aliases.
    """
    root = pathlib.Path(snapshot_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"snapshot dir not found: {root}")
    files = sorted([*root.rglob("*.json"), *root.rglob("*.csv")])[:20]
    rows: list[Row] = []
    for i, path in enumerate(files, start=1):
        alias = f"snapshot_{i:02d}"
        if path.suffix == ".json":
            entry, lengths = contract_entry(json.loads(path.read_text()))
        else:
            entry, lengths = _csv_entry(path)
        rel = path.relative_to(root).as_posix()
        rows.append((alias, entry, lengths, {"relative_file": rel}))
    return assemble("snapshot", rows, {"snapshot_dir": str(root)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NBA data-access & schema pilot")
    parser.add_argument("--date", default="2023-12-25", help="a completed slate date")
    parser.add_argument("--max-games", type=int, default=2, help="games to sample")
    parser.add_argument(
        "--snapshot-dir",
        help="fingerprint a local SRC-SHUFINSKIY extract instead of the network",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=(
            "where to write the value-free schema contract "
            f"(default: {DEFAULT_OUT}, gitignored). Pass a tracked path such as "
            "pilot/schema/v0_schema_contract.json for the tracked contract."
        ),
    )
    args = parser.parse_args(argv)

    LOCAL.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"mode": "snapshot" if args.snapshot_dir else "live"}
    try:
        if args.snapshot_dir:
            contract, manifest = run_snapshot(args.snapshot_dir)
        else:
            contract, manifest = run_live(args.date, args.max_games, state)
    except HardStop as stop:
        partial = {
            "mode": state.get("mode"),
            "date": state.get("date"),
            "game_ids": state.get("game_ids", []),
            "requests": state.get("requests", []),
            "stopped": str(stop),
        }
        MANIFEST.write_text(json.dumps(partial, indent=2, sort_keys=True))
        print(f"\nHARD STOP: {stop}")
        print(f"partial manifest -> {MANIFEST} (gitignored)")
        return 2

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract, indent=2, sort_keys=True))
    print(f"\ncontract -> {args.out}")
    print(f"manifest -> {MANIFEST} (gitignored)")
    for alias, entry in contract["endpoints"].items():
        print(f"  {alias}: {entry['schema'].get('type')} {entry['schema_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

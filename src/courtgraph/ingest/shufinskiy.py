"""Build a ``stats_nba_pbpstats/v2`` snapshot from local SRC-SHUFINSKIY archives.

``shufinskiy/nba_data`` re-packages ``stats.nba.com`` / ``data.nba.com`` payloads
as flat CSVs. ``DATA_SOURCES.md`` designates it the **local-dev-only** bulk
source (SRC-SHUFINSKIY; §8 pilot check 1). This module reconstructs, for the
games across one or more archive directories (each may span several seasons --
every ``nbastats_*.csv`` / ``datanba_*.csv`` / ``shotdetail_*.csv`` under each is
read and concatenated), exactly the files
:mod:`courtgraph.ingest.pipeline` consumes:

* ``pbp/stats_<gid>.json``  <- ``nbastats_*.csv`` (playbyplayv2), **rows kept in
  the archive's order** -- ``EVENTNUM`` is not always monotonic and pbpstats
  fixes ordering itself.
* ``pbp/data_<gid>.json``  <- ``datanba_*.csv`` (data.nba.com feed, ``oftid`` on
  every event) -- the second possession-reconstruction surface (v2), used by
  the pipeline when the playbyplayv2 surface needs a network call.
* ``game_details/stats_{home,away}_shots_<gid>.json`` <- ``shotdetail_*.csv``
* ``courtgraph_snapshot.json`` -- game date and rest days from the **validated
  ``GAME_DATE``** in ``shotdetail_*.csv`` (not the UTC event wall-clock); teams
  and the score-reconciliation target as described in ``reconciliation.source``.
  An operator ``official_totals.json`` (NBA box-score totals) is used when
  present; otherwise the data.nba.com game-feed running score, labelled as such.
* ``display_names.json`` -- id -> name, for the demo report only (ignored by the
  importer).
* ``provenance.json`` -- consumed-CSV sha256s, the pinned ``shufinskiy`` commit,
  and this converter's version.
* ``.gitignore`` (``*``) -- the whole snapshot is NBA-derived data.

Nothing here contacts the network. No value is fabricated: a game whose rest
days or reconciliation totals cannot be determined is written without them and
the importer quarantines it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtgraph.ingest import SNAPSHOT_FORMAT
from courtgraph.ingest._paths import (
    OutputPathError,
    assert_directory_ok,
    assert_not_symlink,
    ensure_gitignore_block,
    reject_overlap,
    safe_mkdir,
    safe_target,
)

CONVERTER_VERSION = "cg-shufinskiy/4"

# Each provider's CSVs are matched by this glob in the archive directory. One
# archive may hold several seasons (e.g. ``nbastats_2020.csv`` … ``_2024.csv``)
# or a playoffs file (``nbastats_po_2024.csv``); every match is read and
# concatenated -- NBA game ids are globally unique, so merging cannot collide.
# The trailing ``_`` keeps ``nbastats_*`` from also matching ``nbastatsv3_*``
# (a different surface: playbyplayv3, not consumed by this importer).
_PROVIDER_GLOBS = {
    "nbastats": "nbastats_*.csv",
    "datanba": "datanba_*.csv",
    "shotdetail": "shotdetail_*.csv",
}
_SNAPSHOT_GITIGNORE_HEADER = (
    "# Written by `courtgraph snapshot-from-shufinskiy`. NBA-derived data\n"
    "# (DATA_SOURCES.md 1 / 5.1) -- never commit."
)

# playbyplayv2 columns pbpstats reads; kept in this order in the emitted JSON.
_PBP_COLUMNS = (
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "WCTIMESTRING",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
    "PERSON1TYPE",
    "PLAYER1_ID",
    "PLAYER1_NAME",
    "PLAYER1_TEAM_ID",
    "PLAYER1_TEAM_CITY",
    "PLAYER1_TEAM_NICKNAME",
    "PLAYER1_TEAM_ABBREVIATION",
    "PERSON2TYPE",
    "PLAYER2_ID",
    "PLAYER2_NAME",
    "PLAYER2_TEAM_ID",
    "PERSON3TYPE",
    "PLAYER3_ID",
    "PLAYER3_NAME",
    "PLAYER3_TEAM_ID",
    "PLAYER3_TEAM_CITY",
    "PLAYER3_TEAM_NICKNAME",
    "PLAYER3_TEAM_ABBREVIATION",
    "VIDEO_AVAILABLE_FLAG",
)
_PBP_INT_COLUMNS = frozenset(
    {
        "EVENTNUM",
        "EVENTMSGTYPE",
        "EVENTMSGACTIONTYPE",
        "PERIOD",
        "PERSON1TYPE",
        "PERSON2TYPE",
        "PERSON3TYPE",
        "PLAYER1_ID",
        "PLAYER2_ID",
        "PLAYER3_ID",
        "VIDEO_AVAILABLE_FLAG",
    }
)
_PBP_NULLABLE_INT_COLUMNS = frozenset(
    {"PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"}
)

_SHOT_COLUMNS = (
    "GRID_TYPE",
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_NAME",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "EVENT_TYPE",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_ATTEMPTED_FLAG",
    "SHOT_MADE_FLAG",
    "GAME_DATE",
    "HTM",
    "VTM",
)

_GENERATED_SNAPSHOT_FILES = (
    "courtgraph_snapshot.json",
    "display_names.json",
    "provenance.json",
    ".gitignore",
)


class ShufinskiyArchiveError(ValueError):
    """Raised when the archive cannot supply what a requested game needs."""


def _pad_game_id(game_id: str) -> str:
    gid = str(game_id).strip()
    return gid if len(gid) == 10 else gid.zfill(10)


def _pbp_cell(column: str, value: str) -> Any:
    value = value.strip()
    if column in _PBP_INT_COLUMNS:
        return int(value) if value else 0
    if column in _PBP_NULLABLE_INT_COLUMNS:
        return int(value) if value else None
    if column in {"GAME_ID", "WCTIMESTRING", "PCTIMESTRING"}:
        return value
    return value or None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ShufinskiyArchiveError(f"missing archive file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _provider_files(archive_dirs: list[Path], provider: str) -> list[Path]:
    """Every CSV for one provider across all archive dirs, sorted by name. Raises
    if the provider contributes nothing anywhere -- a partial archive is never
    silently ingested. NBA game ids are globally unique, so merging dirs (each a
    season range) cannot collide."""

    files = sorted(p for d in archive_dirs for p in d.glob(_PROVIDER_GLOBS[provider]))
    if not files:
        glob = _PROVIDER_GLOBS[provider]
        raise ShufinskiyArchiveError(
            f"no {provider} CSV in archive dir(s) (expected {glob})"
        )
    return files


def _read_provider(
    archive_dirs: list[Path], provider: str
) -> tuple[list[dict[str, str]], list[Path]]:
    files = _provider_files(archive_dirs, provider)
    rows: list[dict[str, str]] = []
    for path in files:
        rows.extend(_read_csv(path))
    return rows, files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_commit(archive_dirs: list[Path]) -> str | list[str] | None:
    """The pinned commit recorded in the archive dirs' ``SOURCE.md`` files. One
    string if they agree; the sorted list if they disagree (recorded verbatim so
    a mixed pull is visible in provenance); ``None`` if none record one."""

    found: set[str] = set()
    for archive_dir in archive_dirs:
        source_md = archive_dir / "SOURCE.md"
        if not source_md.is_file():
            continue
        match = re.search(
            r"[Pp]inned commit:\s*`?([0-9a-f]{40})`?", source_md.read_text()
        )
        if match:
            found.add(match.group(1))
    if not found:
        return None
    return found.pop() if len(found) == 1 else sorted(found)


def _archive_provenance(
    archive_dirs: list[Path], consumed: list[Path]
) -> dict[str, Any]:
    return {
        "source": "SRC-SHUFINSKIY (shufinskiy/nba_data) — local, not redistributable",
        "pinned_commit": _pinned_commit(archive_dirs),
        "archive_dirs": [str(d) for d in archive_dirs],
        "converter_version": CONVERTER_VERSION,
        "consumed_csv_sha256": {
            path.name: _sha256(path) for path in sorted(consumed, key=lambda p: p.name)
        },
    }


def _iso_game_date(raw: str) -> str | None:
    """``shotdetail`` ``GAME_DATE`` is ``YYYYMMDD`` -- the validated game date,
    unaffected by the UTC wall-clock rolling past midnight."""

    raw = (raw or "").strip()
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    return None


def _game_date_from_shots(rows: list[dict[str, str]]) -> str | None:
    for row in rows:
        parsed = _iso_game_date(row.get("GAME_DATE", ""))
        if parsed is not None:
            return parsed
    return None


def _derive_teams_and_scores(
    rows: list[dict[str, str]],
) -> tuple[int, int, dict[int, int], dict[int, list[int]]]:
    """From one game's data.nba.com rows: home/away team id and the final +
    per-period score tracked by the ``hs`` (home) / ``vs`` (visitor) columns."""

    home_id: int | None = None
    away_id: int | None = None
    prev_hs = prev_vs = 0
    for row in rows:
        hs, vs = int(row["hs"] or 0), int(row["vs"] or 0)
        team = row["oftid"] or row["tid"]
        if team and team != "0":
            if hs > prev_hs and home_id is None:
                home_id = int(team)
            elif vs > prev_vs and away_id is None:
                away_id = int(team)
        prev_hs, prev_vs = hs, vs
    all_teams = {int(r["tid"]) for r in rows if r["tid"] and r["tid"] != "0"}
    if home_id is None and away_id is not None:
        home_id = next((t for t in all_teams if t != away_id), None)
    if away_id is None and home_id is not None:
        away_id = next((t for t in all_teams if t != home_id), None)
    if home_id is None or away_id is None:
        raise ShufinskiyArchiveError("could not identify home/away team from datanba")

    period_cumulative: dict[int, tuple[int, int]] = {}
    last = (0, 0)
    for row in rows:
        last = (int(row["hs"] or 0), int(row["vs"] or 0))
        period_cumulative[int(row["PERIOD"])] = last
    home_periods: list[int] = []
    away_periods: list[int] = []
    ph = pv = 0
    for p in sorted(period_cumulative):
        ch, cv = period_cumulative[p]
        home_periods.append(ch - ph)
        away_periods.append(cv - pv)
        ph, pv = ch, cv
    final = {home_id: last[0], away_id: last[1]}
    period_scores = {home_id: home_periods, away_id: away_periods}
    return home_id, away_id, final, period_scores


def _season_from_game_id(game_id: str) -> tuple[str, str]:
    yy = int(game_id[3:5])
    season = f"20{yy:02d}-{(yy + 1) % 100:02d}"
    season_type = {
        "1": "Pre Season",
        "2": "Regular Season",
        "4": "Playoffs",
        "5": "PlayIn",
    }.get(game_id[2], "Regular Season")
    return season, season_type


def _load_official_totals(archive_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Optional operator-supplied NBA box-score totals, keyed by game id:
    ``{game_id: {"final_score": {team_id: pts}, "period_scores": {...},
    "source": "..."}}``. Preferred over the data.nba.com game feed. Merged across
    archive dirs; a later dir wins on a duplicate key."""

    merged: dict[str, dict[str, Any]] = {}
    for archive_dir in archive_dirs:
        path = archive_dir / "official_totals.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        merged.update({_pad_game_id(str(k)): v for k, v in data.items()})
    return merged


@dataclass(frozen=True)
class ShufinskiySnapshot:
    out_dir: Path
    game_ids: tuple[str, ...]
    provenance: dict[str, Any]
    quarantine_expected: dict[str, str]
    archive_coverage: dict[str, Any]


def build_snapshot(
    archive_dir: str | Path | list[str | Path],
    game_ids: list[str] | None,
    out_dir: str | Path,
) -> ShufinskiySnapshot:
    raw_dirs = archive_dir if isinstance(archive_dir, list) else [archive_dir]
    archives = [Path(d) for d in raw_dirs]
    out = Path(out_dir)

    # Destination safety, before creating or writing anything. Every generated
    # file -- including the ones under pbp/ and game_details/ -- is written only
    # after `safe_target` confirms no path component is a symlink and the target
    # resolves inside `out`, so a symlinked intermediate directory cannot
    # redirect a write into a source file.
    for archive in archives:
        reject_overlap(archive, out, in_label="--archive-dir", out_label="--out-dir")
    assert_directory_ok(out)
    if out.exists():
        assert_not_symlink(*(out / name for name in _GENERATED_SNAPSHOT_FILES))

    out.mkdir(parents=True, exist_ok=True)
    safe_mkdir(out, out / "pbp")
    safe_mkdir(out, out / "game_details")
    ensure_gitignore_block(out, ["*"], header=_SNAPSHOT_GITIGNORE_HEADER)

    nbastats, nbastats_files = _read_provider(archives, "nbastats")
    datanba, datanba_files = _read_provider(archives, "datanba")
    shotdetail, shotdetail_files = _read_provider(archives, "shotdetail")
    official_totals = _load_official_totals(archives)
    provenance = _archive_provenance(
        archives, [*nbastats_files, *datanba_files, *shotdetail_files]
    )

    pbp_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nbastats:
        pbp_by_game[_pad_game_id(row["GAME_ID"])].append(row)
    datanba_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in datanba:
        datanba_by_game[_pad_game_id(row["GAME_ID"])].append(row)
    shots_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in shotdetail:
        shots_by_game[_pad_game_id(row["GAME_ID"])].append(row)

    # Per-season schedule for rest days, using the validated GAME_DATE only.
    # Rest is counted only against a team's earlier game *in the same season* --
    # pooling several seasons must not turn a season opener into a game with
    # ~150 days' "rest" off the prior postseason.
    schedule: dict[str, tuple[str, set[int]]] = {}
    for gid, rows in shots_by_game.items():
        date = _game_date_from_shots(rows)
        teams = {int(r["TEAM_ID"]) for r in rows if (r.get("TEAM_ID") or "").strip()}
        if date is not None and len(teams) == 2:
            schedule[gid] = (date, teams)

    available = {
        "nbastats": set(pbp_by_game),
        "datanba": set(datanba_by_game),
        "shotdetail": set(shots_by_game),
    }
    archive_games = set().union(*available.values())
    complete_games = set.intersection(*available.values())
    excluded_games = []
    for gid in sorted(archive_games - complete_games):
        rows = shots_by_game.get(gid, [])
        excluded_games.append(
            {
                "game_id": gid,
                "game_date": _game_date_from_shots(rows) or "",
                "team_ids": sorted(
                    {
                        int(row["TEAM_ID"])
                        for row in rows
                        if (row.get("TEAM_ID") or "").strip()
                    }
                ),
                "missing_inputs": sorted(
                    name for name, games in available.items() if gid not in games
                ),
            }
        )
    archive_coverage = {
        "archive_games": len(archive_games),
        "complete_games": len(complete_games),
        "games_by_input": {name: len(games) for name, games in available.items()},
        "excluded_games": excluded_games,
        "selection": "all complete games" if game_ids is None else "explicit game ids",
    }
    provenance = {**provenance, "archive_coverage": archive_coverage}
    requested = (
        sorted(complete_games)
        if game_ids is None
        else [_pad_game_id(g) for g in game_ids]
    )
    games_meta: list[dict[str, Any]] = []
    names: dict[str, dict[str, str]] = {"teams": {}, "players": {}}
    quarantine_expected: dict[str, str] = {}
    if game_ids is None:
        for rows in pbp_by_game.values():
            _collect_names(names, rows)

    for gid in requested:
        if gid not in pbp_by_game:
            raise ShufinskiyArchiveError(
                f"game {gid} not in the archive's nbastats CSVs"
            )
        if gid not in datanba_by_game:
            raise ShufinskiyArchiveError(
                f"game {gid} not in the archive's datanba CSVs"
            )

        pbp_rows = pbp_by_game[gid]
        _write_pbp(out, gid, pbp_rows)

        datanba_rows = datanba_by_game[gid]
        _write_data_nba_pbp(out, gid, datanba_rows)

        home_id, away_id, feed_final, feed_periods = _derive_teams_and_scores(
            datanba_rows
        )
        _write_shots(out, gid, shots_by_game.get(gid, []), home_id, away_id)
        _collect_names(names, pbp_rows)

        game_date = _game_date_from_shots(shots_by_game.get(gid, []))
        season, season_type = _season_from_game_id(gid)
        entry: dict[str, Any] = {
            "game_id": gid,
            "season": season,
            "season_type": season_type,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "data_nba_pbp": f"pbp/data_{gid}.json",
        }
        gaps: list[str] = []

        if game_date is not None:
            entry["game_date"] = game_date
        else:
            entry["game_date"] = ""
            gaps.append("game_date (no validated GAME_DATE in shotdetail)")

        recon = _reconciliation(
            gid, home_id, away_id, feed_final, feed_periods, official_totals
        )
        entry["reconciliation"] = recon

        rest = (
            _rest_days(gid, game_date, season, (home_id, away_id), schedule)
            if game_date is not None
            else None
        )
        if rest is not None:
            entry["days_rest"] = {str(k): v for k, v in rest.items()}
        else:
            gaps.append(
                "days_rest (no earlier same-season game for a team in the archive)"
            )

        if gaps:
            quarantine_expected[gid] = "missing_context — not fabricated: " + "; ".join(
                gaps
            )
        games_meta.append(entry)

    safe_target(out, out / "courtgraph_snapshot.json").write_text(
        json.dumps({"snapshot_format": SNAPSHOT_FORMAT, "games": games_meta}, indent=2),
        encoding="utf-8",
    )
    safe_target(out, out / "display_names.json").write_text(
        json.dumps(names, indent=2), encoding="utf-8"
    )
    safe_target(out, out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return ShufinskiySnapshot(
        out_dir=out,
        game_ids=tuple(requested),
        provenance=provenance,
        quarantine_expected=quarantine_expected,
        archive_coverage=archive_coverage,
    )


_FEED_SOURCE = (
    "data.nba.com game feed (the archive's datanba play-by-play) — the NBA's "
    "own running score; a second NBA surface, not an independent provider"
)


def _reconciliation(
    gid: str,
    home_id: int,
    away_id: int,
    feed_final: dict[int, int],
    feed_periods: dict[int, list[int]],
    official_totals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    supplied = official_totals.get(gid)
    if supplied and supplied.get("final_score"):
        final = {str(k): int(v) for k, v in supplied["final_score"].items()}
        periods = {
            str(k): [int(x) for x in v]
            for k, v in (supplied.get("period_scores") or {}).items()
        }
        source = str(
            supplied.get("source")
            or "operator-supplied NBA official box-score totals (official_totals.json)"
        )
        return {"final_score": final, "period_scores": periods, "source": source}
    return {
        "final_score": {str(k): v for k, v in feed_final.items()},
        "period_scores": {str(k): v for k, v in feed_periods.items()},
        "source": _FEED_SOURCE,
    }


def _write_pbp(out: Path, gid: str, rows: list[dict[str, str]]) -> None:
    # Rows are emitted in the archive's order -- EVENTNUM is not sorted.
    row_set = []
    for row in rows:
        record = dict(row)
        record["GAME_ID"] = gid
        row_set.append([_pbp_cell(col, record.get(col, "")) for col in _PBP_COLUMNS])
    payload = {
        "resource": "playbyplayv2",
        "parameters": {"GameID": gid},
        "resultSets": [
            {"name": "PlayByPlay", "headers": list(_PBP_COLUMNS), "rowSet": row_set}
        ],
    }
    safe_target(out, out / "pbp" / f"stats_{gid}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


# data.nba.com (v2015 mobile_teams) event keys; the datanba CSV columns map 1:1
# to what pbpstats' ``DataEnhancedPbpItem`` reads (KEY_ATTR_MAPPER). ``oftid`` is
# the offense team id on every event -- the reason this surface reconstructs
# games the playbyplayv2 surface cannot without a network call.
_DATANBA_EVENT_KEYS = (
    "evt",
    "cl",
    "de",
    "locX",
    "locY",
    "opt1",
    "opt2",
    "mtype",
    "etype",
    "opid",
    "tid",
    "pid",
    "hs",
    "vs",
    "epid",
    "oftid",
    "ord",
)


def _write_data_nba_pbp(out: Path, gid: str, rows: list[dict[str, str]]) -> None:
    """Emit ``pbp/data_<gid>.json`` in the nested ``g.pd[].pla[]`` shape
    ``pbpstats``' ``data_nba`` provider reads, from the archive's datanba rows.
    Rows are kept in archive order; ``pbpstats`` sorts by the ``ord`` field."""

    by_period: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            period = int(row["PERIOD"])
        except (KeyError, ValueError):
            continue
        event = {
            key: row[key] for key in _DATANBA_EVENT_KEYS if (row.get(key) or "") != ""
        }
        by_period[period].append(event)
    payload = {
        "g": {
            "gid": gid,
            "pd": [
                {"p": period, "pla": by_period[period]} for period in sorted(by_period)
            ],
        }
    }
    safe_target(out, out / "pbp" / f"data_{gid}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _shot_payload(rows: list[dict[str, str]], gid: str) -> dict[str, Any]:
    row_set = []
    for row in rows:
        cells: list[Any] = []
        for col in _SHOT_COLUMNS:
            value = (row.get(col) or "").strip()
            if col == "GAME_ID":
                cells.append(gid)
            elif col in {"GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "PERIOD"}:
                cells.append(int(value) if value else 0)
            elif col in {
                "LOC_X",
                "LOC_Y",
                "SHOT_DISTANCE",
                "SHOT_ATTEMPTED_FLAG",
                "SHOT_MADE_FLAG",
            }:
                cells.append(int(float(value)) if value else 0)
            else:
                cells.append(value or None)
        row_set.append(cells)
    return {
        "resource": "shotchartdetail",
        "parameters": {"GameID": gid},
        "resultSets": [
            {
                "name": "Shot_Chart_Detail",
                "headers": list(_SHOT_COLUMNS),
                "rowSet": row_set,
            }
        ],
    }


def _write_shots(
    out: Path, gid: str, rows: list[dict[str, str]], home_id: int, away_id: int
) -> None:
    home_rows = [r for r in rows if (r.get("TEAM_ID") or "").strip() == str(home_id)]
    away_rows = [r for r in rows if (r.get("TEAM_ID") or "").strip() == str(away_id)]
    safe_target(out, out / "game_details" / f"stats_home_shots_{gid}.json").write_text(
        json.dumps(_shot_payload(home_rows, gid)), encoding="utf-8"
    )
    safe_target(out, out / "game_details" / f"stats_away_shots_{gid}.json").write_text(
        json.dumps(_shot_payload(away_rows, gid)), encoding="utf-8"
    )


def _collect_names(
    names: dict[str, dict[str, str]], rows: list[dict[str, str]]
) -> None:
    for row in rows:
        for i in (1, 2, 3):
            pid = (row.get(f"PLAYER{i}_ID") or "").strip()
            pname = (row.get(f"PLAYER{i}_NAME") or "").strip()
            if pid and pid != "0" and pname:
                names["players"].setdefault(pid, pname)
        tid = (row.get("PLAYER1_TEAM_ID") or "").strip()
        city = (row.get("PLAYER1_TEAM_CITY") or "").strip()
        nick = (row.get("PLAYER1_TEAM_NICKNAME") or "").strip()
        if tid and tid != "0" and city and nick:
            names["teams"].setdefault(tid, f"{city} {nick}")


def _rest_days(
    gid: str,
    game_date: str,
    season: str,
    team_ids: tuple[int, int],
    schedule: dict[str, tuple[str, set[int]]],
) -> dict[int, int] | None:
    """0-based days of rest (back-to-back = 0) for each team, counted only
    against that team's earlier game *in the same season*. ``None`` if either
    team has no earlier same-season game in the archive (e.g. a season opener)."""

    target = _dt.date.fromisoformat(game_date)
    out: dict[int, int] = {}
    for team_id in team_ids:
        prior = [
            _dt.date.fromisoformat(date)
            for other_gid, (date, teams) in schedule.items()
            if other_gid != gid
            and team_id in teams
            and date < game_date
            and _season_from_game_id(other_gid)[0] == season
        ]
        if not prior:
            return None
        out[team_id] = max((target - max(prior)).days - 1, 0)
    return out


__all__ = [
    "CONVERTER_VERSION",
    "OutputPathError",
    "ShufinskiyArchiveError",
    "ShufinskiySnapshot",
    "build_snapshot",
]

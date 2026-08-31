"""Build a ``stats_nba_pbpstats/v1`` snapshot from a local SRC-SHUFINSKIY archive.

``shufinskiy/nba_data`` re-packages ``stats.nba.com`` / ``data.nba.com`` payloads
as flat CSVs. ``DATA_SOURCES.md`` designates it the **local-dev-only** fallback
when the live endpoints are unreachable (SRC-SHUFINSKIY; §8 pilot check 1). This
module reconstructs, for one or more games, exactly the files
:mod:`courtgraph.ingest.pipeline` consumes:

* ``pbp/stats_<gid>.json``                  <- ``nbastats_*.csv``  (playbyplayv2)
* ``game_details/stats_{home,away}_shots_<gid>.json`` <- ``shotdetail_*.csv``
* ``courtgraph_snapshot.json``              -- game date + teams from the CSVs;
  ``reconciliation`` from ``datanba_*.csv`` (the **second NBA lineage**, not an
  independent provider); rest days from the archive's own game dates.
* ``display_names.json``                    -- id -> name, for the demo report
  only (never a modeling input; ignored by the importer).

Nothing here contacts the network. No value is fabricated: a game whose rest
days cannot be derived from the archive is written without ``days_rest`` and the
importer quarantines it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class _GameFacts:
    game_id: str  # 10-digit
    game_date: str  # ISO
    home_team_id: int
    away_team_id: int
    final_score: dict[int, int]
    period_scores: dict[int, list[int]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _derive_datanba_facts(
    rows: list[dict[str, str]],
) -> tuple[str, dict[int, int], dict[int, list[int]], int, int]:
    """From one game's data.nba.com rows: date, final + per-period score, and
    which team id is home (the one tracked by the ``hs`` column)."""

    date_iso = rows[0]["wallclk"][:10]
    home_id: int | None = None
    away_id: int | None = None
    prev_hs = prev_vs = 0
    for row in rows:
        hs = int(row["hs"] or 0)
        vs = int(row["vs"] or 0)
        team = row["oftid"] or row["tid"]
        if team and team != "0":
            if hs > prev_hs and home_id is None:
                home_id = int(team)
            elif vs > prev_vs and away_id is None:
                away_id = int(team)
        prev_hs, prev_vs = hs, vs
    all_teams = {int(r["tid"]) for r in rows if r["tid"] and r["tid"] != "0"}
    if home_id is None and away_id is not None:
        home_id = next(t for t in all_teams if t != away_id)
    if away_id is None and home_id is not None:
        away_id = next(t for t in all_teams if t != home_id)
    if home_id is None or away_id is None:
        raise ShufinskiyArchiveError(
            "could not identify home/away team from datanba rows"
        )

    period_cumulative: dict[int, tuple[int, int]] = {}
    last = (0, 0)
    for row in rows:
        last = (int(row["hs"] or 0), int(row["vs"] or 0))
        period_cumulative[int(row["PERIOD"])] = last
    periods = sorted(period_cumulative)
    home_periods: list[int] = []
    away_periods: list[int] = []
    ph = pv = 0
    for p in periods:
        ch, cv = period_cumulative[p]
        home_periods.append(ch - ph)
        away_periods.append(cv - pv)
        ph, pv = ch, cv
    final = {home_id: last[0], away_id: last[1]}
    period_scores = {home_id: home_periods, away_id: away_periods}
    return date_iso, final, period_scores, home_id, away_id


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


@dataclass(frozen=True)
class ShufinskiySnapshot:
    out_dir: Path
    game_ids: tuple[str, ...]
    quarantine_expected: dict[str, str]  # game_id -> reason for a game with a gap


def build_snapshot(
    archive_dir: str | Path,
    game_ids: list[str],
    out_dir: str | Path,
) -> ShufinskiySnapshot:
    archive = Path(archive_dir)
    out = Path(out_dir)
    (out / "pbp").mkdir(parents=True, exist_ok=True)
    (out / "game_details").mkdir(parents=True, exist_ok=True)

    nbastats = _read_csv(archive / "nbastats_po_2024.csv")
    datanba = _read_csv(archive / "datanba_po_2024.csv")
    shotdetail = _read_csv(archive / "shotdetail_po_2024.csv")

    pbp_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nbastats:
        pbp_by_game[_pad_game_id(row["GAME_ID"])].append(row)
    datanba_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in datanba:
        datanba_by_game[_pad_game_id(row["GAME_ID"])].append(row)
    shots_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in shotdetail:
        shots_by_game[_pad_game_id(row["GAME_ID"])].append(row)

    # A minimal schedule (game -> date, teams) across the whole archive, for rest days.
    schedule: dict[str, tuple[str, set[int]]] = {}
    for gid, rows in datanba_by_game.items():
        teams = {int(r["tid"]) for r in rows if r["tid"] and r["tid"] != "0"}
        schedule[gid] = (rows[0]["wallclk"][:10], teams)

    requested = [_pad_game_id(g) for g in game_ids]
    games_meta: list[dict[str, Any]] = []
    names: dict[str, dict[str, str]] = {"teams": {}, "players": {}}
    quarantine_expected: dict[str, str] = {}

    for gid in requested:
        if gid not in pbp_by_game:
            raise ShufinskiyArchiveError(
                f"game {gid} not present in nbastats_po_2024.csv"
            )
        if gid not in datanba_by_game:
            raise ShufinskiyArchiveError(
                f"game {gid} not present in datanba_po_2024.csv"
            )

        pbp_rows = pbp_by_game[gid]
        _write_pbp(out, gid, pbp_rows)
        date_iso, final, period_scores, home_id, away_id = _derive_datanba_facts(
            datanba_by_game[gid]
        )
        _write_shots(out, gid, shots_by_game.get(gid, []), home_id, away_id)
        _collect_names(names, pbp_rows)

        season, season_type = _season_from_game_id(gid)
        entry: dict[str, Any] = {
            "game_id": gid,
            "game_date": date_iso,
            "season": season,
            "season_type": season_type,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "reconciliation": {
                "final_score": {str(k): v for k, v in final.items()},
                "period_scores": {str(k): v for k, v in period_scores.items()},
                "source": "data.nba.com lineage (datanba_*.csv) — a second NBA "
                "surface, not an independent provider",
            },
        }
        rest = _rest_days(gid, date_iso, (home_id, away_id), schedule)
        if rest is not None:
            entry["days_rest"] = {str(k): v for k, v in rest.items()}
        else:
            quarantine_expected[gid] = (
                "missing_context — no prior game for one team in the archive; "
                "days_rest cannot be derived and is not fabricated"
            )
        games_meta.append(entry)

    (out / "courtgraph_snapshot.json").write_text(
        json.dumps(
            {"snapshot_format": "stats_nba_pbpstats/v1", "games": games_meta},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "display_names.json").write_text(
        json.dumps(names, indent=2), encoding="utf-8"
    )
    return ShufinskiySnapshot(
        out_dir=out,
        game_ids=tuple(requested),
        quarantine_expected=quarantine_expected,
    )


def _write_pbp(out: Path, gid: str, rows: list[dict[str, str]]) -> None:
    row_set = []
    for row in sorted(rows, key=lambda r: int(r["EVENTNUM"])):
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
    (out / "pbp" / f"stats_{gid}.json").write_text(
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
    (out / "game_details" / f"stats_home_shots_{gid}.json").write_text(
        json.dumps(_shot_payload(home_rows, gid)), encoding="utf-8"
    )
    (out / "game_details" / f"stats_away_shots_{gid}.json").write_text(
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
    team_ids: tuple[int, int],
    schedule: dict[str, tuple[str, set[int]]],
) -> dict[int, int] | None:
    target = _dt.date.fromisoformat(game_date)
    out: dict[int, int] = {}
    for team_id in team_ids:
        prior_dates = [
            _dt.date.fromisoformat(date)
            for other_gid, (date, teams) in schedule.items()
            if other_gid != gid and team_id in teams and date < game_date
        ]
        if not prior_dates:
            return None
        gap = (target - max(prior_dates)).days - 1
        out[team_id] = max(gap, 0)
    return out

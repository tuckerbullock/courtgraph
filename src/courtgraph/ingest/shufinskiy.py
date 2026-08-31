"""Build a ``stats_nba_pbpstats/v1`` snapshot from a local SRC-SHUFINSKIY archive.

``shufinskiy/nba_data`` re-packages ``stats.nba.com`` / ``data.nba.com`` payloads
as flat CSVs. ``DATA_SOURCES.md`` designates it the **local-dev-only** fallback
when the live endpoints are unreachable (SRC-SHUFINSKIY; §8 pilot check 1). This
module reconstructs, for one or more games, exactly the files
:mod:`courtgraph.ingest.pipeline` consumes:

* ``pbp/stats_<gid>.json``  <- ``nbastats_*.csv`` (playbyplayv2), **rows kept in
  the archive's order** -- ``EVENTNUM`` is not always monotonic and pbpstats
  fixes ordering itself.
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

from courtgraph.ingest._paths import (
    OutputPathError,
    assert_directory_ok,
    assert_not_symlink,
    ensure_gitignore_block,
    reject_overlap,
    safe_mkdir,
    safe_target,
)

CONVERTER_VERSION = "cg-shufinskiy/2"
_CONSUMED_CSVS = (
    "nbastats_po_2024.csv",
    "datanba_po_2024.csv",
    "shotdetail_po_2024.csv",
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_commit(archive_dir: Path) -> str | None:
    source_md = archive_dir / "SOURCE.md"
    if not source_md.is_file():
        return None
    match = re.search(r"[Pp]inned commit:\s*`?([0-9a-f]{40})`?", source_md.read_text())
    return match.group(1) if match else None


def _archive_provenance(archive_dir: Path) -> dict[str, Any]:
    return {
        "source": "SRC-SHUFINSKIY (shufinskiy/nba_data) — local, not redistributable",
        "pinned_commit": _pinned_commit(archive_dir),
        "converter_version": CONVERTER_VERSION,
        "consumed_csv_sha256": {
            name: _sha256(archive_dir / name) for name in _CONSUMED_CSVS
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


def _load_official_totals(archive_dir: Path) -> dict[str, dict[str, Any]]:
    """Optional operator-supplied NBA box-score totals, keyed by game id:
    ``{game_id: {"final_score": {team_id: pts}, "period_scores": {...},
    "source": "..."}}``. Preferred over the data.nba.com game feed."""

    path = archive_dir / "official_totals.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {_pad_game_id(str(k)): v for k, v in data.items()}


@dataclass(frozen=True)
class ShufinskiySnapshot:
    out_dir: Path
    game_ids: tuple[str, ...]
    provenance: dict[str, Any]
    quarantine_expected: dict[str, str]


def build_snapshot(
    archive_dir: str | Path,
    game_ids: list[str],
    out_dir: str | Path,
) -> ShufinskiySnapshot:
    archive = Path(archive_dir)
    out = Path(out_dir)

    # Destination safety, before creating or writing anything. Every generated
    # file -- including the ones under pbp/ and game_details/ -- is written only
    # after `safe_target` confirms no path component is a symlink and the target
    # resolves inside `out`, so a symlinked intermediate directory cannot
    # redirect a write into a source file.
    reject_overlap(archive, out, in_label="--archive-dir", out_label="--out-dir")
    assert_directory_ok(out)
    if out.exists():
        assert_not_symlink(*(out / name for name in _GENERATED_SNAPSHOT_FILES))

    out.mkdir(parents=True, exist_ok=True)
    safe_mkdir(out, out / "pbp")
    safe_mkdir(out, out / "game_details")
    ensure_gitignore_block(out, ["*"], header=_SNAPSHOT_GITIGNORE_HEADER)

    nbastats = _read_csv(archive / "nbastats_po_2024.csv")
    datanba = _read_csv(archive / "datanba_po_2024.csv")
    shotdetail = _read_csv(archive / "shotdetail_po_2024.csv")
    official_totals = _load_official_totals(archive)
    provenance = _archive_provenance(archive)

    pbp_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nbastats:
        pbp_by_game[_pad_game_id(row["GAME_ID"])].append(row)
    datanba_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in datanba:
        datanba_by_game[_pad_game_id(row["GAME_ID"])].append(row)
    shots_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in shotdetail:
        shots_by_game[_pad_game_id(row["GAME_ID"])].append(row)

    # Whole-archive schedule for rest days, using the validated GAME_DATE only.
    schedule: dict[str, tuple[str, set[int]]] = {}
    for gid, rows in shots_by_game.items():
        date = _game_date_from_shots(rows)
        teams = {int(r["TEAM_ID"]) for r in rows if (r.get("TEAM_ID") or "").strip()}
        if date is not None and len(teams) == 2:
            schedule[gid] = (date, teams)

    requested = [_pad_game_id(g) for g in game_ids]
    games_meta: list[dict[str, Any]] = []
    names: dict[str, dict[str, str]] = {"teams": {}, "players": {}}
    quarantine_expected: dict[str, str] = {}

    for gid in requested:
        if gid not in pbp_by_game:
            raise ShufinskiyArchiveError(f"game {gid} not in nbastats_po_2024.csv")
        if gid not in datanba_by_game:
            raise ShufinskiyArchiveError(f"game {gid} not in datanba_po_2024.csv")

        pbp_rows = pbp_by_game[gid]
        _write_pbp(out, gid, pbp_rows)

        home_id, away_id, feed_final, feed_periods = _derive_teams_and_scores(
            datanba_by_game[gid]
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
            _rest_days(gid, game_date, (home_id, away_id), schedule)
            if game_date is not None
            else None
        )
        if rest is not None:
            entry["days_rest"] = {str(k): v for k, v in rest.items()}
        else:
            gaps.append("days_rest (no prior game for a team in the archive)")

        if gaps:
            quarantine_expected[gid] = "missing_context — not fabricated: " + "; ".join(
                gaps
            )
        games_meta.append(entry)

    safe_target(out, out / "courtgraph_snapshot.json").write_text(
        json.dumps(
            {"snapshot_format": "stats_nba_pbpstats/v1", "games": games_meta}, indent=2
        ),
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
    )


_FEED_SOURCE = (
    "data.nba.com game feed (datanba_po_2024.csv) — the NBA's own running "
    "score; a second NBA surface, not an independent provider"
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
    team_ids: tuple[int, int],
    schedule: dict[str, tuple[str, set[int]]],
) -> dict[int, int] | None:
    target = _dt.date.fromisoformat(game_date)
    out: dict[int, int] = {}
    for team_id in team_ids:
        prior = [
            _dt.date.fromisoformat(date)
            for other_gid, (date, teams) in schedule.items()
            if other_gid != gid and team_id in teams and date < game_date
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

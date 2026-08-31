"""Tiny hand-authored SRC-SHUFINSKIY-shaped CSV archive for tests.

Writes the three CSVs :mod:`courtgraph.ingest.shufinskiy` reads, for a couple of
toy playoff games, reusing the ``playbyplayv2`` row builder from
``_nba_fixtures``. Not a test module.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nba_fixtures import (  # noqa: E402
    AWAY_TEAM,
    HOME_TEAM,
    PBP_COLUMNS,
    GameBuilder,
    ordinary_game,
)

_NBASTATS_EXTRA = [
    "PLAYER1_TEAM_CITY",
    "PLAYER1_TEAM_NICKNAME",
    "PLAYER1_TEAM_ABBREVIATION",
    "PLAYER3_TEAM_CITY",
    "PLAYER3_TEAM_NICKNAME",
    "PLAYER3_TEAM_ABBREVIATION",
]
_NBASTATS_COLUMNS = list(PBP_COLUMNS) + _NBASTATS_EXTRA
_DATANBA_COLUMNS = ["wallclk", "hs", "vs", "tid", "oftid", "etype", "PERIOD", "GAME_ID"]
_SHOT_COLUMNS = [
    "GRID_TYPE",
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_NAME",
    "PERIOD",
    "LOC_X",
    "LOC_Y",
    "GAME_DATE",
    "HTM",
    "VTM",
]

TEAM_NAMES = {
    HOME_TEAM: ("Rivertown", "Otters", "RIV"),
    AWAY_TEAM: ("Hillside", "Foxes", "HIL"),
}


def _pbp_rows(builder: GameBuilder, game_id: str) -> list[dict[str, str]]:
    payload: Any = builder.pbp_payload()
    rows: list[dict[str, str]] = []
    for values in payload["resultSets"][0]["rowSet"]:
        record = dict(zip(PBP_COLUMNS, values, strict=False))
        out = {c: "" for c in _NBASTATS_COLUMNS}
        for key, value in record.items():
            out[key] = "" if value is None else str(value)
        out["GAME_ID"] = game_id[2:] if game_id.startswith("00") else game_id
        team_raw = out.get("PLAYER1_TEAM_ID", "")
        if team_raw:
            city, nick, abbr = TEAM_NAMES[int(team_raw)]
            out["PLAYER1_TEAM_CITY"], out["PLAYER1_TEAM_NICKNAME"] = city, nick
            out["PLAYER1_TEAM_ABBREVIATION"] = abbr
        rows.append(out)
    return rows


def _datanba_rows(
    builder: GameBuilder, game_id: str, date_iso: str, wallclk_date: str | None = None
) -> list[dict[str, str]]:
    """A compatible data.nba.com track: same running score, home = HOME_TEAM
    (its scoring drives the ``hs`` column, which is how the converter tells the
    two teams apart). ``wallclk_date`` (defaulting to ``date_iso``) is the UTC
    calendar date of the event timestamps -- distinct from the game's local
    date when a tip-off rolls past UTC midnight."""

    final = builder.final_score()
    periods = builder.period_scores()
    short = game_id[2:] if game_id.startswith("00") else game_id
    wallclk_date = wallclk_date or date_iso

    def row(
        clock: str, hs: int, vs: int, tid: int, etype: str, period: int
    ) -> dict[str, str]:
        return {
            "wallclk": f"{wallclk_date}T{clock}Z",
            "hs": str(hs),
            "vs": str(vs),
            "tid": str(tid),
            "oftid": str(tid),
            "etype": etype,
            "PERIOD": str(period),
            "GAME_ID": short,
        }

    rows = [row("23:00:00.000", 0, 0, 0, "12", 1)]
    hs = vs = 0
    for period, (h_pts, a_pts) in enumerate(
        zip(periods[HOME_TEAM], periods[AWAY_TEAM], strict=False), start=1
    ):
        hs += h_pts
        rows.append(row(f"23:{period:02d}:10.000", hs, vs, HOME_TEAM, "1", period))
        vs += a_pts
        rows.append(row(f"23:{period:02d}:40.000", hs, vs, AWAY_TEAM, "1", period))
    assert (hs, vs) == (final[HOME_TEAM], final[AWAY_TEAM])
    return rows


def _shot_rows(game_id: str, date_iso: str) -> list[dict[str, str]]:
    short = game_id[2:] if game_id.startswith("00") else game_id
    base = {
        "GRID_TYPE": "Shot Chart Detail",
        "GAME_ID": short,
        "PLAYER_NAME": "A Player",
        "PERIOD": "1",
        "LOC_X": "10",
        "LOC_Y": "20",
        "GAME_DATE": date_iso.replace("-", ""),
        "HTM": "RIV",
        "VTM": "HIL",
    }
    return [
        {**base, "GAME_EVENT_ID": "8", "PLAYER_ID": "101", "TEAM_ID": str(HOME_TEAM)},
        {**base, "GAME_EVENT_ID": "9", "PLAYER_ID": "201", "TEAM_ID": str(AWAY_TEAM)},
    ]


def write_archive(
    directory: str | Path,
    games: list[tuple[Any, ...]],
) -> Path:
    """``games`` = list of (game_id, game_date_iso, builder[, wallclk_date_iso])."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / "SOURCE.md").write_text(
        "# fixture\nPinned commit: `0123456789abcdef0123456789abcdef01234567`\n"
    )
    nbastats: list[dict[str, str]] = []
    datanba: list[dict[str, str]] = []
    shots: list[dict[str, str]] = []
    for entry in games:
        game_id, date_iso, builder = entry[0], entry[1], entry[2]
        wallclk_date = entry[3] if len(entry) > 3 else None
        assert isinstance(builder, GameBuilder)
        nbastats += _pbp_rows(builder, game_id)
        datanba += _datanba_rows(builder, game_id, date_iso, wallclk_date)
        shots += _shot_rows(game_id, date_iso)

    write_raw_archive(root, nbastats=nbastats, datanba=datanba, shots=shots)
    return root


def write_raw_archive(
    directory: str | Path,
    *,
    nbastats: list[dict[str, str]],
    datanba: list[dict[str, str]],
    shots: list[dict[str, str]],
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "nbastats_po_2024.csv", _NBASTATS_COLUMNS, nbastats)
    _write(root / "datanba_po_2024.csv", _DATANBA_COLUMNS, datanba)
    _write(root / "shotdetail_po_2024.csv", _SHOT_COLUMNS, shots)
    return root


def write_official_totals(
    directory: str | Path, totals: dict[str, dict[str, object]]
) -> Path:
    path = Path(directory) / "official_totals.json"
    path.write_text(json.dumps(totals, indent=2))
    return path


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def sample_games() -> list[tuple[Any, ...]]:
    """Three playoff games for the same two teams: G2 has a prior game (rest
    days derivable), G1 does not (converter omits days_rest -> quarantine)."""

    g1 = ordinary_game("0042400081").builder
    g2 = ordinary_game("0042400082").builder
    g3 = ordinary_game("0042400083").builder
    return [
        ("0042400081", "2025-04-19", g1),
        ("0042400082", "2025-04-22", g2),
        ("0042400083", "2025-04-25", g3),
    ]

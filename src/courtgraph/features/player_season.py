"""Per-(player, season) role / skill profiles derived from a local snapshot.

Counting stats come from the raw ``playbyplayv2`` and ``shotchartdetail``
payloads in the snapshot; possession exposure comes from the stint file
(so the denominator matches the reconstruction the models train on). Only
games present in **both** the snapshot and the stint file are counted.

The output feeds the role-conditioned interaction model: usage, shot profile
(rim / mid / three / corner-three share), playmaking, and turnover rate per
100 offensive possessions.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from courtgraph.chemistry.stints import StintTable
from courtgraph.ingest.snapshot import Snapshot, SnapshotGame, load_snapshot

PLAYER_PROFILE_SCHEMA_VERSION = 1

# stats.nba.com EVENTMSGTYPE
_MADE_SHOT = 1
_MISSED_SHOT = 2
_FREE_THROW = 3
_REBOUND = 4
_TURNOVER = 5
_FOUL = 6

_OFF_DEF_RE = re.compile(r"Off:(\d+)\s+Def:(\d+)")
_DEFAULT_MIN_RATE_POSS = 200  # below this on-court exposure, rates are left null


@dataclass(frozen=True)
class PlayerSeasonProfile:
    """One player's counting stats, exposure, and derived per-possession rates
    for one season. Rates are ``None`` when on-court exposure is too thin."""

    player_id: int
    season: str
    games: int
    off_possessions: int
    def_possessions: int
    # raw counts (regular field goals from the shot chart; the rest from PBP)
    fga: int
    fgm: int
    fg3a: int
    fg3m: int
    fta: int
    ftm: int
    assists: int
    turnovers: int
    off_rebounds: int
    def_rebounds: int
    steals: int
    blocks: int
    personal_fouls: int
    rim_fga: int
    mid_fga: int
    corner3_fga: int
    # derived rates (points-per-100 style; None when exposure is below the floor)
    usage: float | None
    assist_per100: float | None
    turnover_per100: float | None
    oreb_per100: float | None
    dreb_per100: float | None
    steal_per100: float | None
    block_per100: float | None
    three_rate: float | None  # share of FGA taken from three
    rim_rate: float | None  # share of FGA at the rim
    corner3_rate: float | None  # share of FGA from the corner three
    ft_rate: float | None  # FTA per FGA

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["schema_version"] = PLAYER_PROFILE_SCHEMA_VERSION
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerSeasonProfile:
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**fields)


# --------------------------------------------------------------------------- #
# Raw-payload parsing
# --------------------------------------------------------------------------- #


def _result_rows(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return [], []
    first = sets[0]
    return list(first.get("headers", [])), list(first.get("rowSet", []))


def _event_description(row: dict[str, Any]) -> str:
    return str(
        row.get("HOMEDESCRIPTION")
        or row.get("VISITORDESCRIPTION")
        or row.get("NEUTRALDESCRIPTION")
        or ""
    )


@dataclass
class _SeasonTally:
    """Mutable accumulator for one (player, season) cell."""

    fga: int = 0
    fgm: int = 0
    fg3a: int = 0
    fg3m: int = 0
    fta: int = 0
    ftm: int = 0
    assists: int = 0
    turnovers: int = 0
    off_rebounds: int = 0
    def_rebounds: int = 0
    steals: int = 0
    blocks: int = 0
    personal_fouls: int = 0
    rim_fga: int = 0
    mid_fga: int = 0
    corner3_fga: int = 0


def _accumulate_pbp(
    game: SnapshotGame,
    season: str,
    tally: dict[tuple[int, str], _SeasonTally],
) -> None:
    headers, rows = _result_rows(json.loads(game.pbp_path.read_text(encoding="utf-8")))
    if not headers:
        return
    idx = {h: i for i, h in enumerate(headers)}
    reb_totals: dict[int, tuple[int, int]] = {}

    for raw in rows:
        row = {h: raw[idx[h]] for h in headers}
        etype = row.get("EVENTMSGTYPE")
        p1 = int(row.get("PLAYER1_ID") or 0)
        p2 = int(row.get("PLAYER2_ID") or 0)
        p3 = int(row.get("PLAYER3_ID") or 0)
        desc = _event_description(row)

        if etype == _MADE_SHOT and p2:
            tally[(p2, season)].assists += 1
        elif etype == _FREE_THROW and p1 and "Technical" not in desc:
            tally[(p1, season)].fta += 1
            if not desc.startswith("MISS"):
                tally[(p1, season)].ftm += 1
        elif etype == _TURNOVER:
            if p1:
                tally[(p1, season)].turnovers += 1
            if p2:
                tally[(p2, season)].steals += 1
        elif etype == _MISSED_SHOT and p3:
            tally[(p3, season)].blocks += 1
        elif etype == _FOUL and p1:
            tally[(p1, season)].personal_fouls += 1
        elif etype == _REBOUND and p1:
            match = _OFF_DEF_RE.search(desc)
            if match:
                reb_totals[p1] = (int(match.group(1)), int(match.group(2)))

    for pid, (off, dfn) in reb_totals.items():
        tally[(pid, season)].off_rebounds += off
        tally[(pid, season)].def_rebounds += dfn


def _accumulate_shots(
    game: SnapshotGame,
    season: str,
    tally: dict[tuple[int, str], _SeasonTally],
) -> None:
    for path in (game.home_shots_path, game.away_shots_path):
        headers, rows = _result_rows(json.loads(path.read_text(encoding="utf-8")))
        if not headers:
            continue
        idx = {h: i for i, h in enumerate(headers)}
        for raw in rows:
            row = {h: raw[idx[h]] for h in headers}
            if not int(row.get("SHOT_ATTEMPTED_FLAG") or 0):
                continue
            pid = int(row.get("PLAYER_ID") or 0)
            if not pid:
                continue
            cell = tally[(pid, season)]
            made = int(row.get("SHOT_MADE_FLAG") or 0)
            is_three = str(row.get("SHOT_TYPE", "")).startswith("3")
            zone = str(row.get("SHOT_ZONE_BASIC", ""))
            cell.fga += 1
            cell.fgm += made
            if is_three:
                cell.fg3a += 1
                cell.fg3m += made
                if "Corner 3" in zone:
                    cell.corner3_fga += 1
            if zone == "Restricted Area":
                cell.rim_fga += 1
            elif zone in ("In The Paint (Non-RA)", "Mid-Range"):
                cell.mid_fga += 1


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _share(numer: int, denom: int) -> float | None:
    return None if denom == 0 else numer / denom


def build_player_profiles(
    snapshot: Snapshot,
    stints: StintTable,
    *,
    min_off_possessions: int = _DEFAULT_MIN_RATE_POSS,
) -> list[PlayerSeasonProfile]:
    """One :class:`PlayerSeasonProfile` per (player, season). Only games present
    in both the snapshot and ``stints`` are counted; ``min_off_possessions``
    sets the exposure floor below which rates are left ``None``."""

    def rate(numer: float, denom: int) -> float | None:
        return None if denom < min_off_possessions else numer / denom

    # exposure + season labels + game set from the stints
    off_poss: dict[tuple[int, str], int] = defaultdict(int)
    def_poss: dict[tuple[int, str], int] = defaultdict(int)
    games_seen: dict[tuple[int, str], set[str]] = defaultdict(set)
    stint_games: set[str] = set()
    for stint in stints:
        stint_games.add(stint.game_id)
        season = stint.season
        for pid in stint.offense_player_ids:
            off_poss[(pid, season)] += stint.offensive_possessions
            games_seen[(pid, season)].add(stint.game_id)
        for pid in stint.defense_player_ids:
            def_poss[(pid, season)] += stint.offensive_possessions
            games_seen[(pid, season)].add(stint.game_id)

    tally: dict[tuple[int, str], _SeasonTally] = defaultdict(_SeasonTally)
    for game in snapshot:
        if game.metadata.game_id not in stint_games:
            continue
        season = game.metadata.season
        _accumulate_pbp(game, season, tally)
        _accumulate_shots(game, season, tally)

    keys = sorted(set(off_poss) | set(def_poss) | set(tally))
    profiles: list[PlayerSeasonProfile] = []
    for pid, season in keys:
        cell = tally.get((pid, season), _SeasonTally())
        o = off_poss.get((pid, season), 0)
        d = def_poss.get((pid, season), 0)
        profiles.append(
            PlayerSeasonProfile(
                player_id=pid,
                season=season,
                games=len(games_seen.get((pid, season), set())),
                off_possessions=o,
                def_possessions=d,
                fga=cell.fga,
                fgm=cell.fgm,
                fg3a=cell.fg3a,
                fg3m=cell.fg3m,
                fta=cell.fta,
                ftm=cell.ftm,
                assists=cell.assists,
                turnovers=cell.turnovers,
                off_rebounds=cell.off_rebounds,
                def_rebounds=cell.def_rebounds,
                steals=cell.steals,
                blocks=cell.blocks,
                personal_fouls=cell.personal_fouls,
                rim_fga=cell.rim_fga,
                mid_fga=cell.mid_fga,
                corner3_fga=cell.corner3_fga,
                usage=rate(cell.fga + 0.44 * cell.fta + cell.turnovers, o),
                assist_per100=rate(100.0 * cell.assists, o),
                turnover_per100=rate(100.0 * cell.turnovers, o),
                oreb_per100=rate(100.0 * cell.off_rebounds, o),
                dreb_per100=rate(100.0 * cell.def_rebounds, d),
                steal_per100=rate(100.0 * cell.steals, d),
                block_per100=rate(100.0 * cell.blocks, d),
                three_rate=_share(cell.fg3a, cell.fga),
                rim_rate=_share(cell.rim_fga, cell.fga),
                corner3_rate=_share(cell.corner3_fga, cell.fga),
                ft_rate=_share(cell.fta, cell.fga),
            )
        )
    return profiles


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def write_player_profiles(
    profiles: list[PlayerSeasonProfile], path: str | Path
) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile.to_dict(), sort_keys=True) + "\n")
    return path


def read_player_profiles(path: str | Path) -> list[PlayerSeasonProfile]:
    out: list[PlayerSeasonProfile] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(PlayerSeasonProfile.from_dict(json.loads(line)))
    return out


def build_from_paths(
    snapshot_dir: str | Path,
    stints_path: str | Path,
    *,
    min_off_possessions: int = _DEFAULT_MIN_RATE_POSS,
) -> list[PlayerSeasonProfile]:
    """Convenience: load the snapshot and stint file, then build."""

    from courtgraph.chemistry.stints import read_stints

    snapshot = load_snapshot(snapshot_dir)
    stints = read_stints(stints_path)
    return build_player_profiles(
        snapshot, stints, min_off_possessions=min_off_possessions
    )

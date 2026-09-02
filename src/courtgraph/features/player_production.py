"""Per-(player, stint) offensive production, attributed from the snapshot.

Master plan §45 Phase B needs a teammate's **individual** on-court production,
not the lineup's net rating -- the one estimand the symmetric lineup-value
models cannot separate from the giver's own talent. This module builds it from
the same ``stats_nba_pbpstats`` snapshot the stint file came from: no new
download.

For each offense stint and each of its five offensive players:

* ``fg_points``       -- points from made field goals (shot chart, time-window
  join, same machinery as :func:`courtgraph.features.stint_shots.attribute_shots`);
* ``ft_points``       -- made non-technical free throws (play-by-play);
* ``assisted_points`` -- points of teammates' made field goals this player
  assisted (play-by-play ``PLAYER2_ID``).

``points = fg_points + ft_points``; a **registered** research choice
(``ProductionConfig.assist_credit``) sets how much assisted-teammate scoring is
credited back to the passer, and Phase B reports the result at both 0.0 and the
configured value. A player's own production is never attributed to anyone else.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtgraph.chemistry.stints import StintTable
from courtgraph.features.stint_shots import _match, _period_length, _windows
from courtgraph.ingest.snapshot import Snapshot

PRODUCTION_SCHEMA_VERSION = 1

_MADE_SHOT = 1
_FREE_THROW = 3


@dataclass(frozen=True)
class ProductionConfig:
    """Registered research choices for production attribution."""

    assist_credit: float = 0.5  # fraction of assisted-teammate points to the passer
    exclude_technical_ft: bool = True


@dataclass(frozen=True)
class PlayerStintProduction:
    stint_id: str
    game_id: str
    season: str
    season_index: int
    player_id: int
    team_id: int
    offensive_possessions: int  # the stint's -- the exposure denominator
    fg_points: int
    ft_points: int
    assisted_points: int  # full teammate points assisted (scale by assist_credit)

    @property
    def points(self) -> int:
        return self.fg_points + self.ft_points

    def credited(self, config: ProductionConfig) -> float:
        return self.points + config.assist_credit * self.assisted_points

    def to_dict(self) -> dict[str, Any]:
        return {
            "stint_id": self.stint_id,
            "game_id": self.game_id,
            "season": self.season,
            "season_index": self.season_index,
            "player_id": self.player_id,
            "team_id": self.team_id,
            "offensive_possessions": self.offensive_possessions,
            "fg_points": self.fg_points,
            "ft_points": self.ft_points,
            "assisted_points": self.assisted_points,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlayerStintProduction:
        return cls(
            stint_id=str(d["stint_id"]),
            game_id=str(d["game_id"]),
            season=str(d["season"]),
            season_index=int(d["season_index"]),
            player_id=int(d["player_id"]),
            team_id=int(d["team_id"]),
            offensive_possessions=int(d["offensive_possessions"]),
            fg_points=int(d["fg_points"]),
            ft_points=int(d["ft_points"]),
            assisted_points=int(d["assisted_points"]),
        )


@dataclass(frozen=True)
class ProductionTable:
    rows: tuple[PlayerStintProduction, ...]
    config: ProductionConfig
    fg_events_total: int
    fg_events_matched: int
    ft_events_total: int
    ft_events_matched: int
    assist_events_total: int
    assist_events_matched: int
    off_roster_events: int  # scoring events landing on a non-offense player -- dropped

    @property
    def match_rate(self) -> float:
        tot = self.fg_events_total + self.ft_events_total + self.assist_events_total
        m = self.fg_events_matched + self.ft_events_matched + self.assist_events_matched
        return m / tot if tot else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.rows),
            "config": {
                "assist_credit": self.config.assist_credit,
                "exclude_technical_ft": self.config.exclude_technical_ft,
            },
            "fg_events": [self.fg_events_matched, self.fg_events_total],
            "ft_events": [self.ft_events_matched, self.ft_events_total],
            "assist_events": [self.assist_events_matched, self.assist_events_total],
            "off_roster_events_dropped": self.off_roster_events,
            "match_rate": self.match_rate,
        }


def _clock_seconds(clock: str) -> float | None:
    parts = str(clock).split(":")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]) * 60.0 + float(parts[1])
    except ValueError:
        return None


def _pbp_rows(payload: dict[str, Any]) -> tuple[dict[str, int], list[list[Any]]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return {}, []
    first = sets[0]
    headers = list(first.get("headers", []))
    return {h: i for i, h in enumerate(headers)}, list(first.get("rowSet", []))


def _shot_rows(payload: dict[str, Any]) -> tuple[dict[str, int], list[list[Any]]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return {}, []
    first = sets[0]
    headers = list(first.get("headers", []))
    return {h: i for i, h in enumerate(headers)}, list(first.get("rowSet", []))


def _description(raw: list[Any], idx: dict[str, int]) -> str:
    for col in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
        i = idx.get(col)
        if i is not None and raw[i]:
            return str(raw[i])
    return ""


def attribute_player_production(
    snapshot: Snapshot,
    table: StintTable,
    *,
    config: ProductionConfig | None = None,
) -> ProductionTable:
    cfg = config or ProductionConfig()
    windows = _windows(table)
    stint_by_id = {s.stint_id: s for s in table}
    stint_games = {s.game_id for s in table}

    # (stint_id, player_id) -> [fg_points, ft_points, assisted_points]
    acc: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    fg_t = fg_m = ft_t = ft_m = as_t = as_m = off_roster = 0

    for game in snapshot:
        gid = game.metadata.game_id
        if gid not in stint_games:
            continue

        # 1. made field goals -> fg_points per (stint, shooter)
        for path in (game.home_shots_path, game.away_shots_path):
            idx, rows = _shot_rows(json.loads(path.read_text(encoding="utf-8")))
            if not idx:
                continue
            for raw in rows:
                if not int(raw[idx["SHOT_ATTEMPTED_FLAG"]] or 0):
                    continue
                if not int(raw[idx["SHOT_MADE_FLAG"]] or 0):
                    continue
                fg_t += 1
                team = int(raw[idx["TEAM_ID"]] or 0)
                period = int(raw[idx["PERIOD"]] or 0)
                remaining = 60.0 * float(raw[idx["MINUTES_REMAINING"]] or 0) + float(
                    raw[idx["SECONDS_REMAINING"]] or 0
                )
                elapsed = _period_length(period) - remaining
                spans = windows.get((gid, period, team))
                if not spans:
                    continue
                sid = _match(spans, elapsed)
                if sid is None:
                    continue
                pid = int(raw[idx["PLAYER_ID"]] or 0)
                three = str(raw[idx["SHOT_TYPE"]] or "").startswith("3")
                if pid in stint_by_id[sid].offense_player_ids:
                    acc[(sid, pid)][0] += 3 if three else 2
                    fg_m += 1
                else:
                    off_roster += 1

        # 2. play-by-play -> free throws + assist credit
        idx, rows = _pbp_rows(json.loads(game.pbp_path.read_text(encoding="utf-8")))
        if not idx:
            continue
        for raw in rows:
            etype = raw[idx["EVENTMSGTYPE"]] if "EVENTMSGTYPE" in idx else None
            if etype not in (_MADE_SHOT, _FREE_THROW):
                continue
            period = int(raw[idx["PERIOD"]] or 0)
            pc_i = idx.get("PCTIMESTRING")
            clock = _clock_seconds(raw[pc_i]) if pc_i is not None else None
            if clock is None or period == 0:
                continue
            elapsed = _period_length(period) - clock
            desc = _description(raw, idx)
            t_i = idx.get("PLAYER1_TEAM_ID")
            team = int(raw[t_i] or 0) if t_i is not None else 0
            spans = windows.get((gid, period, team))

            if etype == _FREE_THROW:
                if cfg.exclude_technical_ft and "Technical" in desc:
                    continue
                ft_t += 1
                if not spans:
                    continue
                sid = _match(spans, elapsed)
                if sid is None:
                    continue
                p1 = int(raw[idx["PLAYER1_ID"]] or 0)
                if not p1 or p1 not in stint_by_id[sid].offense_player_ids:
                    off_roster += 1
                    continue
                made = 0 if desc.upper().startswith("MISS") else 1
                acc[(sid, p1)][1] += made
                ft_m += 1
            else:  # made shot -- credit the assister (PLAYER2)
                p2 = int(raw[idx["PLAYER2_ID"]] or 0) if "PLAYER2_ID" in idx else 0
                if not p2:
                    continue
                as_t += 1
                if not spans:
                    continue
                sid = _match(spans, elapsed)
                if sid is None:
                    continue
                if p2 not in stint_by_id[sid].offense_player_ids:
                    off_roster += 1
                    continue
                pts = 3 if "3PT" in desc else 2
                acc[(sid, p2)][2] += pts
                as_m += 1

    # 3. emit a row for every (stint, offensive player) -- zero-fill so the
    #    exposure denominator is complete for a downstream RAPM design
    out: list[PlayerStintProduction] = []
    for stint in table:
        for pid in stint.offense_player_ids:
            fg, ft, ast = acc.get((stint.stint_id, pid), (0, 0, 0))
            out.append(
                PlayerStintProduction(
                    stint_id=stint.stint_id,
                    game_id=stint.game_id,
                    season=stint.season,
                    season_index=stint.season_index,
                    player_id=pid,
                    team_id=stint.offense_team_id,
                    offensive_possessions=stint.offensive_possessions,
                    fg_points=fg,
                    ft_points=ft,
                    assisted_points=ast,
                )
            )

    return ProductionTable(
        rows=tuple(out),
        config=cfg,
        fg_events_total=fg_t,
        fg_events_matched=fg_m,
        ft_events_total=ft_t,
        ft_events_matched=ft_m,
        assist_events_total=as_t,
        assist_events_matched=as_m,
        off_roster_events=off_roster,
    )


def write_production(table: ProductionTable, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": table.summary()}) + "\n")
        for row in table.rows:
            fh.write(json.dumps(row.to_dict()) + "\n")
    return p


def read_production(path: str | Path) -> list[PlayerStintProduction]:
    rows: list[PlayerStintProduction] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                continue
            rows.append(PlayerStintProduction.from_dict(d))
    return rows

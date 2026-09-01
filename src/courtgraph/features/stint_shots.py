"""Attribute shot-chart shots to stints by a time-window join.

Each offense stint (`<game>-P<period>-R<run>-O<team>`) covers a
``[start_time_seconds, next_start_or_period_end)`` window within its own
``(period, offense_team_id)`` sequence -- the sequences differ per team
because a stint boundary is cut whenever *either* team subs. A shot from the
raw ``shotchartdetail`` payload is placed by (game, period, elapsed clock) and
credited to the stint in the **shooter's** team sequence.

No re-ingest and no new download: the same snapshot the stint file was built
from. Shots outside every window (clock rounding, malformed rows) are counted
and dropped, never guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from courtgraph.chemistry.stints import StintTable
from courtgraph.ingest.snapshot import Snapshot

_REGULATION_SECONDS = 720.0
_OVERTIME_SECONDS = 300.0


def _period_length(period: int) -> float:
    return _REGULATION_SECONDS if period <= 4 else _OVERTIME_SECONDS


@dataclass(frozen=True)
class StintShots:
    """Shot aggregates for one offense stint."""

    fga: int
    fgm: int
    fg3a: int
    fg3m: int
    rim_fga: int
    mid_fga: int
    corner3_fga: int
    fg_points: int  # 2 * fg2m + 3 * fg3m

    @property
    def points_per_shot(self) -> float:
        return self.fg_points / self.fga if self.fga else 0.0

    @property
    def rim_share(self) -> float:
        return self.rim_fga / self.fga if self.fga else 0.0

    @property
    def three_share(self) -> float:
        return self.fg3a / self.fga if self.fga else 0.0


@dataclass(frozen=True)
class ShotAttribution:
    per_stint: dict[str, StintShots]
    shots_total: int
    shots_matched: int
    shots_unmatched: int

    @property
    def match_rate(self) -> float:
        return self.shots_matched / self.shots_total if self.shots_total else 0.0


class _Bucket:
    __slots__ = (
        "fga",
        "fgm",
        "fg3a",
        "fg3m",
        "rim_fga",
        "mid_fga",
        "corner3_fga",
        "fg_points",
    )

    def __init__(self) -> None:
        self.fga = self.fgm = self.fg3a = self.fg3m = 0
        self.rim_fga = self.mid_fga = self.corner3_fga = self.fg_points = 0

    def add(self, made: int, three: bool, zone: str) -> None:
        self.fga += 1
        self.fgm += made
        if three:
            self.fg3a += 1
            self.fg3m += made
            self.fg_points += 3 * made
            if "Corner 3" in zone:
                self.corner3_fga += 1
        else:
            self.fg_points += 2 * made
        if zone == "Restricted Area":
            self.rim_fga += 1
        elif zone in ("In The Paint (Non-RA)", "Mid-Range"):
            self.mid_fga += 1

    def freeze(self) -> StintShots:
        return StintShots(
            fga=self.fga,
            fgm=self.fgm,
            fg3a=self.fg3a,
            fg3m=self.fg3m,
            rim_fga=self.rim_fga,
            mid_fga=self.mid_fga,
            corner3_fga=self.corner3_fga,
            fg_points=self.fg_points,
        )


def _windows(
    table: StintTable,
) -> dict[tuple[str, int, int], list[tuple[float, float, str]]]:
    """(game, period, offense_team) -> sorted [(start, end, stint_id), ...]."""

    raw: dict[tuple[str, int, int], list[tuple[float, str]]] = {}
    for stint in table:
        key = (stint.game_id, stint.period, stint.offense_team_id)
        raw.setdefault(key, []).append((stint.start_time_seconds, stint.stint_id))

    out: dict[tuple[str, int, int], list[tuple[float, float, str]]] = {}
    for key, entries in raw.items():
        entries.sort()
        period_end = _period_length(key[1])
        spans: list[tuple[float, float, str]] = []
        for i, (start, sid) in enumerate(entries):
            end = entries[i + 1][0] if i + 1 < len(entries) else period_end
            spans.append((start, end, sid))
        out[key] = spans
    return out


def _shot_rows(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return [], []
    first = sets[0]
    return list(first.get("headers", [])), list(first.get("rowSet", []))


def attribute_shots(snapshot: Snapshot, table: StintTable) -> ShotAttribution:
    windows = _windows(table)
    buckets: dict[str, _Bucket] = {}
    stint_games = {s.game_id for s in table}
    total = matched = 0

    for game in snapshot:
        if game.metadata.game_id not in stint_games:
            continue
        gid = game.metadata.game_id
        for path in (game.home_shots_path, game.away_shots_path):
            headers, rows = _shot_rows(json.loads(path.read_text(encoding="utf-8")))
            if not headers:
                continue
            idx = {h: i for i, h in enumerate(headers)}
            for raw in rows:
                if not int(raw[idx["SHOT_ATTEMPTED_FLAG"]] or 0):
                    continue
                total += 1
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
                matched += 1
                made = int(raw[idx["SHOT_MADE_FLAG"]] or 0)
                three = str(raw[idx["SHOT_TYPE"]] or "").startswith("3")
                zone = str(raw[idx["SHOT_ZONE_BASIC"]] or "")
                buckets.setdefault(sid, _Bucket()).add(made, three, zone)

    per_stint = {sid: b.freeze() for sid, b in buckets.items()}
    return ShotAttribution(
        per_stint=per_stint,
        shots_total=total,
        shots_matched=matched,
        shots_unmatched=total - matched,
    )


def _match(spans: list[tuple[float, float, str]], elapsed: float) -> str | None:
    for start, end, sid in spans:
        if start <= elapsed < end:
            return sid
    # a shot at the exact tail of the last window
    if spans and abs(elapsed - spans[-1][1]) < 1.0:
        return spans[-1][2]
    return None

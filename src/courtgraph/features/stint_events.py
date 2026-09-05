"""Attribute turnovers and assisted makes to stints, for the mechanistic
outcome ladder's turnover-rate and assist-rate outcomes.

Same time-window join as :mod:`courtgraph.features.stint_shots` --
(game, period, elapsed clock) placed against each offense stint's
``[start, next_start_or_period_end)`` window -- applied to raw
``playbyplayv2`` events instead of ``shotchartdetail`` shots, since turnovers
and assists are not on the shot chart.

Research choices, made explicit (`AGENTS.md`: store configuration, don't hide
it in constants):

* ``turnover_rate = turnovers / offensive_possessions`` -- the stint's own
  possession count is already a validated field on every stint record (the
  same exposure denominator RAPM training uses), so no additional join is
  needed for the denominator, only for attributing individual turnover
  events to the right stint.
* ``assist_rate = assisted_fgm / fgm`` -- both counted from the same
  play-by-play walk (``EVENTMSGTYPE == 1`` made-shot rows; assisted when
  ``PLAYER2_ID`` is present), so the ratio is self-contained and does not
  depend on the separate shot-chart-based ``StintShots.fgm``.

A turnover or made-shot event outside every window (clock rounding,
malformed rows) is counted and dropped, never guessed -- the same fail-closed
posture as `stint_shots.py` and `player_production.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from courtgraph.chemistry.stints import StintTable
from courtgraph.features.stint_shots import _match, _period_length, _windows
from courtgraph.ingest.snapshot import Snapshot

_MADE_SHOT = 1
_TURNOVER = 5


@dataclass(frozen=True)
class StintPlayEvents:
    """Turnover and assisted-make aggregates for one offense stint."""

    turnovers: int
    offensive_possessions: int
    fgm: int
    assisted_fgm: int

    @property
    def turnover_rate(self) -> float:
        return (
            self.turnovers / self.offensive_possessions
            if self.offensive_possessions
            else 0.0
        )

    @property
    def assist_rate(self) -> float:
        return self.assisted_fgm / self.fgm if self.fgm else 0.0


@dataclass(frozen=True)
class EventAttribution:
    per_stint: dict[str, StintPlayEvents]
    events_total: int
    events_matched: int
    events_unmatched: int

    @property
    def match_rate(self) -> float:
        return self.events_matched / self.events_total if self.events_total else 0.0


class _Bucket:
    __slots__ = ("turnovers", "fgm", "assisted_fgm")

    def __init__(self) -> None:
        self.turnovers = self.fgm = self.assisted_fgm = 0

    def freeze(self, offensive_possessions: int) -> StintPlayEvents:
        return StintPlayEvents(
            turnovers=self.turnovers,
            offensive_possessions=offensive_possessions,
            fgm=self.fgm,
            assisted_fgm=self.assisted_fgm,
        )


def _pbp_rows(payload: dict[str, Any]) -> tuple[dict[str, int], list[list[Any]]]:
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


def attribute_play_events(snapshot: Snapshot, table: StintTable) -> EventAttribution:
    windows = _windows(table)
    buckets: dict[str, _Bucket] = {}
    stint_games = {s.game_id for s in table}
    total = matched = 0

    for game in snapshot:
        gid = game.metadata.game_id
        if gid not in stint_games:
            continue
        idx, rows = _pbp_rows(json.loads(game.pbp_path.read_text(encoding="utf-8")))
        if not idx:
            continue

        for raw in rows:
            etype = raw[idx["EVENTMSGTYPE"]] if "EVENTMSGTYPE" in idx else None
            if etype not in (_MADE_SHOT, _TURNOVER):
                continue
            period = int(raw[idx["PERIOD"]] or 0)
            pc_i = idx.get("PCTIMESTRING")
            clock = _clock_seconds(str(raw[pc_i])) if pc_i is not None else None
            if clock is None or period == 0:
                continue
            elapsed = _period_length(period) - clock

            t_i = idx.get("PLAYER1_TEAM_ID")
            team = int(raw[t_i] or 0) if t_i is not None else 0
            spans = windows.get((gid, period, team))

            total += 1
            if not spans:
                continue
            sid = _match(spans, elapsed)
            if sid is None:
                continue
            matched += 1
            bucket = buckets.setdefault(sid, _Bucket())

            if etype == _TURNOVER:
                bucket.turnovers += 1
            else:  # made shot
                bucket.fgm += 1
                p2_i = idx.get("PLAYER2_ID")
                if p2_i is not None and int(raw[p2_i] or 0):
                    bucket.assisted_fgm += 1

    per_stint = {
        stint.stint_id: buckets[stint.stint_id].freeze(stint.offensive_possessions)
        for stint in table
        if stint.stint_id in buckets
    }
    return EventAttribution(
        per_stint=per_stint,
        events_total=total,
        events_matched=matched,
        events_unmatched=total - matched,
    )


def _clock_seconds(clock: str) -> float | None:
    parts = clock.split(":")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]) * 60.0 + float(parts[1])
    except ValueError:
        return None

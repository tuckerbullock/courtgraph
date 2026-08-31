"""The ``pbpstats`` boundary: reconstruct possessions from a working copy.

``pbpstats`` is used **only here**, **only in file mode**, and **only as a
tool** -- its possession/lineup output is derived data that CourtGraph then
validates independently (``DATA_SOURCES.md`` 5, master plan 7.7). Nothing in
this module is trusted downstream without passing :mod:`courtgraph.ingest.validate`.

A hard offline guard wraps every reconstruction: if any code path (e.g.
``pbpstats`` falling back to a stats.nba.com boxscore request for period
starters) attempts a socket connection, it is turned into
:class:`IngestNetworkAttempt` and the game is quarantined -- the adapter never
makes a network request.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REAL_ENDING_EVENT_TYPES = frozenset(
    {"FieldGoal", "FreeThrow", "Rebound", "Turnover", "JumpBall"}
)
# Events that can precede live play in a possession without being "live" -- a
# substitution among these is the normal case, not a split-lineup possession.
_NON_LIVE_EVENT_TYPES = frozenset(
    {"Substitution", "Timeout", "StartOfPeriod", "EndOfPeriod", "Replay"}
)


class IngestNetworkAttempt(RuntimeError):
    """Raised when file-mode reconstruction tries to reach the network."""


class PossessionReconstructionError(RuntimeError):
    """Wraps any ``pbpstats`` failure for a single game (kept, not fatal)."""

    def __init__(self, game_id: str, cause: BaseException) -> None:
        self.game_id = game_id
        self.cause_type = type(cause).__name__
        super().__init__(f"game {game_id}: {self.cause_type}: {cause}")


@dataclass(frozen=True)
class PossessionView:
    """A plain, typed snapshot of one reconstructed possession.

    Deliberately free of ``pbpstats`` objects so the rest of the adapter has a
    stable, inspectable contract.
    """

    game_id: str
    period: int
    number: int
    sequence_index: int  # 0-based position in the game's reconstructed possessions
    offense_team_id: int
    defense_team_id: int
    lineups: dict[int, frozenset[int]]  # 5 per team at the first *live* event
    end_lineups: dict[int, frozenset[int]]  # 5 per team at the last event
    lineup_changed_during_live_play: bool  # a sub happened between live events
    start_seconds_remaining: float
    end_seconds_remaining: float
    start_score: dict[int, int]  # team_id -> points before this possession
    end_score: dict[int, int]  # team_id -> points after this possession
    offense_technical_ft_points: int
    technical_ft_points_by_team: dict[int, int]  # any team's made technical FTs here
    has_real_ending_event: bool
    event_type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def offense_points_raw(self) -> int:
        return int(self.end_score.get(self.offense_team_id, 0)) - int(
            self.start_score.get(self.offense_team_id, 0)
        )

    @property
    def is_split_lineup(self) -> bool:
        """True if the ten on the floor changed at any point during live play.

        Catches a player subbing out, another taking part in live play, and the
        original returning before the possession-ending event -- which a
        first-vs-last comparison misses.
        """

        return self.lineup_changed_during_live_play or self.lineups != self.end_lineups

    def lineup_5(self, team_id: int) -> frozenset[int]:
        return self.lineups.get(team_id, frozenset())


@contextmanager
def offline_guard() -> Iterator[None]:
    """Turn any socket connect / DNS attempt into :class:`IngestNetworkAttempt`."""

    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def _blocked_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise IngestNetworkAttempt("file-mode ingestion attempted a network connection")

    def _blocked_getaddrinfo(*args: Any, **kwargs: Any) -> Any:
        raise IngestNetworkAttempt("file-mode ingestion attempted a DNS lookup")

    _patch(socket.socket, "connect", _blocked_connect)
    _patch(socket, "getaddrinfo", _blocked_getaddrinfo)
    try:
        yield
    finally:
        _patch(socket.socket, "connect", real_connect)
        _patch(socket, "getaddrinfo", real_getaddrinfo)


def _patch(obj: object, name: str, value: object) -> None:
    """Rebind an attribute (used only to install/remove the offline guard)."""

    setattr(obj, name, value)


def _clock_to_seconds(clock: str) -> float:
    minutes, _, seconds = clock.partition(":")
    return float(minutes) * 60.0 + float(seconds)


def _lineups_from_event(event: Any) -> dict[int, frozenset[int]]:
    current = getattr(event, "current_players", {}) or {}
    return {
        int(team): frozenset(int(p) for p in players)
        for team, players in current.items()
    }


def _score_dict(event: Any) -> dict[int, int]:
    return {int(team): int(points) for team, points in dict(event.score).items()}


def reconstruct_game(work_dir: str | Path, game_id: str) -> list[PossessionView]:
    """Reconstruct one game's possessions from files in ``work_dir``.

    :raises IngestNetworkAttempt: if reconstruction touches the network.
    :raises PossessionReconstructionError: for any other ``pbpstats`` failure.
    """

    from pbpstats.client import Client  # lazy: keeps `doctor` third-party-free

    settings = {
        "dir": str(work_dir),
        "Possessions": {"source": "file", "data_provider": "stats_nba"},
    }
    with offline_guard():
        try:
            game = Client(settings).Game(game_id)
            possessions = list(game.possessions.items)
        except IngestNetworkAttempt:
            raise
        except Exception as exc:  # noqa: BLE001 - kept as a quarantine, never fatal
            raise PossessionReconstructionError(game_id, exc) from exc

    views: list[PossessionView] = []
    for sequence_index, possession in enumerate(possessions):
        events = list(possession.events)
        last = events[-1]
        first_live_idx = next(
            (
                i
                for i, event in enumerate(events)
                if _real_event_base_name(event) not in _NON_LIVE_EVENT_TYPES
            ),
            0,
        )
        first_live = events[first_live_idx]

        # Track every ten-player set from the first live event onward: any
        # change (a sub between live events, even one later undone) makes the
        # possession's lineup attribution ambiguous.
        live_lineups = [_lineups_from_event(e) for e in events[first_live_idx:]]
        base_lineup = live_lineups[0] if live_lineups else {}
        lineup_changed_during_live_play = any(
            candidate and candidate != base_lineup for candidate in live_lineups[1:]
        )

        # Score carried into the possession: the previous possession's final
        # score, or -- for a period opener -- the StartOfPeriod event's score.
        # (Never a mid-possession event, whose ``.score`` already includes the
        # points we are trying to measure.)
        prev = getattr(possession, "previous_possession", None)
        start_score = (
            _score_dict(prev.events[-1]) if prev is not None else _score_dict(events[0])
        )

        raw_offense = getattr(possession, "offense_team_id", None)
        offense_for_tech = int(raw_offense) if raw_offense else 0

        type_counts: dict[str, int] = {}
        tech_ft_points = 0
        tech_ft_by_team: dict[int, int] = {}
        has_real_ending = False
        for event in events:
            base = _real_event_base_name(event)
            type_counts[base] = type_counts.get(base, 0) + 1
            if base in _REAL_ENDING_EVENT_TYPES:
                has_real_ending = True
            if (
                base == "FreeThrow"
                and getattr(event, "is_technical_ft", False)
                and getattr(event, "is_made", False)
            ):
                team = int(getattr(event, "team_id", 0))
                tech_ft_by_team[team] = tech_ft_by_team.get(team, 0) + 1
                if team == offense_for_tech:
                    tech_ft_points += 1

        offense = offense_for_tech
        end_lineups = _lineups_from_event(last)
        others = sorted(t for t in end_lineups if t != offense)
        defense = others[0] if others else offense

        views.append(
            PossessionView(
                game_id=game_id,
                period=int(possession.period),
                number=int(possession.number),
                sequence_index=sequence_index,
                offense_team_id=offense,
                defense_team_id=int(defense),
                lineups=_lineups_from_event(first_live),
                end_lineups=end_lineups,
                lineup_changed_during_live_play=lineup_changed_during_live_play,
                start_seconds_remaining=_clock_to_seconds(possession.start_time),
                end_seconds_remaining=_clock_to_seconds(possession.end_time),
                start_score=start_score,
                end_score=_score_dict(last),
                offense_technical_ft_points=tech_ft_points,
                technical_ft_points_by_team=tech_ft_by_team,
                has_real_ending_event=has_real_ending,
                event_type_counts=type_counts,
            )
        )
    return views


def _real_event_base_name(event: Any) -> str:
    """Map a ``pbpstats`` event class to a provider-agnostic base type name."""

    for base in (
        "FieldGoal",
        "FreeThrow",
        "Rebound",
        "Turnover",
        "JumpBall",
        "Substitution",
        "Timeout",
        "Foul",
        "Violation",
        "Ejection",
        "Replay",
        "StartOfPeriod",
        "EndOfPeriod",
    ):
        if base.lower() in type(event).__name__.lower():
            return base
    return type(event).__name__

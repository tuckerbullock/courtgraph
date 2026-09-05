"""Turnover / assisted-make -> stint attribution by time-window join."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

_GID = "0022300001"
_MADE_SHOT = 1
_TURNOVER = 5

_PBP_HEADERS = [
    "GAME_ID",
    "EVENTMSGTYPE",
    "PERIOD",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "PLAYER1_ID",
    "PLAYER1_TEAM_ID",
    "PLAYER2_ID",
]


def _event(
    etype: int, period: int, elapsed: float, *, p1: int, t1: int, p2: int = 0
) -> dict[str, Any]:
    plen = 720.0 if period <= 4 else 300.0
    rem = plen - elapsed
    return {
        "GAME_ID": _GID,
        "EVENTMSGTYPE": etype,
        "PERIOD": period,
        "PCTIMESTRING": f"{int(rem // 60)}:{int(rem % 60):02d}",
        "HOMEDESCRIPTION": "",
        "PLAYER1_ID": p1,
        "PLAYER1_TEAM_ID": t1,
        "PLAYER2_ID": p2,
    }


def _payload(headers: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resultSets": [
            {
                "headers": headers,
                "rowSet": [[r.get(h, 0) for h in headers] for r in rows],
            }
        ]
    }


def _write_snapshot(root: Path, pbp: list[dict[str, Any]]) -> None:
    (root / "pbp").mkdir(parents=True)
    (root / "game_details").mkdir(parents=True)
    (root / "pbp" / f"stats_{_GID}.json").write_text(
        json.dumps(_payload(_PBP_HEADERS, pbp))
    )
    empty_shots = json.dumps({"resultSets": [{"headers": ["GAME_ID"], "rowSet": []}]})
    (root / "game_details" / f"stats_home_shots_{_GID}.json").write_text(empty_shots)
    (root / "game_details" / f"stats_away_shots_{_GID}.json").write_text(empty_shots)
    (root / "courtgraph_snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_format": "stats_nba_pbpstats/v1",
                "games": [
                    {
                        "game_id": _GID,
                        "game_date": "2024-01-10",
                        "season": "2023-24",
                        "season_type": "Regular Season",
                        "home_team_id": 10,
                        "away_team_id": 20,
                    }
                ],
            }
        )
    )


def _stints() -> Any:
    from courtgraph.chemistry.stints import Stint, StintTable

    def mk(sid: str, team: int, opp: int, start: float) -> Stint:
        return Stint(
            stint_id=sid,
            game_id=_GID,
            game_date="2024-01-10",
            season="2023-24",
            season_index=0,
            period=1,
            start_time_seconds=start,
            offense_team_id=team,
            defense_team_id=opp,
            offense_player_ids=(1, 2, 3, 4, 5) if team == 10 else (6, 7, 8, 9, 10),
            defense_player_ids=(6, 7, 8, 9, 10) if team == 10 else (1, 2, 3, 4, 5),
            offensive_possessions=10,
            points_scored=10,
            home_offense=(team == 10),
            score_margin_offense=0,
            playoff=False,
            days_rest_offense=1,
            garbage_time_weight=1.0,
        )

    return StintTable.from_stints(
        [
            # team 10 offense windows: [0, 300) and [300, 720)
            mk(f"{_GID}-P1-R001-O10", 10, 20, 0.0),
            mk(f"{_GID}-P1-R002-O10", 10, 20, 300.0),
        ]
    )


class StintPlayEventAttributionTests(unittest.TestCase):
    def test_turnovers_and_assists_land_in_the_right_window(self) -> None:
        from courtgraph.features.stint_events import attribute_play_events
        from courtgraph.ingest.snapshot import load_snapshot

        pbp = [
            # R001: one assisted make, one turnover
            _event(_MADE_SHOT, 1, 100.0, p1=1, t1=10, p2=2),
            _event(_TURNOVER, 1, 150.0, p1=3, t1=10),
            # R002: one unassisted make
            _event(_MADE_SHOT, 1, 500.0, p1=1, t1=10),
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_snapshot(root, pbp)
            att = attribute_play_events(load_snapshot(root), _stints())

        self.assertEqual(att.events_total, 3)
        self.assertEqual(att.events_matched, 3)
        r1 = att.per_stint[f"{_GID}-P1-R001-O10"]
        self.assertEqual(r1.fgm, 1)
        self.assertEqual(r1.assisted_fgm, 1)
        self.assertEqual(r1.turnovers, 1)
        self.assertEqual(r1.offensive_possessions, 10)
        self.assertAlmostEqual(r1.assist_rate, 1.0)
        self.assertAlmostEqual(r1.turnover_rate, 0.1)

        r2 = att.per_stint[f"{_GID}-P1-R002-O10"]
        self.assertEqual(r2.fgm, 1)
        self.assertEqual(r2.assisted_fgm, 0)
        self.assertEqual(r2.turnovers, 0)
        self.assertAlmostEqual(r2.assist_rate, 0.0)
        self.assertAlmostEqual(r2.turnover_rate, 0.0)

    def test_event_outside_every_window_is_dropped_not_guessed(self) -> None:
        from courtgraph.features.stint_events import attribute_play_events
        from courtgraph.ingest.snapshot import load_snapshot

        # a period-5 (OT) turnover with no matching stint window
        pbp = [_event(_TURNOVER, 5, 100.0, p1=1, t1=10)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_snapshot(root, pbp)
            att = attribute_play_events(load_snapshot(root), _stints())
        self.assertEqual(att.events_total, 1)
        self.assertEqual(att.events_matched, 0)
        self.assertEqual(att.events_unmatched, 1)
        self.assertEqual(att.per_stint, {})

    def test_stint_with_zero_possessions_has_zero_rates_not_a_crash(self) -> None:
        from courtgraph.features.stint_events import StintPlayEvents

        empty = StintPlayEvents(
            turnovers=0, offensive_possessions=0, fgm=0, assisted_fgm=0
        )
        self.assertEqual(empty.turnover_rate, 0.0)
        self.assertEqual(empty.assist_rate, 0.0)


if __name__ == "__main__":
    unittest.main()

"""Shot -> stint attribution by time-window join."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

if TYPE_CHECKING:
    from courtgraph.chemistry.stints import StintTable

_SHOT_HEADERS = [
    "GAME_ID",
    "TEAM_ID",
    "PLAYER_ID",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ATTEMPTED_FLAG",
    "SHOT_MADE_FLAG",
]


def _shot(
    team: int,
    period: int,
    elapsed: float,
    *,
    three: bool = False,
    zone: str = "Mid-Range",
    made: int = 0,
) -> dict[str, object]:
    period_len = 720.0 if period <= 4 else 300.0
    remaining = period_len - elapsed
    return {
        "TEAM_ID": team,
        "PERIOD": period,
        "MINUTES_REMAINING": int(remaining // 60),
        "SECONDS_REMAINING": int(remaining % 60),
        "SHOT_TYPE": "3PT Field Goal" if three else "2PT Field Goal",
        "SHOT_ZONE_BASIC": zone,
        "SHOT_ATTEMPTED_FLAG": 1,
        "SHOT_MADE_FLAG": made,
    }


def _shots_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "resultSets": [
            {
                "headers": _SHOT_HEADERS,
                "rowSet": [[r.get(h, 0) for h in _SHOT_HEADERS] for r in rows],
            }
        ]
    }


def _write_snapshot(
    root: Path,
    home_shots: list[dict[str, object]],
    away_shots: list[dict[str, object]],
) -> None:
    (root / "pbp").mkdir(parents=True)
    (root / "game_details").mkdir(parents=True)
    gid = "0022300001"
    (root / "pbp" / f"stats_{gid}.json").write_text(
        json.dumps({"resultSets": [{"headers": ["GAME_ID"], "rowSet": []}]})
    )
    (root / "game_details" / f"stats_home_shots_{gid}.json").write_text(
        json.dumps(_shots_payload(home_shots))
    )
    (root / "game_details" / f"stats_away_shots_{gid}.json").write_text(
        json.dumps(_shots_payload(away_shots))
    )
    (root / "courtgraph_snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_format": "stats_nba_pbpstats/v1",
                "games": [
                    {
                        "game_id": gid,
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


def _stints() -> StintTable:
    from courtgraph.chemistry.stints import Stint, StintTable

    def mk(sid: str, team: int, opp: int, start: float) -> Stint:
        return Stint(
            stint_id=sid,
            game_id="0022300001",
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
            mk("0022300001-P1-R001-O10", 10, 20, 0.0),
            mk("0022300001-P1-R002-O10", 10, 20, 300.0),
            # team 20 offense windows: [0, 400) and [400, 720)
            mk("0022300001-P1-R001-O20", 20, 10, 0.0),
            mk("0022300001-P1-R002-O20", 20, 10, 400.0),
        ]
    )


class StintShotAttributionTests(unittest.TestCase):
    def test_shots_land_in_the_right_window_and_team(self) -> None:
        from courtgraph.features.stint_shots import attribute_shots
        from courtgraph.ingest.snapshot import load_snapshot

        home = [
            _shot(10, 1, 100.0, zone="Restricted Area", made=1),  # -> O10 R001
            _shot(10, 1, 250.0, three=True, zone="Above the Break 3"),  # -> O10 R001
            _shot(10, 1, 500.0, zone="Mid-Range", made=1),  # -> O10 R002
        ]
        away = [
            _shot(20, 1, 50.0, zone="Restricted Area"),  # -> O20 R001
            _shot(20, 1, 600.0, three=True, zone="Left Corner 3", made=1),  # O20 R002
            _shot(20, 1, 719.7, zone="Mid-Range"),  # tail of last window -> O20 R002
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_snapshot(root, home, away)
            att = attribute_shots(load_snapshot(root), _stints())

        self.assertEqual(att.shots_total, 6)
        self.assertEqual(att.shots_matched, 6)
        r1 = att.per_stint["0022300001-P1-R001-O10"]
        self.assertEqual(r1.fga, 2)
        self.assertEqual(r1.rim_fga, 1)
        self.assertEqual(r1.fg3a, 1)
        self.assertEqual(r1.fg_points, 2)  # the made rim 2
        r2 = att.per_stint["0022300001-P1-R002-O10"]
        self.assertEqual(r2.fga, 1)
        self.assertEqual(r2.fg_points, 2)
        a2 = att.per_stint["0022300001-P1-R002-O20"]
        self.assertEqual(a2.fga, 2)
        self.assertEqual(a2.corner3_fga, 1)
        self.assertEqual(a2.fg_points, 3)
        self.assertAlmostEqual(r1.rim_share, 0.5)

    def test_shot_outside_every_window_is_dropped_not_guessed(self) -> None:
        from courtgraph.features.stint_shots import attribute_shots
        from courtgraph.ingest.snapshot import load_snapshot

        # a period-5 (OT) shot with no matching stint window
        home = [_shot(10, 5, 100.0, zone="Mid-Range")]
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_snapshot(root, home, [])
            att = attribute_shots(load_snapshot(root), _stints())
        self.assertEqual(att.shots_total, 1)
        self.assertEqual(att.shots_matched, 0)
        self.assertEqual(att.shots_unmatched, 1)
        self.assertEqual(att.per_stint, {})


if __name__ == "__main__":
    unittest.main()

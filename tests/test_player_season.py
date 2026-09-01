"""Per-(player, season) role profiles derived from a snapshot + stint file."""

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

_PBP_HEADERS = [
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "HOMEDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "VISITORDESCRIPTION",
    "PLAYER1_ID",
    "PLAYER2_ID",
    "PLAYER3_ID",
]
_SHOT_HEADERS = [
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_DISTANCE",
    "SHOT_ATTEMPTED_FLAG",
    "SHOT_MADE_FLAG",
]


def _pbp(rows: list[dict[str, object]]) -> dict[str, object]:
    row_set = [[r.get(h, 0) for h in _PBP_HEADERS] for r in rows]
    return {"resultSets": [{"headers": _PBP_HEADERS, "rowSet": row_set}]}


def _shots(rows: list[dict[str, object]]) -> dict[str, object]:
    row_set = [[r.get(h, 0) for h in _SHOT_HEADERS] for r in rows]
    return {"resultSets": [{"headers": _SHOT_HEADERS, "rowSet": row_set}]}


def _ev(
    msg: int, p1: int = 0, p2: int = 0, p3: int = 0, desc: str = ""
) -> dict[str, object]:
    return {
        "EVENTMSGTYPE": msg,
        "PLAYER1_ID": p1,
        "PLAYER2_ID": p2,
        "PLAYER3_ID": p3,
        "VISITORDESCRIPTION": desc,
    }


def _shot(
    pid: int, three: bool, zone: str, made: int, dist: int = 5
) -> dict[str, object]:
    return {
        "PLAYER_ID": pid,
        "SHOT_TYPE": "3PT Field Goal" if three else "2PT Field Goal",
        "SHOT_ZONE_BASIC": zone,
        "SHOT_DISTANCE": dist,
        "SHOT_ATTEMPTED_FLAG": 1,
        "SHOT_MADE_FLAG": made,
    }


def _write_mini_snapshot(root: Path) -> None:
    (root / "pbp").mkdir(parents=True)
    (root / "game_details").mkdir(parents=True)
    games = []
    for gid, season in (("0022300001", "2023-24"), ("0022300002", "2023-24")):
        pbp_rows = [
            # A: 3 assists (made shots crediting player 10 as assister)
            _ev(1, p1=20, p2=10),
            _ev(1, p1=21, p2=10),
            _ev(1, p1=22, p2=10),
            # A: 2 turnovers, one with a steal by 30
            _ev(5, p1=10),
            _ev(5, p1=10, p2=30),
            # A: 4 FTA, 3 made
            _ev(3, p1=10, desc="Player Free Throw 1 of 2 (1 PTS)"),
            _ev(3, p1=10, desc="Player Free Throw 2 of 2 (2 PTS)"),
            _ev(3, p1=10, desc="Player Free Throw 1 of 2 (3 PTS)"),
            _ev(3, p1=10, desc="MISS Player Free Throw 2 of 2"),
            # B: a technical FT that must NOT count toward FTA
            _ev(3, p1=11, desc="Player Free Throw Technical"),
            # B: final rebound line of the game -> Off:2 Def:5
            _ev(4, p1=11, desc="Player REBOUND (Off:1 Def:3)"),
            _ev(4, p1=11, desc="Player REBOUND (Off:2 Def:5)"),
            # B: one block on a missed shot by 22
            _ev(2, p1=22, p3=11),
            # a team turnover (PLAYER1_ID 0) -> ignored
            _ev(5, p1=0),
        ]
        shot_rows_home = [
            # A: 10 FGA -> 4 threes (1 corner), 2 rim, 4 mid
            *[_shot(10, True, "Above the Break 3", i % 2) for i in range(3)],
            _shot(10, True, "Left Corner 3", 1),
            _shot(10, False, "Restricted Area", 1),
            _shot(10, False, "Restricted Area", 0),
            *[_shot(10, False, "Mid-Range", 0) for _ in range(4)],
        ]
        shot_rows_away = [
            # B: 6 FGA, all corner threes, 2 made
            *[_shot(11, True, "Right Corner 3", 1 if i < 2 else 0) for i in range(6)],
        ]
        (root / "pbp" / f"stats_{gid}.json").write_text(json.dumps(_pbp(pbp_rows)))
        (root / "game_details" / f"stats_home_shots_{gid}.json").write_text(
            json.dumps(_shots(shot_rows_home))
        )
        (root / "game_details" / f"stats_away_shots_{gid}.json").write_text(
            json.dumps(_shots(shot_rows_away))
        )
        games.append(
            {
                "game_id": gid,
                "game_date": "2024-01-10",
                "season": season,
                "season_type": "Regular Season",
                "home_team_id": 1,
                "away_team_id": 2,
            }
        )
    (root / "courtgraph_snapshot.json").write_text(
        json.dumps({"snapshot_format": "stats_nba_pbpstats/v1", "games": games})
    )


def _stint_table(off_poss_per_game: int) -> StintTable:
    from courtgraph.chemistry.stints import Stint, StintTable

    stints = []
    for gid in ("0022300001", "0022300002"):
        stints.append(
            Stint(
                stint_id=f"{gid}-1",
                game_id=gid,
                game_date="2024-01-10",
                season="2023-24",
                season_index=0,
                period=1,
                start_time_seconds=0.0,
                offense_team_id=1,
                defense_team_id=2,
                offense_player_ids=(10, 11, 12, 13, 14),
                defense_player_ids=(15, 16, 17, 18, 19),
                offensive_possessions=off_poss_per_game,
                points_scored=off_poss_per_game,
                home_offense=True,
                score_margin_offense=0,
                playoff=False,
                days_rest_offense=1,
                garbage_time_weight=1.0,
            )
        )
    return StintTable.from_stints(stints)


class PlayerSeasonProfileTests(unittest.TestCase):
    def test_counts_and_rates_match_the_hand_built_snapshot(self) -> None:
        from courtgraph.features.player_season import (
            build_player_profiles,
            read_player_profiles,
            write_player_profiles,
        )
        from courtgraph.ingest.snapshot import load_snapshot

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_mini_snapshot(root)
            snapshot = load_snapshot(root)
            # 600 offensive possessions each game -> 1200 for player 10 (above
            # the 200 floor); low floor so rates are populated.
            profiles = build_player_profiles(
                snapshot, _stint_table(600), min_off_possessions=200
            )
            by = {(p.player_id, p.season): p for p in profiles}

            a = by[(10, "2023-24")]
            # counts double because the two games are identical
            self.assertEqual(a.assists, 6)
            self.assertEqual(a.turnovers, 4)
            self.assertEqual(a.fta, 8)  # technical FT excluded
            self.assertEqual(a.ftm, 6)
            self.assertEqual(a.fga, 20)
            self.assertEqual(a.fg3a, 8)
            self.assertEqual(a.rim_fga, 4)
            self.assertEqual(a.corner3_fga, 2)
            self.assertEqual(a.off_possessions, 1200)
            self.assertEqual(a.games, 2)
            assert a.three_rate is not None and a.rim_rate is not None
            assert a.usage is not None and a.assist_per100 is not None
            self.assertAlmostEqual(a.three_rate, 8 / 20)
            self.assertAlmostEqual(a.rim_rate, 4 / 20)
            self.assertAlmostEqual(a.usage, (20 + 0.44 * 8 + 4) / 1200, places=6)
            self.assertAlmostEqual(a.assist_per100, 600.0 / 1200)

            b = by[(11, "2023-24")]
            self.assertEqual(b.fta, 0)  # only a technical, excluded
            self.assertEqual(b.off_rebounds, 4)  # Off:2 per game x2
            self.assertEqual(b.def_rebounds, 10)  # Def:5 per game x2
            self.assertEqual(b.blocks, 2)
            self.assertEqual(b.fg3a, 12)
            self.assertEqual(b.corner3_rate, 1.0)

            path = Path(tmp) / "profiles.jsonl"
            write_player_profiles(profiles, path)
            restored = read_player_profiles(path)
            self.assertEqual(
                {(p.player_id, p.season) for p in restored},
                {(p.player_id, p.season) for p in profiles},
            )
            self.assertEqual(restored[0].to_dict(), profiles[0].to_dict())

    def test_cli_writes_profiles_and_reports_a_summary(self) -> None:
        from io import StringIO

        from courtgraph.chemistry.stints import write_stints
        from courtgraph.cli import main
        from courtgraph.features.player_season import read_player_profiles

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_mini_snapshot(root)
            stints_path = Path(tmp) / "stints.jsonl"
            write_stints(_stint_table(600), stints_path)
            out_path = Path(tmp) / "profiles.jsonl"

            buf = StringIO()
            code = main(
                [
                    "player-features",
                    "--snapshot-dir",
                    str(root),
                    "--stints",
                    str(stints_path),
                    "--out",
                    str(out_path),
                    "--min-possessions",
                    "200",
                    "--json",
                ],
                output=buf,
            )
            self.assertEqual(code, 0)
            self.assertTrue(out_path.is_file())
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["seasons"], ["2023-24"])
            self.assertGreaterEqual(payload["with_rates"], 1)
            self.assertEqual(len(read_player_profiles(out_path)), payload["profiles"])

    def test_rates_are_none_below_the_exposure_floor(self) -> None:
        from courtgraph.features.player_season import build_player_profiles
        from courtgraph.ingest.snapshot import load_snapshot

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "snap"
            _write_mini_snapshot(root)
            profiles = build_player_profiles(
                load_snapshot(root), _stint_table(50), min_off_possessions=200
            )
            a = next(p for p in profiles if p.player_id == 10)
            self.assertEqual(a.off_possessions, 100)
            self.assertIsNone(a.usage)
            self.assertIsNone(a.assist_per100)
            # shot-share ratios do not depend on possession exposure
            self.assertIsNotNone(a.three_rate)


if __name__ == "__main__":
    unittest.main()

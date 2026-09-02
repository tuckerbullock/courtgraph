"""Per-(player, stint) production attribution from the snapshot."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

_GID = "0022300001"
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
_PBP_HEADERS = [
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "PERIOD",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "PLAYER1_ID",
    "PLAYER1_TEAM_ID",
    "PLAYER2_ID",
    "PLAYER2_TEAM_ID",
]


def _shot(
    team: int, pid: int, period: int, elapsed: float, *, made: int, three: bool = False
) -> dict[str, Any]:
    plen = 720.0 if period <= 4 else 300.0
    rem = plen - elapsed
    return {
        "GAME_ID": _GID,
        "TEAM_ID": team,
        "PLAYER_ID": pid,
        "PERIOD": period,
        "MINUTES_REMAINING": int(rem // 60),
        "SECONDS_REMAINING": int(rem % 60),
        "SHOT_TYPE": "3PT Field Goal" if three else "2PT Field Goal",
        "SHOT_ZONE_BASIC": "Above the Break 3" if three else "Mid-Range",
        "SHOT_ATTEMPTED_FLAG": 1,
        "SHOT_MADE_FLAG": made,
    }


def _pbp(
    etype: int,
    period: int,
    elapsed: float,
    *,
    p1: int,
    t1: int,
    p2: int = 0,
    desc: str = "",
) -> dict[str, Any]:
    plen = 720.0 if period <= 4 else 300.0
    rem = plen - elapsed
    return {
        "GAME_ID": _GID,
        "EVENTNUM": 0,
        "EVENTMSGTYPE": etype,
        "PERIOD": period,
        "PCTIMESTRING": f"{int(rem // 60)}:{int(rem % 60):02d}",
        "HOMEDESCRIPTION": desc,
        "VISITORDESCRIPTION": "",
        "NEUTRALDESCRIPTION": "",
        "PLAYER1_ID": p1,
        "PLAYER1_TEAM_ID": t1,
        "PLAYER2_ID": p2,
        "PLAYER2_TEAM_ID": t1 if p2 else 0,
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


def _write_snapshot(
    root: Path, shots: list[dict[str, Any]], pbp: list[dict[str, Any]]
) -> None:
    (root / "pbp").mkdir(parents=True)
    (root / "game_details").mkdir(parents=True)
    (root / "pbp" / f"stats_{_GID}.json").write_text(
        json.dumps(_payload(_PBP_HEADERS, pbp))
    )
    home = [s for s in shots if s["TEAM_ID"] == 10]
    away = [s for s in shots if s["TEAM_ID"] == 20]
    (root / "game_details" / f"stats_home_shots_{_GID}.json").write_text(
        json.dumps(_payload(_SHOT_HEADERS, home))
    )
    (root / "game_details" / f"stats_away_shots_{_GID}.json").write_text(
        json.dumps(_payload(_SHOT_HEADERS, away))
    )
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

    def mk(sid: str, team: int, opp: int, start: float, off: tuple[int, ...]) -> Stint:
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
            offense_player_ids=off,  # type: ignore[arg-type]
            defense_player_ids=(91, 92, 93, 94, 95),
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
            mk(f"{_GID}-P1-R001-O10", 10, 20, 0.0, (1, 2, 3, 4, 5)),
            mk(f"{_GID}-P1-R002-O10", 10, 20, 360.0, (1, 2, 3, 6, 7)),
        ]
    )


class PlayerProductionTests(unittest.TestCase):
    def test_attributes_fg_ft_and_assist_to_the_right_player_and_stint(self) -> None:
        from courtgraph.features.player_production import (
            ProductionConfig,
            attribute_player_production,
        )
        from courtgraph.ingest.snapshot import load_snapshot

        shots = [
            _shot(10, 1, 1, 100.0, made=1, three=True),  # player 1, 3 pts, stint R001
            _shot(10, 2, 1, 200.0, made=1),  # player 2, 2 pts, stint R001
            _shot(10, 1, 1, 400.0, made=1),  # player 1, 2 pts, stint R002
            _shot(10, 3, 1, 150.0, made=0),  # miss -> no points
        ]
        pbp = [
            _pbp(3, 1, 120.0, p1=2, t1=10, desc="Player2 Free Throw 1 of 2 (1 PTS)"),
            _pbp(3, 1, 121.0, p1=2, t1=10, desc="MISS Player2 Free Throw 2 of 2"),
            _pbp(3, 1, 300.0, p1=9, t1=10, desc="Player Technical Free Throw"),
            _pbp(
                1,
                1,
                100.0,
                p1=1,
                t1=10,
                p2=4,
                desc="Player1 3PT (3 PTS) (Player4 1 AST)",
            ),
            _pbp(
                1, 1, 400.0, p1=1, t1=10, p2=6, desc="Player1 (2 PTS) (Player6 1 AST)"
            ),
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_snapshot(root, shots, pbp)
            prod = attribute_player_production(
                load_snapshot(root),
                _stints(),
                config=ProductionConfig(assist_credit=0.5),
            )

        by = {(r.stint_id, r.player_id): r for r in prod.rows}
        r001 = f"{_GID}-P1-R001-O10"
        r002 = f"{_GID}-P1-R002-O10"

        self.assertEqual(by[(r001, 1)].fg_points, 3)  # the made three
        self.assertEqual(by[(r001, 2)].fg_points, 2)
        self.assertEqual(by[(r001, 2)].ft_points, 1)  # one made FT (the other missed)
        self.assertEqual(by[(r001, 4)].assisted_points, 3)  # assisted the three
        self.assertEqual(by[(r002, 1)].fg_points, 2)
        self.assertEqual(by[(r002, 6)].assisted_points, 2)
        self.assertEqual(by[(r001, 3)].fg_points, 0)  # missed shot

        # technical FT credited to nobody (player 9 not on the R001 offense)
        self.assertNotIn((r001, 9), by)

        cfg = ProductionConfig(assist_credit=0.5)
        self.assertEqual(by[(r001, 1)].points, 3)
        self.assertEqual(by[(r001, 4)].credited(cfg), 1.5)  # 0 + 0.5 * 3

        # every offensive player of every stint has a row (zero-filled)
        self.assertEqual(len(prod.rows), 10)  # 2 stints x 5 players

    def test_round_trips_through_jsonl(self) -> None:
        from courtgraph.features.player_production import (
            ProductionConfig,
            attribute_player_production,
            read_production,
            write_production,
        )
        from courtgraph.ingest.snapshot import load_snapshot

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_snapshot(
                root,
                [_shot(10, 1, 1, 100.0, made=1)],
                [
                    _pbp(
                        3,
                        1,
                        120.0,
                        p1=1,
                        t1=10,
                        desc="Player1 Free Throw 1 of 1 (1 PTS)",
                    )
                ],
            )
            prod = attribute_player_production(
                load_snapshot(root), _stints(), config=ProductionConfig()
            )
            out = write_production(prod, root / "prod.jsonl")
            back = read_production(out)
        self.assertEqual(len(back), len(prod.rows))
        self.assertEqual(
            {(r.stint_id, r.player_id, r.fg_points) for r in back},
            {(r.stint_id, r.player_id, r.fg_points) for r in prod.rows},
        )


if __name__ == "__main__":
    unittest.main()

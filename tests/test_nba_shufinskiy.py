"""The SRC-SHUFINSKIY CSV -> ``stats_nba_pbpstats/v1`` snapshot converter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nba_fixtures import AWAY_TEAM, HOME_TEAM  # noqa: E402
from _shufinskiy_fixture import sample_games, write_archive  # noqa: E402
from courtgraph.ingest.shufinskiy import (  # noqa: E402
    ShufinskiyArchiveError,
    build_snapshot,
)


class ShufinskiyConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.archive = write_archive(base / "arc", sample_games())
        self.snap = build_snapshot(
            self.archive,
            ["42400081", "0042400082", "42400083"],  # 8- and 10-digit both accepted
            base / "snap",
        )
        self.index = json.loads(
            (self.snap.out_dir / "courtgraph_snapshot.json").read_text()
        )

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_emits_the_v1_layout_with_padded_game_ids(self) -> None:
        self.assertEqual(self.index["snapshot_format"], "stats_nba_pbpstats/v1")
        self.assertEqual(
            {g["game_id"] for g in self.index["games"]},
            {"0042400081", "0042400082", "0042400083"},
        )
        for gid in ("0042400081", "0042400082", "0042400083"):
            self.assertTrue((self.snap.out_dir / "pbp" / f"stats_{gid}.json").is_file())
            self.assertTrue(
                (
                    self.snap.out_dir / "game_details" / f"stats_home_shots_{gid}.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    self.snap.out_dir / "game_details" / f"stats_away_shots_{gid}.json"
                ).is_file()
            )

    def test_pbp_file_is_a_playbyplayv2_result_set(self) -> None:
        payload = json.loads(
            (self.snap.out_dir / "pbp" / "stats_0042400082.json").read_text()
        )
        self.assertEqual(payload["resource"], "playbyplayv2")
        result_set = payload["resultSets"][0]
        self.assertIn("EVENTMSGTYPE", result_set["headers"])
        self.assertIn("PCTIMESTRING", result_set["headers"])
        gid_index = result_set["headers"].index("GAME_ID")
        self.assertTrue(
            all(row[gid_index] == "0042400082" for row in result_set["rowSet"])
        )

    def test_metadata_is_derived_not_fabricated(self) -> None:
        by_id = {g["game_id"]: g for g in self.index["games"]}
        g2 = by_id["0042400082"]
        self.assertEqual(g2["game_date"], "2025-04-22")
        self.assertEqual(g2["season"], "2024-25")
        self.assertEqual(g2["season_type"], "Playoffs")
        self.assertEqual(g2["home_team_id"], HOME_TEAM)
        self.assertEqual(g2["away_team_id"], AWAY_TEAM)
        # reconciliation target is the data.nba.com lineage, labelled as such
        self.assertIn("data.nba.com", g2["reconciliation"]["source"])
        self.assertEqual(
            set(g2["reconciliation"]["final_score"]),
            {str(HOME_TEAM), str(AWAY_TEAM)},
        )
        # G2's prior game (G1) is in the archive -> rest days are derivable
        self.assertEqual(g2["days_rest"], {str(HOME_TEAM): 2, str(AWAY_TEAM): 2})

    def test_missing_prior_game_omits_days_rest_and_is_flagged(self) -> None:
        by_id = {g["game_id"]: g for g in self.index["games"]}
        self.assertNotIn("days_rest", by_id["0042400081"])  # no prior game in archive
        self.assertIn("0042400081", self.snap.quarantine_expected)
        self.assertIn("days_rest", self.snap.quarantine_expected["0042400081"])

    def test_display_names_sidecar_is_populated(self) -> None:
        names = json.loads((self.snap.out_dir / "display_names.json").read_text())
        self.assertIn(str(HOME_TEAM), names["teams"])
        self.assertIn(str(AWAY_TEAM), names["teams"])
        self.assertTrue(names["players"])  # id -> name

    def test_unknown_game_is_a_clear_error(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ShufinskiyArchiveError),
        ):
            build_snapshot(self.archive, ["0049999999"], Path(directory) / "s")


if __name__ == "__main__":
    unittest.main()

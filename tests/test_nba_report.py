"""The readable HTML report for a ``courtgraph ingest`` run."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nba_fixtures import (  # noqa: E402
    AWAY_TEAM,
    HOME_TEAM,
    ordinary_game,
    split_lineup_game,
    write_snapshot,
)
from courtgraph.ingest.report import render_report  # noqa: E402

HAS_PBPSTATS = importlib.util.find_spec("pbpstats") is not None


@unittest.skipUnless(HAS_PBPSTATS, "report is rendered from a real ingest run")
class IngestReportTests(unittest.TestCase):
    def setUp(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.snapshot = base / "snap"
        write_snapshot(self.snapshot, [ordinary_game(), split_lineup_game()])
        # a display-name sidecar, as `snapshot-from-shufinskiy` would write
        (self.snapshot / "display_names.json").write_text(
            json.dumps(
                {
                    "teams": {
                        str(HOME_TEAM): "Rivertown Otters",
                        str(AWAY_TEAM): "Hillside Foxes",
                    },
                    "players": {str(p): f"Player {p}" for p in range(100, 211)},
                }
            )
        )
        self.out = base / "out"
        run_ingest(self.snapshot, self.out)
        self.html = render_report(self.out, snapshot_dir=self.snapshot)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_report_is_self_contained(self) -> None:
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<script", self.html)
        self.assertIn("<style>", self.html)

    def test_report_shows_teams_lineups_and_the_score_check(self) -> None:
        self.assertIn("Rivertown Otters", self.html)
        self.assertIn("Hillside Foxes", self.html)
        self.assertIn("Score check", self.html)
        self.assertIn("data.nba.com lineage", self.html)  # honest labelling
        self.assertIn("evidence of predictive accuracy", self.html)  # disclaimer
        self.assertIn("Player 101", self.html)  # a real lineup name in a stint row

    def test_report_lists_exclusions_and_totals(self) -> None:
        self.assertIn("Excluded possessions", self.html)
        self.assertIn("split_lineup_possession", self.html)
        self.assertIn("stints emitted", self.html)


if __name__ == "__main__":
    unittest.main()

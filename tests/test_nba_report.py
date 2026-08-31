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
from courtgraph.ingest._paths import OutputPathError  # noqa: E402
from courtgraph.ingest.report import render_report, write_report  # noqa: E402

HAS_PBPSTATS = importlib.util.find_spec("pbpstats") is not None

_RECON_SOURCE = "operator-supplied NBA official box score (nba.com), as of 2026-08-31"


@unittest.skipUnless(HAS_PBPSTATS, "report is rendered from a real ingest run")
class IngestReportTests(unittest.TestCase):
    def setUp(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.snapshot = base / "snap"
        write_snapshot(self.snapshot, [ordinary_game(), split_lineup_game()])

        # a display-name sidecar + a per-game reconciliation source + a
        # provenance sidecar, exactly as `snapshot-from-shufinskiy` writes them
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
        (self.snapshot / "provenance.json").write_text(
            json.dumps(
                {
                    "source": "SRC-SHUFINSKIY (test)",
                    "pinned_commit": "0123456789abcdef0123456789abcdef01234567",
                    "converter_version": "cg-shufinskiy/2",
                    "consumed_csv_sha256": {"nbastats_po_2024.csv": "0" * 64},
                }
            )
        )
        index_path = self.snapshot / "courtgraph_snapshot.json"
        index = json.loads(index_path.read_text())
        for game in index["games"]:
            game["reconciliation"]["source"] = _RECON_SOURCE
        index_path.write_text(json.dumps(index))

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
        self.assertIn("evidence of predictive accuracy", self.html)
        self.assertIn("Player 101", self.html)

    def test_report_shows_provenance_and_the_recorded_score_source(self) -> None:
        self.assertIn("Source provenance", self.html)
        self.assertIn("cg-shufinskiy/2", self.html)
        self.assertIn("0123456789abcdef0123456789abcdef01234567", self.html)
        self.assertIn("sha256 nbastats_po_2024.csv", self.html)
        # the per-game score check names the actual recorded source
        self.assertIn(_RECON_SOURCE, self.html)

    def test_report_lists_exclusions_and_totals(self) -> None:
        self.assertIn("Excluded possessions", self.html)
        self.assertIn("split_lineup_possession", self.html)
        self.assertIn("stints emitted", self.html)

    def test_write_report_refuses_a_symlinked_or_snapshot_path(self) -> None:
        target = self.snapshot / "courtgraph_snapshot.json"
        link = Path(self._dir.name) / "report_link.html"
        link.symlink_to(target)
        with self.assertRaises(OutputPathError):
            write_report(self.out, link, snapshot_dir=self.snapshot)
        with self.assertRaises(OutputPathError):
            write_report(
                self.out, self.snapshot / "report.html", snapshot_dir=self.snapshot
            )


@unittest.skipUnless(HAS_PBPSTATS, "report is rendered from a real ingest run")
class ReportSourceLabelTests(unittest.TestCase):
    """The banner's origin sentence is derived from recorded provenance; a run
    with no provenance sidecar is never labelled real SRC-SHUFINSKIY data."""

    def setUp(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        self._dir = tempfile.TemporaryDirectory()
        self.base = Path(self._dir.name)
        self.snapshot = self.base / "snap"
        write_snapshot(self.snapshot, [ordinary_game()])
        self.out = self.base / "out"
        run_ingest(self.snapshot, self.out)  # no provenance.json written

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_no_provenance_is_not_labelled_shufinskiy(self) -> None:
        html = render_report(self.out, snapshot_dir=self.snapshot)
        self.assertNotIn("SHUFINSKIY", html.upper())
        self.assertIn("Source not recorded", html)
        # the honest-limits language still stands
        self.assertIn("evidence of predictive accuracy", html)

    def test_synthetic_source_label_is_not_promoted_to_shufinskiy(self) -> None:
        (self.snapshot / "provenance.json").write_text(
            json.dumps({"source": "hand-authored synthetic fixture"})
        )
        self.out2 = self.base / "out2"
        from courtgraph.ingest.pipeline import run_ingest

        run_ingest(self.snapshot, self.out2)
        html = render_report(self.out2, snapshot_dir=self.snapshot)
        self.assertNotIn("SHUFINSKIY", html.upper())
        self.assertIn("hand-authored synthetic fixture", html)

    def test_write_report_auto_gitignores_and_preserves_existing_rules(self) -> None:
        dest = self.base / "reports"
        dest.mkdir()
        (dest / ".gitignore").write_text("# keep me\n!notes.md\n", encoding="utf-8")
        write_report(self.out, dest / "game_report.html", snapshot_dir=self.snapshot)
        text = (dest / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/game_report.html", text)
        self.assertIn("# keep me", text)
        self.assertIn("!notes.md", text)

    def test_write_report_creates_a_gitignore_when_none_exists(self) -> None:
        dest = self.base / "fresh"
        write_report(self.out, dest / "r.html", snapshot_dir=self.snapshot)
        self.assertIn("/r.html", (dest / ".gitignore").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

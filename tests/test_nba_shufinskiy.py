"""The SRC-SHUFINSKIY CSV -> ``stats_nba_pbpstats/v1`` snapshot converter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nba_fixtures import AWAY_TEAM, HOME_TEAM, ordinary_game  # noqa: E402
from _shufinskiy_fixture import (  # noqa: E402
    sample_games,
    write_archive,
    write_multi_season_archive,
    write_official_totals,
    write_raw_archive,
)
from courtgraph.ingest._paths import OutputPathError  # noqa: E402
from courtgraph.ingest.shufinskiy import (  # noqa: E402
    CONVERTER_VERSION,
    ShufinskiyArchiveError,
    build_snapshot,
)


def _minimal_nbastats_rows(game_id: str, eventnums: list[int]) -> list[dict[str, str]]:
    """A few playbyplayv2-shaped rows with caller-chosen EVENTNUM values."""

    short = game_id[2:]
    rows = [
        {
            "GAME_ID": short,
            "EVENTNUM": "2",
            "EVENTMSGTYPE": "12",
            "PERIOD": "1",
            "PCTIMESTRING": "12:00",
            "NEUTRALDESCRIPTION": "Start of 1st Period",
        }
    ]
    for i, num in enumerate(eventnums):
        rows.append(
            {
                "GAME_ID": short,
                "EVENTNUM": str(num),
                "EVENTMSGTYPE": "1",
                "PERIOD": "1",
                "PCTIMESTRING": f"{11 - i}:00",
                "HOMEDESCRIPTION": f"Player 10{i} 2PT Shot",
                "PLAYER1_ID": "101",
                "PLAYER1_TEAM_ID": str(HOME_TEAM),
                "SCORE": f"0 - {2 * (i + 1)}",
            }
        )
    return rows


def _minimal_datanba_rows(
    game_id: str, wallclk_date: str, home_final: int, away_final: int
) -> list[dict[str, str]]:
    short = game_id[2:]

    def row(clock: str, hs: int, vs: int, tid: int, etype: str) -> dict[str, str]:
        return {
            "wallclk": f"{wallclk_date}T{clock}Z",
            "hs": str(hs),
            "vs": str(vs),
            "tid": str(tid),
            "oftid": str(tid),
            "etype": etype,
            "PERIOD": "1",
            "GAME_ID": short,
        }

    return [
        row("23:00:00.000", 0, 0, 0, "12"),
        row("23:05:00.000", home_final, 0, HOME_TEAM, "1"),
        row("23:10:00.000", home_final, away_final, AWAY_TEAM, "1"),
    ]


def _minimal_shot_rows(game_id: str, game_date_yyyymmdd: str) -> list[dict[str, str]]:
    short = game_id[2:]
    return [
        {
            "GRID_TYPE": "Shot Chart Detail",
            "GAME_ID": short,
            "GAME_EVENT_ID": "3",
            "PLAYER_ID": "101",
            "TEAM_ID": str(HOME_TEAM),
            "PERIOD": "1",
            "LOC_X": "0",
            "LOC_Y": "10",
            "GAME_DATE": game_date_yyyymmdd,
            "HTM": "RIV",
            "VTM": "HIL",
        },
        {
            "GRID_TYPE": "Shot Chart Detail",
            "GAME_ID": short,
            "GAME_EVENT_ID": "4",
            "PLAYER_ID": "201",
            "TEAM_ID": str(AWAY_TEAM),
            "PERIOD": "1",
            "LOC_X": "0",
            "LOC_Y": "10",
            "GAME_DATE": game_date_yyyymmdd,
            "HTM": "RIV",
            "VTM": "HIL",
        },
    ]


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

    def test_emits_the_v1_layout_with_padded_game_ids_and_gitignore(self) -> None:
        self.assertEqual(self.index["snapshot_format"], "stats_nba_pbpstats/v1")
        self.assertEqual(
            {g["game_id"] for g in self.index["games"]},
            {"0042400081", "0042400082", "0042400083"},
        )
        self.assertIn("*", (self.snap.out_dir / ".gitignore").read_text())

    def test_metadata_is_derived_not_fabricated(self) -> None:
        by_id = {g["game_id"]: g for g in self.index["games"]}
        g2 = by_id["0042400082"]
        self.assertEqual(g2["game_date"], "2025-04-22")
        self.assertEqual(g2["season"], "2024-25")
        self.assertEqual(g2["season_type"], "Playoffs")
        self.assertEqual(g2["home_team_id"], HOME_TEAM)
        self.assertEqual(g2["away_team_id"], AWAY_TEAM)
        self.assertIn("data.nba.com game feed", g2["reconciliation"]["source"])
        self.assertEqual(g2["days_rest"], {str(HOME_TEAM): 2, str(AWAY_TEAM): 2})

    def test_missing_prior_game_omits_days_rest_and_is_flagged(self) -> None:
        by_id = {g["game_id"]: g for g in self.index["games"]}
        self.assertNotIn("days_rest", by_id["0042400081"])
        self.assertIn("0042400081", self.snap.quarantine_expected)

    def test_provenance_records_hashes_commit_and_version(self) -> None:
        prov = json.loads((self.snap.out_dir / "provenance.json").read_text())
        self.assertEqual(prov["converter_version"], CONVERTER_VERSION)
        self.assertRegex(prov["pinned_commit"], r"^[0-9a-f]{40}$")
        digests = prov["consumed_csv_sha256"]
        self.assertEqual(
            set(digests),
            {
                "nbastats_po_2024.csv",
                "datanba_po_2024.csv",
                "shotdetail_po_2024.csv",
            },
        )
        for digest in digests.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_display_names_sidecar_is_populated(self) -> None:
        names = json.loads((self.snap.out_dir / "display_names.json").read_text())
        self.assertIn(str(HOME_TEAM), names["teams"])
        self.assertTrue(names["players"])

    def test_all_games_uses_complete_archive_intersection_and_records_gap(self) -> None:
        datanba = self.archive / "datanba_po_2024.csv"
        with datanba.open() as stream:
            rows = list(csv.DictReader(stream))
        # Rewrite with the original header while removing one game's feed rows.
        with datanba.open() as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
        kept = [row for row in rows if row["GAME_ID"].zfill(10) != "0042400083"]
        with datanba.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        with tempfile.TemporaryDirectory() as directory:
            snap = build_snapshot(self.archive, None, Path(directory) / "all")
            self.assertEqual(snap.game_ids, ("0042400081", "0042400082"))
            self.assertEqual(snap.archive_coverage["archive_games"], 3)
            self.assertEqual(snap.archive_coverage["complete_games"], 2)
            excluded = snap.archive_coverage["excluded_games"]
            self.assertEqual(excluded[0]["game_id"], "0042400083")
            self.assertEqual(excluded[0]["missing_inputs"], ["datanba"])
            provenance = json.loads((snap.out_dir / "provenance.json").read_text())
            self.assertEqual(provenance["archive_coverage"], snap.archive_coverage)

    def test_unknown_game_is_a_clear_error(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ShufinskiyArchiveError),
        ):
            build_snapshot(self.archive, ["0049999999"], Path(directory) / "s")


class ShufinskiyMultiSeasonTests(unittest.TestCase):
    """One archive holding several seasons of regular-season CSVs."""

    def _archive(self, directory: Path) -> Path:
        # Season 2020-21: two games for the same two teams (game 2 has rest).
        # Season 2021-22: one game -> its own season opener.
        return write_multi_season_archive(
            directory / "arc",
            {
                "2020": [
                    ("0022000001", "2020-12-22", ordinary_game("0022000001").builder),
                    ("0022000015", "2020-12-25", ordinary_game("0022000015").builder),
                ],
                "2021": [
                    ("0022100001", "2021-10-19", ordinary_game("0022100001").builder),
                ],
            },
        )

    def test_every_season_file_is_discovered_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(Path(directory))
            snap = build_snapshot(archive, None, Path(directory) / "snap")
            self.assertEqual(snap.game_ids, ("0022000001", "0022000015", "0022100001"))
            index = json.loads((snap.out_dir / "courtgraph_snapshot.json").read_text())
            seasons = {g["season"] for g in index["games"]}
            self.assertEqual(seasons, {"2020-21", "2021-22"})
            digests = json.loads((snap.out_dir / "provenance.json").read_text())[
                "consumed_csv_sha256"
            ]
            self.assertEqual(
                set(digests),
                {
                    "nbastats_2020.csv",
                    "nbastats_2021.csv",
                    "datanba_2020.csv",
                    "datanba_2021.csv",
                    "shotdetail_2020.csv",
                    "shotdetail_2021.csv",
                },
            )

    def test_rest_days_are_bounded_to_the_same_season(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(Path(directory))
            snap = build_snapshot(archive, None, Path(directory) / "snap")
            index = json.loads((snap.out_dir / "courtgraph_snapshot.json").read_text())
            by_id = {g["game_id"]: g for g in index["games"]}
            # Game 2 of 2020-21: rest counted from the 2020-12-22 game (2 days).
            self.assertEqual(
                by_id["0022000015"]["days_rest"],
                {str(HOME_TEAM): 2, str(AWAY_TEAM): 2},
            )
            # The 2021-22 opener must NOT borrow the prior season's last game as
            # a "prior game" (~300 days) -- it has no same-season predecessor.
            self.assertNotIn("days_rest", by_id["0022100001"])
            self.assertIn("0022100001", snap.quarantine_expected)

    def test_missing_one_provider_for_a_season_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(Path(directory))
            (archive / "datanba_2020.csv").unlink()
            (archive / "datanba_2021.csv").unlink()
            with self.assertRaises(ShufinskiyArchiveError):
                build_snapshot(archive, None, Path(directory) / "snap")


class ShufinskiyEdgeCaseTests(unittest.TestCase):
    def test_csv_event_order_is_preserved_not_sorted(self) -> None:
        # EVENTNUM is not always monotonic; the emitted rowSet must keep the
        # archive's row order (pbpstats fixes ordering itself).
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            eventnums = [10, 40, 20, 50, 30]
            write_raw_archive(
                base / "arc",
                nbastats=_minimal_nbastats_rows("0042400099", eventnums),
                datanba=_minimal_datanba_rows("0042400099", "2025-05-01", 10, 8),
                shots=_minimal_shot_rows("0042400099", "20250501"),
            )
            build_snapshot(base / "arc", ["0042400099"], base / "snap")
            pbp = json.loads(
                (base / "snap" / "pbp" / "stats_0042400099.json").read_text()
            )
            headers = pbp["resultSets"][0]["headers"]
            idx = headers.index("EVENTNUM")
            got = [row[idx] for row in pbp["resultSets"][0]["rowSet"]]
            self.assertEqual(got, [2, *eventnums])  # CSV order, not sorted

    def test_game_date_uses_validated_game_date_across_utc_midnight(self) -> None:
        # Local game date 2025-04-30; tip rolls past UTC midnight so the event
        # wall-clock lands on 2025-05-01. GAME_DATE (2025-04-30) must win, for
        # both the game date and the rest calculation.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            builder = sample_games()[0][2]  # reuse a toy game builder
            write_archive(
                base / "arc",
                [
                    ("0042400091", "2025-04-27", builder),
                    ("0042400092", "2025-04-30", builder, "2025-05-01"),
                ],
            )
            snap = build_snapshot(base / "arc", ["0042400092"], base / "snap")
            game = json.loads((snap.out_dir / "courtgraph_snapshot.json").read_text())[
                "games"
            ][0]
            self.assertEqual(game["game_date"], "2025-04-30")  # not 2025-05-01
            self.assertEqual(game["days_rest"], {str(HOME_TEAM): 2, str(AWAY_TEAM): 2})

    def test_official_totals_are_preferred_over_the_feed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_archive(base / "arc", sample_games()[1:])  # G2, G3
            write_official_totals(
                base / "arc",
                {
                    "0042400082": {
                        "final_score": {str(HOME_TEAM): 111, str(AWAY_TEAM): 99},
                        "period_scores": {
                            str(HOME_TEAM): [111],
                            str(AWAY_TEAM): [99],
                        },
                        "source": "NBA official box score (nba.com), as of 2026-08-31",
                    }
                },
            )
            snap = build_snapshot(base / "arc", ["0042400082"], base / "snap")
            recon = json.loads((snap.out_dir / "courtgraph_snapshot.json").read_text())[
                "games"
            ][0]["reconciliation"]
            self.assertEqual(
                recon["final_score"], {str(HOME_TEAM): 111, str(AWAY_TEAM): 99}
            )
            self.assertIn("NBA official box score", recon["source"])


class ShufinskiyDestinationSafetyTests(unittest.TestCase):
    def test_out_dir_inside_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = write_archive(base / "arc", sample_games())
            for out_dir in (archive, archive / "snap", archive / "deep" / "x"):
                with self.subTest(out_dir=out_dir), self.assertRaises(OutputPathError):
                    build_snapshot(archive, ["0042400082"], out_dir)
            # the archive was not touched
            self.assertFalse((archive / "snap").exists())

    def test_generated_file_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = write_archive(base / "arc", sample_games())
            out = base / "snap"
            out.mkdir()
            victim = archive / "nbastats_po_2024.csv"
            (out / "courtgraph_snapshot.json").symlink_to(victim)
            before = victim.read_bytes()
            with self.assertRaises(OutputPathError):
                build_snapshot(archive, ["0042400082"], out)
            self.assertEqual(victim.read_bytes(), before)

    def test_symlinked_intermediate_directory_is_rejected(self) -> None:
        # Codex repro: a pre-created `pbp/` (or `game_details/`) symlink pointing
        # at a directory of source files. `mkdir(exist_ok=True)` would silently
        # succeed and every `_write_pbp` would land in the linked directory.
        for linked in ("pbp", "game_details"):
            with self.subTest(linked=linked), tempfile.TemporaryDirectory() as d:
                base = Path(d)
                archive = write_archive(base / "arc", sample_games())
                victim_dir = base / "src"
                victim_dir.mkdir()
                victim = victim_dir / "stats_0042400082.json"
                victim.write_bytes(b"ORIGINAL")
                out = base / "snap"
                out.mkdir()
                (out / linked).symlink_to(victim_dir, target_is_directory=True)
                with self.assertRaises(OutputPathError):
                    build_snapshot(archive, ["0042400082"], out)
                self.assertEqual(victim.read_bytes(), b"ORIGINAL")

    def test_existing_gitignore_contents_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = write_archive(base / "arc", sample_games())
            out = base / "snap"
            out.mkdir()
            (out / ".gitignore").write_text(
                "# operator rules\n!keep-this.txt\n", encoding="utf-8"
            )
            build_snapshot(archive, ["0042400082"], out)
            first = (out / ".gitignore").read_text(encoding="utf-8")
            self.assertIn("# operator rules", first)
            self.assertIn("!keep-this.txt", first)
            self.assertIn("*", first)
            # a second run must not keep growing the file
            build_snapshot(archive, ["0042400082"], out)
            self.assertEqual((out / ".gitignore").read_text(encoding="utf-8"), first)

    def test_snapshot_dir_is_git_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            archive = write_archive(repo / "arc", sample_games())
            build_snapshot(archive, ["0042400082"], repo / "snap")
            for rel in ("courtgraph_snapshot.json", "provenance.json"):
                checked = subprocess.run(
                    ["git", "check-ignore", "-q", f"snap/{rel}"], cwd=repo
                )
                self.assertEqual(checked.returncode, 0, rel)


if __name__ == "__main__":
    unittest.main()

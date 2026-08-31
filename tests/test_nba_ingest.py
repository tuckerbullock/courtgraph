"""Offline NBA snapshot -> stint ingestion, exercised through the real
``pbpstats`` file-mode parser on hand-authored, NBA-shaped fixtures."""

from __future__ import annotations

import importlib.util
import io
import json
import socket
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nba_fixtures import (  # noqa: E402
    AWAY_TEAM,
    HOME,
    HOME_TEAM,
    GameSpec,
    free_throw_game,
    noncontiguous_lineup_game,
    ordinary_game,
    overtime_game,
    returning_player_split_game,
    rotation_season,
    split_lineup_game,
    write_snapshot,
)
from courtgraph.chemistry.stints import read_stints  # noqa: E402
from courtgraph.ingest.policy import IngestPolicy  # noqa: E402

HAS_PBPSTATS = importlib.util.find_spec("pbpstats") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@dataclass
class _Run:
    result: Any
    manifest: dict[str, Any]
    quarantine: list[dict[str, Any]]
    table: Any
    out: Path

    def game(self, index: int = 0) -> dict[str, Any]:
        games: list[dict[str, Any]] = self.manifest["games"]
        return games[index]


def _run(specs: list[GameSpec], **policy_kwargs: Any) -> _Run:
    from courtgraph.ingest.pipeline import run_ingest

    tmp = tempfile.mkdtemp()
    root = Path(tmp) / "snap"
    out = Path(tmp) / "out"
    write_snapshot(root, specs)
    result = run_ingest(root, out, policy=IngestPolicy(**policy_kwargs))
    manifest = json.loads(result.manifest_path.read_text())
    quarantine = [
        json.loads(line)
        for line in result.quarantine_path.read_text().splitlines()
        if line
    ]
    table = read_stints(result.stints_path) if result.stints_written else None
    return _Run(
        result=result,
        manifest=manifest,
        quarantine=quarantine,
        table=table,
        out=out,
    )


@unittest.skipUnless(HAS_PBPSTATS, "ingestion requires pbpstats")
class IngestScenarioTests(unittest.TestCase):
    def test_ordinary_game_emits_well_formed_stints(self) -> None:
        run = _run([ordinary_game()])
        table = run.table
        self.assertIsNotNone(table)
        assert table is not None
        self.assertGreater(len(table), 0)
        for stint in table:
            self.assertEqual(len(set(stint.offense_player_ids)), 5)
            self.assertEqual(len(set(stint.defense_player_ids)), 5)
            self.assertFalse(
                set(stint.offense_player_ids) & set(stint.defense_player_ids)
            )
            self.assertNotEqual(stint.offense_team_id, stint.defense_team_id)
            self.assertGreater(stint.offensive_possessions, 0)
            self.assertGreaterEqual(stint.points_scored, 0)
            self.assertEqual(stint.source, "nba-stats-pbpstats")
            self.assertEqual(stint.game_date, "2023-10-24")

    def test_offensive_rebound_stays_in_the_same_possession(self) -> None:
        # P1: HOME has 4 possessions (one prolonged by an offensive rebound);
        # the starters never change, so it is a single stint of 4 possessions.
        run = _run([ordinary_game()])
        table = run.table
        assert table is not None
        p1_home = [s for s in table if s.period == 1 and s.offense_team_id == HOME_TEAM]
        self.assertEqual(len(p1_home), 1)
        self.assertEqual(p1_home[0].offensive_possessions, 4)

    def test_substitution_starts_a_new_stint(self) -> None:
        # ordinary_game subs HOME 105 -> 106 in period 2.
        run = _run([ordinary_game()])
        table = run.table
        assert table is not None
        home_lineups = {
            s.offense_player_ids for s in table if s.offense_team_id == HOME_TEAM
        }
        self.assertIn(tuple(sorted(HOME)), home_lineups)  # the starters
        after_sub = tuple(sorted(HOME[:4] + [106]))
        self.assertIn(after_sub, home_lineups)  # 105 out, 106 in -> a new stint
        for lineup in home_lineups:
            if 106 in lineup:
                self.assertNotIn(105, lineup)

    def test_noncontiguous_same_lineup_is_never_merged(self) -> None:
        run = _run([noncontiguous_lineup_game()])
        table = run.table
        assert table is not None
        starters = tuple(sorted(HOME))
        spells = [
            s
            for s in table
            if s.offense_team_id == HOME_TEAM and s.offense_player_ids == starters
        ]
        self.assertEqual(len(spells), 2, "the two spells must not be merged")
        self.assertNotEqual(spells[0].stint_id, spells[1].stint_id)

    def test_split_lineup_with_a_returning_player_is_excluded(self) -> None:
        # possessions 3 (HOME) and 4 (AWAY) sub a player out, use a substitute
        # in live play, then sub the original back before the shot.
        run = _run([returning_player_split_game()])
        game = run.game()
        self.assertEqual(game["status"], "accepted")
        split = [
            e
            for e in game["excluded_possessions"]
            if e["reason"] == "split_lineup_possession"
        ]
        self.assertEqual(len(split), 2)
        self.assertEqual({e["possession_number"] for e in split}, {3, 4})

    def test_excluded_possessions_break_stint_continuity(self) -> None:
        # accepted possessions 1-2 and 5-6 share the starting five but sit on
        # opposite sides of the excluded possessions 3-4 -> two runs, not one.
        run = _run([returning_player_split_game()])
        table = run.table
        assert table is not None
        starters = tuple(sorted(HOME))
        home_spells = [
            s
            for s in table
            if s.offense_team_id == HOME_TEAM and s.offense_player_ids == starters
        ]
        self.assertEqual(len(home_spells), 2)
        self.assertNotEqual(home_spells[0].stint_id, home_spells[1].stint_id)
        for stint in home_spells:
            self.assertEqual(stint.offensive_possessions, 1)

    def test_overtime_period_is_ingested(self) -> None:
        run = _run([overtime_game()])
        table = run.table
        assert table is not None
        ot = [s for s in table if s.period == 5]
        self.assertTrue(ot)
        for stint in ot:
            self.assertGreaterEqual(stint.start_time_seconds, 0.0)
            self.assertLessEqual(stint.start_time_seconds, 300.0)

    def test_free_throws_and_technical_are_handled_not_fabricated(self) -> None:
        run = _run([free_throw_game()])
        game = run.game()
        self.assertEqual(game["status"], "accepted")
        self.assertTrue(game["reconciliation"]["final_score_matched"])
        self.assertIn("technical_free_throws_in_game", game["flags"])
        table = run.table
        assert table is not None
        for stint in table:
            self.assertGreaterEqual(stint.points_scored, 0)
            self.assertLessEqual(stint.points_scored, 4 * stint.offensive_possessions)

    def test_split_lineup_possession_is_quarantined_but_still_reconciled(self) -> None:
        run = _run([split_lineup_game()])
        game = run.game()
        self.assertEqual(game["status"], "accepted")
        self.assertTrue(game["reconciliation"]["final_score_matched"])
        reasons = {e["reason"] for e in game["excluded_possessions"]}
        self.assertIn("split_lineup_possession", reasons)
        self.assertIn("split_lineup_possession", {q["reason"] for q in run.quarantine})

    def test_score_reconciliation_failure_quarantines_the_game(self) -> None:
        spec = ordinary_game()
        spec.final_score_override = {HOME_TEAM: 999, AWAY_TEAM: 1}
        run = _run([spec])
        result = run.result
        self.assertEqual(result.stints_written, 0)
        self.assertEqual(result.games_quarantined, 1)
        game = run.game()
        self.assertEqual(game["quarantine_reason"], "score_reconciliation_failed")
        self.assertFalse(game["reconciliation"]["final_score_matched"])
        self.assertIn(
            "score_reconciliation_failed",
            {q["reason"] for q in run.quarantine},
        )

    def test_allow_score_mismatch_emits_with_a_flag(self) -> None:
        spec = ordinary_game()
        real = spec.builder.final_score()
        spec.final_score_override = {
            HOME_TEAM: real[HOME_TEAM] + 2,
            AWAY_TEAM: real[AWAY_TEAM],
        }
        run = _run([spec], allow_score_mismatch=True)
        game = run.game()
        self.assertEqual(game["status"], "accepted")
        self.assertIn("score_reconciliation_mismatch_allowed", game["flags"])
        self.assertGreater(run.result.stints_written, 0)

    def test_incomplete_final_score_fails_closed(self) -> None:
        # Removing one team from reconciliation.final_score must not reconcile,
        # even when score mismatches are explicitly allowed.
        for allow in (False, True):
            with self.subTest(allow_score_mismatch=allow):
                tmp = tempfile.mkdtemp()
                root = Path(tmp) / "snap"
                write_snapshot(root, [ordinary_game()])
                index = root / "courtgraph_snapshot.json"
                payload = json.loads(index.read_text())
                final = payload["games"][0]["reconciliation"]["final_score"]
                del final[str(AWAY_TEAM)]
                index.write_text(json.dumps(payload))
                from courtgraph.ingest.pipeline import run_ingest

                result = run_ingest(
                    root,
                    Path(tmp) / "out",
                    policy=IngestPolicy(allow_score_mismatch=allow),
                )
                self.assertEqual(result.stints_written, 0)
                game = json.loads(result.manifest_path.read_text())["games"][0]
                self.assertEqual(game["quarantine_reason"], "missing_context")

    def test_override_change_changes_recorded_provenance(self) -> None:
        # missing_period_starters.json controls reconstructed lineups; changing
        # it must change the recorded input hashes and the correction-set id.
        from courtgraph.ingest.pipeline import run_ingest

        tmp = Path(tempfile.mkdtemp())
        root = tmp / "snap"
        write_snapshot(root, [ordinary_game()])
        override = root / "overrides" / "missing_period_starters.json"
        self.assertTrue(override.is_file())

        m1 = json.loads(run_ingest(root, tmp / "out1").manifest_path.read_text())
        payload = json.loads(override.read_text())
        game_id = next(iter(payload))
        payload[game_id]["1"][str(HOME_TEAM)] = [901, 902, 903, 904, 905]
        override.write_text(json.dumps(payload))
        m2 = json.loads(run_ingest(root, tmp / "out2").manifest_path.read_text())

        self.assertNotEqual(
            m1["corrections"]["correction_set_id"],
            m2["corrections"]["correction_set_id"],
        )
        self.assertNotEqual(
            m1["corrections"]["override_files"], m2["corrections"]["override_files"]
        )
        override_rel = "overrides/missing_period_starters.json"
        self.assertIn(override_rel, m1["games"][0]["input_files"])
        self.assertNotEqual(
            m1["games"][0]["input_files"][override_rel],
            m2["games"][0]["input_files"][override_rel],
        )

    def test_missing_context_quarantines_the_game(self) -> None:
        for drop in ("game_date", "days_rest"):
            with self.subTest(field=drop):
                tmp = tempfile.mkdtemp()
                root = Path(tmp) / "snap"
                write_snapshot(root, [ordinary_game()])
                index = root / "courtgraph_snapshot.json"
                payload = json.loads(index.read_text())
                if drop == "game_date":
                    payload["games"][0]["game_date"] = ""
                else:
                    payload["games"][0]["days_rest"] = {}
                index.write_text(json.dumps(payload))
                from courtgraph.ingest.pipeline import run_ingest

                result = run_ingest(root, Path(tmp) / "out")
                self.assertEqual(result.stints_written, 0)
                manifest = json.loads(result.manifest_path.read_text())
                self.assertEqual(
                    manifest["games"][0]["quarantine_reason"], "missing_context"
                )

    def test_ambiguous_reconstruction_is_quarantined_kept_as_audit(self) -> None:
        # a pbp file that pbpstats cannot turn into alternating possessions
        tmp = tempfile.mkdtemp()
        root = Path(tmp) / "snap"
        write_snapshot(root, [ordinary_game()])
        pbp_path = next(root.glob("pbp/stats_*.json"))
        payload = json.loads(pbp_path.read_text())
        rows = payload["resultSets"][0]["rowSet"]
        payload["resultSets"][0]["rowSet"] = rows[:3]  # truncate mid-first-period
        pbp_path.write_text(json.dumps(payload))
        from courtgraph.ingest.pipeline import run_ingest

        result = run_ingest(root, Path(tmp) / "out")
        manifest = json.loads(result.manifest_path.read_text())
        game = manifest["games"][0]
        self.assertEqual(game["status"], "quarantined")
        self.assertTrue(
            game["quarantine_reason"].startswith("pbpstats_reconstruction_failed")
            or game["quarantine_reason"] == "network_required"
        )

    def test_manifest_is_a_complete_audit_trail(self) -> None:
        run = _run([ordinary_game(), free_throw_game()])
        manifest = run.manifest
        self.assertEqual(manifest["snapshot_format"], "stats_nba_pbpstats/v1")
        self.assertEqual(manifest["parser"]["tool"], "pbpstats")
        self.assertEqual(manifest["parser"]["mode"], "file")
        self.assertFalse(manifest["parser"]["hosted_archive_used"])
        self.assertRegex(manifest["parser"]["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(manifest["policy"]["policy_version"], "cg-ingest-policy/1")
        self.assertIn("created_utc", manifest)
        self.assertEqual(manifest["totals"]["games_in"], 2)
        corrections = manifest["corrections"]
        self.assertRegex(
            corrections["correction_set_id"], r"^cg-corrections/[0-9a-f]{16}$"
        )
        override_rel = "overrides/missing_period_starters.json"
        self.assertRegex(corrections["override_files"][override_rel], r"^[0-9a-f]{64}$")
        for game in manifest["games"]:
            self.assertTrue(game["input_files"])
            self.assertIn(override_rel, game["input_files"])  # overrides in provenance
            self.assertEqual(
                game["correction_set_id"], corrections["correction_set_id"]
            )
            for rel, digest in game["input_files"].items():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
                self.assertTrue(rel.endswith(".json"))
            self.assertIn("source_event_counts", game)
        self.assertRegex(manifest["outputs"]["stints_sha256"], r"^[0-9a-f]{64}$")


@unittest.skipUnless(HAS_PBPSTATS, "ingestion requires pbpstats")
class IngestOfflineTests(unittest.TestCase):
    def test_reconstruction_makes_no_network_connection(self) -> None:
        from unittest import mock

        from courtgraph.ingest.pipeline import run_ingest

        def _fail_connect(*a: Any, **k: Any) -> Any:
            raise AssertionError("ingestion opened a socket connection")

        def _fail_getaddrinfo(*a: Any, **k: Any) -> Any:
            raise AssertionError("ingestion performed a DNS lookup")

        with (
            mock.patch.object(socket.socket, "connect", _fail_connect),
            mock.patch.object(socket, "getaddrinfo", _fail_getaddrinfo),
            tempfile.TemporaryDirectory() as directory,
        ):
            root = Path(directory) / "snap"
            write_snapshot(root, [ordinary_game(), overtime_game(), free_throw_game()])
            result = run_ingest(root, Path(directory) / "out")
        self.assertGreater(result.stints_written, 0)

    def test_a_game_that_would_need_the_network_is_quarantined(self) -> None:
        # drop the period-starter override so pbpstats cannot resolve all five
        # starters from this short pbp -> its only fallback is a network request.
        spec = ordinary_game("0022300777")
        spec.starters = {}
        run = _run([spec])
        game = run.game()
        self.assertEqual(game["status"], "quarantined")
        self.assertEqual(game["quarantine_reason"], "network_required")
        self.assertEqual(run.result.stints_written, 0)


@unittest.skipUnless(HAS_PBPSTATS and HAS_NUMPY, "needs pbpstats + numpy")
class IngestToModelTests(unittest.TestCase):
    def test_fixture_ingestion_feeds_fit_and_predict(self) -> None:
        from courtgraph.cli import main

        tmp = Path(tempfile.mkdtemp())
        root = tmp / "snap"
        write_snapshot(root, rotation_season(4))

        out = tmp / "ingest"
        rc = main(["ingest", "--snapshot-dir", str(root), "--out-dir", str(out)])
        self.assertEqual(rc, 0)
        stints = out / "stints.jsonl"
        table = read_stints(stints)
        self.assertGreaterEqual(len(table), 50)

        model_path = tmp / "model.json"
        rc = main(
            [
                "fit",
                "--input",
                str(stints),
                "--model-out",
                str(model_path),
                "--rank",
                "2",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue(model_path.is_file())

        buffer = io.StringIO()
        rc = main(
            [
                "predict",
                "--model",
                str(model_path),
                "--offense",
                "101,102,103,104,105",
                "--defense",
                "201,202,203,204,205",
                "--json",
            ],
            output=buffer,
        )
        self.assertEqual(rc, 0)
        payload = json.loads(buffer.getvalue())
        decomposition = payload["decomposition"]
        self.assertAlmostEqual(
            decomposition["talent"]
            + decomposition["interaction"]
            + decomposition["context"],
            decomposition["total"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

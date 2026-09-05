"""Local app data boundaries, model wiring, and loopback HTTP behavior."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from dataclasses import replace
from http.client import HTTPConnection
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from courtgraph.app.observations import Observations
from courtgraph.chemistry.stints import Stint, StintTable, write_stints
from courtgraph.cli import build_parser

if TYPE_CHECKING:
    from http.server import ThreadingHTTPServer

    from courtgraph.app.sandbox import Sandbox

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture(directory: Path) -> Path:
    first = Stint(
        stint_id="one",
        game_id="game1",
        game_date="2025-01-01",
        season="2024-25",
        season_index=0,
        period=1,
        start_time_seconds=0,
        offense_team_id=10,
        defense_team_id=20,
        offense_player_ids=(1, 2, 3, 4, 5),
        defense_player_ids=(6, 7, 8, 9, 10),
        offensive_possessions=10,
        points_scored=20,
        home_offense=True,
        score_margin_offense=0,
        playoff=False,
        days_rest_offense=1,
        garbage_time_weight=1,
        source="hand-authored-fixture",
    )
    table = StintTable.from_stints(
        [
            first,
            replace(
                first,
                stint_id="two",
                period=2,
                offensive_possessions=30,
                points_scored=30,
                garbage_time_weight=0.2,
            ),
        ]
    )
    write_stints(table, directory / "stints.jsonl")
    manifest = {
        "outputs": {
            "stints_written": 2,
            "stints_sha256": hashlib.sha256(
                (directory / "stints.jsonl").read_bytes()
            ).hexdigest(),
        },
        "games": [
            {
                "game_id": "game1",
                "game_date": "2025-01-01",
                "status": "accepted",
                "stints_emitted": 2,
                "accepted_possessions": 40,
                "reconciliation": {},
                "excluded_possessions": [{"reason": "empty_possession"}],
            },
            {
                "game_id": "bad",
                "game_date": "2025-01-02",
                "status": "quarantined",
                "stints_emitted": 0,
                "accepted_possessions": 0,
                "reconciliation": {},
                "quarantine_reason": "network_required",
            },
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    names = directory / "display_names.json"
    names.write_text(
        json.dumps(
            {"teams": {"10": "Fixture team"}, "players": {"1": "A <script> player"}}
        )
    )
    return names


class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.names = fixture(self.path)
        self.data = Observations(self.path, self.names)

    def test_rates_use_possession_totals_not_average_stint_rates(self) -> None:
        result = self.data.query()
        row = result["lineups"][0]
        self.assertEqual(
            (row["rating"], row["possessions"], row["points"]), (125, 40, 50)
        )
        self.assertEqual(row["downweighted_stints"], 1)
        self.assertEqual(row["team"], "Fixture team")
        self.assertEqual(row["players"][0], "A <script> player")

    def test_filters_and_empty_quarantined_game(self) -> None:
        self.assertEqual(len(self.data.query(player="1", team="10")["lineups"]), 1)
        for filters in ({"game": "bad"}, {"team": "20"}, {"player": "6"}):
            result = self.data.query(
                game=filters.get("game", ""),
                team=filters.get("team", ""),
                player=filters.get("player", ""),
            )
            self.assertEqual(result["lineups"], [])
            self.assertEqual(result["possessions"], 0)
        result = self.data.query(minimum=41)
        self.assertEqual(result["lineups"], [])
        self.assertEqual(result["possessions"], 40)
        with self.assertRaises(ValueError):
            self.data.query(minimum=0)

    def test_player_pool_returns_the_observed_pool_not_a_roster(self) -> None:
        offense_pool = self.data.player_pool("10")
        self.assertEqual(
            sorted(p["id"] for p in offense_pool["players"]), [1, 2, 3, 4, 5]
        )
        self.assertTrue(all(p["possessions"] == 40 for p in offense_pool["players"]))
        self.assertIn("not an official roster", offense_pool["source"])

        defense_pool = self.data.player_pool("20")
        self.assertEqual(
            sorted(p["id"] for p in defense_pool["players"]), [6, 7, 8, 9, 10]
        )

        with self.assertRaises(ValueError):
            self.data.player_pool("")
        with self.assertRaises(ValueError):
            self.data.player_pool("999")

    def test_provenance_never_assumes_real_nba_or_independent_score_source(
        self,
    ) -> None:
        overview = self.data.overview()
        self.assertIn("Source not recorded", overview["source"])
        self.assertEqual(overview["games"][0]["score_source"], "Not recorded")
        self.assertEqual(overview["games"][1]["status"], "quarantined")
        self.assertEqual(overview["cutoff"], "2025-01-01")
        self.assertEqual(
            Observations(self.path).query()["lineups"][0]["team"], "Team 10"
        )

    def test_archive_coverage_includes_unattempted_source_game(self) -> None:
        path = self.path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["games"][0].update(
            {"home_team_id": 10, "away_team_id": 20, "season_type": "Playoffs"}
        )
        manifest["source_provenance"] = {
            "archive_coverage": {
                "archive_games": 3,
                "complete_games": 2,
                "excluded_games": [
                    {
                        "game_id": "source-only",
                        "game_date": "2025-01-03",
                        "team_ids": [10, 20],
                        "missing_inputs": ["datanba.csv"],
                    }
                ],
            }
        }
        path.write_text(json.dumps(manifest))
        overview = Observations(self.path, self.names).overview()
        self.assertEqual(
            overview["coverage"],
            {
                "archive_games": 3,
                "complete_games": 2,
                "attempted_games": 2,
                "accepted_games": 1,
                "quarantined_games": 1,
                "source_incomplete_games": 1,
            },
        )
        source_only = next(g for g in overview["games"] if g["id"] == "source-only")
        self.assertEqual(source_only["status"], "source_incomplete")
        self.assertIn("datanba.csv", source_only["quarantine_reason"])
        accepted = next(g for g in overview["games"] if g["id"] == "game1")
        self.assertEqual(accepted["home_team"], "Fixture team")
        self.assertEqual(accepted["away_team"], "Team 20")

    def test_source_files_unchanged(self) -> None:
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        self.data.overview()
        self.data.query()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir()})

    def test_mismatched_hash_fails_closed(self) -> None:
        with (self.path / "stints.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Observations(self.path)

    def test_manifest_cannot_promote_quarantined_stints(self) -> None:
        path = self.path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["games"][0]["status"] = "quarantined"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "accepted game"):
            Observations(self.path)

    def test_manifest_exposure_and_dates_must_match(self) -> None:
        path = self.path / "manifest.json"
        original = path.read_text()
        for field, value in [
            ("accepted_possessions", 41),
            ("stints_emitted", 3),
            ("game_date", "2024-01-01"),
        ]:
            manifest = json.loads(original)
            manifest["games"][0][field] = value
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                Observations(self.path)

    def test_duplicate_games_are_rejected(self) -> None:
        path = self.path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["games"].append(manifest["games"][0])
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "duplicate games"):
            Observations(self.path)

    def test_app_cli_options_are_explicit(self) -> None:
        args = build_parser().parse_args(
            ["app", "--port", "9000", "--ingest-dir", "out", "--names", "names.json"]
        )
        self.assertEqual(
            (args.port, args.ingest_dir, args.names),
            (9000, Path("out"), Path("names.json")),
        )


@unittest.skipUnless(HAS_NUMPY, "synthetic app requires numpy")
class SandboxTests(unittest.TestCase):
    sandbox: Sandbox

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.app.sandbox import Sandbox

        cls.sandbox = Sandbox()

    def payload(self) -> dict[str, Any]:
        return {
            "offense": self.sandbox.offense,
            "alternative": self.sandbox.alternative,
            "defense": self.sandbox.defense,
            "home": True,
            "playoff": False,
            "rest": 1,
        }

    def test_comparison_matches_existing_model_and_decomposition(self) -> None:
        result = self.sandbox.compare(self.payload())
        for i, key in enumerate(("offense", "alternative")):
            row = result["results"][i]
            expected = self.sandbox.model.decompose(
                tuple(sorted(self.payload()[key])),
                tuple(sorted(self.sandbox.defense)),
                result["context"],
            )
            self.assertEqual(row["decomposition"], expected.as_dict())
            self.assertAlmostEqual(
                expected.total,
                expected.talent + expected.interaction + expected.context,
            )
        self.assertAlmostEqual(
            result["delta"]["total"],
            result["results"][1]["decomposition"]["total"]
            - result["results"][0]["decomposition"]["total"],
        )
        self.assertEqual(result["mode"], "synthetic")
        self.assertEqual(self.sandbox.catalog()["bootstrap_members"], 8)

    def test_identical_lineups_have_zero_difference_and_order_does_not_matter(
        self,
    ) -> None:
        payload = self.payload()
        payload["alternative"] = list(reversed(payload["offense"]))
        self.assertTrue(
            all(value == 0 for value in self.sandbox.compare(payload)["delta"].values())
        )

    def test_duplicate_overlap_unknown_and_noninteger_players_rejected(self) -> None:
        for invalid in (
            [self.sandbox.offense[0]] * 5,
            self.sandbox.defense,
            [1629636, 2, 3, 4, 5],
            [True, 2, 3, 4, 5],
            [1, 2, 3, 4],
            "1,2,3,4,5",
        ):
            payload = {**self.payload(), "offense": invalid}
            with self.assertRaises(ValueError):
                self.sandbox.compare(payload)

    def test_unrecognized_context_and_artifacts_rejected(self) -> None:
        for changes in (
            {"rest": -1},
            {"rest": 8},
            {"rest": True},
            {"home": "true"},
            {"playoff": 1},
            {"model_path": "NBA-model.json"},
        ):
            with self.assertRaises(ValueError):
                self.sandbox.compare({**self.payload(), **changes})

    def test_context_applies_equally_to_both_lineups(self) -> None:
        result = self.sandbox.compare(
            {**self.payload(), "rest": 3, "home": False, "playoff": True}
        )
        self.assertEqual(result["context"]["days_rest_offense"], 3)
        self.assertEqual(
            result["results"][0]["decomposition"]["context"],
            result["results"][1]["decomposition"]["context"],
        )

    def test_sandbox_recovers_a_non_zero_interaction_surplus(self) -> None:
        # The synthetic pool must be large enough for the cross-fitted
        # interaction fit to clear its out-of-fold selection gate; otherwise
        # the sandbox's headline "interaction surplus" is identically zero and
        # the A/B comparison is meaningless.
        players = self.sandbox.players
        interactions = []
        for start in range(0, 40, 10):
            offense = players[start : start + 5]
            alternative = players[start + 5 : start + 10]
            defense = players[start + 10 : start + 15]
            row = self.sandbox.compare(
                {
                    "offense": offense,
                    "alternative": alternative,
                    "defense": defense,
                    "home": True,
                    "playoff": False,
                    "rest": 1,
                }
            )
            for result in row["results"]:
                interactions.append(abs(result["decomposition"]["interaction"]))
        self.assertTrue(any(value > 1e-6 for value in interactions))


@unittest.skipUnless(HAS_NUMPY, "local app requires numpy")
class LocalHTTPTests(unittest.TestCase):
    server: ThreadingHTTPServer
    thread: Thread
    sandbox: Sandbox
    temp: TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.app.sandbox import Sandbox
        from courtgraph.app.server import make_server

        cls.temp = TemporaryDirectory()
        path = Path(cls.temp.name)
        names = fixture(path)
        cls.sandbox = Sandbox()
        cls.server = make_server(0, Observations(path, names), cls.sandbox)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_assets_and_api_are_served_locally_without_external_assets(self) -> None:
        for path in ("/", "/app.js", "/style.css", "/api/state", "/api/observations"):
            status, headers, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertNotIn("Access-Control-Allow-Origin", headers)
            self.assertTrue(body)
        state = json.loads(self.request("/api/state")[2])
        self.assertNotIn("snapshot_root", state["real"])
        self.assertEqual(state["synthetic"]["mode"], "synthetic")
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_local_comparison_and_invalid_json(self) -> None:
        headers = {"Content-Type": "application/json", "X-CourtGraph-Request": "local"}
        payload = {
            "offense": self.sandbox.offense,
            "alternative": self.sandbox.alternative,
            "defense": self.sandbox.defense,
        }
        result = self.request(
            "/api/compare", method="POST", body=json.dumps(payload), headers=headers
        )
        self.assertEqual(result[0], 200)
        for body in ("[]", "{broken", "{}", "x" * 17000):
            self.assertEqual(
                self.request("/api/compare", method="POST", body=body, headers=headers)[
                    0
                ],
                400,
            )
        self.assertEqual(self.request("/api/compare", method="POST", body="{}")[0], 415)

    def test_cross_origin_rebinding_and_path_traversal_blocked(self) -> None:
        for headers in (
            {"Host": "attacker.example"},
            {"Origin": "https://attacker.example"},
            {"Sec-Fetch-Site": "cross-site"},
        ):
            self.assertEqual(self.request("/api/state", headers=headers)[0], 403)
        for path in (
            "/../../pyproject.toml",
            "/%2e%2e/README.md",
            "/api/model",
            "/manifest.json",
        ):
            self.assertEqual(self.request(path)[0], 404)

    def test_player_pool_endpoint(self) -> None:
        status, _headers, body = self.request("/api/player-pool?team=10")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(sorted(p["id"] for p in payload["players"]), [1, 2, 3, 4, 5])
        self.assertEqual(self.request("/api/player-pool?team=999")[0], 400)
        self.assertEqual(self.request("/api/player-pool?bogus=1")[0], 400)

    def test_predict_real_endpoint_has_no_chemistry_field(self) -> None:
        headers = {"Content-Type": "application/json", "X-CourtGraph-Request": "local"}
        payload = {"offense": [1, 2, 3, 4, 5], "defense": [6, 7, 8, 9, 10]}
        status, _headers, body = self.request(
            "/api/predict-real",
            method="POST",
            body=json.dumps(payload),
            headers=headers,
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertNotIn("interaction", result)
        self.assertIn("total", result)
        self.assertIn("interval_95", result)
        self.assertIn("no interaction/chemistry term", result["note"])

        bad = {"offense": [1, 2, 3, 4], "defense": [6, 7, 8, 9, 10]}
        self.assertEqual(
            self.request(
                "/api/predict-real",
                method="POST",
                body=json.dumps(bad),
                headers=headers,
            )[0],
            400,
        )
        self.assertEqual(
            self.request("/api/predict-real", method="POST", body="{}")[0], 415
        )

    def test_compare_real_endpoint(self) -> None:
        headers = {"Content-Type": "application/json", "X-CourtGraph-Request": "local"}
        payload = {
            "offense": [1, 2, 3, 4, 5],
            "alternative": [1, 2, 3, 4, 5],
            "defense": [6, 7, 8, 9, 10],
        }
        status, _headers, body = self.request(
            "/api/compare-real",
            method="POST",
            body=json.dumps(payload),
            headers=headers,
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertIn("delta", result)
        self.assertNotIn("interaction", result)
        self.assertAlmostEqual(
            result["delta"]["total"],
            result["b"]["total"] - result["a"]["total"],
            places=9,
        )
        bad = {"offense": [1, 2, 3, 4, 5], "defense": [6, 7, 8, 9, 10]}
        self.assertEqual(
            self.request(
                "/api/compare-real",
                method="POST",
                body=json.dumps(bad),
                headers=headers,
            )[0],
            400,
        )

    def test_bad_filters_fail_and_empty_results_are_valid(self) -> None:
        for query in (
            "minimum=no",
            "minimum=-1",
            "path=/private/file",
            "&".join(f"x{i}=1" for i in range(10)),
        ):
            self.assertEqual(self.request("/api/observations?" + query)[0], 400)
        self.assertEqual(
            json.loads(self.request("/api/observations?game=bad")[2])["lineups"], []
        )

    def test_missing_dataset_and_startup_errors(self) -> None:
        from courtgraph.app.server import make_server, serve

        server = make_server(0, None, self.sandbox)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/api/state")
            self.assertFalse(
                json.loads(connection.getresponse().read())["real"]["loaded"]
            )
            connection.request("GET", "/api/player-pool?team=10")
            self.assertEqual(connection.getresponse().status, 200)
            predict_body = json.dumps(
                {"offense": [1, 2, 3, 4, 5], "defense": [6, 7, 8, 9, 10]}
            )
            connection.request(
                "POST",
                "/api/predict-real",
                body=predict_body,
                headers={
                    "Content-Type": "application/json",
                    "X-CourtGraph-Request": "local",
                },
            )
            self.assertEqual(connection.getresponse().status, 400)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()
        with patch("courtgraph.app.server.Sandbox") as fit:
            for port, ingest, names in [
                (-1, None, None),
                (8765, None, Path("names.json")),
                (8765, Path("missing-dir"), None),
            ]:
                self.assertEqual(serve(port, ingest, names, StringIO()), 2)
            fit.assert_not_called()

#!/usr/bin/env python3
"""Offline checks for the pilot -- no network, no third-party install.

    python3 pilot/test_pilot.py
    python3 -m unittest discover -s pilot -p "test_*.py"

Covers:
  * the recursive, value-free schema fingerprinter (unit);
  * end-to-end separation: a live-shaped run and a snapshot run each produce a
    tracked schema contract with NO provenance, and an ignored run manifest
    that carries the provenance.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
import unittest.mock as mock
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nba_access_pilot import (  # noqa: E402
    Row,
    assemble,
    contract_entry,
    fingerprint,
    run_snapshot,
    schema_of,
)

# data.nba.com-style nested payload: game -> periods[] -> events[] -> fields.
NESTED_A = {
    "g": {
        "gid": "0022300001",
        "pd": [
            {
                "p": 1,
                "pla": [
                    {"evt": 1, "cl": "12:00", "de": "Jump Ball", "opt1": 0, "hs": 0},
                    {"evt": 2, "cl": "11:44", "de": "Missed Shot", "opt1": 2, "hs": 0},
                ],
            }
        ],
    }
}
# Same structure, every scalar value changed, and a different event count.
NESTED_B = {
    "g": {
        "gid": "0022500987",
        "pd": [
            {
                "p": 4,
                "pla": [
                    {"evt": 511, "cl": "0:03", "de": "Made 3PT", "opt1": 3, "hs": 118},
                    {"evt": 512, "cl": "0:01", "de": "Foul", "opt1": 0, "hs": 118},
                    {"evt": 513, "cl": "0:00", "de": "End", "opt1": 0, "hs": 118},
                ],
            }
        ],
    }
}
NESTED_C = json.loads(json.dumps(NESTED_A))
NESTED_C["g"]["pd"][0]["pla"][0]["assist_pid"] = 201939

RESULT_SETS_A = {
    "resource": "playbyplay",
    "resultSets": [
        {
            "name": "PlayByPlay",
            "headers": ["GAME_ID", "EVENTNUM", "PCTIMESTRING", "SCORE", "SCOREMARGIN"],
            "rowSet": [
                ["0022300001", 1, "12:00", None, None],
                ["0022300001", 2, "0:34.5", "110 - 108", "2"],
            ],
        }
    ],
}


class RecursiveFingerprintTests(unittest.TestCase):
    def test_includes_nested_event_fields(self) -> None:
        schema = fingerprint(NESTED_A)["schema"]
        blob = json.dumps(schema)
        for field in ("gid", "pd", "pla", "evt", "cl", "de", "opt1", "hs"):
            self.assertIn(f'"{field}"', blob, f"missing nested field {field!r}")
        pla = schema["fields"]["g"]["fields"]["pd"]["items"]["fields"]["pla"]
        self.assertEqual(pla["type"], "array")
        self.assertEqual(pla["items"]["type"], "object")
        self.assertIn("evt", pla["items"]["fields"])

    def test_excludes_cell_values(self) -> None:
        schema = fingerprint(NESTED_A)["schema"]
        blob = json.dumps(schema)
        for value in ("0022300001", "12:00", "Jump Ball", "Missed Shot", "11:44"):
            self.assertNotIn(value, blob, f"value {value!r} leaked into the schema")
        self.assertEqual(schema["fields"]["g"]["fields"]["gid"], {"type": "string"})
        pla = schema["fields"]["g"]["fields"]["pd"]["items"]["fields"]["pla"]
        self.assertEqual(pla["items"]["fields"]["evt"], {"type": "integer"})

    def test_stable_when_values_change_structure_does_not(self) -> None:
        fp_a, fp_b = fingerprint(NESTED_A), fingerprint(NESTED_B)
        self.assertEqual(fp_a["schema_sha256"], fp_b["schema_sha256"])
        self.assertEqual(schema_of(NESTED_A), schema_of(NESTED_B))
        self.assertNotEqual(
            fp_a["array_lengths"]["$.g.pd[].pla"],
            fp_b["array_lengths"]["$.g.pd[].pla"],
        )

    def test_changes_when_structure_changes(self) -> None:
        self.assertNotEqual(
            fingerprint(NESTED_A)["schema_sha256"],
            fingerprint(NESTED_C)["schema_sha256"],
        )
        type_change = json.loads(json.dumps(NESTED_A))
        type_change["g"]["pd"][0]["pla"][0]["evt"] = "1"  # int -> str
        self.assertNotEqual(
            fingerprint(NESTED_A)["schema_sha256"],
            fingerprint(type_change)["schema_sha256"],
        )

    def test_result_sets_view_is_value_free(self) -> None:
        fp = fingerprint(RESULT_SETS_A)
        names = [c["name"] for c in fp["nba_view"]["PlayByPlay"]["columns"]]
        self.assertEqual(
            names, ["GAME_ID", "EVENTNUM", "PCTIMESTRING", "SCORE", "SCOREMARGIN"]
        )
        blob = json.dumps(fp)
        for value in ("0022300001", "12:00", "0:34.5", "110 - 108"):
            self.assertNotIn(value, blob)
        self.assertEqual(fp["array_lengths"].get("$.resultSets[].rowSet"), 2)

    def test_deterministic_and_key_order_independent(self) -> None:
        self.assertEqual(fingerprint(NESTED_A), fingerprint(NESTED_A))
        reordered = {"g": {"pd": NESTED_A["g"]["pd"], "gid": NESTED_A["g"]["gid"]}}
        self.assertEqual(
            fingerprint(NESTED_A)["schema_sha256"],
            fingerprint(reordered)["schema_sha256"],
        )


# Provenance values that must never reach the tracked contract.
FORBIDDEN_IN_CONTRACT = (
    "0022300001",  # game id
    "2023-12-25",  # date
    "Jump Ball",  # event description
    "12:00",  # clock value
    "0:34.5",  # clock value
    "110 - 108",  # score value
    "34:12",  # minutes value
)


class EndToEndSeparationTests(unittest.TestCase):
    def _write_both(
        self, contract: dict[str, Any], manifest: dict[str, Any], root: pathlib.Path
    ) -> tuple[str, str]:
        out = root / "v0_schema_contract.json"
        local = root / "_local"
        local.mkdir()
        out.write_text(json.dumps(contract, indent=2, sort_keys=True))
        (local / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        return out.read_text(), (local / "run_manifest.json").read_text()

    def test_live_shaped_run_separates_provenance(self) -> None:
        sched = {
            "resultSets": [
                {
                    "name": "GameHeader",
                    "headers": ["GAME_DATE_EST", "GAME_ID"],
                    "rowSet": [["2023-12-25T00:00:00", "0022300001"]],
                }
            ]
        }
        spbp = {
            "resultSets": [
                {
                    "name": "PlayByPlay",
                    "headers": ["GAME_ID", "PCTIMESTRING", "HOMEDESCRIPTION", "SCORE"],
                    "rowSet": [
                        ["0022300001", "12:00", "Jump Ball", None],
                        ["0022300001", "0:34.5", "Made Shot", "110 - 108"],
                    ],
                }
            ]
        }
        box = {
            "resultSets": [
                {
                    "name": "PlayerStats",
                    "headers": ["GAME_ID", "MIN"],
                    "rowSet": [["0022300001", "34:12"]],
                }
            ]
        }
        dpbp = {
            "g": {
                "gid": "0022300001",
                "pd": [
                    {
                        "p": 1,
                        "pla": [
                            {"evt": 1, "cl": "12:00", "de": "Jump Ball", "hs": 0},
                            {"evt": 2, "cl": "0:34.5", "de": "Made Shot", "hs": 110},
                        ],
                    }
                ],
            }
        }
        rows: list[Row] = [
            (
                "stats_scoreboard",
                *contract_entry(sched),
                {
                    "url": "https://stats.nba.com/stats/scoreboardv2",
                    "params": {"GameDate": "2023-12-25"},
                    "raw_file": "raw/stats_scoreboard.json",
                },
            ),
            (
                "stats_playbyplay_01",
                *contract_entry(spbp),
                {"game_id": "0022300001", "url": "s/playbyplayv2", "raw_file": "raw/x"},
            ),
            (
                "stats_boxscore_traditional_01",
                *contract_entry(box),
                {"game_id": "0022300001", "url": "s/box", "raw_file": "raw/y"},
            ),
            (
                "data_playbyplay_01",
                *contract_entry(dpbp),
                {
                    "game_id": "0022300001",
                    "url": "d/.../0022300001_full_pbp.json",
                    "raw_file": "raw/z",
                },
            ),
        ]
        contract, manifest = assemble(
            "live",
            rows,
            {"date": "2023-12-25", "game_ids": ["0022300001"], "requests": []},
        )
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ctext, mtext = self._write_both(contract, manifest, root)

            for token in (*FORBIDDEN_IN_CONTRACT, td, str(root)):
                self.assertNotIn(token, ctext, f"{token!r} leaked into the contract")
            self.assertEqual(
                sorted(contract["endpoints"]),
                [
                    "data_playbyplay_01",
                    "stats_boxscore_traditional_01",
                    "stats_playbyplay_01",
                    "stats_scoreboard",
                ],
            )
            # provenance lives in the ignored manifest
            self.assertIn("0022300001", mtext)
            self.assertIn("2023-12-25", mtext)
            # response *content* stays out of the manifest (raw payloads only)
            self.assertNotIn("Jump Ball", mtext)
            self.assertNotIn("110 - 108", mtext)

    def test_snapshot_run_separates_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = pathlib.Path(td) / "src" / "datanba"
            src.mkdir(parents=True)
            (src / "0022300001_full_pbp.json").write_text(
                json.dumps(
                    {
                        "g": {
                            "gid": "0022300001",
                            "pd": [
                                {
                                    "p": 1,
                                    "pla": [
                                        {"evt": 1, "cl": "12:00", "de": "Jump Ball"}
                                    ],
                                }
                            ],
                        }
                    }
                )
            )
            csv_text = "GAME_ID,EVENTNUM\n0022300001,1\n"
            (src.parent / "nbastats_pbp.csv").write_text(csv_text)
            root_src = str(src.parent)

            contract, manifest = run_snapshot(root_src)
            out_root = pathlib.Path(td) / "out"
            out_root.mkdir()
            ctext, mtext = self._write_both(contract, manifest, out_root)

            for token in ("0022300001", "Jump Ball", "12:00", root_src, td):
                self.assertNotIn(token, ctext, f"{token!r} leaked into the contract")
            self.assertTrue(
                all(a.startswith("snapshot_") for a in contract["endpoints"])
            )
            # provenance in the ignored manifest: dir + the game-id-bearing name
            self.assertIn(root_src, mtext)
            self.assertIn("0022300001_full_pbp.json", mtext)
            self.assertNotIn("Jump Ball", mtext)


# --------------------------------------------------------------------------- #
# probe_access.py: 5.1 immediate-stop -- a first-provider 403/429 must prevent
# the second one-shot request; a timeout may proceed.
# --------------------------------------------------------------------------- #


class _FakeRequestException(Exception):
    pass


class _FakeTimeout(_FakeRequestException):
    pass


class _FakeConnectionError(_FakeRequestException):
    pass


class _FakeHTTPError(_FakeRequestException):
    pass


class _Resp:
    def __init__(
        self, status_code: int, content: bytes = b"{}", headers: dict | None = None
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


def _fake_requests(get_impl: Any) -> types.ModuleType:
    """A stand-in for the ``requests`` module: fake exceptions plus ``get``."""
    mod = types.ModuleType("requests")
    exc = types.ModuleType("requests.exceptions")
    for target in (mod, exc):
        target.RequestException = _FakeRequestException  # type: ignore[attr-defined]
        target.Timeout = _FakeTimeout  # type: ignore[attr-defined]
        target.ConnectionError = _FakeConnectionError  # type: ignore[attr-defined]
        target.HTTPError = _FakeHTTPError  # type: ignore[attr-defined]
    mod.exceptions = exc  # type: ignore[attr-defined]
    mod.get = get_impl  # type: ignore[attr-defined]
    return mod


class ProbeImmediateStopTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: sys.modules.pop("probe_access", None))
        self.addCleanup(lambda: sys.modules.pop("requests", None))

    def _load(self, get_impl: Any) -> tuple[Any, list[str]]:
        calls: list[str] = []

        def tracking_get(url: str, headers: Any = None, timeout: Any = None) -> Any:
            calls.append(url)
            return get_impl(url, len(calls))

        sys.modules["requests"] = _fake_requests(tracking_get)
        sys.modules.pop("probe_access", None)
        return importlib.import_module("probe_access"), calls

    def test_first_probe_403_skips_second_provider(self) -> None:
        probe_access, calls = self._load(
            lambda url, n: _Resp(403, b"<Error/>", {"Content-Type": "application/xml"})
        )
        probe_access.main()
        self.assertEqual(len(calls), 1, "second provider was contacted after a 403")
        self.assertIn("stats.nba.com", calls[0])

    def test_first_probe_429_skips_second_provider(self) -> None:
        probe_access, calls = self._load(lambda url, n: _Resp(429))
        probe_access.main()
        self.assertEqual(len(calls), 1, "second provider was contacted after a 429")

    def test_first_probe_timeout_still_reaches_second_provider(self) -> None:
        def impl(url: str, n: int) -> Any:
            if n == 1:
                raise sys.modules["requests"].exceptions.Timeout("read timeout")
            return _Resp(200)

        probe_access, calls = self._load(impl)
        probe_access.main()
        self.assertEqual(len(calls), 2, "timeout on first probe blocked the second")
        self.assertIn("data.nba.com", calls[1])


# --------------------------------------------------------------------------- #
# nba_access_pilot.py: a hard-stopped live run must still write the partial
# request provenance to the ignored manifest.
# --------------------------------------------------------------------------- #


class HardStopManifestTests(unittest.TestCase):
    def _patch(self, obj: Any, name: str, value: Any) -> None:
        old = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(setattr, obj, name, old)

    def test_hard_stopped_live_run_preserves_partial_manifest(self) -> None:
        import nba_access_pilot as pilot

        patcher = mock.patch("time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

        with tempfile.TemporaryDirectory() as td:
            local = pathlib.Path(td) / "_local"
            self._patch(pilot, "LOCAL", local)
            self._patch(pilot, "MANIFEST", local / "run_manifest.json")
            self._patch(pilot, "RAW", local / "raw")
            self._patch(pilot, "DEFAULT_OUT", local / "schema_contract.json")

            class FakeRequests:
                Timeout = _FakeTimeout
                ConnectionError = _FakeConnectionError
                HTTPError = _FakeHTTPError

                @staticmethod
                def get(*args: Any, **kwargs: Any) -> Any:
                    raise _FakeTimeout("simulated read timeout")

            self._patch(pilot, "_requests", lambda: FakeRequests)

            rc = pilot.main(["--date", "2023-12-25", "--max-games", "2"])

            self.assertEqual(rc, 2)
            manifest = json.loads((local / "run_manifest.json").read_text())
            self.assertEqual(manifest["mode"], "live")
            self.assertEqual(manifest["date"], "2023-12-25")
            self.assertIn("stopped", manifest)
            self.assertTrue(
                manifest["requests"], "attempted requests were discarded on hard stop"
            )
            self.assertTrue(
                any("scoreboardv2" in r.get("url", "") for r in manifest["requests"])
            )
            self.assertTrue(
                all(r.get("alias") == "stats_scoreboard" for r in manifest["requests"])
            )
            # no schema contract is written when the run hard-stops
            self.assertFalse((local / "schema_contract.json").exists())


if __name__ == "__main__":
    unittest.main()

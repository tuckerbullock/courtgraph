"""Snapshot-format contract: structure, malformed inputs, missing files, and
that the immutable snapshot is never modified by ingestion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nba_fixtures import (  # noqa: E402
    free_throw_game,
    ordinary_game,
    overtime_game,
    split_lineup_game,
    write_snapshot,
)
from courtgraph.ingest import SNAPSHOT_FORMAT  # noqa: E402
from courtgraph.ingest.snapshot import SnapshotError, load_snapshot  # noqa: E402

HAS_PBPSTATS = importlib.util.find_spec("pbpstats") is not None


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class SnapshotStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name) / "snap"
        write_snapshot(self.root, [ordinary_game()])

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_valid_snapshot_loads_with_hashes_and_metadata(self) -> None:
        snapshot = load_snapshot(self.root)
        self.assertEqual(len(snapshot.games), 1)
        game = snapshot.games[0]
        self.assertEqual(game.metadata.game_date, "2023-10-24")
        self.assertEqual(game.metadata.season, "2023-24")
        self.assertIn("pbp/stats_0022300001.json", game.file_hashes)
        # every recorded hash matches the file on disk
        for rel, digest in game.file_hashes.items():
            self.assertEqual(
                hashlib.sha256((self.root / rel).read_bytes()).hexdigest(), digest
            )

    def test_missing_index_is_a_snapshot_error(self) -> None:
        (self.root / "courtgraph_snapshot.json").unlink()
        with self.assertRaises(SnapshotError):
            load_snapshot(self.root)

    def test_wrong_snapshot_format_is_rejected(self) -> None:
        index = self.root / "courtgraph_snapshot.json"
        payload = json.loads(index.read_text())
        payload["snapshot_format"] = "something/else"
        index.write_text(json.dumps(payload))
        with self.assertRaises(SnapshotError):
            load_snapshot(self.root)
        self.assertEqual(SNAPSHOT_FORMAT, "stats_nba_pbpstats/v1")

    def test_missing_pbp_file_is_a_snapshot_error(self) -> None:
        next(self.root.glob("pbp/stats_*.json")).unlink()
        with self.assertRaises(SnapshotError) as ctx:
            load_snapshot(self.root)
        self.assertIn("pbp", str(ctx.exception))

    def test_missing_shot_chart_file_is_a_snapshot_error(self) -> None:
        next(self.root.glob("game_details/stats_away_shots_*.json")).unlink()
        with self.assertRaises(SnapshotError):
            load_snapshot(self.root)

    def test_malformed_index_json_is_a_snapshot_error(self) -> None:
        (self.root / "courtgraph_snapshot.json").write_text("{ not json ]")
        with self.assertRaises(SnapshotError):
            load_snapshot(self.root)

    def test_metadata_missing_required_field_is_a_snapshot_error(self) -> None:
        index = self.root / "courtgraph_snapshot.json"
        payload = json.loads(index.read_text())
        del payload["games"][0]["home_team_id"]
        index.write_text(json.dumps(payload))
        with self.assertRaises(SnapshotError):
            load_snapshot(self.root)


class FixtureShapeTests(unittest.TestCase):
    """The fixtures are real stats.nba.com-shaped JSON, not mock parser output."""

    def test_pbp_fixture_is_a_playbyplayv2_result_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snap"
            write_snapshot(root, [ordinary_game(), overtime_game()])
            for pbp_path in sorted(root.glob("pbp/stats_*.json")):
                payload = json.loads(pbp_path.read_text())
                self.assertEqual(payload["resource"], "playbyplayv2")
                result_set = payload["resultSets"][0]
                self.assertIn("EVENTMSGTYPE", result_set["headers"])
                self.assertIn("PCTIMESTRING", result_set["headers"])
                type_index = result_set["headers"].index("EVENTMSGTYPE")
                event_types = {row[type_index] for row in result_set["rowSet"]}
                self.assertTrue(event_types <= set(range(1, 21)))
                self.assertIn(12, event_types)  # start of period
                self.assertIn(13, event_types)  # end of period


class IngestImportIsLightTests(unittest.TestCase):
    def test_importing_the_ingest_package_pulls_in_no_heavy_deps(self) -> None:
        # `courtgraph doctor` must stay third-party-free even with ingest present.
        code = (
            "import courtgraph.ingest, courtgraph.ingest.snapshot, "
            "courtgraph.ingest.policy, sys; "
            "assert 'pbpstats' not in sys.modules, 'pbpstats imported eagerly'; "
            "assert 'numpy' not in sys.modules, 'numpy imported eagerly'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(HAS_PBPSTATS, "ingestion requires pbpstats")
class SnapshotImmutabilityTests(unittest.TestCase):
    def test_ingestion_never_modifies_the_snapshot(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snap"
            write_snapshot(
                root,
                [
                    ordinary_game(),
                    free_throw_game(),
                    overtime_game(),
                    split_lineup_game(),
                ],
            )
            before = _hash_tree(root)
            out = Path(directory) / "out"
            run_ingest(root, out)
            run_ingest(root, Path(directory) / "out2")  # idempotent
            self.assertEqual(before, _hash_tree(root))
            # the pbpstats working copy is a private temp dir, discarded, and
            # never written under the snapshot or the output directory
            self.assertFalse((out / "_work").exists())
            self.assertFalse(any(p.name == "_work" for p in root.rglob("*")))

    def test_overlapping_snapshot_and_out_dir_is_rejected_before_writing(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snap"
            write_snapshot(root, [ordinary_game()])
            before = _hash_tree(root)
            for out_dir in (root, root / "_work", root / "nested" / "deep"):
                with self.subTest(out_dir=out_dir), self.assertRaises(SnapshotError):
                    run_ingest(root, out_dir)
            self.assertEqual(before, _hash_tree(root))

    def test_output_directory_is_git_ignored(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        repo_gitignore = (
            Path(__file__).resolve().parent.parent / ".gitignore"
        ).read_text()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            (repo / "snap").mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / ".gitignore").write_text(repo_gitignore)
            write_snapshot(repo / "snap", [ordinary_game()])
            result = run_ingest(repo / "snap", repo / "ingest_out")
            for name in ("stints.jsonl", "quarantine.jsonl", "manifest.json"):
                checked = subprocess.run(
                    ["git", "check-ignore", "-q", f"ingest_out/{name}"], cwd=repo
                )
                self.assertEqual(checked.returncode, 0, f"{name} is not git-ignored")
            self.assertGreaterEqual(result.stints_written, 0)

    def test_existing_output_gitignore_and_unrelated_files_are_preserved(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            (repo / "snap").mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            write_snapshot(repo / "snap", [ordinary_game()])

            out = repo / "ingest_out"
            out.mkdir()
            (out / ".gitignore").write_text("# caller's rules\n*.tmp\nbuild/\n")
            (out / "keep.txt").write_text("do not touch")

            run_ingest(repo / "snap", out)

            self.assertEqual((out / "keep.txt").read_text(), "do not touch")
            gitignore = (out / ".gitignore").read_text()
            self.assertIn("# caller's rules", gitignore)
            self.assertIn("*.tmp", gitignore)
            self.assertIn("build/", gitignore)
            # ...and the generated data is now ignored too
            for name in ("stints.jsonl", "quarantine.jsonl", "manifest.json"):
                checked = subprocess.run(
                    ["git", "check-ignore", "-q", f"ingest_out/{name}"], cwd=repo
                )
                self.assertEqual(checked.returncode, 0, f"{name} is not git-ignored")


class OutputSafetyValidationTests(unittest.TestCase):
    """Destination validation happens before any directory is created or byte
    written -- so an invalid snapshot or unsafe destination mutates nothing.
    These need no parser and run on the dependency-free path."""

    def test_invalid_snapshot_creates_no_output_directory(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "snap"
            bad.mkdir()
            (bad / "courtgraph_snapshot.json").write_text("{ not json ]")
            out = Path(directory) / "was_never_here"
            with self.assertRaises(SnapshotError):
                run_ingest(bad, out)
            self.assertFalse(out.exists())

    def test_rejected_nested_output_creates_nothing_and_mutates_nothing(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snap"
            write_snapshot(root, [ordinary_game()])
            before = _hash_tree(root)
            with self.assertRaises(SnapshotError):
                run_ingest(root, root / "deep" / "nested" / "out")
            self.assertFalse((root / "deep").exists())
            self.assertEqual(before, _hash_tree(root))

    def test_output_symlink_targeting_snapshot_metadata_is_rejected(self) -> None:
        from courtgraph.ingest.pipeline import run_ingest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snap"
            write_snapshot(root, [ordinary_game()])
            metadata = root / "courtgraph_snapshot.json"
            metadata_before = metadata.read_bytes()

            out = Path(directory) / "out"
            out.mkdir()
            (out / "stints.jsonl").symlink_to(metadata)

            with self.assertRaises(SnapshotError) as ctx:
                run_ingest(root, out)
            self.assertIn("symlink", str(ctx.exception))
            self.assertEqual(metadata.read_bytes(), metadata_before)
            self.assertTrue((out / "stints.jsonl").is_symlink())  # not followed


class OutputGitignoreTests(unittest.TestCase):
    """`_ensure_output_gitignore` keeps the generated data files ignored even
    when the caller's `.gitignore` negates them, and does not grow on re-runs.
    Parser-free -- runs on the dependency-free path."""

    DATA_FILES = ("stints.jsonl", "quarantine.jsonl", "manifest.json")

    def _ignored(self, repo: Path, rel: str) -> bool:
        return (
            subprocess.run(["git", "check-ignore", "-q", rel], cwd=repo).returncode == 0
        )

    def _prepare(self, gitignore_text: str) -> tuple[Path, Path]:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
        out = directory / "ingest_out"
        out.mkdir()
        (out / ".gitignore").write_text(gitignore_text)
        return directory, out

    def test_wildcard_then_negations_are_overridden(self) -> None:
        from courtgraph.ingest.pipeline import _ensure_output_gitignore

        repo, out = self._prepare(
            "*\n!stints.jsonl\n!quarantine.jsonl\n!manifest.json\n!caller_notes.txt\n"
        )
        _ensure_output_gitignore(out)

        for name in self.DATA_FILES:
            self.assertTrue(self._ignored(repo, f"ingest_out/{name}"), name)
        # the caller's unrelated re-inclusion survives
        self.assertFalse(self._ignored(repo, "ingest_out/caller_notes.txt"))
        self.assertIn("!caller_notes.txt", (out / ".gitignore").read_text())

    def test_exact_rules_then_negations_are_overridden(self) -> None:
        from courtgraph.ingest.pipeline import _ensure_output_gitignore

        repo, out = self._prepare(
            "/stints.jsonl\n/quarantine.jsonl\n/manifest.json\n"
            "!/stints.jsonl\n!/quarantine.jsonl\n!/manifest.json\n"
            "caller-data/\n"
        )
        _ensure_output_gitignore(out)

        for name in self.DATA_FILES:
            self.assertTrue(self._ignored(repo, f"ingest_out/{name}"), name)
        text = (out / ".gitignore").read_text()
        self.assertIn("caller-data/", text)  # unrelated caller rule preserved
        self.assertTrue(self._ignored(repo, "ingest_out/caller-data/x"))

    def test_repeated_runs_do_not_grow_the_file(self) -> None:
        from courtgraph.ingest.pipeline import _ensure_output_gitignore

        _repo, out = self._prepare("# caller rules\n*.log\n")
        _ensure_output_gitignore(out)
        first = (out / ".gitignore").read_text()
        _ensure_output_gitignore(out)
        _ensure_output_gitignore(out)
        self.assertEqual((out / ".gitignore").read_text(), first)
        self.assertEqual(first.count("BEGIN courtgraph ingest"), 1)


if __name__ == "__main__":
    unittest.main()

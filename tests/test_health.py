"""Tests for the dependency-free CourtGraph bootstrap checks."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from courtgraph.cli import main
from courtgraph.health import (
    REQUIRED_PROJECT_PATHS,
    check_project_layout,
    check_python_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PythonVersionCheckTests(unittest.TestCase):
    def test_supported_version_passes(self) -> None:
        result = check_python_version((3, 11, 0))

        self.assertTrue(result.passed)
        self.assertIn("Python 3.11.0", result.detail)

    def test_unsupported_version_fails(self) -> None:
        result = check_python_version((3, 10, 14))

        self.assertFalse(result.passed)
        self.assertIn("requires Python 3.11+", result.detail)


class ProjectLayoutCheckTests(unittest.TestCase):
    def test_shared_agent_handoff_files_are_required(self) -> None:
        required = {str(path) for path in REQUIRED_PROJECT_PATHS}

        self.assertTrue(
            {"AGENTS.md", "CLAUDE.md", "docs/CURRENT_TASK.md"}.issubset(required)
        )

    def test_repository_layout_passes(self) -> None:
        result = check_project_layout(PROJECT_ROOT)

        self.assertTrue(result.passed, result.detail)

    def test_missing_files_are_sorted_and_reported(self) -> None:
        with TemporaryDirectory() as directory:
            result = check_project_layout(
                Path(directory),
                required_paths=(Path("z.txt"), Path("a.txt")),
            )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.detail,
            "Missing required files: a.txt, z.txt",
        )


class DoctorCommandTests(unittest.TestCase):
    def test_json_output_is_machine_readable(self) -> None:
        output = StringIO()

        exit_code = main(
            ["doctor", "--root", str(PROJECT_ROOT), "--json"],
            output=output,
        )
        result = json.loads(output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(
            [check["name"] for check in result["checks"]],
            ["python_version", "project_layout"],
        )

    def test_human_output_and_failure_exit_code(self) -> None:
        output = StringIO()
        with TemporaryDirectory() as directory:
            exit_code = main(
                ["doctor", "--root", directory],
                output=output,
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("CourtGraph 0.1.0: unhealthy", output.getvalue())
        self.assertIn("[FAIL] project_layout", output.getvalue())


if __name__ == "__main__":
    unittest.main()

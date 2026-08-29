"""CourtGraph's dependency-free bootstrap command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence, TextIO

from courtgraph.health import run_health_checks


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without performing any side effects."""

    parser = argparse.ArgumentParser(
        prog="courtgraph",
        description="CourtGraph research and data-pipeline tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="verify the Python runtime and required project files",
    )
    doctor.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project root to inspect (default: current directory)",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit a stable machine-readable result",
    )
    return parser


def _render_human(result: dict[str, object], output: TextIO) -> None:
    """Render a compact result for a developer at the terminal."""

    print(f"CourtGraph {result['courtgraph_version']}: {result['status']}", file=output)
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['name']}: {check['detail']}", file=output)


def main(argv: Sequence[str] | None = None, output: TextIO | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    args = build_parser().parse_args(argv)
    stream = output or sys.stdout

    if args.command == "doctor":
        result = run_health_checks(args.root)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True), file=stream)
        else:
            _render_human(result, stream)
        return 0 if result["status"] == "healthy" else 1

    raise AssertionError(f"Unhandled command: {args.command}")

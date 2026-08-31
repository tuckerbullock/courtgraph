"""Shared output-safety helpers for the ingest tools.

The pipeline validates its ``--out-dir`` this way already; the SRC-SHUFINSKIY
converter and the report writer reuse the same checks so a destination can
never overwrite an input and a write never follows a symlink.
"""

from __future__ import annotations

from pathlib import Path


class OutputPathError(ValueError):
    """Raised when a destination would overwrite an input or follow a symlink."""


def readlink(path: Path) -> str:
    try:
        return str(path.readlink())
    except OSError:  # pragma: no cover - defensive
        return "?"


def reject_overlap(inp: Path, out: Path, *, in_label: str, out_label: str) -> None:
    """Fail if ``out`` resolves to, inside, or a parent of ``inp`` (symlinks
    resolved first)."""

    a = inp.resolve()
    b = out.resolve()
    if a == b or b.is_relative_to(a) or a.is_relative_to(b):
        raise OutputPathError(
            f"{out_label} ({b}) overlaps {in_label} ({a}); "
            f"choose a {out_label} outside the inputs"
        )


def assert_directory_ok(out_dir: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise OutputPathError(f"destination exists and is not a directory: {out_dir}")


def assert_not_symlink(*paths: Path) -> None:
    for path in paths:
        if path.is_symlink():
            raise OutputPathError(
                f"refusing to write through a symlink: {path} -> {readlink(path)}"
            )


def writable(path: Path) -> Path:
    """Return ``path`` iff it is safe to write; refuse a symlink outright."""

    assert_not_symlink(path)
    return path

"""Shared output-safety helpers for the ingest tools.

The pipeline validates its ``--out-dir`` this way already; the SRC-SHUFINSKIY
converter and the report writer reuse the same checks so a destination can
never overwrite an input and a write never follows a symlink -- through a file
symlink *or* a directory symlink somewhere in the path.
"""

from __future__ import annotations

from pathlib import Path

# Distinct from the ingest pipeline's own managed-block markers
# (``# BEGIN courtgraph ingest ...``) so the two blocks never strip each other
# when both a snapshot/report writer and the pipeline touch one ``.gitignore``.
_GITIGNORE_BLOCK_START = "# BEGIN courtgraph generated outputs (keep last)"
_GITIGNORE_BLOCK_END = "# END courtgraph generated outputs"


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


def safe_target(base: Path, target: Path) -> Path:
    """Return ``target`` iff it is safe to write under ``base``.

    Rejects ``target`` (or any directory between it and ``base``) being a
    symlink, and rejects a ``target`` that resolves outside ``base`` -- so a
    symlinked ``pbp/`` or ``game_details/`` cannot redirect a write into a
    source directory. ``base`` itself is trusted (the caller validated it with
    :func:`reject_overlap`).
    """

    try:
        parts = target.relative_to(base).parts
    except ValueError as exc:
        raise OutputPathError(f"{target} is not under {base}") from exc
    current = base
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise OutputPathError(
                "refusing to write through a symlinked path component: "
                f"{current} -> {readlink(current)}"
            )
    resolved = target.resolve()
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise OutputPathError(f"{target} resolves outside {base}")
    return target


def safe_mkdir(base: Path, subdir: Path) -> Path:
    """``mkdir`` ``subdir`` only after confirming it is safe under ``base``."""

    safe_target(base, subdir)
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def writable(path: Path) -> Path:
    """Return ``path`` iff it is safe to write; refuse a symlink outright."""

    assert_not_symlink(path)
    return path


def _strip_managed_block(lines: list[str]) -> list[str]:
    if _GITIGNORE_BLOCK_START not in lines:
        return list(lines)
    start = lines.index(_GITIGNORE_BLOCK_START)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i] == _GITIGNORE_BLOCK_END),
        len(lines) - 1,
    )
    return lines[:start] + lines[end + 1 :]


def ensure_gitignore_block(
    directory: Path, entries: list[str], *, header: str = ""
) -> None:
    """Guarantee ``directory/.gitignore`` ignores ``entries`` (e.g. ``["*"]``
    for a whole snapshot, or ``["/report.html"]`` for one file), **preserving**
    anything the caller already put there.

    The managed block is (re)written at the end of the file so its anchored
    patterns win over any earlier negation; rewriting it in place keeps repeat
    runs from growing the file.
    """

    path = directory / ".gitignore"
    assert_not_symlink(path)
    managed = [
        _GITIGNORE_BLOCK_START,
        "# Anchored last so these win over any earlier negation "
        "(Git: last match wins).",
        *entries,
        _GITIGNORE_BLOCK_END,
    ]
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        prefix = [*header.splitlines(), ""] if header else []
        path.write_text("\n".join([*prefix, *managed]) + "\n", encoding="utf-8")
        return
    original = path.read_text(encoding="utf-8")
    kept = _strip_managed_block(original.splitlines())
    while kept and not kept[-1].strip():
        kept.pop()
    new_lines = kept + (["", *managed] if kept else list(managed))
    new_text = "\n".join(new_lines) + "\n"
    if new_text != original:
        path.write_text(new_text, encoding="utf-8")

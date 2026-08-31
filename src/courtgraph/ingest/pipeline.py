"""End-to-end offline ingestion: snapshot -> validated stints + audit manifest.

Outputs written under ``out_dir``:

* ``stints.jsonl``    -- records accepted by :mod:`courtgraph.chemistry.stints`
* ``quarantine.jsonl``-- one line per excluded possession / quarantined game
* ``manifest.json``   -- the audit trail (hashes, versions, reconciliation)
* ``.gitignore``      -- created as ``*`` when absent; otherwise the caller's
  contents are kept and a marked block anchoring ``/stints.jsonl`` /
  ``/quarantine.jsonl`` / ``/manifest.json`` is (re)written at the end so it
  wins over any earlier negation. Repeated runs rewrite the block in place

The snapshot and the destination are validated **before any directory is
created or byte written**: an ``out_dir`` that resolves into the snapshot, is a
non-directory, or holds a symlinked generated file is rejected, and every
generated-file write refuses to follow a symlink. The ``pbpstats`` working copy
lives in a private, uniquely owned ``tempfile`` directory outside both the
snapshot and ``out_dir`` and is deleted on the way out.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from courtgraph.chemistry.stints import Stint, StintTable, write_stints
from courtgraph.ingest.manifest import AuditManifest, GameManifest
from courtgraph.ingest.policy import IngestPolicy
from courtgraph.ingest.possessions import (
    IngestNetworkAttempt,
    PossessionReconstructionError,
    reconstruct_game,
)
from courtgraph.ingest.snapshot import (
    GameMetadata,
    Snapshot,
    SnapshotError,
    load_snapshot,
    sha256_file,
    stage_working_copy,
)
from courtgraph.ingest.stints import possessions_to_stints
from courtgraph.ingest.validate import Exclusion, validate_game

_OUT_DIR_GITIGNORE_HEADER = (
    "# Written by `courtgraph ingest`. Everything here is NBA-derived data\n"
    "# (DATA_SOURCES.md 1 / 5.1) and must never be committed.\n"
    "*\n"
)
# Every file the run writes into --out-dir. All get the same protection:
# never followed through a symlink, and all covered by the output .gitignore.
_GENERATED_FILES = (".gitignore", "stints.jsonl", "quarantine.jsonl", "manifest.json")
# The generated *data* files whose exclusion the run guarantees in .gitignore.
_GENERATED_DATA_FILES = ("stints.jsonl", "quarantine.jsonl", "manifest.json")
# Markers around the block `courtgraph ingest` owns and rewrites in place.
_GITIGNORE_BLOCK_START = "# BEGIN courtgraph ingest (generated data — keep last)"
_GITIGNORE_BLOCK_END = "# END courtgraph ingest"


@dataclass(frozen=True)
class IngestResult:
    out_dir: Path
    stints_path: Path
    quarantine_path: Path
    manifest_path: Path
    stints_written: int
    games_accepted: int
    games_quarantined: int
    possessions_excluded: int


def _validate_destination(snapshot_dir: Path, out_dir: Path) -> None:
    """Reject an unsafe ``--out-dir`` **before anything is created or written**.

    Rejects: an output directory that resolves to (or inside, or a parent of)
    the snapshot -- symlinks resolved first -- so ingestion can never clobber
    the immutable inputs; an ``--out-dir`` that exists as a non-directory; and
    any generated-file path that already exists as a symlink (writing would
    follow it, potentially into the snapshot).
    """

    snap = snapshot_dir.resolve()
    out = out_dir.resolve()
    if snap == out or out.is_relative_to(snap) or snap.is_relative_to(out):
        raise SnapshotError(
            f"--snapshot-dir ({snap}) and --out-dir ({out}) overlap; "
            "choose an output directory outside the snapshot"
        )
    if out_dir.exists() and not out_dir.is_dir():
        raise SnapshotError(f"--out-dir exists and is not a directory: {out_dir}")
    for name in _GENERATED_FILES:
        target = out_dir / name
        if target.is_symlink():
            raise SnapshotError(
                f"refusing to write generated file through a symlink: {target} "
                f"-> {_readlink(target)}"
            )


def _readlink(path: Path) -> str:
    try:
        return str(path.readlink())
    except OSError:  # pragma: no cover - defensive
        return "?"


def _writable(path: Path) -> Path:
    """Return ``path`` only if it is safe to write; refuse a symlink outright.

    A last-line check right before every generated-file write, so a symlink
    planted after :func:`_validate_destination` still cannot redirect a write
    into the snapshot.
    """

    if path.is_symlink():
        raise SnapshotError(
            f"refusing to write generated file through a symlink: {path} "
            f"-> {_readlink(path)}"
        )
    return path


def _managed_gitignore_block() -> list[str]:
    """The lines ``courtgraph ingest`` owns: anchored exclusions for the
    generated data files, wrapped in markers so a later run rewrites them in
    place instead of appending again.
    """

    return [
        _GITIGNORE_BLOCK_START,
        "# Anchored last so these win over any earlier negation "
        "(Git: last match wins).",
        *(f"/{name}" for name in _GENERATED_DATA_FILES),
        _GITIGNORE_BLOCK_END,
    ]


def _without_managed_block(lines: list[str]) -> list[str]:
    if _GITIGNORE_BLOCK_START not in lines:
        return list(lines)
    start = lines.index(_GITIGNORE_BLOCK_START)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i] == _GITIGNORE_BLOCK_END),
        len(lines) - 1,
    )
    return lines[:start] + lines[end + 1 :]


def _ensure_output_gitignore(out: Path) -> None:
    """Guarantee the output directory ignores the generated data files in Git
    while **preserving** everything the caller already put in ``.gitignore``.

    The managed block of anchored exclusions is always (re)written at the end
    of the file, so it overrides any earlier ``!`` negation of a generated data
    file. Protection is never inferred from the mere presence of ``*`` or of an
    exact rule (either can be undone by a later negation). Rewriting the block
    in place keeps repeated runs from growing the file.
    """

    path = _writable(out / ".gitignore")
    managed = _managed_gitignore_block()

    if not path.exists():
        lines = _OUT_DIR_GITIGNORE_HEADER.splitlines() + ["", *managed]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    original = path.read_text(encoding="utf-8")
    kept = _without_managed_block(original.splitlines())
    while kept and not kept[-1].strip():
        kept.pop()
    new_lines = kept + (["", *managed] if kept else list(managed))
    new_text = "\n".join(new_lines) + "\n"
    if new_text != original:
        path.write_text(new_text, encoding="utf-8")


def run_ingest(
    snapshot_dir: str | Path,
    out_dir: str | Path,
    *,
    policy: IngestPolicy | None = None,
) -> IngestResult:
    policy = policy or IngestPolicy()
    snapshot_path = Path(snapshot_dir)
    out = Path(out_dir)

    # Validate everything that can be validated by reading only -- the snapshot
    # and the destination -- before creating a directory or writing a byte.
    snapshot: Snapshot = load_snapshot(snapshot_path)
    _validate_destination(snapshot_path, out)

    out.mkdir(parents=True, exist_ok=True)
    _ensure_output_gitignore(out)

    # A private, uniquely owned working directory outside snapshot and out_dir.
    work_dir = Path(tempfile.mkdtemp(prefix="courtgraph-ingest-work-"))
    try:
        stage_working_copy(snapshot, work_dir)
        return _ingest(snapshot, snapshot_path, out, work_dir, policy)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _ingest(
    snapshot: Snapshot,
    snapshot_path: Path,
    out: Path,
    work_dir: Path,
    policy: IngestPolicy,
) -> IngestResult:
    season_index = {
        season: idx
        for idx, season in enumerate(
            sorted({g.metadata.season for g in snapshot.games})
        )
    }

    manifest = AuditManifest(
        snapshot_root=str(snapshot_path.resolve()),
        parser={
            "tool": "pbpstats",
            "version": _pbpstats_version(),
            "mode": "file",
            "data_provider": "stats_nba",
            "hosted_archive_used": False,
        },
        policy=policy.to_dict(),
        corrections={
            "correction_set_id": snapshot.correction_set_id,
            "override_files": dict(snapshot.override_hashes),
        },
    )

    all_stints: list[Stint] = []
    quarantine_lines: list[dict[str, object]] = []

    for game in snapshot.games:
        meta = game.metadata
        idx = season_index[meta.season]
        try:
            views = reconstruct_game(work_dir, meta.game_id)
        except (PossessionReconstructionError, IngestNetworkAttempt) as exc:
            reason = (
                "network_required"
                if isinstance(exc, IngestNetworkAttempt)
                else f"pbpstats_reconstruction_failed:{exc.cause_type}"
            )
            exclusion = Exclusion(
                game_id=meta.game_id, level="game", reason=reason, detail=str(exc)
            )
            quarantine_lines.append(exclusion.to_dict())
            manifest.games.append(
                _game_manifest(
                    meta,
                    idx,
                    "quarantined",
                    reason,
                    0,
                    [exclusion],
                    0,
                    {},
                    {},
                    [],
                    game.file_hashes,
                    snapshot.correction_set_id,
                )
            )
            continue

        validation = validate_game(views, meta, policy)
        stints = (
            possessions_to_stints(validation.accepted, meta, policy, idx)
            if not validation.game_quarantined
            else []
        )
        _validate_stint_ids(stints)
        all_stints.extend(stints)

        for exclusion in validation.exclusions:
            # period-end empty possessions are an expected pbpstats artifact:
            # recorded in the manifest, not in the quarantine feed.
            if exclusion.reason != "empty_possession":
                quarantine_lines.append(exclusion.to_dict())

        manifest.games.append(
            _game_manifest(
                meta,
                idx,
                "quarantined" if validation.game_quarantined else "accepted",
                validation.quarantine_reason,
                validation.reconstructed_possessions,
                validation.exclusions,
                len(validation.accepted),
                validation.source_event_counts,
                validation.reconciliation,
                validation.flags,
                game.file_hashes,
                snapshot.correction_set_id,
                stints_emitted=len(stints),
            )
        )

    table = StintTable.from_stints(all_stints)
    stints_path = write_stints(table, _writable(out / "stints.jsonl"))
    quarantine_path = _write_jsonl(
        _writable(out / "quarantine.jsonl"), quarantine_lines
    )

    manifest.outputs = {
        "stints_path": stints_path.name,
        "stints_sha256": sha256_file(stints_path) if stints_path.stat().st_size else "",
        "stints_written": len(all_stints),
        "quarantine_path": quarantine_path.name,
        "quarantine_sha256": (
            sha256_file(quarantine_path) if quarantine_path.stat().st_size else ""
        ),
        "working_copy": "private tempfile directory, discarded after the run",
    }
    manifest_path = manifest.write(_writable(out / "manifest.json"))

    return IngestResult(
        out_dir=out,
        stints_path=stints_path,
        quarantine_path=quarantine_path,
        manifest_path=manifest_path,
        stints_written=len(all_stints),
        games_accepted=manifest.totals["games_accepted"],
        games_quarantined=manifest.totals["games_quarantined"],
        possessions_excluded=manifest.totals["possessions_excluded"],
    )


def _game_manifest(
    meta: GameMetadata,
    season_idx: int,
    status: str,
    quarantine_reason: str,
    reconstructed: int,
    exclusions: list[Exclusion],
    accepted: int,
    source_event_counts: dict[str, int],
    reconciliation: dict[str, object],
    flags: list[str],
    file_hashes: dict[str, str],
    correction_set_id: str,
    *,
    stints_emitted: int = 0,
) -> GameManifest:
    return GameManifest(
        game_id=meta.game_id,
        game_date=meta.game_date,
        season=meta.season,
        season_index=season_idx,
        status=status,
        quarantine_reason=quarantine_reason,
        reconstructed_possessions=reconstructed,
        accepted_possessions=accepted,
        excluded_possessions=[e.to_dict() for e in exclusions],
        stints_emitted=stints_emitted,
        source_event_counts=source_event_counts,
        reconciliation=reconciliation,
        flags=list(flags),
        input_files=dict(file_hashes),
        correction_set_id=correction_set_id,
    )


def _validate_stint_ids(stints: list[Stint]) -> None:
    seen: set[str] = set()
    for stint in stints:
        if stint.stint_id in seen:
            raise ValueError(f"ingest produced a duplicate stint_id: {stint.stint_id}")
        seen.add(stint.stint_id)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return path


def _pbpstats_version() -> str:
    try:
        return importlib.metadata.version("pbpstats")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__all__ = ["IngestResult", "run_ingest", "load_snapshot"]

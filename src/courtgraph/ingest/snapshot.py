"""The one documented offline snapshot layout, and safe access to it.

Snapshot format ``stats_nba_pbpstats/v1`` -- a directory:

```
<snapshot-dir>/
  courtgraph_snapshot.json                     # index + required game metadata
  pbp/stats_<game_id>.json                     # raw playbyplayv2 (unmodified)
  game_details/stats_home_shots_<game_id>.json # raw shotchartdetail, home (unmodified)
  game_details/stats_away_shots_<game_id>.json # raw shotchartdetail, away (unmodified)
  game_details/stats_boxscore_<game_id>.json   # optional boxscoretraditionalv2
  overrides/*.json                             # optional pbpstats overrides, verbatim
```

``pbp/`` and ``game_details/*_shots_*.json`` are raw stats.nba.com response
payloads consumed by ``pbpstats`` unchanged. ``courtgraph_snapshot.json``
carries exactly the facts that are *not* in play-by-play and that a real
pipeline would take from ``boxscoresummaryv2`` (date, teams, official period /
final scores) and ``leaguegamelog`` (rest days):

```json
{
  "snapshot_format": "stats_nba_pbpstats/v1",
  "games": [
    {
      "game_id": "0022300001",
      "game_date": "2023-10-24",
      "season": "2023-24",
      "season_type": "Regular Season",
      "home_team_id": 1610612739,
      "away_team_id": 1610612744,
      "days_rest": {"1610612739": 3, "1610612744": 2},
      "reconciliation": {
        "final_score": {"1610612739": 114, "1610612744": 110},
        "period_scores": {
          "1610612739": [28, 30, 27, 29],
          "1610612744": [25, 31, 26, 28]
        }
      }
    }
  ]
}
```

The adapter treats the snapshot as **immutable**: it hashes every file it reads
(``pbpstats`` overrides included) and runs ``pbpstats`` against a private,
uniquely owned ``tempfile`` *working copy* -- ``pbpstats`` rewrites pbp files in
place when it fixes event order, and the copy is discarded after the run.
:func:`courtgraph.ingest.pipeline.run_ingest` also rejects an ``out_dir`` that
overlaps the snapshot.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from courtgraph.ingest import SNAPSHOT_FORMAT

INDEX_FILENAME = "courtgraph_snapshot.json"
_REQUIRED_META_FIELDS = (
    "game_id",
    "game_date",
    "season",
    "season_type",
    "home_team_id",
    "away_team_id",
)


class SnapshotError(ValueError):
    """Raised when a snapshot directory does not satisfy the v1 contract."""


@dataclass(frozen=True)
class GameMetadata:
    """The non-play-by-play facts required to place a game on the timeline."""

    game_id: str
    game_date: str  # ISO "YYYY-MM-DD"
    season: str
    season_type: str
    home_team_id: int
    away_team_id: int
    days_rest: dict[int, int]  # team_id -> whole days of rest (>= 0)
    final_score: dict[int, int]  # team_id -> official final points
    period_scores: dict[int, list[int]]  # team_id -> official points by period
    reconciliation_source: str = ""  # provenance of final_score / period_scores

    @property
    def playoff(self) -> bool:
        return self.season_type.lower() in {"playoffs", "playin", "play-in"}

    def missing_context(self) -> list[str]:
        """Names of required-but-absent context fields (never fabricated).

        Reconciliation fails closed: an official final score is required for
        **both** of the game's teams, not just one.
        """

        gaps: list[str] = []
        if not self.game_date:
            gaps.append("game_date")
        for team in (self.home_team_id, self.away_team_id):
            if team not in self.days_rest:
                gaps.append(f"days_rest[{team}]")
            if team not in self.final_score:
                gaps.append(f"reconciliation.final_score[{team}]")
        return gaps


@dataclass(frozen=True)
class SnapshotGame:
    """One game's file set plus content hashes, ready to ingest."""

    metadata: GameMetadata
    pbp_path: Path
    home_shots_path: Path
    away_shots_path: Path
    boxscore_path: Path | None
    file_hashes: dict[str, str]  # relative path -> sha256


@dataclass(frozen=True)
class Snapshot:
    """A validated view over a snapshot directory."""

    root: Path
    games: tuple[SnapshotGame, ...]
    override_files: tuple[Path, ...] = field(default_factory=tuple)
    override_hashes: dict[str, str] = field(default_factory=dict)  # rel path -> sha256
    source_provenance: dict[str, Any] = field(default_factory=dict)  # provenance.json

    def __iter__(self) -> Iterator[SnapshotGame]:
        return iter(self.games)

    @property
    def correction_set_id(self) -> str:
        """A reproducible identity for the consumed ``pbpstats`` override set.

        ``pbpstats`` overrides (``missing_period_starters.json``,
        ``bad_pbp_possessions.json``) change reconstructed lineups and
        possessions, so their content is part of every derived row's provenance.
        """

        if not self.override_hashes:
            return "cg-corrections/none"
        payload = json.dumps(self.override_hashes, sort_keys=True)
        return "cg-corrections/" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    """Content hash of a file, used for the immutable-input audit trail."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_int_keyed(mapping: Any) -> dict[int, Any]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[int, Any] = {}
    for key, value in mapping.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out


def _read_metadata(entry: dict[str, Any]) -> GameMetadata:
    for name in _REQUIRED_META_FIELDS:
        if name not in entry:
            raise SnapshotError(f"game metadata missing required field {name!r}")
    recon = entry.get("reconciliation") or {}
    days_rest_raw = _coerce_int_keyed(entry.get("days_rest") or {})
    days_rest = {k: int(v) for k, v in days_rest_raw.items() if v is not None}
    final_raw = _coerce_int_keyed(recon.get("final_score") or {})
    final_score = {k: int(v) for k, v in final_raw.items()}
    period_raw = _coerce_int_keyed(recon.get("period_scores") or {})
    period_scores = {
        k: [int(x) for x in v] for k, v in period_raw.items() if isinstance(v, list)
    }
    return GameMetadata(
        game_id=str(entry["game_id"]),
        game_date=str(entry["game_date"]),
        season=str(entry["season"]),
        season_type=str(entry["season_type"]),
        home_team_id=int(entry["home_team_id"]),
        away_team_id=int(entry["away_team_id"]),
        days_rest=days_rest,
        final_score=final_score,
        period_scores=period_scores,
        reconciliation_source=str(recon.get("source") or ""),
    )


def load_snapshot(snapshot_dir: str | Path) -> Snapshot:
    """Parse and structurally validate a ``stats_nba_pbpstats/v1`` snapshot.

    Structural problems (missing index, unknown format, missing pbp/shots files)
    raise :class:`SnapshotError`. *Content* problems (bad event order, missing
    context, reconstruction failure) are not raised here -- they become
    per-game quarantine entries during ingestion.
    """

    root = Path(snapshot_dir)
    if not root.is_dir():
        raise SnapshotError(f"snapshot directory not found: {root}")
    index_path = root / INDEX_FILENAME
    if not index_path.is_file():
        raise SnapshotError(f"{root}: missing {INDEX_FILENAME}")

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"{index_path}: invalid JSON ({exc})") from exc

    fmt = index.get("snapshot_format")
    if fmt != SNAPSHOT_FORMAT:
        raise SnapshotError(
            f"{index_path}: snapshot_format {fmt!r} != {SNAPSHOT_FORMAT!r}"
        )
    raw_games = index.get("games")
    if not isinstance(raw_games, list) or not raw_games:
        raise SnapshotError(f"{index_path}: 'games' must be a non-empty list")

    # Overrides are game-agnostic but they steer reconstructed lineups and
    # possessions, so their hashes belong in every game's provenance.
    overrides_dir = root / "overrides"
    override_files = (
        tuple(sorted(p for p in overrides_dir.glob("*.json")))
        if overrides_dir.is_dir()
        else ()
    )
    override_hashes = {str(p.relative_to(root)): sha256_file(p) for p in override_files}

    # Optional provenance sidecar (written by `snapshot-from-shufinskiy`).
    provenance_path = root / "provenance.json"
    source_provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"{provenance_path}: invalid JSON ({exc})") from exc
        if isinstance(loaded, dict):
            source_provenance = loaded

    games: list[SnapshotGame] = []
    seen: set[str] = set()
    for entry in raw_games:
        if not isinstance(entry, dict):
            raise SnapshotError(f"{index_path}: each game entry must be an object")
        meta = _read_metadata(entry)
        if meta.game_id in seen:
            raise SnapshotError(f"{index_path}: duplicate game_id {meta.game_id!r}")
        seen.add(meta.game_id)

        pbp = root / "pbp" / f"stats_{meta.game_id}.json"
        home_shots = root / "game_details" / f"stats_home_shots_{meta.game_id}.json"
        away_shots = root / "game_details" / f"stats_away_shots_{meta.game_id}.json"
        boxscore = root / "game_details" / f"stats_boxscore_{meta.game_id}.json"
        for required in (pbp, home_shots, away_shots):
            if not required.is_file():
                raise SnapshotError(
                    f"game {meta.game_id}: required file not found: "
                    f"{required.relative_to(root)}"
                )

        hashes = {
            str(index_path.relative_to(root)): sha256_file(index_path),
            str(pbp.relative_to(root)): sha256_file(pbp),
            str(home_shots.relative_to(root)): sha256_file(home_shots),
            str(away_shots.relative_to(root)): sha256_file(away_shots),
        }
        if boxscore.is_file():
            hashes[str(boxscore.relative_to(root))] = sha256_file(boxscore)
        hashes.update(override_hashes)

        games.append(
            SnapshotGame(
                metadata=meta,
                pbp_path=pbp,
                home_shots_path=home_shots,
                away_shots_path=away_shots,
                boxscore_path=boxscore if boxscore.is_file() else None,
                file_hashes=hashes,
            )
        )

    return Snapshot(
        root=root,
        games=tuple(games),
        override_files=override_files,
        override_hashes=override_hashes,
        source_provenance=source_provenance,
    )


def stage_working_copy(snapshot: Snapshot, work_dir: str | Path) -> Path:
    """Populate an **already-created, uniquely owned, empty** ``work_dir`` with
    the files ``pbpstats`` needs, leaving the snapshot untouched (``pbpstats``
    rewrites pbp files in place when it fixes event order).

    This function never deletes anything: the caller owns ``work_dir`` (see
    :func:`courtgraph.ingest.pipeline.run_ingest`, which uses a private
    ``tempfile`` directory) and unrelated files elsewhere are preserved.
    """

    work = Path(work_dir)
    (work / "pbp").mkdir(parents=True, exist_ok=True)
    (work / "game_details").mkdir(parents=True, exist_ok=True)
    for game in snapshot.games:
        shutil.copy2(game.pbp_path, work / "pbp" / game.pbp_path.name)
        shutil.copy2(
            game.home_shots_path, work / "game_details" / game.home_shots_path.name
        )
        shutil.copy2(
            game.away_shots_path, work / "game_details" / game.away_shots_path.name
        )
        if game.boxscore_path is not None:
            shutil.copy2(
                game.boxscore_path, work / "game_details" / game.boxscore_path.name
            )
    if snapshot.override_files:
        (work / "overrides").mkdir(exist_ok=True)
        for override in snapshot.override_files:
            shutil.copy2(override, work / "overrides" / override.name)
    return work

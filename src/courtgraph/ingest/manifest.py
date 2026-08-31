"""The ingestion audit manifest (master plan 6.1, research contract 24).

Every run writes a ``manifest.json`` recording the immutable-input hashes, the
tool/policy versions, the run timestamp, per-game source-event counts, the
reconciliation result, and every exclusion -- enough to re-derive or dispute any
emitted stint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from courtgraph.ingest import INGEST_PIPELINE_VERSION, SNAPSHOT_FORMAT


@dataclass
class GameManifest:
    game_id: str
    game_date: str
    season: str
    season_index: int
    status: str  # "accepted" | "quarantined"
    quarantine_reason: str
    reconstructed_possessions: int
    accepted_possessions: int
    excluded_possessions: list[dict[str, Any]]
    stints_emitted: int
    source_event_counts: dict[str, int]
    reconciliation: dict[str, Any]
    flags: list[str]
    input_files: dict[str, str]
    correction_set_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "game_date": self.game_date,
            "season": self.season,
            "season_index": self.season_index,
            "status": self.status,
            "quarantine_reason": self.quarantine_reason,
            "reconstructed_possessions": self.reconstructed_possessions,
            "accepted_possessions": self.accepted_possessions,
            "excluded_possessions": self.excluded_possessions,
            "stints_emitted": self.stints_emitted,
            "source_event_counts": self.source_event_counts,
            "reconciliation": self.reconciliation,
            "flags": self.flags,
            "input_files": self.input_files,
            "correction_set_id": self.correction_set_id,
        }


@dataclass
class AuditManifest:
    snapshot_root: str
    snapshot_format: str = SNAPSHOT_FORMAT
    ingest_pipeline_version: int = INGEST_PIPELINE_VERSION
    created_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    parser: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    corrections: dict[str, Any] = field(default_factory=dict)
    source_provenance: dict[str, Any] = field(default_factory=dict)
    games: list[GameManifest] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def totals(self) -> dict[str, int]:
        return {
            "games_in": len(self.games),
            "games_accepted": sum(g.status == "accepted" for g in self.games),
            "games_quarantined": sum(g.status == "quarantined" for g in self.games),
            "possessions_reconstructed": sum(
                g.reconstructed_possessions for g in self.games
            ),
            "possessions_accepted": sum(g.accepted_possessions for g in self.games),
            "possessions_excluded": sum(
                len(g.excluded_possessions) for g in self.games
            ),
            "stints_emitted": sum(g.stints_emitted for g in self.games),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_format": self.snapshot_format,
            "ingest_pipeline_version": self.ingest_pipeline_version,
            "created_utc": self.created_utc,
            "snapshot_root": self.snapshot_root,
            "parser": self.parser,
            "policy": self.policy,
            "corrections": self.corrections,
            "source_provenance": self.source_provenance,
            "totals": self.totals,
            "games": [g.to_dict() for g in self.games],
            "outputs": self.outputs,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

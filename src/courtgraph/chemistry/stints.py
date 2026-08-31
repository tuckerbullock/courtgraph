"""Versioned stint-data input format for CourtGraph lineup-chemistry modeling.

A *stint* is a maximal interval of unchanged ten-player membership (master plan
7.5). It is the parallel unit to the possession for the RAPM / chemistry design
matrix: an offensive five, a defensive five, an exposure weight (possessions),
an outcome (points), and enough game / season / time / context to place it on a
leakage-safe timeline.

This module defines the on-disk contract only. It has no NumPy dependency and no
knowledge of any model, so real NBA stints produced by a future ingestion
pipeline can be dropped in by emitting the same records.

On disk
-------
Two interchangeable encodings, chosen by file suffix:

* ``.jsonl`` -- one JSON object per line, each carrying ``schema_version``. The
  streaming format intended for real, large datasets.
* ``.json``  -- a single object ``{"schema_version": N, "stints": [ ... ]}``.

Both round-trip through :func:`read_stints` / :func:`write_stints` exactly.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2  # v2 added the required game_date field for chronological order
LINEUP_SIZE = 5

# Context fields that models are allowed to condition on. Ordering is part of the
# contract: feature builders and serialized artifacts refer to context by name,
# never by position, but a stable order keeps reports and tests deterministic.
CONTEXT_FIELDS: tuple[str, ...] = (
    "home_offense",
    "score_margin_offense",
    "period",
    "playoff",
    "days_rest_offense",
    "garbage_time_weight",
)


class StintSchemaError(ValueError):
    """Raised when a stint record violates the versioned contract."""


@dataclass(frozen=True, slots=True)
class Stint:
    """One immutable stint record.

    All ID fields are integers or short strings; no player, team, or coach names
    are stored (master plan 6.4 -- names are never primary keys). Player id
    tuples are stored in canonical ascending order so that lineup identity is an
    unordered set (research contract 13).
    """

    stint_id: str
    game_id: str
    game_date: str  # ISO date "YYYY-MM-DD"; the chronological ordering key
    season: str
    season_index: int
    period: int
    start_time_seconds: float
    offense_team_id: int
    defense_team_id: int
    offense_player_ids: tuple[int, int, int, int, int]
    defense_player_ids: tuple[int, int, int, int, int]
    offensive_possessions: int
    points_scored: int
    # --- context (CONTEXT_FIELDS) --------------------------------------------
    home_offense: bool
    score_margin_offense: int
    playoff: bool
    days_rest_offense: int
    garbage_time_weight: float
    # --- provenance --------------------------------------------------------
    source: str = "unknown"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Lineup identity is an unordered set: canonicalise on construction so
        # every downstream comparison and hash is order-independent.
        object.__setattr__(
            self, "offense_player_ids", tuple(sorted(self.offense_player_ids))
        )
        object.__setattr__(
            self, "defense_player_ids", tuple(sorted(self.defense_player_ids))
        )
        _validate_stint(self)

    @property
    def offensive_rating(self) -> float:
        """Offensive points per 100 possessions -- the modeled lineup value."""

        return 100.0 * self.points_scored / self.offensive_possessions

    @property
    def offense_lineup_id(self) -> str:
        """Canonical string id for the offensive five-player set."""

        return lineup_id(self.offense_player_ids)

    def context_vector(self) -> dict[str, float]:
        """Context as a name -> float mapping in ``CONTEXT_FIELDS`` order."""

        return {
            "home_offense": float(self.home_offense),
            "score_margin_offense": float(self.score_margin_offense),
            "period": float(self.period),
            "playoff": float(self.playoff),
            "days_rest_offense": float(self.days_rest_offense),
            "garbage_time_weight": float(self.garbage_time_weight),
        }

    def to_record(self) -> dict[str, Any]:
        """JSON-serializable mapping (tuples become lists)."""

        record = asdict(self)
        record["offense_player_ids"] = list(self.offense_player_ids)
        record["defense_player_ids"] = list(self.defense_player_ids)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Stint:
        """Build a :class:`Stint` from a decoded JSON mapping."""

        version = record.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise StintSchemaError(
                f"unsupported stint schema_version {version!r}; "
                f"this build reads {SCHEMA_VERSION}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(record) - known
        if unknown:
            raise StintSchemaError(f"unknown stint field(s): {sorted(unknown)}")
        data = dict(record)
        data["offense_player_ids"] = _as_lineup(data.get("offense_player_ids"))
        data["defense_player_ids"] = _as_lineup(data.get("defense_player_ids"))
        try:
            return cls(**data)
        except TypeError as exc:  # missing required field
            raise StintSchemaError(str(exc)) from exc


def lineup_id(player_ids: Iterable[int]) -> str:
    """Canonical id for an unordered player set: sorted ids joined by ``-``."""

    return "-".join(str(pid) for pid in sorted(player_ids))


def pair_id(player_a: int, player_b: int) -> str:
    """Canonical id for an unordered player pair."""

    low, high = sorted((int(player_a), int(player_b)))
    return f"{low}-{high}"


def _as_lineup(value: Any) -> tuple[int, int, int, int, int]:
    if not isinstance(value, (list, tuple)):
        raise StintSchemaError(f"lineup must be a list of 5 ids, got {value!r}")
    ids = tuple(int(v) for v in value)
    if len(ids) != LINEUP_SIZE:
        raise StintSchemaError(f"lineup must have {LINEUP_SIZE} ids, got {len(ids)}")
    return tuple(sorted(ids))  # type: ignore[return-value]


def _validate_stint(stint: Stint) -> None:
    if len(stint.offense_player_ids) != LINEUP_SIZE:
        raise StintSchemaError("offense_player_ids must have 5 entries")
    if len(stint.defense_player_ids) != LINEUP_SIZE:
        raise StintSchemaError("defense_player_ids must have 5 entries")
    if len(set(stint.offense_player_ids)) != LINEUP_SIZE:
        raise StintSchemaError("offense_player_ids must be distinct")
    if len(set(stint.defense_player_ids)) != LINEUP_SIZE:
        raise StintSchemaError("defense_player_ids must be distinct")
    if set(stint.offense_player_ids) & set(stint.defense_player_ids):
        raise StintSchemaError("a player cannot be on offense and defense at once")
    if stint.offensive_possessions <= 0:
        raise StintSchemaError("offensive_possessions must be positive")
    if stint.points_scored < 0:
        raise StintSchemaError("points_scored must be non-negative")
    if stint.period <= 0:
        raise StintSchemaError("period must be positive")
    if not 0.0 < stint.garbage_time_weight <= 1.0:
        raise StintSchemaError("garbage_time_weight must be in (0, 1]")
    if stint.season_index < 0:
        raise StintSchemaError("season_index must be non-negative")
    if stint.offense_team_id == stint.defense_team_id:
        raise StintSchemaError("offense and defense teams must differ")
    try:
        _dt.date.fromisoformat(stint.game_date)
    except ValueError as exc:
        raise StintSchemaError(
            f"game_date must be an ISO date 'YYYY-MM-DD', got {stint.game_date!r}"
        ) from exc


@dataclass(frozen=True)
class StintTable:
    """An ordered, validated collection of stints from one snapshot."""

    stints: tuple[Stint, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise StintSchemaError(
                f"unsupported table schema_version {self.schema_version!r}"
            )
        seen: set[str] = set()
        for stint in self.stints:
            if stint.stint_id in seen:
                raise StintSchemaError(f"duplicate stint_id {stint.stint_id!r}")
            seen.add(stint.stint_id)

    def __len__(self) -> int:
        return len(self.stints)

    def __iter__(self) -> Iterator[Stint]:
        return iter(self.stints)

    def __getitem__(self, index: int) -> Stint:
        return self.stints[index]

    @classmethod
    def from_stints(cls, stints: Iterable[Stint]) -> StintTable:
        return cls(tuple(stints))

    def subset(self, stint_ids: Iterable[str]) -> StintTable:
        """A new table with only the named stints, order preserved."""

        wanted = set(stint_ids)
        return StintTable(tuple(s for s in self.stints if s.stint_id in wanted))

    def player_ids(self) -> tuple[int, ...]:
        """Every player id that appears on offense or defense, sorted."""

        ids: set[int] = set()
        for stint in self.stints:
            ids.update(stint.offense_player_ids)
            ids.update(stint.defense_player_ids)
        return tuple(sorted(ids))

    def season_order(self) -> tuple[str, ...]:
        """Distinct seasons, ordered by their ``season_index``."""

        by_index: dict[int, str] = {}
        for stint in self.stints:
            by_index.setdefault(stint.season_index, stint.season)
        return tuple(by_index[i] for i in sorted(by_index))

    def chronological_key(self, stint: Stint) -> tuple[str, str, int, float]:
        """Sort key placing a stint on the global timeline.

        ``game_date`` (an explicit ISO date) is the primary key -- never the
        ``game_id`` string, which real providers do not guarantee to sort
        chronologically. ``game_id`` is only a within-day tiebreak so a day's
        games stay grouped; ``period`` and ``start_time_seconds`` order stints
        within a game.
        """

        return (
            stint.game_date,
            stint.game_id,
            stint.period,
            stint.start_time_seconds,
        )

    def sorted_chronologically(self) -> StintTable:
        return StintTable(tuple(sorted(self.stints, key=self.chronological_key)))

    def total_possessions(self) -> int:
        return sum(s.offensive_possessions for s in self.stints)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def write_stints(table: StintTable, path: str | Path) -> Path:
    """Write ``table`` to ``path``; ``.jsonl`` streams, ``.json`` wraps."""

    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for stint in table.stints:
                handle.write(json.dumps(stint.to_record(), sort_keys=True))
                handle.write("\n")
    else:
        payload = {
            "schema_version": table.schema_version,
            "stints": [s.to_record() for s in table.stints],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_stints(path: str | Path) -> StintTable:
    """Read a stint table written by :func:`write_stints`."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stint file not found: {path}")
    if path.suffix == ".jsonl":
        records = _read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise StintSchemaError(
                f"unsupported stint schema_version {version!r} in {path}"
            )
        raw = payload.get("stints")
        if not isinstance(raw, list):
            raise StintSchemaError(f"{path}: 'stints' must be a list")
        records = raw
    stints = [Stint.from_record(record) for record in records]
    return StintTable.from_stints(stints)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StintSchemaError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise StintSchemaError(f"{path}:{line_number}: expected an object")
            records.append(record)
    return records


def stint_records(stints: Sequence[Stint]) -> list[dict[str, Any]]:
    """Convenience: plain-dict view of a stint sequence (used by reports)."""

    return [s.to_record() for s in stints]

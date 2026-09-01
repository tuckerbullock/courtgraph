"""Leakage-safe holdout construction and verification.

Three splits, all **outcome-blind** (chosen by date and exposure, never by
performance) and **exhaustively removed** (research contract 12-13):

* ``chronological``   -- train on early games, test on later ones by explicit
  ``game_date``, cut on a game boundary so no game straddles the split.
* ``unseen_pair``     -- **structurally** unseen teammate pairs: every training
  stint with both players on offense is removed, while **each player individually
  stays observed in training**. This is not the research-contract "strong"
  variant (which additionally requires the pair's first-ever partnership to fall
  in the test period).
* ``unseen_lineup``   -- pick exact offensive five-player sets with enough test
  exposure; remove every training stint with that exact set. Players and pairs
  may remain observed.

:func:`verify_split` is the leakage gate: it re-derives the forbidden overlaps
from the raw stints and returns a list of violations (empty == safe). Tests and
the demo both run it; a real pipeline would wire it into CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtgraph.chemistry.stints import StintTable, lineup_id, pair_id

SPLIT_CODE_VERSION = 1
SPLIT_KINDS = ("chronological", "unseen_pair", "unseen_lineup")


class UnsatisfiableSplitError(ValueError):
    """Raised when no leakage-safe holdout can be built under the given limits.

    Preferred over returning an empty or budget-busting split: a caller that
    asked for an impossible holdout gets a descriptive failure, not a
    misleading manifest.
    """


@dataclass(frozen=True)
class SplitManifest:
    """An immutable, serializable record of one train/test partition."""

    kind: str
    split_code_version: int
    train_stint_ids: tuple[str, ...]
    test_stint_ids: tuple[str, ...]
    held_out_pairs: tuple[str, ...]
    held_out_lineups: tuple[str, ...]
    selection_reason: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in SPLIT_KINDS:
            raise ValueError(f"unknown split kind {self.kind!r}")

    @property
    def n_train(self) -> int:
        return len(self.train_stint_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_stint_ids)

    def train_table(self, table: StintTable) -> StintTable:
        return table.subset(self.train_stint_ids)

    def test_table(self, table: StintTable) -> StintTable:
        return table.subset(self.test_stint_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split_code_version": self.split_code_version,
            "train_stint_ids": list(self.train_stint_ids),
            "test_stint_ids": list(self.test_stint_ids),
            "held_out_pairs": list(self.held_out_pairs),
            "held_out_lineups": list(self.held_out_lineups),
            "selection_reason": self.selection_reason,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitManifest:
        return cls(
            kind=data["kind"],
            split_code_version=int(data["split_code_version"]),
            train_stint_ids=tuple(data["train_stint_ids"]),
            test_stint_ids=tuple(data["test_stint_ids"]),
            held_out_pairs=tuple(data["held_out_pairs"]),
            held_out_lineups=tuple(data["held_out_lineups"]),
            selection_reason=data["selection_reason"],
            parameters=dict(data.get("parameters", {})),
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def make_chronological_split(
    table: StintTable, *, train_fraction: float = 0.6
) -> SplitManifest:
    ordered = table.sorted_chronologically()
    total = ordered.total_possessions()
    target = train_fraction * total
    running = 0
    train_ids: list[str] = []
    test_ids: list[str] = []
    boundary_game: str | None = None
    for stint in ordered:
        in_train = running < target or stint.game_id == boundary_game
        if in_train:
            train_ids.append(stint.stint_id)
            running += stint.offensive_possessions
            boundary_game = stint.game_id
        else:
            test_ids.append(stint.stint_id)
    return SplitManifest(
        kind="chronological",
        split_code_version=SPLIT_CODE_VERSION,
        train_stint_ids=tuple(train_ids),
        test_stint_ids=tuple(test_ids),
        held_out_pairs=(),
        held_out_lineups=(),
        selection_reason=(
            f"train = earliest games covering >= {train_fraction:.0%} of "
            "possessions, cut on a game boundary; test = the rest"
        ),
        parameters={"train_fraction": train_fraction},
    )


def make_unseen_pair_split(
    table: StintTable,
    *,
    n_pairs: int = 60,
    min_test_stints: int = 3,
    min_solo_train_stints: int = 8,
    max_test_fraction: float = 0.15,
    max_stints_per_pair_fraction: float = 0.01,
    seed: int = 0,
) -> SplitManifest:
    """Structurally unseen teammate pairs.

    Every training stint with both players on offense is removed; a pair is only
    admitted if **each member still has at least ``min_solo_train_stints``
    offensive training stints without the partner**, so both players stay
    individually observed. Pairs are added greedily by shared-floor-time
    (descending, tiebreak by id) until ``n_pairs`` are held. ``max_test_fraction``
    is a hard upper bound on the cumulative held-out size;
    ``max_stints_per_pair_fraction`` caps any single pair so a handful of
    very-high-minute pairs cannot consume the whole budget and starve the macro
    group count. Outcome-blind.

    Raises :class:`UnsatisfiableSplitError` when the budget (and the
    solo-observation constraint) admit no pair at all, rather than returning an
    empty, unusable split.
    """

    ordered = table.sorted_chronologically()
    n_total = len(ordered)
    co_stints: dict[str, list[str]] = {}
    player_offense_stints: dict[int, set[str]] = {}
    for stint in ordered:
        ids = stint.offense_player_ids
        for pid in ids:
            player_offense_stints.setdefault(pid, set()).add(stint.stint_id)
        for a in range(5):
            for b in range(a + 1, 5):
                co_stints.setdefault(pair_id(ids[a], ids[b]), []).append(stint.stint_id)

    per_pair_cap = max(min_test_stints, int(max_stints_per_pair_fraction * n_total))
    candidates = sorted(
        (
            key
            for key, s in co_stints.items()
            if min_test_stints <= len(s) <= per_pair_cap
        ),
        key=lambda k: (-len(co_stints[k]), k),
    )

    budget = int(max_test_fraction * n_total)
    held: list[str] = []
    removed: set[str] = set()
    held_players: set[int] = set()
    for key in candidates:
        if len(held) >= n_pairs:
            break
        a, b = (int(x) for x in key.split("-"))
        trial_removed = removed | set(co_stints[key])
        # Hard upper bound for every candidate, the first one included.
        if len(trial_removed) > budget:
            continue
        players_to_check = held_players | {a, b}
        if all(
            len(player_offense_stints[p] - trial_removed) >= min_solo_train_stints
            for p in players_to_check
        ):
            held.append(key)
            removed = trial_removed
            held_players |= {a, b}

    if not held:
        smallest = min((len(co_stints[k]) for k in candidates), default=0)
        raise UnsatisfiableSplitError(
            "unseen_pair: no teammate pair fits max_test_fraction="
            f"{max_test_fraction:g} (budget {budget} of {n_total} stints) while "
            f"keeping each player to >= {min_solo_train_stints} solo training "
            "stints"
            + (
                f"; the smallest eligible pair alone needs {smallest} test stints"
                if smallest
                else f" (no pair reaches min_test_stints={min_test_stints})"
            )
            + ". Raise max_test_fraction, lower min_test_stints, or use more data."
        )

    test_ids = tuple(s.stint_id for s in ordered if s.stint_id in removed)
    train_ids = tuple(s.stint_id for s in ordered if s.stint_id not in removed)
    return SplitManifest(
        kind="unseen_pair",
        split_code_version=SPLIT_CODE_VERSION,
        train_stint_ids=train_ids,
        test_stint_ids=test_ids,
        held_out_pairs=tuple(held),
        held_out_lineups=(),
        selection_reason=(
            f"{len(held)} structurally unseen teammate pairs (most shared floor "
            "time first, outcome-blind); every training stint with both on "
            f"offense removed, each player kept to >= {min_solo_train_stints} "
            "solo training stints"
        ),
        parameters={
            "n_pairs": n_pairs,
            "held": len(held),
            "min_test_stints": min_test_stints,
            "min_solo_train_stints": min_solo_train_stints,
            "max_test_fraction": max_test_fraction,
            "max_stints_per_pair_fraction": max_stints_per_pair_fraction,
            "per_pair_cap": per_pair_cap,
            "seed": seed,
        },
    )


def make_unseen_lineup_split(
    table: StintTable,
    *,
    n_lineups: int = 60,
    min_test_stints: int = 3,
    seed: int = 0,
) -> SplitManifest:
    ordered = table.sorted_chronologically()
    lineup_stints: dict[str, list[str]] = {}
    lineup_first: dict[str, int] = {}
    for position, stint in enumerate(ordered):
        lid = stint.offense_lineup_id
        lineup_stints.setdefault(lid, []).append(stint.stint_id)
        lineup_first.setdefault(lid, position)

    eligible = [
        lid for lid, stints in lineup_stints.items() if len(stints) >= min_test_stints
    ]
    # outcome-blind: most exposure first (stable test groups), late first-seen as
    # tie-break, id last -- never sorted by the outcome.
    eligible.sort(key=lambda lid: (-len(lineup_stints[lid]), -lineup_first[lid], lid))
    held = tuple(eligible[:n_lineups])
    held_set = set(held)

    test_ids = tuple(s.stint_id for s in ordered if s.offense_lineup_id in held_set)
    test_lookup = set(test_ids)
    train_ids = tuple(s.stint_id for s in ordered if s.stint_id not in test_lookup)
    return SplitManifest(
        kind="unseen_lineup",
        split_code_version=SPLIT_CODE_VERSION,
        train_stint_ids=train_ids,
        test_stint_ids=test_ids,
        held_out_pairs=(),
        held_out_lineups=held,
        selection_reason=(
            f"{len(held)} exact offensive five-player sets with the most test "
            "exposure; every training stint with that exact set removed"
        ),
        parameters={
            "n_lineups": n_lineups,
            "held": len(held),
            "min_test_stints": min_test_stints,
            "seed": seed,
        },
    )


def make_all_splits(table: StintTable, **kwargs: Any) -> dict[str, SplitManifest]:
    return {
        "chronological": make_chronological_split(table),
        "unseen_pair": make_unseen_pair_split(table),
        "unseen_lineup": make_unseen_lineup_split(table),
    }


# --------------------------------------------------------------------------- #
# Verification (the leakage gate)
# --------------------------------------------------------------------------- #


def verify_split(table: StintTable, manifest: SplitManifest) -> list[str]:
    """Return a list of leakage violations; an empty list means the split is safe."""

    violations: list[str] = []
    all_ids = {s.stint_id for s in table}
    train = set(manifest.train_stint_ids)
    test = set(manifest.test_stint_ids)

    if train & test:
        violations.append(f"{len(train & test)} stint id(s) are in both train and test")
    unknown = (train | test) - all_ids
    if unknown:
        violations.append(f"{len(unknown)} manifest stint id(s) are not in the table")
    missing = all_ids - (train | test)
    if missing:
        violations.append(
            f"{len(missing)} table stint(s) are in neither train nor test"
        )
    if not test:
        violations.append("test set is empty")

    by_id = {s.stint_id: s for s in table}
    train_stints = [by_id[i] for i in manifest.train_stint_ids if i in by_id]
    test_stints = [by_id[i] for i in manifest.test_stint_ids if i in by_id]

    if manifest.kind == "chronological":
        violations += _verify_chronological(table, train_stints, test_stints)
    elif manifest.kind == "unseen_pair":
        violations += _verify_unseen_pair(manifest, train_stints, test_stints)
    elif manifest.kind == "unseen_lineup":
        violations += _verify_unseen_lineup(manifest, train_stints, test_stints)

    return violations


def _verify_chronological(
    table: StintTable, train_stints: list[Any], test_stints: list[Any]
) -> list[str]:
    if not train_stints or not test_stints:
        return []
    key = table.chronological_key
    max_train = max(key(s) for s in train_stints)
    min_test = min(key(s) for s in test_stints)
    problems: list[str] = []
    if max_train >= min_test:
        problems.append("a training stint is not strictly before every test stint")
    train_games = {s.game_id for s in train_stints}
    test_games = {s.game_id for s in test_stints}
    straddle = train_games & test_games
    if straddle:
        problems.append(f"{len(straddle)} game(s) appear in both train and test")
    return problems


def _verify_unseen_pair(
    manifest: SplitManifest, train_stints: list[Any], test_stints: list[Any]
) -> list[str]:
    problems: list[str] = []
    held = [tuple(int(x) for x in key.split("-")) for key in manifest.held_out_pairs]
    train_offense_players: set[int] = set()
    for s in train_stints:
        train_offense_players.update(s.offense_player_ids)
    for a, b in held:
        leaked = sum(
            1
            for s in train_stints
            if a in s.offense_player_ids and b in s.offense_player_ids
        )
        if leaked:
            problems.append(
                f"pair {a}-{b}: {leaked} training stint(s) have both on offense"
            )
        appears = sum(
            1
            for s in test_stints
            if a in s.offense_player_ids and b in s.offense_player_ids
        )
        if not appears:
            problems.append(f"pair {a}-{b}: no test stint contains the pair")
        for player in (a, b):
            if player not in train_offense_players:
                problems.append(
                    f"pair {a}-{b}: player {player} never appears on offense in "
                    "training (must stay individually observed)"
                )
    return problems


def _verify_unseen_lineup(
    manifest: SplitManifest, train_stints: list[Any], test_stints: list[Any]
) -> list[str]:
    problems: list[str] = []
    held = set(manifest.held_out_lineups)
    for lid in held:
        leaked = sum(1 for s in train_stints if s.offense_lineup_id == lid)
        if leaked:
            problems.append(f"lineup {lid}: {leaked} training stint(s) use it exactly")
        if not any(s.offense_lineup_id == lid for s in test_stints):
            problems.append(f"lineup {lid}: no test stint uses it")
    return problems


def novelty_of_lineup(train_table: StintTable, offense_ids: tuple[int, ...]) -> str:
    """Novelty class of an offensive set relative to a training table."""

    seen_lineups = {s.offense_lineup_id for s in train_table}
    if lineup_id(offense_ids) in seen_lineups:
        return "seen"
    seen_pairs: set[str] = set()
    seen_players: set[int] = set()
    for stint in train_table:
        seen_players.update(stint.offense_player_ids)
        ids = stint.offense_player_ids
        for a in range(5):
            for b in range(a + 1, 5):
                seen_pairs.add(pair_id(ids[a], ids[b]))
    if any(p not in seen_players for p in offense_ids):
        return "unseen"
    all_pairs_seen = all(
        pair_id(offense_ids[a], offense_ids[b]) in seen_pairs
        for a in range(len(offense_ids))
        for b in range(a + 1, len(offense_ids))
    )
    return "partially-seen" if all_pairs_seen else "unseen"

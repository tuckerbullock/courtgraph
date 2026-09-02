"""Derive a roster-change cohort from the stint data itself.

The contract's T4 test (roster changes as natural experiments) needs a
transaction dataset. Rather than scrape one, this module reconstructs the
clean, unambiguous subset directly: a player whose **team of record changes
between two consecutive seasons**, with meaningful exposure on both sides.

Mid-season trades (a player appearing for two teams within one season) are a
documented follow-up -- the cutover date is fuzzier (buyout, 10-day, two-way)
and the leakage-safe split is messier. v1 is cross-season only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from courtgraph.chemistry.stints import StintTable


@dataclass(frozen=True)
class Transaction:
    """One player's clean cross-season move from ``from_team`` to ``to_team``.

    ``cutover_season`` is the first season on the new team; the leakage-safe
    training window is every stint in a season strictly earlier than it, so a
    model fit on that window has never seen the player on the new team.
    """

    player_id: int
    from_team_id: int
    to_team_id: int
    from_season: str
    cutover_season: str
    pre_possessions: int
    post_possessions: int
    post_stint_ids: tuple[str, ...]  # new team, cutover season, player on floor


def _player_team_possessions(
    table: StintTable,
) -> dict[int, dict[str, dict[int, int]]]:
    """player -> season -> team -> offensive possessions on the floor."""

    out: dict[int, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for s in table:
        for pid in s.offense_player_ids:
            out[pid][s.season][s.offense_team_id] += s.offensive_possessions
    return out


def _season_rank(table: StintTable) -> dict[str, int]:
    rank: dict[str, int] = {}
    for s in table:
        rank.setdefault(s.season, s.season_index)
    # normalise to a dense chronological order
    order = sorted(rank, key=lambda k: rank[k])
    return {season: i for i, season in enumerate(order)}


def find_transactions(
    table: StintTable, *, min_poss_each_side: int = 500
) -> list[Transaction]:
    """Every clean cross-season team switch: the player's dominant team (by
    offensive possessions) differs between consecutive seasons, both teams are
    disjoint, and exposure clears ``min_poss_each_side`` on each side."""

    ptp = _player_team_possessions(table)
    rank = _season_rank(table)
    by_rank = sorted(rank, key=lambda k: rank[k])

    # index of each (team, season, player) -> stint ids, for the post window
    post_index: dict[tuple[int, str, int], list[str]] = defaultdict(list)
    for s in table:
        for pid in s.offense_player_ids:
            post_index[(s.offense_team_id, s.season, pid)].append(s.stint_id)

    txns: list[Transaction] = []
    for pid, seasons in ptp.items():
        for a, b in zip(by_rank, by_rank[1:], strict=False):
            if a not in seasons or b not in seasons:
                continue
            ta = {t: c for t, c in seasons[a].items() if c >= min_poss_each_side}
            tb = {t: c for t, c in seasons[b].items() if c >= min_poss_each_side}
            if len(ta) != 1 or len(tb) != 1:
                continue  # split seasons on either side -> not a clean switch
            (from_team, pre_poss), (to_team, post_poss) = (
                next(iter(ta.items())),
                next(iter(tb.items())),
            )
            if from_team == to_team:
                continue
            txns.append(
                Transaction(
                    player_id=pid,
                    from_team_id=from_team,
                    to_team_id=to_team,
                    from_season=a,
                    cutover_season=b,
                    pre_possessions=pre_poss,
                    post_possessions=post_poss,
                    post_stint_ids=tuple(sorted(post_index[(to_team, b, pid)])),
                )
            )
    txns.sort(key=lambda t: (rank[t.cutover_season], t.player_id))
    return txns


def leakage_safe_train(table: StintTable, cutover_season: str) -> StintTable:
    """Every stint in a season strictly earlier than ``cutover_season``."""

    rank = _season_rank(table)
    cut = rank[cutover_season]
    return StintTable.from_stints(s for s in table if rank[s.season] < cut)


def phantom_transactions(
    table: StintTable,
    real: list[Transaction],
    *,
    min_poss_each_side: int = 500,
    seed: int = 0,
    n: int | None = None,
) -> list[Transaction]:
    """Players who did **not** switch teams: for each, pretend the current team
    is a move from the same team in the prior season. The identical backtest on
    these is the null band -- a real roster-fit effect must beat it."""

    import numpy as np

    rng = np.random.default_rng(seed)
    ptp = _player_team_possessions(table)
    rank = _season_rank(table)
    by_rank = sorted(rank, key=lambda k: rank[k])
    moved = {(t.player_id, t.cutover_season) for t in real}

    post_index: dict[tuple[int, str, int], list[str]] = defaultdict(list)
    for s in table:
        for pid in s.offense_player_ids:
            post_index[(s.offense_team_id, s.season, pid)].append(s.stint_id)

    candidates: list[Transaction] = []
    for pid, seasons in ptp.items():
        for a, b in zip(by_rank, by_rank[1:], strict=False):
            if (pid, b) in moved or a not in seasons or b not in seasons:
                continue
            ta = {t: c for t, c in seasons[a].items() if c >= min_poss_each_side}
            tb = {t: c for t, c in seasons[b].items() if c >= min_poss_each_side}
            if len(ta) != 1 or len(tb) != 1:
                continue
            (team_a, pre), (team_b, post) = (
                next(iter(ta.items())),
                next(iter(tb.items())),
            )
            if team_a != team_b:
                continue  # they actually moved by our own rule -> skip
            candidates.append(
                Transaction(
                    player_id=pid,
                    from_team_id=team_a,
                    to_team_id=team_b,
                    from_season=a,
                    cutover_season=b,
                    pre_possessions=pre,
                    post_possessions=post,
                    post_stint_ids=tuple(sorted(post_index[(team_b, b, pid)])),
                )
            )
    if n is not None and n < len(candidates):
        pick = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[i] for i in sorted(pick)]
    return candidates

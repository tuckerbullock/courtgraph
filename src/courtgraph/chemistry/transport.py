"""Phase-transport evaluation: train on the regular season, test on the playoffs.

Unlike the three in-sample holdouts (``chronological`` / ``unseen_pair`` /
``unseen_lineup``, all built by partitioning one :class:`StintTable`), this
compares two independently ingested tables -- the full regular-season stint set
and a held-out playoff set. Every playoff player is already observed in the
regular season, so this is a clean test of whether *season phase* (tighter
rotations, a series-long scheme battle) exposes teammate-interaction structure
the regular season does not.

It fits rung 2 (additive ridge RAPM), rung 3 (hierarchical EB) and -- when a
``rung4_config`` is given -- rung 4 (explicit teammate-pair interaction) on the
regular season, then evaluates all of them on the playoffs:

* macro RMSE + interval calibration over recurring playoff lineups, split by
  how novel each lineup is relative to the regular season;
* a pair-level "do teammate terms help" test over regular-season-admitted pairs
  that recur in the playoffs, with the same placebo control as the rung-4
  chronological exit test (:mod:`courtgraph.chemistry.baseline_ladder`);
* the same three models' possession-weighted error on the clutch subset
  (one-possession game, fourth quarter or later) -- where lineup fit is most
  often argued to matter.

The regular season contains no ``playoff=1`` rows, so the fitted models carry
**no playoff main effect**: any systematic phase shift (the playoffs are
lower-scoring, defenses tighten) surfaces as a calibration offset, not a
correction the model can make. That is a reported limitation, not a bug.

Preserves whatever it finds (research contract 17, 26). Does not modify the
in-sample ladder path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.baseline_ladder import (
    _group_realized,
    _pair_level_breakdown,
    _placebo_vocab,
    _rung2_band,
)
from courtgraph.chemistry.calibration import calibration_report
from courtgraph.chemistry.evaluate import _rmse
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge
from courtgraph.chemistry.pair_interaction import (
    PairHierarchicalConfig,
    PairHierarchicalRidge,
    PairVocabulary,
)
from courtgraph.chemistry.stints import StintTable, lineup_id, pair_id

_CLUTCH_MARGIN = 6
_CLUTCH_PERIOD = 4


def _seen_sets(train: StintTable) -> tuple[set[str], set[str], set[int]]:
    """(exact offensive lineup ids, offensive pair ids, offensive player ids)
    observed anywhere in the training table -- computed once."""

    lineups: set[str] = set()
    pairs: set[str] = set()
    players: set[int] = set()
    for stint in train:
        ids = stint.offense_player_ids
        lineups.add(stint.offense_lineup_id)
        players.update(ids)
        for a in range(5):
            for b in range(a + 1, 5):
                pairs.add(pair_id(ids[a], ids[b]))
    return lineups, pairs, players


def _novelty(
    offense_ids: tuple[int, ...],
    seen_lineups: set[str],
    seen_pairs: set[str],
    seen_players: set[int],
) -> str:
    """``seen`` (exact five played together), ``partially-seen`` (every pair
    seen but not the exact five), or ``unseen`` (a pair or a player never seen).
    Inlined from :func:`splits.novelty_of_lineup` so the seen sets are built
    once, not per lineup."""

    if lineup_id(offense_ids) in seen_lineups:
        return "seen"
    if any(p not in seen_players for p in offense_ids):
        return "unseen"
    all_pairs_seen = all(
        pair_id(offense_ids[a], offense_ids[b]) in seen_pairs
        for a in range(len(offense_ids))
        for b in range(a + 1, len(offense_ids))
    )
    return "partially-seen" if all_pairs_seen else "unseen"


@dataclass(frozen=True)
class TransportResult:
    n_train: int
    n_test: int
    leakage_violations: tuple[str, ...]
    zeroed_context_columns: tuple[str, ...]
    coverage: dict[str, float]
    variance_components: dict[str, Any]
    # macro over recurring playoff lineups
    n_lineup_groups: int
    rung2_macro_rmse: float
    rung3_macro_rmse: float
    rung3_calibration: dict[str, float]
    rung2_band_calibration: dict[str, float]
    # macro RMSE by playoff-lineup novelty relative to the regular season
    by_novelty: dict[str, dict[str, float]]
    # possession-weighted error, all playoff stints / clutch / non-clutch
    micro_rmse: dict[str, dict[str, float]]
    # rung 4 (only when rung4_config is given)
    rung4_macro_rmse: float | None = None
    rung4_calibration: dict[str, float] | None = None
    rung4_n_admitted_pairs: int | None = None
    rung4_pair_level: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "leakage_violations": list(self.leakage_violations),
            "zeroed_context_columns": list(self.zeroed_context_columns),
            "coverage": dict(self.coverage),
            "variance_components": dict(self.variance_components),
            "n_lineup_groups": self.n_lineup_groups,
            "rung2_macro_rmse": self.rung2_macro_rmse,
            "rung3_macro_rmse": self.rung3_macro_rmse,
            "rung3_calibration": dict(self.rung3_calibration),
            "rung2_band_calibration": dict(self.rung2_band_calibration),
            "by_novelty": {k: dict(v) for k, v in self.by_novelty.items()},
            "micro_rmse": {k: dict(v) for k, v in self.micro_rmse.items()},
        }
        if self.rung4_macro_rmse is not None:
            out["rung4_macro_rmse"] = self.rung4_macro_rmse
            out["rung4_calibration"] = dict(self.rung4_calibration or {})
            out["rung4_n_admitted_pairs"] = self.rung4_n_admitted_pairs
            out["rung4_pair_level"] = dict(self.rung4_pair_level or {})
        return out


def _transport_coverage(
    test_table: StintTable,
    seen_lineups: set[str],
    seen_pairs: set[str],
    seen_players: set[int],
    admitted_pairs: set[str],
) -> dict[str, float]:
    n = len(test_table)
    players: set[int] = set()
    po_pairs: set[str] = set()
    lineup_novelty = {"seen": 0, "partially-seen": 0, "unseen": 0}
    admitted_hits = 0
    for stint in test_table:
        ids = stint.offense_player_ids
        players.update(ids)
        lineup_novelty[_novelty(ids, seen_lineups, seen_pairs, seen_players)] += 1
        stint_has_admitted = False
        for a in range(5):
            for b in range(a + 1, 5):
                key = pair_id(ids[a], ids[b])
                po_pairs.add(key)
                if key in admitted_pairs:
                    stint_has_admitted = True
        admitted_hits += int(stint_has_admitted)
    return {
        "test_players": float(len(players)),
        "test_players_unseen_in_train": float(len(players - seen_players)),
        "test_offensive_pairs": float(len(po_pairs)),
        "test_pairs_admitted_in_train": float(len(po_pairs & admitted_pairs)),
        "test_stints_with_an_admitted_pair": float(admitted_hits),
        "test_stints_seen_lineup": float(lineup_novelty["seen"]),
        "test_stints_partially_seen_lineup": float(lineup_novelty["partially-seen"]),
        "test_stints_unseen_lineup": float(lineup_novelty["unseen"]),
        "n_test_stints": float(n),
    }


def _lineup_groups(
    test_table: StintTable, *, min_test_stints: int
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, stint in enumerate(test_table):
        groups.setdefault(stint.offense_lineup_id, []).append(i)
    return {k: rows for k, rows in groups.items() if len(rows) >= min_test_stints}


def evaluate_transport(
    train_table: StintTable,
    test_table: StintTable,
    *,
    seed: int = 0,
    n_boot: int = 150,
    config: HierarchicalConfig | None = None,
    rung4_config: PairHierarchicalConfig | None = None,
    min_lineup_stints: int = 5,
    min_pair_stints: int = 5,
) -> TransportResult:
    """Fit rungs 2/3(/4) on ``train_table`` and evaluate them on ``test_table``.

    ``train_table`` is expected to be the regular season and ``test_table`` the
    held-out playoffs, but nothing here assumes that beyond the leakage check
    (no shared game reaches both sides)."""

    train_games = {s.game_id for s in train_table}
    test_games = {s.game_id for s in test_table}
    train_ids = {s.stint_id for s in train_table}
    test_ids = {s.stint_id for s in test_table}
    violations: list[str] = []
    if train_games & test_games:
        violations.append(
            f"{len(train_games & test_games)} game(s) appear in both train and test"
        )
    if train_ids & test_ids:
        violations.append(
            f"{len(train_ids & test_ids)} stint id(s) appear in both train and test"
        )
    if not test_table:
        violations.append("test table is empty")

    seen_lineups, seen_pairs, seen_players = _seen_sets(train_table)

    space = FeatureSpace.from_training(train_table)
    train_design = space.build(train_table)
    test_design = space.build(test_table)

    # A context column the training data never moved off zero (here: ``playoff``,
    # since the regular season has no playoff rows) carries no fitted
    # information -- its coefficient is pure prior. Left in place it would give
    # every playoff test row a posterior variance of ~tau_c2, wrecking the
    # rung-3/4 intervals. Zero it in the test design: the point predictions are
    # unchanged (the coefficient is ~0) and the intervals become meaningful.
    train_ctx = train_design.context
    dead = (train_ctx.std(axis=0) < 1e-9) & (np.abs(train_ctx[0]) < 1e-9)
    zeroed = tuple(space.context_columns[c] for c in np.flatnonzero(dead))
    if zeroed:
        ctx = test_design.context.copy()
        ctx[:, np.flatnonzero(dead)] = 0.0
        test_design = replace(test_design, context=ctx)

    rung2 = AdditiveRidge.fit(train_design, space)
    rung3 = HierarchicalRidge.fit(train_design, space, config=config)

    admitted: set[str] = set()
    vocab: PairVocabulary | None = None
    rung4: PairHierarchicalRidge | None = None
    rung4_placebo: PairHierarchicalRidge | None = None
    if rung4_config is not None:
        vocab = PairVocabulary.from_training(
            train_table, min_co_stints=rung4_config.min_co_stints
        )
        admitted = set(vocab.pair_ids)
        rung4 = PairHierarchicalRidge.fit(
            train_design, space, vocab, config=rung4_config
        )
        rung4_placebo = PairHierarchicalRidge.fit(
            train_design, space, _placebo_vocab(vocab, seed), config=rung4_config
        )

    coverage = _transport_coverage(
        test_table, seen_lineups, seen_pairs, seen_players, admitted
    )

    # -- macro over recurring playoff lineups -------------------------------- #
    groups = _lineup_groups(test_table, min_test_stints=min_lineup_stints)
    keys = list(groups)
    group_arrays = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}
    realized = _group_realized(test_design, groups)
    y = np.array([realized[k] for k in keys])

    r3 = rung3.group_predictive(test_design, group_arrays)
    p3 = np.array([r3[k][0] for k in keys])
    s3 = np.array([r3[k][1] for k in keys])
    r2 = _rung2_band(
        train_table,
        space,
        train_design,
        test_design,
        groups,
        rung2,
        seed=seed,
        n_boot=n_boot,
    )
    p2 = np.array([r2[k][0] for k in keys])
    s2 = np.array([r2[k][1] for k in keys])

    p4: np.ndarray | None = None
    rung4_kw: dict[str, Any] = {}
    if rung4 is not None and rung4_placebo is not None and vocab is not None:
        r4 = rung4.group_predictive(test_design, group_arrays)
        p4 = np.array([r4[k][0] for k in keys])
        s4 = np.array([r4[k][1] for k in keys])
        rung4_kw = {
            "rung4_macro_rmse": _rmse(p4, y),
            "rung4_calibration": calibration_report(p4, s4, y),
            "rung4_n_admitted_pairs": vocab.n_pairs,
            "rung4_pair_level": _pair_level_breakdown(
                test_table,
                test_design,
                rung2,
                rung4,
                rung4_placebo,
                admitted,
                min_test_stints=min_pair_stints,
            ),
        }

    # -- macro RMSE by playoff-lineup novelty ------------------------------- #
    by_novelty = _macro_by_novelty(
        test_table, keys, y, p2, p3, p4, seen_lineups, seen_pairs, seen_players
    )

    # -- possession-weighted error: all / clutch / non-clutch -------------- #
    micro_rmse = _micro_rmse_by_context(test_table, test_design, rung2, rung3, rung4)

    return TransportResult(
        n_train=len(train_table),
        n_test=len(test_table),
        leakage_violations=tuple(violations),
        zeroed_context_columns=zeroed,
        coverage=coverage,
        variance_components=rung3.variance_components(),
        n_lineup_groups=len(keys),
        rung2_macro_rmse=_rmse(p2, y),
        rung3_macro_rmse=_rmse(p3, y),
        rung3_calibration=calibration_report(p3, s3, y),
        rung2_band_calibration=calibration_report(p2, s2, y),
        by_novelty=by_novelty,
        micro_rmse=micro_rmse,
        **rung4_kw,
    )


def _macro_by_novelty(
    test_table: StintTable,
    keys: list[str],
    y: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray | None,
    seen_lineups: set[str],
    seen_pairs: set[str],
    seen_players: set[int],
) -> dict[str, dict[str, float]]:
    lineup_ids_by_key: dict[str, tuple[int, ...]] = {}
    for stint in test_table:
        lineup_ids_by_key.setdefault(stint.offense_lineup_id, stint.offense_player_ids)

    buckets: dict[str, list[int]] = {"seen": [], "partially-seen": [], "unseen": []}
    for gi, key in enumerate(keys):
        nov = _novelty(lineup_ids_by_key[key], seen_lineups, seen_pairs, seen_players)
        buckets[nov].append(gi)

    out: dict[str, dict[str, float]] = {}
    for nov, idx in buckets.items():
        if not idx:
            out[nov] = {"n_groups": 0.0}
            continue
        sel = np.asarray(idx, dtype=np.int64)
        entry = {
            "n_groups": float(len(idx)),
            "rung2_macro_rmse": _rmse(p2[sel], y[sel]),
            "rung3_macro_rmse": _rmse(p3[sel], y[sel]),
        }
        if p4 is not None:
            entry["rung4_macro_rmse"] = _rmse(p4[sel], y[sel])
        out[nov] = entry
    return out


def _micro_rmse_by_context(
    test_table: StintTable,
    test_design: DesignMatrices,
    rung2: AdditiveRidge,
    rung3: HierarchicalRidge,
    rung4: PairHierarchicalRidge | None,
) -> dict[str, dict[str, float]]:
    clutch = np.array(
        [
            abs(s.score_margin_offense) <= _CLUTCH_MARGIN and s.period >= _CLUTCH_PERIOD
            for s in test_table
        ]
    )
    preds = {
        "rung2": rung2.predict(test_design),
        "rung3": rung3.predict(test_design),
    }
    if rung4 is not None:
        preds["rung4"] = rung4.predict(test_design)

    def _subset(mask: np.ndarray) -> dict[str, float]:
        w = test_design.weight[mask]
        entry = {"n_stints": float(mask.sum()), "possessions": float(w.sum())}
        for name, pred in preds.items():
            entry[f"{name}_micro_rmse"] = _rmse(pred[mask], test_design.y[mask], w)
        return entry

    return {
        "all": _subset(np.ones(len(test_table), dtype=bool)),
        "clutch": _subset(clutch),
        "non_clutch": _subset(~clutch),
    }

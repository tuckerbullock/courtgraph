"""Rung 2 vs rung 3: point accuracy and calibration on the leakage-safe holdouts.

Rung 2 is the additive ridge RAPM baseline
(:class:`~courtgraph.chemistry.baseline.AdditiveRidge`); rung 3 is the
empirical-Bayes hierarchical model
(:class:`~courtgraph.chemistry.hierarchical.HierarchicalRidge`).

For each holdout this fits both on the training rows, buckets the test rows into
groups (held-out lineups / pairs / seasons -- reusing
:func:`courtgraph.chemistry.evaluate._group_index`), and reports:

* possession-weighted macro / micro RMSE of each model's point prediction;
* rung-3 interval calibration (coverage 50/80/95, calibration line, width vs
  error) from its Gaussian posterior + outcome noise;
* rung-2 interval calibration from an **approximate** block-bootstrap-over-games
  predictive band (the same hedge the interaction pathway uses).

The research-contract question (section 11) is whether rung 3 beats rung 2 "on
calibration and stability" -- point RMSE is expected to be about the same.

This module imports from :mod:`courtgraph.chemistry.evaluate` but does not
modify it; the ``ChemistryModel`` evaluation path is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.calibration import calibration_report
from courtgraph.chemistry.evaluate import _group_index, _rmse
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge
from courtgraph.chemistry.pair_interaction import (
    PairHierarchicalConfig,
    PairHierarchicalRidge,
    PairVocabulary,
)
from courtgraph.chemistry.splits import SplitManifest
from courtgraph.chemistry.stints import StintTable, pair_id

FloatArray = NDArray[np.float64]
_HOLDOUTS = ("chronological", "unseen_pair", "unseen_lineup")


@dataclass(frozen=True)
class HoldoutLadderResult:
    kind: str
    n_train: int
    n_test: int
    n_groups: int
    rung2_macro_rmse: float
    rung3_macro_rmse: float
    rung2_micro_rmse: float
    rung3_micro_rmse: float
    rung3_calibration: dict[str, float]
    rung2_band_calibration: dict[str, float]
    # rung 4 (only when compare_rungs is given a rung4_config)
    rung4_macro_rmse: float | None = None
    rung4_calibration: dict[str, float] | None = None
    rung4_n_admitted_pairs: int | None = None
    # chronological only: rung 2 vs rung 4 split by whether every offense pair
    # of a held-out lineup is in rung 4's admitted vocabulary -- the section 11
    # "beat rung 2 on seen pairs" exit test lives in `pair_covered`.
    rung4_pair_covered: dict[str, float] | None = None
    rung4_pair_degraded: dict[str, float] | None = None
    # pair-level "seen pairs" test: rung 2 vs rung 4 vs a placebo-pair rung 4,
    # macro-averaged over every admitted pair that recurs in the test period.
    rung4_pair_level: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_groups": self.n_groups,
            "rung2_macro_rmse": self.rung2_macro_rmse,
            "rung3_macro_rmse": self.rung3_macro_rmse,
            "rung2_micro_rmse": self.rung2_micro_rmse,
            "rung3_micro_rmse": self.rung3_micro_rmse,
            "rung3_calibration": dict(self.rung3_calibration),
            "rung2_band_calibration": dict(self.rung2_band_calibration),
        }
        if self.rung4_macro_rmse is not None:
            out["rung4_macro_rmse"] = self.rung4_macro_rmse
            out["rung4_calibration"] = dict(self.rung4_calibration or {})
            out["rung4_n_admitted_pairs"] = self.rung4_n_admitted_pairs
        if self.rung4_pair_covered is not None:
            out["rung4_pair_covered"] = dict(self.rung4_pair_covered)
            out["rung4_pair_degraded"] = dict(self.rung4_pair_degraded or {})
        if self.rung4_pair_level is not None:
            out["rung4_pair_level"] = dict(self.rung4_pair_level)
        return out


@dataclass(frozen=True)
class LadderComparison:
    variance_components: dict[str, Any]
    holdouts: tuple[HoldoutLadderResult, ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "variance_components": dict(self.variance_components),
            "holdouts": [h.as_dict() for h in self.holdouts],
            "notes": list(self.notes),
        }


def bootstrap_group_delta(
    baseline_point: FloatArray,
    model_point: FloatArray,
    realized: FloatArray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Resample the held-out groups (with replacement) and report the
    distribution of ``rmse(baseline) - rmse(model)`` over the macro group means.
    Positive delta = the model is better. ``frac_gt_0`` near 1 (and a CI that
    excludes 0) means the improvement is unlikely to be group-sampling noise."""

    n = len(realized)
    if n < 3:
        return {
            "n_groups": float(n),
            "mean": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "frac_gt_0": 0.0,
        }
    rng = np.random.default_rng(seed)
    base_err = (baseline_point - realized) ** 2
    model_err = (model_point - realized) ** 2
    deltas = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        rb = float(np.sqrt(base_err[s].mean()))
        rm = float(np.sqrt(model_err[s].mean()))
        deltas[b] = rb - rm
    deltas.sort()
    return {
        "n_groups": float(n),
        "mean": float(deltas.mean()),
        "ci_lo": float(deltas[int(0.025 * n_boot)]),
        "ci_hi": float(deltas[min(int(0.975 * n_boot), n_boot - 1)]),
        "frac_gt_0": float((deltas > 0).mean()),
    }


def _group_realized(
    design: DesignMatrices, groups: dict[str, list[int]]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for gid, rows in groups.items():
        idx = np.asarray(rows, dtype=np.int64)
        weight = design.weight[idx]
        out[gid] = float(np.average(design.y[idx], weights=weight))
    return out


def _rung2_band(
    train_table: StintTable,
    space: FeatureSpace,
    train_design: DesignMatrices,
    test_design: DesignMatrices,
    groups: dict[str, list[int]],
    base: AdditiveRidge,
    *,
    seed: int,
    n_boot: int,
) -> dict[str, tuple[float, float]]:
    """Rung-2 group point + an approximate block-bootstrap-over-games band.

    The point is the base fit's possession-weighted group prediction; the SD is
    ``sqrt(var_b(bootstrap group predictions) + sigma_hat^2 / sum_w)`` with
    ``l2_player`` held at the base pick.
    """

    keys = list(groups)
    point = {}
    per_row = base.predict(test_design)
    for gid in keys:
        idx = np.asarray(groups[gid], dtype=np.int64)
        weight = test_design.weight[idx]
        point[gid] = float(np.average(per_row[idx], weights=weight))

    resid = base.predict(train_design) - train_design.y
    sigma2_hat = float(np.average(resid**2, weights=train_design.weight))

    if n_boot <= 0:
        return {gid: (point[gid], float(np.sqrt(sigma2_hat))) for gid in keys}

    games = np.array(train_design.game_ids)
    unique_games = np.array(sorted(set(train_design.game_ids)))
    game_pos = {g: i for i, g in enumerate(unique_games)}
    row_game = np.array([game_pos[g] for g in games])
    rng = np.random.default_rng(seed)

    columns = np.zeros((n_boot, len(keys)))
    for b in range(n_boot):
        counts = np.bincount(
            rng.integers(0, len(unique_games), size=len(unique_games)),
            minlength=len(unique_games),
        ).astype(np.float64)
        weight_b = train_design.weight * counts[row_game]
        design_b = replace(train_design, weight=weight_b)
        model_b = AdditiveRidge.fit(
            design_b, space, l2_player=base.l2_player, l2_context=base.l2_context
        )
        pred_b = model_b.predict(test_design)
        for gi, gid in enumerate(keys):
            idx = np.asarray(groups[gid], dtype=np.int64)
            weight = test_design.weight[idx]
            columns[b, gi] = np.average(pred_b[idx], weights=weight)

    out: dict[str, tuple[float, float]] = {}
    for gi, gid in enumerate(keys):
        idx = np.asarray(groups[gid], dtype=np.int64)
        var_boot = float(np.var(columns[:, gi], ddof=1))
        sd = float(np.sqrt(var_boot + sigma2_hat / test_design.weight[idx].sum()))
        out[gid] = (point[gid], sd)
    return out


def _lineup_groups(
    test_table: StintTable,
) -> dict[str, tuple[tuple[int, ...], list[int]]]:
    """Held-out test rows bucketed by exact offensive five (id -> (ids, rows))."""

    out: dict[str, tuple[tuple[int, ...], list[int]]] = {}
    for i, stint in enumerate(test_table):
        lid = stint.offense_lineup_id
        if lid not in out:
            out[lid] = (stint.offense_player_ids, [])
        out[lid][1].append(i)
    return out


def _pair_coverage_breakdown(
    test_table: StintTable,
    test_design: DesignMatrices,
    rung2: AdditiveRidge,
    rung4: PairHierarchicalRidge,
    admitted: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Rung 2 vs rung 4 macro RMSE over held-out lineups, split by whether every
    offensive pair of the lineup is in rung 4's admitted vocabulary."""

    lineups = _lineup_groups(test_table)
    covered: dict[str, list[int]] = {}
    degraded: dict[str, list[int]] = {}
    for lid, (ids, rows) in lineups.items():
        full = all(
            pair_id(ids[a], ids[b]) in admitted
            for a in range(5)
            for b in range(a + 1, 5)
        )
        (covered if full else degraded)[lid] = rows

    def _macro(buckets: dict[str, list[int]]) -> dict[str, float]:
        if not buckets:
            return {"n_groups": 0.0}
        p2_pred = rung2.predict(test_design)
        p4_pred = rung4.predict(test_design)
        y_g, p2_g, p4_g = [], [], []
        for rows in buckets.values():
            idx = np.asarray(rows, dtype=np.int64)
            weight = test_design.weight[idx]
            y_g.append(float(np.average(test_design.y[idx], weights=weight)))
            p2_g.append(float(np.average(p2_pred[idx], weights=weight)))
            p4_g.append(float(np.average(p4_pred[idx], weights=weight)))
        y_a = np.array(y_g)
        return {
            "n_groups": float(len(buckets)),
            "rung2_macro_rmse": _rmse(np.array(p2_g), y_a),
            "rung4_macro_rmse": _rmse(np.array(p4_g), y_a),
        }

    return _macro(covered), _macro(degraded)


def _placebo_vocab(vocab: PairVocabulary, seed: int) -> PairVocabulary:
    """Same admitted pair keys, same parameter count, same total pair exposure --
    but each pair's stints are routed to a randomly chosen coefficient row (drawn
    with replacement). Distinct real pairs collide onto shared rows and the
    pair->outcome link is broken, so a rung-4 fit on it cannot carry
    pair-specific signal. The control: real rung-4 error must be meaningfully
    below this placebo error, otherwise the pair terms are only soaking up
    additive misfit and noise."""

    rng = np.random.default_rng(seed)
    targets = tuple(int(x) for x in rng.integers(0, vocab.n_pairs, size=vocab.n_pairs))
    return PairVocabulary(
        pair_ids=vocab.pair_ids,
        min_co_stints=vocab.min_co_stints,
        _index=dict(vocab._index),
        row_override=targets,
    )


def _pair_level_breakdown(
    test_table: StintTable,
    test_design: DesignMatrices,
    rung2: AdditiveRidge,
    rung4: PairHierarchicalRidge,
    rung4_placebo: PairHierarchicalRidge,
    admitted: set[str],
    *,
    min_test_stints: int = 5,
) -> dict[str, float]:
    """Rung 2 vs rung 4 vs a placebo-pair rung 4, macro-averaged over every
    admitted pair that recurs in the held-out period (>= ``min_test_stints``
    test stints with the pair on offense). Far better powered than the
    all-pairs-covered-lineup test: hundreds of pair groups, not a few hundred
    lineups. This is the rung-4 "beat rung 2 on seen pairs" exit test."""

    groups: dict[str, list[int]] = {}
    for i, stint in enumerate(test_table):
        ids = stint.offense_player_ids
        for a in range(5):
            for b in range(a + 1, 5):
                key = pair_id(ids[a], ids[b])
                if key in admitted:
                    groups.setdefault(key, []).append(i)
    groups = {k: rows for k, rows in groups.items() if len(rows) >= min_test_stints}
    if not groups:
        return {"n_pair_groups": 0.0}

    p2 = rung2.predict(test_design)
    p4 = rung4.predict(test_design)
    p4p = rung4_placebo.predict(test_design)
    y_g, p2_g, p4_g, p4p_g = [], [], [], []
    for rows in groups.values():
        idx = np.asarray(rows, dtype=np.int64)
        weight = test_design.weight[idx]
        y_g.append(float(np.average(test_design.y[idx], weights=weight)))
        p2_g.append(float(np.average(p2[idx], weights=weight)))
        p4_g.append(float(np.average(p4[idx], weights=weight)))
        p4p_g.append(float(np.average(p4p[idx], weights=weight)))
    y_a = np.array(y_g)
    return {
        "n_pair_groups": float(len(groups)),
        "min_test_stints": float(min_test_stints),
        "rung2_macro_rmse": _rmse(np.array(p2_g), y_a),
        "rung4_macro_rmse": _rmse(np.array(p4_g), y_a),
        "rung4_placebo_macro_rmse": _rmse(np.array(p4p_g), y_a),
    }


def compare_rungs(
    table: StintTable,
    splits: dict[str, SplitManifest],
    *,
    seed: int = 0,
    n_boot: int = 150,
    config: HierarchicalConfig | None = None,
    rung4_config: PairHierarchicalConfig | None = None,
) -> LadderComparison:
    space_all = FeatureSpace.from_training(table)
    global_rung3 = HierarchicalRidge.fit(
        space_all.build(table), space_all, config=config
    )

    holdouts: list[HoldoutLadderResult] = []
    for kind in _HOLDOUTS:
        manifest = splits[kind]
        train_table = manifest.train_table(table)
        test_table = manifest.test_table(table)
        space = FeatureSpace.from_training(train_table)
        train_design = space.build(train_table)
        test_design = space.build(test_table)

        rung2 = AdditiveRidge.fit(train_design, space)
        rung3 = HierarchicalRidge.fit(train_design, space, config=config)

        groups = _group_index(test_table, manifest)
        realized = _group_realized(test_design, groups)
        keys = list(groups)
        group_arrays = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}

        r3 = rung3.group_predictive(test_design, group_arrays)
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

        y = np.array([realized[k] for k in keys])
        p2 = np.array([r2[k][0] for k in keys])
        s2 = np.array([r2[k][1] for k in keys])
        p3 = np.array([r3[k][0] for k in keys])
        s3 = np.array([r3[k][1] for k in keys])

        rung4_kw: dict[str, Any] = {}
        if rung4_config is not None:
            vocab = PairVocabulary.from_training(
                train_table, min_co_stints=rung4_config.min_co_stints
            )
            rung4 = PairHierarchicalRidge.fit(
                train_design, space, vocab, config=rung4_config
            )
            r4 = rung4.group_predictive(test_design, group_arrays)
            p4 = np.array([r4[k][0] for k in keys])
            s4 = np.array([r4[k][1] for k in keys])
            rung4_kw = {
                "rung4_macro_rmse": _rmse(p4, y),
                "rung4_calibration": calibration_report(p4, s4, y),
                "rung4_n_admitted_pairs": vocab.n_pairs,
            }
            if kind == "chronological":
                rung4_placebo = PairHierarchicalRidge.fit(
                    train_design,
                    space,
                    _placebo_vocab(vocab, seed),
                    config=rung4_config,
                )
                covered, degraded = _pair_coverage_breakdown(
                    test_table, test_design, rung2, rung4, set(vocab.pair_ids)
                )
                rung4_kw["rung4_pair_covered"] = covered
                rung4_kw["rung4_pair_degraded"] = degraded
                rung4_kw["rung4_pair_level"] = _pair_level_breakdown(
                    test_table,
                    test_design,
                    rung2,
                    rung4,
                    rung4_placebo,
                    set(vocab.pair_ids),
                )

        holdouts.append(
            HoldoutLadderResult(
                kind=kind,
                n_train=len(train_table),
                n_test=len(test_table),
                n_groups=len(keys),
                rung2_macro_rmse=_rmse(p2, y),
                rung3_macro_rmse=_rmse(p3, y),
                rung2_micro_rmse=_rmse(
                    rung2.predict(test_design), test_design.y, test_design.weight
                ),
                rung3_micro_rmse=_rmse(
                    rung3.predict(test_design), test_design.y, test_design.weight
                ),
                rung3_calibration=calibration_report(p3, s3, y),
                rung2_band_calibration=calibration_report(p2, s2, y),
                **rung4_kw,
            )
        )

    notes = (
        "Rung-2 intervals are an approximate block-bootstrap-over-games band "
        "with l2_player held at the base pick; rung-3 intervals are the Gaussian "
        "posterior plus outcome noise (empirical Bayes, approximate).",
        "EM recovers the realised player-pool effect SD, not the generative "
        "SyntheticConfig scale.",
    )
    return LadderComparison(
        variance_components=global_rung3.variance_components(),
        holdouts=tuple(holdouts),
        notes=notes,
    )

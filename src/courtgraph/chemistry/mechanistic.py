"""Candidate idea #2 -- mechanistic outcomes instead of points per 100.

Stint points/100 is a very noisy target (sigma ~ 119). A more *mechanical*
quantity -- how a lineup shifts its own shot selection, shot quality, or ball
security -- might carry non-additive structure the aggregate outcome washes
out. This module swaps the design's outcome for one of:

* ``pts_per_shot``    -- field-goal points per shot attempt (an eFG proxy:
  does lineup fit improve shot *quality* beyond the sum of individual
  tendencies?);
* ``rim_share``       -- share of shots at the rim (spacing: does a shooter
  on the floor pull teammates' shots toward the basket?);
* ``three_share``     -- share of shots from three;
* ``turnover_rate``   -- turnovers per offensive possession
  (:mod:`courtgraph.features.stint_events`);
* ``assist_rate``     -- share of made shots that were assisted
  (:mod:`courtgraph.features.stint_events`).

The first three are weighted by field-goal attempts and read off
:class:`~courtgraph.features.stint_shots.StintShots`; the last two are
weighted by offensive possessions / makes respectively and read off
:class:`~courtgraph.features.stint_events.StintPlayEvents`. It then runs the
same rung 2 / rung 3 / role-conditioned / permuted-role-placebo comparison as
:mod:`courtgraph.chemistry.role_eval` on the three leakage-safe holdouts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.baseline_ladder import _group_realized, _rung2_band
from courtgraph.chemistry.calibration import calibration_report
from courtgraph.chemistry.evaluate import _group_index, _rmse
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge
from courtgraph.chemistry.role_interaction import (
    RoleClusterInteraction,
    RoleInteractionConfig,
)
from courtgraph.chemistry.splits import SplitManifest, make_all_splits
from courtgraph.chemistry.stints import StintTable
from courtgraph.features.role_clusters import RoleClustering, permuted_clustering
from courtgraph.features.stint_events import EventAttribution, StintPlayEvents
from courtgraph.features.stint_shots import ShotAttribution, StintShots

FloatArray = NDArray[np.float64]
_HOLDOUTS = ("chronological", "unseen_pair", "unseen_lineup")
OUTCOMES = ("pts_per_shot", "rim_share", "three_share", "turnover_rate", "assist_rate")
EVENT_OUTCOMES = frozenset({"turnover_rate", "assist_rate"})
MechanisticAttribution = ShotAttribution | EventAttribution
_StintRecord = StintShots | StintPlayEvents
_ZERO_SHOTS = StintShots(0, 0, 0, 0, 0, 0, 0, 0)
_ZERO_EVENTS = StintPlayEvents(0, 0, 0, 0)


def _outcome_value(record: _StintRecord, outcome: str) -> float:
    if isinstance(record, StintShots):
        if outcome == "pts_per_shot":
            return record.points_per_shot
        if outcome == "rim_share":
            return record.rim_share
        if outcome == "three_share":
            return record.three_share
    elif isinstance(record, StintPlayEvents):
        if outcome == "turnover_rate":
            return record.turnover_rate
        if outcome == "assist_rate":
            return record.assist_rate
    raise ValueError(f"unknown mechanistic outcome {outcome!r}")


def _outcome_weight(record: _StintRecord, outcome: str) -> float:
    """The design weight for ``outcome`` -- the rate's own denominator, so a
    stint with more exposure to the outcome counts for more, mirroring how
    rung 2/3 weight by offensive possessions."""

    if isinstance(record, StintShots):
        return float(record.fga)
    if outcome == "turnover_rate":
        return float(record.offensive_possessions)
    return float(record.fgm)  # assist_rate


def _zero_for(attribution: MechanisticAttribution) -> _StintRecord:
    return _ZERO_EVENTS if isinstance(attribution, EventAttribution) else _ZERO_SHOTS


def mechanistic_table_and_design(
    space: FeatureSpace,
    table: StintTable,
    attribution: MechanisticAttribution,
    outcome: str,
    *,
    min_fga: int,
) -> tuple[StintTable, DesignMatrices]:
    """Filter to stints with ``>= min_fga`` of the outcome's own exposure
    (FGA for the shot outcomes; possessions / FGM for turnover / assist
    rate), then build a design whose ``y`` is the mechanistic outcome and
    ``weight`` is that same exposure."""

    zero = _zero_for(attribution)
    kept = [
        s
        for s in table
        if _outcome_weight(attribution.per_stint.get(s.stint_id, zero), outcome)
        >= min_fga
    ]
    kt = StintTable.from_stints(kept)
    design = space.build(kt)
    y = np.array(
        [_outcome_value(attribution.per_stint[s.stint_id], outcome) for s in kt],
        dtype=np.float64,
    )
    w = np.array(
        [_outcome_weight(attribution.per_stint[s.stint_id], outcome) for s in kt],
        dtype=np.float64,
    )
    return kt, replace(design, y=y, weight=w)


@dataclass(frozen=True)
class MechanisticHoldoutResult:
    kind: str
    n_train: int
    n_test: int
    n_groups: int
    rung2_macro_rmse: float
    rung3_macro_rmse: float
    role_macro_rmse: float
    role_placebo_macro_rmse: float
    rung3_calibration: dict[str, float]
    role_calibration: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_groups": self.n_groups,
            "rung2_macro_rmse": self.rung2_macro_rmse,
            "rung3_macro_rmse": self.rung3_macro_rmse,
            "role_macro_rmse": self.role_macro_rmse,
            "role_placebo_macro_rmse": self.role_placebo_macro_rmse,
            "rung3_calibration": dict(self.rung3_calibration),
            "role_calibration": dict(self.role_calibration),
        }


@dataclass(frozen=True)
class MechanisticComparison:
    outcome: str
    min_fga: int
    n_stints_kept: int
    mean_outcome: float
    match_rate: float
    role_pair_matrix: list[list[float]]
    role_variance_components: dict[str, Any]
    holdouts: tuple[MechanisticHoldoutResult, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "min_fga": self.min_fga,
            "n_stints_kept": self.n_stints_kept,
            "mean_outcome": self.mean_outcome,
            "match_rate": self.match_rate,
            "role_pair_matrix": self.role_pair_matrix,
            "role_variance_components": dict(self.role_variance_components),
            "holdouts": [h.as_dict() for h in self.holdouts],
            "notes": list(self.notes),
        }


def evaluate_mechanistic(
    table: StintTable,
    attribution: MechanisticAttribution,
    clustering: RoleClustering,
    *,
    outcome: str = "pts_per_shot",
    min_fga: int = 3,
    seed: int = 0,
    n_boot: int = 120,
    config: HierarchicalConfig | None = None,
    role_config: RoleInteractionConfig | None = None,
) -> MechanisticComparison:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    role_cfg = role_config or RoleInteractionConfig()

    full_space = FeatureSpace.from_training(table)
    full_kt, full_design = mechanistic_table_and_design(
        full_space, table, attribution, outcome, min_fga=min_fga
    )
    full_role = RoleClusterInteraction.fit(
        full_design, full_space, clustering, config=role_cfg
    )
    mean_outcome = float(np.average(full_design.y, weights=full_design.weight))

    # splits are built on the filtered table so group sizes are right
    splits: dict[str, SplitManifest] = make_all_splits(full_kt)

    holdouts: list[MechanisticHoldoutResult] = []
    for kind in _HOLDOUTS:
        manifest = splits[kind]
        train_table = manifest.train_table(full_kt)
        test_table = manifest.test_table(full_kt)
        space = FeatureSpace.from_training(train_table)
        _, train_design = mechanistic_table_and_design(
            space, train_table, attribution, outcome, min_fga=min_fga
        )
        test_kt, test_design = mechanistic_table_and_design(
            space, test_table, attribution, outcome, min_fga=min_fga
        )

        rung2 = AdditiveRidge.fit(train_design, space)
        rung3 = HierarchicalRidge.fit(train_design, space, config=config)
        role = RoleClusterInteraction.fit(
            train_design, space, clustering, config=role_cfg
        )
        role_placebo = RoleClusterInteraction.fit(
            train_design,
            space,
            permuted_clustering(clustering, seed + 1),
            config=role_cfg,
        )

        groups = _group_index(test_kt, manifest)
        realized = _group_realized(test_design, groups)
        keys = list(groups)
        group_arrays = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}
        y = np.array([realized[k] for k in keys])

        r3 = rung3.group_predictive(test_design, group_arrays)
        p3 = np.array([r3[k][0] for k in keys])
        s3 = np.array([r3[k][1] for k in keys])
        rl = role.group_predictive(test_design, group_arrays)
        prole = np.array([rl[k][0] for k in keys])
        srole = np.array([rl[k][1] for k in keys])
        rlp = role_placebo.group_predictive(test_design, group_arrays)
        prolep = np.array([rlp[k][0] for k in keys])
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

        holdouts.append(
            MechanisticHoldoutResult(
                kind=kind,
                n_train=len(train_table),
                n_test=len(test_kt),
                n_groups=len(keys),
                rung2_macro_rmse=_rmse(p2, y),
                rung3_macro_rmse=_rmse(p3, y),
                role_macro_rmse=_rmse(prole, y),
                role_placebo_macro_rmse=_rmse(prolep, y),
                rung3_calibration=calibration_report(p3, s3, y),
                role_calibration=calibration_report(prole, srole, y),
            )
        )

    notes = (
        f"Outcome {outcome!r}: y is weighted by its own exposure denominator "
        "(FGA for the shot outcomes; possessions/FGM for turnover/assist "
        "rate); stints below min_fga of that exposure are dropped.",
        "Role clusters fit once on the full player-profile set (outcome-blind); "
        "the permuted-role placebo is the control.",
    )
    return MechanisticComparison(
        outcome=outcome,
        min_fga=min_fga,
        n_stints_kept=len(full_kt),
        mean_outcome=mean_outcome,
        match_rate=attribution.match_rate,
        role_pair_matrix=[list(row) for row in full_role.role_pair_matrix()],
        role_variance_components=full_role.variance_components(),
        holdouts=tuple(holdouts),
        notes=notes,
    )


def transport_mechanistic(
    train_table: StintTable,
    test_table: StintTable,
    train_attribution: MechanisticAttribution,
    test_attribution: MechanisticAttribution,
    clustering: RoleClustering,
    *,
    outcome: str = "three_share",
    min_fga: int = 3,
    n_boot: int = 2000,
    seed: int = 0,
    config: HierarchicalConfig | None = None,
    role_config: RoleInteractionConfig | None = None,
) -> dict[str, Any]:
    """Train the rung-3 / role / permuted-role-placebo models on ``train_table``'s
    mechanistic outcome and evaluate them on the disjoint ``test_table`` (e.g.
    the held-out playoffs), macro over recurring test lineups, with a bootstrap
    CI on ``rmse(rung 3) - rmse(role)``."""

    from courtgraph.chemistry.baseline_ladder import bootstrap_group_delta

    role_cfg = role_config or RoleInteractionConfig()
    space = FeatureSpace.from_training(train_table)
    _, train_design = mechanistic_table_and_design(
        space, train_table, train_attribution, outcome, min_fga=min_fga
    )
    test_kt, test_design = mechanistic_table_and_design(
        space, test_table, test_attribution, outcome, min_fga=min_fga
    )

    rung3 = HierarchicalRidge.fit(train_design, space, config=config)
    role = RoleClusterInteraction.fit(train_design, space, clustering, config=role_cfg)
    role_placebo = RoleClusterInteraction.fit(
        train_design, space, permuted_clustering(clustering, seed + 1), config=role_cfg
    )

    groups: dict[str, list[int]] = {}
    for i, stint in enumerate(test_kt):
        groups.setdefault(stint.offense_lineup_id, []).append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= 5}
    keys = list(groups)
    ga = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}
    realized = _group_realized(test_design, groups)
    y = np.array([realized[k] for k in keys])

    def _pt(model: Any) -> FloatArray:
        pr = model.group_predictive(test_design, ga)
        return np.array([pr[k][0] for k in keys])

    p3, prole, pplc = _pt(rung3), _pt(role), _pt(role_placebo)
    return {
        "outcome": outcome,
        "n_test_lineups": len(keys),
        "rung3_macro_rmse": _rmse(p3, y),
        "role_macro_rmse": _rmse(prole, y),
        "role_placebo_macro_rmse": _rmse(pplc, y),
        "delta_role_vs_rung3": bootstrap_group_delta(
            p3, prole, y, n_boot=n_boot, seed=seed
        ),
        "delta_role_vs_placebo": bootstrap_group_delta(
            pplc, prole, y, n_boot=n_boot, seed=seed + 1
        ),
    }

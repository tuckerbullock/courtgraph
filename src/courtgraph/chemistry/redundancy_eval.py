"""Evaluate the redundancy / anti-synergy model (candidate idea #3).

Rung 2 vs rung 3 vs the concentration-feature model vs a permuted-role
placebo, on the three leakage-safe holdouts. The role clustering is fit once
on the full profile set (outcome-blind) and reused per fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.baseline_ladder import _group_realized, _rung2_band
from courtgraph.chemistry.calibration import calibration_report
from courtgraph.chemistry.evaluate import _group_index, _rmse
from courtgraph.chemistry.features import FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge
from courtgraph.chemistry.redundancy import RedundancyConfig, RedundancyInteraction
from courtgraph.chemistry.splits import SplitManifest
from courtgraph.chemistry.stints import StintTable
from courtgraph.features.role_clusters import RoleClustering, permuted_clustering

_HOLDOUTS = ("chronological", "unseen_pair", "unseen_lineup")


@dataclass(frozen=True)
class RedundancyHoldoutResult:
    kind: str
    n_train: int
    n_test: int
    n_groups: int
    rung2_macro_rmse: float
    rung3_macro_rmse: float
    redundancy_macro_rmse: float
    redundancy_placebo_macro_rmse: float
    rung3_calibration: dict[str, float]
    redundancy_calibration: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_groups": self.n_groups,
            "rung2_macro_rmse": self.rung2_macro_rmse,
            "rung3_macro_rmse": self.rung3_macro_rmse,
            "redundancy_macro_rmse": self.redundancy_macro_rmse,
            "redundancy_placebo_macro_rmse": self.redundancy_placebo_macro_rmse,
            "rung3_calibration": dict(self.rung3_calibration),
            "redundancy_calibration": dict(self.redundancy_calibration),
        }


@dataclass(frozen=True)
class RedundancyComparison:
    rho: dict[str, float]
    rho_placebo: dict[str, float]
    variance_components: dict[str, Any]
    holdouts: tuple[RedundancyHoldoutResult, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho": dict(self.rho),
            "rho_placebo": dict(self.rho_placebo),
            "variance_components": dict(self.variance_components),
            "holdouts": [h.as_dict() for h in self.holdouts],
            "notes": list(self.notes),
        }


def evaluate_redundancy(
    table: StintTable,
    splits: dict[str, SplitManifest],
    clustering: RoleClustering,
    *,
    seed: int = 0,
    n_boot: int = 120,
    config: HierarchicalConfig | None = None,
    redundancy_config: RedundancyConfig | None = None,
) -> RedundancyComparison:
    red_cfg = redundancy_config or RedundancyConfig()
    placebo_clustering = permuted_clustering(clustering, seed + 1)

    space_all = FeatureSpace.from_training(table)
    full_red = RedundancyInteraction.fit(
        space_all.build(table), space_all, clustering, config=red_cfg
    )
    full_placebo = RedundancyInteraction.fit(
        space_all.build(table), space_all, placebo_clustering, config=red_cfg
    )

    holdouts: list[RedundancyHoldoutResult] = []
    for kind in _HOLDOUTS:
        manifest = splits[kind]
        train_table = manifest.train_table(table)
        test_table = manifest.test_table(table)
        space = FeatureSpace.from_training(train_table)
        train_design = space.build(train_table)
        test_design = space.build(test_table)

        rung2 = AdditiveRidge.fit(train_design, space)
        rung3 = HierarchicalRidge.fit(train_design, space, config=config)
        red = RedundancyInteraction.fit(train_design, space, clustering, config=red_cfg)
        red_placebo = RedundancyInteraction.fit(
            train_design, space, placebo_clustering, config=red_cfg
        )

        groups = _group_index(test_table, manifest)
        realized = _group_realized(test_design, groups)
        keys = list(groups)
        group_arrays = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}
        y = np.array([realized[k] for k in keys])

        r3 = rung3.group_predictive(test_design, group_arrays)
        p3 = np.array([r3[k][0] for k in keys])
        s3 = np.array([r3[k][1] for k in keys])
        rr = red.group_predictive(test_design, group_arrays)
        pr = np.array([rr[k][0] for k in keys])
        sr = np.array([rr[k][1] for k in keys])
        rrp = red_placebo.group_predictive(test_design, group_arrays)
        prp = np.array([rrp[k][0] for k in keys])
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
            RedundancyHoldoutResult(
                kind=kind,
                n_train=len(train_table),
                n_test=len(test_table),
                n_groups=len(keys),
                rung2_macro_rmse=_rmse(p2, y),
                rung3_macro_rmse=_rmse(p3, y),
                redundancy_macro_rmse=_rmse(pr, y),
                redundancy_placebo_macro_rmse=_rmse(prp, y),
                rung3_calibration=calibration_report(p3, s3, y),
                redundancy_calibration=calibration_report(pr, sr, y),
            )
        )

    notes = (
        "conc_d = (sum z_id)^2 - sum z_id^2 over the offensive lineup's "
        "standardized role vectors; rho_d < 0 means concentrating skill d hurts.",
        "Role vectors fit once on the full profile set (outcome-blind); the "
        "permuted-role placebo is the control.",
    )
    return RedundancyComparison(
        rho=full_red.rho_by_feature(),
        rho_placebo=full_placebo.rho_by_feature(),
        variance_components=full_red.variance_components(),
        holdouts=tuple(holdouts),
        notes=notes,
    )

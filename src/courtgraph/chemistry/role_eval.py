"""Evaluate the role-conditioned interaction model (candidate idea #1).

Rung 2 (additive ridge) vs rung 3 (hierarchical EB) vs the role-conditioned
interaction model vs a **permuted-role placebo**, on the three leakage-safe
holdouts. The role clustering is fit once on the full player-profile set --
outcome-blind (it uses only usage / shot mix / playmaking rates, never lineup
value), so this is not leakage -- and reused for every fold.

The question: does keying the offensive interaction on role-cluster pairs
(15 pooled parameters, each backed by thousands of stints) beat additive
talent where the ~2-3k thin per-identity pairs of rung 4 did not, and does it
beat its own placebo?
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
from courtgraph.chemistry.role_interaction import (
    RoleClusterInteraction,
    RoleInteractionConfig,
)
from courtgraph.chemistry.splits import SplitManifest
from courtgraph.chemistry.stints import StintTable
from courtgraph.features.role_clusters import RoleClustering, permuted_clustering

_HOLDOUTS = ("chronological", "unseen_pair", "unseen_lineup")


@dataclass(frozen=True)
class RoleHoldoutResult:
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
    role_micro_rmse: float
    rung3_micro_rmse: float

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
            "role_micro_rmse": self.role_micro_rmse,
            "rung3_micro_rmse": self.rung3_micro_rmse,
        }


@dataclass(frozen=True)
class RoleComparison:
    n_clusters: int
    n_clustered_players: int
    role_pair_matrix: list[list[float]]
    role_variance_components: dict[str, Any]
    cluster_centers: dict[str, dict[str, float]]
    holdouts: tuple[RoleHoldoutResult, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "n_clustered_players": self.n_clustered_players,
            "role_pair_matrix": self.role_pair_matrix,
            "role_variance_components": dict(self.role_variance_components),
            "cluster_centers": {k: dict(v) for k, v in self.cluster_centers.items()},
            "holdouts": [h.as_dict() for h in self.holdouts],
            "notes": list(self.notes),
        }


def evaluate_role_interaction(
    table: StintTable,
    splits: dict[str, SplitManifest],
    clustering: RoleClustering,
    *,
    seed: int = 0,
    n_boot: int = 120,
    config: HierarchicalConfig | None = None,
    role_config: RoleInteractionConfig | None = None,
) -> RoleComparison:
    role_cfg = role_config or RoleInteractionConfig()

    space_all = FeatureSpace.from_training(table)
    full_role = RoleClusterInteraction.fit(
        space_all.build(table), space_all, clustering, config=role_cfg
    )

    holdouts: list[RoleHoldoutResult] = []
    for kind in _HOLDOUTS:
        manifest = splits[kind]
        train_table = manifest.train_table(table)
        test_table = manifest.test_table(table)
        space = FeatureSpace.from_training(train_table)
        train_design = space.build(train_table)
        test_design = space.build(test_table)

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

        groups = _group_index(test_table, manifest)
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
            RoleHoldoutResult(
                kind=kind,
                n_train=len(train_table),
                n_test=len(test_table),
                n_groups=len(keys),
                rung2_macro_rmse=_rmse(p2, y),
                rung3_macro_rmse=_rmse(p3, y),
                role_macro_rmse=_rmse(prole, y),
                role_placebo_macro_rmse=_rmse(prolep, y),
                rung3_calibration=calibration_report(p3, s3, y),
                role_calibration=calibration_report(prole, srole, y),
                role_micro_rmse=_rmse(
                    role.predict(test_design), test_design.y, test_design.weight
                ),
                rung3_micro_rmse=_rmse(
                    rung3.predict(test_design), test_design.y, test_design.weight
                ),
            )
        )

    centers = {
        f"cluster_{c}": clustering.center_profile(c)
        for c in range(clustering.n_clusters)
    }
    notes = (
        "Role clusters are fit once on the full player-profile set (usage / "
        "shot mix / playmaking rates only -- outcome-blind) and reused per fold.",
        "role_pair_matrix[a][b] is the fitted offensive surplus (points per "
        "100) for a lineup pair of a cluster-a and a cluster-b player.",
    )
    return RoleComparison(
        n_clusters=clustering.n_clusters,
        n_clustered_players=len(clustering.player_cluster),
        role_pair_matrix=[list(row) for row in full_role.role_pair_matrix()],
        role_variance_components=full_role.variance_components(),
        cluster_centers=centers,
        holdouts=tuple(holdouts),
        notes=notes,
    )

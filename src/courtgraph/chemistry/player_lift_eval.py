"""Evaluate master plan §45 Phase A -- pooled player-lift on lineup value.

rung 2 vs rung 3 vs the lift model vs a player-permutation placebo, on the four
leakage-safe tasks. Supported only if the lift model beats rung 3 out of sample
**and** beats its placebo, with maintained calibration (contract §17, §45.4).
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
from courtgraph.chemistry.player_lift import PlayerLift, PlayerLiftConfig
from courtgraph.chemistry.splits import SplitManifest
from courtgraph.chemistry.stints import StintTable

_HOLDOUTS = ("chronological", "unseen_pair", "unseen_lineup")


@dataclass(frozen=True)
class PlayerLiftHoldoutResult:
    kind: str
    n_train: int
    n_test: int
    n_groups: int
    rung2_macro_rmse: float
    rung3_macro_rmse: float
    lift_macro_rmse: float
    lift_placebo_macro_rmse: float
    tau_lambda: float
    tau_lambda_placebo: float
    rung3_calibration: dict[str, float]
    lift_calibration: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_groups": self.n_groups,
            "rung2_macro_rmse": self.rung2_macro_rmse,
            "rung3_macro_rmse": self.rung3_macro_rmse,
            "lift_macro_rmse": self.lift_macro_rmse,
            "lift_placebo_macro_rmse": self.lift_placebo_macro_rmse,
            "tau_lambda": self.tau_lambda,
            "tau_lambda_placebo": self.tau_lambda_placebo,
            "rung3_calibration": dict(self.rung3_calibration),
            "lift_calibration": dict(self.lift_calibration),
        }


@dataclass(frozen=True)
class PlayerLiftComparison:
    variance_components: dict[str, Any]
    top_lifts: list[dict[str, float]]
    holdouts: tuple[PlayerLiftHoldoutResult, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "variance_components": dict(self.variance_components),
            "top_lifts": [dict(t) for t in self.top_lifts],
            "holdouts": [h.as_dict() for h in self.holdouts],
            "notes": list(self.notes),
        }


def evaluate_player_lift(
    table: StintTable,
    splits: dict[str, SplitManifest],
    *,
    seed: int = 0,
    n_boot: int = 120,
    config: HierarchicalConfig | None = None,
    lift_config: PlayerLiftConfig | None = None,
) -> PlayerLiftComparison:
    cfg = lift_config or PlayerLiftConfig()

    space_all = FeatureSpace.from_training(table)
    full = PlayerLift.fit(space_all.build(table), space_all, config=cfg, seed=seed)

    holdouts: list[PlayerLiftHoldoutResult] = []
    for kind in _HOLDOUTS:
        manifest = splits[kind]
        train_table = manifest.train_table(table)
        test_table = manifest.test_table(table)
        space = FeatureSpace.from_training(train_table)
        train_design = space.build(train_table)
        test_design = space.build(test_table)

        rung2 = AdditiveRidge.fit(train_design, space)
        rung3 = HierarchicalRidge.fit(train_design, space, config=config)
        lift = PlayerLift.fit(train_design, space, config=cfg, seed=seed)
        lift_p = PlayerLift.fit(
            train_design, space, config=cfg, seed=seed, permuted=True
        )

        groups = _group_index(test_table, manifest)
        realized = _group_realized(test_design, groups)
        keys = list(groups)
        ga = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}
        y = np.array([realized[k] for k in keys])

        r3 = rung3.group_predictive(test_design, ga)
        p3 = np.array([r3[k][0] for k in keys])
        s3 = np.array([r3[k][1] for k in keys])
        rl = lift.group_predictive(test_design, ga)
        pl = np.array([rl[k][0] for k in keys])
        sl = np.array([rl[k][1] for k in keys])
        rlp = lift_p.group_predictive(test_design, ga)
        plp = np.array([rlp[k][0] for k in keys])
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
            PlayerLiftHoldoutResult(
                kind=kind,
                n_train=len(train_table),
                n_test=len(test_table),
                n_groups=len(keys),
                rung2_macro_rmse=_rmse(p2, y),
                rung3_macro_rmse=_rmse(p3, y),
                lift_macro_rmse=_rmse(pl, y),
                lift_placebo_macro_rmse=_rmse(plp, y),
                tau_lambda=float(np.sqrt(lift.tau_lambda2)),
                tau_lambda_placebo=float(np.sqrt(lift_p.tau_lambda2)),
                rung3_calibration=calibration_report(p3, s3, y),
                lift_calibration=calibration_report(pl, sl, y),
            )
        )

    top = [
        {"player_id": float(pid), "lift": lv, "sd": sd}
        for pid, lv, sd in full.top_lifts(20)
    ]
    notes = (
        "lift term: sum_i lambda_i * (A_off,s - alpha_i); two-stage fit with "
        "alpha frozen from rung 3, tau_lambda by marginal likelihood.",
        "placebo permutes the lambda_i -> player assignment (bijection; same "
        "count and exposure, teammate-talent correspondence broken).",
        "rank-1 provision/need with the receiver pinned to observed talent -- "
        "rung 5's general form already failed, so this has a negative prior.",
    )
    return PlayerLiftComparison(
        variance_components=full.variance_components(),
        top_lifts=top,
        holdouts=tuple(holdouts),
        notes=notes,
    )

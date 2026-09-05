"""Better-powered confirmation of the three interaction positives.

Directions #1 (role-cluster interaction), #2 (mechanistic ``three_share``) and
#3 (redundancy) each beat the rung-3 baseline and a permuted-role placebo by a
small margin on the structural holdouts, but on 40-60 group means with no
confidence interval. This module re-runs all three with

* a **wider** ``unseen_lineup`` holdout (default 120 groups);
* a **bootstrap CI** on ``rmse(rung 3) - rmse(model)`` and
  ``rmse(placebo) - rmse(model)`` over the held-out group means
  (:func:`courtgraph.chemistry.baseline_ladder.bootstrap_group_delta`);
* a **K sweep** for the role model, to check the effect is not a K = 5 artefact.

``unseen_pair`` stays capped near 40 groups by the 15 %-of-stints exposure
budget; the bootstrap CI carries that uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.baseline_ladder import (
    _group_realized,
    _rung2_band,
    bootstrap_group_delta,
)
from courtgraph.chemistry.evaluate import _group_index, _rmse
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalRidge
from courtgraph.chemistry.mechanistic import (
    EVENT_OUTCOMES,
    mechanistic_table_and_design,
)
from courtgraph.chemistry.redundancy import RedundancyInteraction
from courtgraph.chemistry.role_interaction import RoleClusterInteraction
from courtgraph.chemistry.splits import SplitManifest, make_all_splits
from courtgraph.chemistry.stints import StintTable
from courtgraph.features.role_clusters import RoleClustering
from courtgraph.features.stint_events import EventAttribution
from courtgraph.features.stint_shots import ShotAttribution

FloatArray = NDArray[np.float64]
_STRUCTURAL = ("unseen_pair", "unseen_lineup")


@dataclass(frozen=True)
class ConfirmRow:
    model: str
    k: int
    holdout: str
    outcome: str
    n_groups: int
    rmse_model: float
    rmse_rung3: float
    rmse_placebo: float
    delta_vs_rung3: dict[str, float]
    delta_vs_placebo: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "k": self.k,
            "holdout": self.holdout,
            "outcome": self.outcome,
            "n_groups": self.n_groups,
            "rmse_model": self.rmse_model,
            "rmse_rung3": self.rmse_rung3,
            "rmse_placebo": self.rmse_placebo,
            "delta_vs_rung3": dict(self.delta_vs_rung3),
            "delta_vs_placebo": dict(self.delta_vs_placebo),
        }


@dataclass(frozen=True)
class ConfirmationResult:
    k_values: tuple[int, ...]
    n_boot: int
    holdout_groups: dict[str, int]
    rows: tuple[ConfirmRow, ...]
    mediation: dict[str, float]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_values": list(self.k_values),
            "n_boot": self.n_boot,
            "holdout_groups": dict(self.holdout_groups),
            "rows": [r.as_dict() for r in self.rows],
            "mediation": dict(self.mediation),
            "notes": list(self.notes),
        }


def _group_points(
    model: Any, test_design: DesignMatrices, group_arrays: dict[str, NDArray[np.int64]]
) -> FloatArray:
    preds = model.group_predictive(test_design, group_arrays)
    return np.array([preds[k][0] for k in group_arrays])


def _row(
    model_name: str,
    k: int,
    holdout: str,
    outcome: str,
    p_model: FloatArray,
    p_rung3: FloatArray,
    p_placebo: FloatArray,
    y: FloatArray,
    *,
    n_boot: int,
    seed: int,
) -> ConfirmRow:
    return ConfirmRow(
        model=model_name,
        k=k,
        holdout=holdout,
        outcome=outcome,
        n_groups=len(y),
        rmse_model=_rmse(p_model, y),
        rmse_rung3=_rmse(p_rung3, y),
        rmse_placebo=_rmse(p_placebo, y),
        delta_vs_rung3=bootstrap_group_delta(
            p_rung3, p_model, y, n_boot=n_boot, seed=seed
        ),
        delta_vs_placebo=bootstrap_group_delta(
            p_placebo, p_model, y, n_boot=n_boot, seed=seed + 1
        ),
    )


def run_confirmation(
    table: StintTable,
    clustering_by_k: dict[int, RoleClustering],
    attribution: ShotAttribution | None,
    *,
    event_attribution: EventAttribution | None = None,
    outcomes: tuple[str, ...] = ("three_share",),
    n_lineups: int = 120,
    n_boot: int = 2000,
    boot_seed: int = 0,
    min_fga: int = 3,
    mediation_k: int = 5,
) -> ConfirmationResult:
    from courtgraph.features.role_clusters import permuted_clustering

    k_values = tuple(sorted(clustering_by_k))
    placebo_by_k = {
        k: permuted_clustering(c, boot_seed + 1) for k, c in clustering_by_k.items()
    }

    splits: dict[str, SplitManifest] = make_all_splits(table, n_lineups=n_lineups)
    holdout_groups: dict[str, int] = {}
    rows: list[ConfirmRow] = []
    mediation: dict[str, float] = {}

    for holdout in _STRUCTURAL:
        manifest = splits[holdout]
        train_table = manifest.train_table(table)
        test_table = manifest.test_table(table)
        space = FeatureSpace.from_training(train_table)
        train_design = space.build(train_table)
        test_design = space.build(test_table)

        groups = _group_index(test_table, manifest)
        keys = list(groups)
        holdout_groups[holdout] = len(keys)
        ga = {g: np.asarray(groups[g], dtype=np.int64) for g in keys}
        y = np.array([_group_realized(test_design, groups)[k] for k in keys])

        rung3 = HierarchicalRidge.fit(train_design, space)
        p3 = _group_points(rung3, test_design, ga)
        rung2 = AdditiveRidge.fit(train_design, space)
        r2 = _rung2_band(
            train_table,
            space,
            train_design,
            test_design,
            groups,
            rung2,
            seed=boot_seed,
            n_boot=0,
        )
        p2 = np.array([r2[k][0] for k in keys])
        rows.append(
            _row(
                "rung2",
                0,
                holdout,
                "points_per_100",
                p2,
                p3,
                p3,
                y,
                n_boot=n_boot,
                seed=boot_seed,
            )
        )

        for k in k_values:
            clustering = clustering_by_k[k]
            placebo = placebo_by_k[k]
            role = RoleClusterInteraction.fit(train_design, space, clustering)
            role_p = RoleClusterInteraction.fit(train_design, space, placebo)
            rows.append(
                _row(
                    "role",
                    k,
                    holdout,
                    "points_per_100",
                    _group_points(role, test_design, ga),
                    p3,
                    _group_points(role_p, test_design, ga),
                    y,
                    n_boot=n_boot,
                    seed=boot_seed,
                )
            )
            red = RedundancyInteraction.fit(train_design, space, clustering)
            red_p = RedundancyInteraction.fit(train_design, space, placebo)
            rows.append(
                _row(
                    "redundancy",
                    k,
                    holdout,
                    "points_per_100",
                    _group_points(red, test_design, ga),
                    p3,
                    _group_points(red_p, test_design, ga),
                    y,
                    n_boot=n_boot,
                    seed=boot_seed,
                )
            )

        # mechanistic outcomes on the same holdout
        for outcome in outcomes:
            is_event = outcome in EVENT_OUTCOMES
            attr = event_attribution if is_event else attribution
            if attr is None:
                needed = "event_attribution" if is_event else "attribution"
                raise ValueError(f"outcome {outcome!r} needs {needed}")
            _, mech_train = mechanistic_table_and_design(
                space, train_table, attr, outcome, min_fga=min_fga
            )
            mech_test_kt, mech_test = mechanistic_table_and_design(
                space, test_table, attr, outcome, min_fga=min_fga
            )
            m_groups = _group_index(mech_test_kt, manifest)
            m_keys = list(m_groups)
            m_ga = {g: np.asarray(m_groups[g], dtype=np.int64) for g in m_keys}
            m_y = np.array([_group_realized(mech_test, m_groups)[k] for k in m_keys])
            m_rung3 = HierarchicalRidge.fit(mech_train, space)
            m_p3 = _group_points(m_rung3, mech_test, m_ga)
            for k in k_values:
                m_role = RoleClusterInteraction.fit(
                    mech_train, space, clustering_by_k[k]
                )
                m_role_p = RoleClusterInteraction.fit(
                    mech_train, space, placebo_by_k[k]
                )
                m_role_point = _group_points(m_role, mech_test, m_ga)
                rows.append(
                    _row(
                        "mechanistic_role",
                        k,
                        holdout,
                        outcome,
                        m_role_point,
                        m_p3,
                        _group_points(m_role_p, mech_test, m_ga),
                        m_y,
                        n_boot=n_boot,
                        seed=boot_seed,
                    )
                )
                # mediation: does the role model's incremental three_share
                # prediction line up with the lineup's scoring surprise?
                if (
                    outcome == "three_share"
                    and holdout == "unseen_lineup"
                    and k == mediation_k
                ):
                    dts = {
                        m_keys[i]: float(m_role_point[i] - m_p3[i])
                        for i in range(len(m_keys))
                    }
                    dpts = {keys[i]: float(y[i] - p3[i]) for i in range(len(keys))}
                    common = [g for g in m_keys if g in dpts]
                    if len(common) >= 5:
                        a = np.array([dts[g] for g in common])
                        b = np.array([dpts[g] for g in common])
                        mediation = {
                            "n_lineups": float(len(common)),
                            "corr_d_three_share_vs_d_points": float(
                                np.corrcoef(a, b)[0, 1]
                            ),
                            "mean_abs_d_three_share": float(np.abs(a).mean()),
                        }

    notes = (
        "delta = rmse(baseline) - rmse(model) over the held-out group means; "
        "positive = model better. CI is the 95 % bootstrap interval over "
        f"{n_boot} group resamples; frac_gt_0 is P(delta > 0).",
        "unseen_pair stays near 40 groups (exposure budget); unseen_lineup is "
        f"widened to {n_lineups}.",
        "mediation: correlation over held-out unseen lineups between the role "
        "model's (three_share prediction - rung 3) and the lineup's (realized "
        "points/100 - rung 3). ~0 means the shot-mix non-additivity is real "
        "but does not move scoring.",
    )
    return ConfirmationResult(
        k_values=tuple(k_values),
        n_boot=n_boot,
        holdout_groups=holdout_groups,
        rows=tuple(rows),
        mediation=mediation,
        notes=notes,
    )

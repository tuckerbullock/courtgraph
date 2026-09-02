"""Evaluate master plan §45 Phase B -- the per-player-production lift model.

Base-only vs base + pooled lift vs base + giver-shuffle placebo, on a
chronological holdout, macro over held-out receiver players, with a bootstrap CI
on ``RMSE(base) - RMSE(lift)`` and ``RMSE(placebo) - RMSE(lift)`` (contract
§45.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from courtgraph.chemistry.baseline_ladder import bootstrap_group_delta
from courtgraph.chemistry.phase_b import (
    PhaseBConfig,
    PhaseBModel,
    build_phase_b_design,
)
from courtgraph.chemistry.stints import StintTable
from courtgraph.features.player_production import PlayerStintProduction


@dataclass(frozen=True)
class PhaseBComparison:
    assist_credit: float
    n_train: int
    n_test: int
    n_test_receivers: int
    variance_components: dict[str, Any]
    base_macro_rmse: float
    lift_macro_rmse: float
    placebo_macro_rmse: float
    delta_vs_base: dict[str, float]
    delta_vs_placebo: dict[str, float]
    top_lifts: list[dict[str, float]]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "assist_credit": self.assist_credit,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_test_receivers": self.n_test_receivers,
            "variance_components": dict(self.variance_components),
            "base_macro_rmse": self.base_macro_rmse,
            "lift_macro_rmse": self.lift_macro_rmse,
            "placebo_macro_rmse": self.placebo_macro_rmse,
            "delta_vs_base": dict(self.delta_vs_base),
            "delta_vs_placebo": dict(self.delta_vs_placebo),
            "top_lifts": [dict(t) for t in self.top_lifts],
            "notes": list(self.notes),
        }


def _macro_by_receiver(design: Any, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Possession-weighted realised and predicted production per receiver."""

    keys = np.unique(design.receiver)
    real = np.empty(len(keys))
    prd = np.empty(len(keys))
    for i, k in enumerate(keys):
        m = design.receiver == k
        w = design.w[m]
        tw = w.sum()
        real[i] = float((design.y[m] * w).sum() / tw)
        prd[i] = float((pred[m] * w).sum() / tw)
    return real, prd


def evaluate_phase_b(
    table: StintTable,
    production: list[PlayerStintProduction],
    *,
    seed: int = 0,
    n_boot: int = 3000,
    config: PhaseBConfig | None = None,
    train_fraction: float = 0.7,
) -> PhaseBComparison:
    cfg = config or PhaseBConfig()

    # chronological split of the stints, then filter production to each side
    by_date = sorted(table, key=table.chronological_key)
    cut = by_date[int(len(by_date) * train_fraction)].game_date
    train_ids = {s.stint_id for s in table if s.game_date < cut}
    test_ids = {s.stint_id for s in table if s.game_date >= cut}
    train_table = StintTable.from_stints(s for s in table if s.stint_id in train_ids)
    test_table = StintTable.from_stints(s for s in table if s.stint_id in test_ids)
    train_prod = [r for r in production if r.stint_id in train_ids]
    test_prod = [r for r in production if r.stint_id in test_ids]

    train_design = build_phase_b_design(train_table, train_prod, config=cfg)
    model = PhaseBModel.fit(train_design, config=cfg, seed=seed)
    placebo = PhaseBModel.fit(train_design, config=cfg, seed=seed, permuted=True)

    # build the test design on the SAME player vocabulary as train
    test_design = build_phase_b_design(test_table, test_prod, config=cfg)
    # restrict test rows to players present in the train vocabulary
    train_pset = set(train_design.player_ids)
    keep = np.array(
        [
            test_design.player_ids[r] in train_pset
            and all(
                (c < 0) or (test_design.player_ids[c] in train_pset)
                for c in test_design.teammates[i]
            )
            for i, r in enumerate(test_design.receiver)
        ]
    )
    # remap test player rows to the train vocabulary
    tmap = {p: i for i, p in enumerate(train_design.player_ids)}

    def _remap(col: int) -> int:
        return tmap.get(test_design.player_ids[col], -1) if col >= 0 else -1

    rec = np.array([_remap(int(c)) for c in test_design.receiver[keep]], dtype=np.int64)
    team = np.array(
        [[_remap(int(c)) for c in row] for row in test_design.teammates[keep]],
        dtype=np.int64,
    )
    from courtgraph.chemistry.phase_b import PhaseBDesign

    td = PhaseBDesign(
        y=test_design.y[keep],
        w=test_design.w[keep],
        receiver=rec,
        teammates=team,
        context=test_design.context[keep],
        player_ids=train_design.player_ids,
        context_names=test_design.context_names,
    )

    real_b, pred_base = _macro_by_receiver(td, model.predict_base_only(td))
    _, pred_lift = _macro_by_receiver(td, model.predict(td))
    _, pred_plc = _macro_by_receiver(td, placebo.predict(td))

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    notes = (
        "outcome: credited production per 100 offensive possessions "
        f"(assist_credit={cfg.assist_credit}); macro over held-out receivers.",
        "base-only = mu + context + base_k; lift adds sum_i lift_i over the "
        "receiver's four teammates. placebo permutes lift_i -> player.",
        "chronological holdout only; unseen giver-receiver and transaction-cohort "
        "checks are follow-ups.",
    )
    return PhaseBComparison(
        assist_credit=cfg.assist_credit,
        n_train=len(train_design.y),
        n_test=int(keep.sum()),
        n_test_receivers=len(real_b),
        variance_components=model.variance_components(),
        base_macro_rmse=_rmse(pred_base, real_b),
        lift_macro_rmse=_rmse(pred_lift, real_b),
        placebo_macro_rmse=_rmse(pred_plc, real_b),
        delta_vs_base=bootstrap_group_delta(
            pred_base, pred_lift, real_b, n_boot=n_boot, seed=seed
        ),
        delta_vs_placebo=bootstrap_group_delta(
            pred_plc, pred_lift, real_b, n_boot=n_boot, seed=seed + 1
        ),
        top_lifts=[
            {"player_id": float(pid), "lift": lv, "sd": sd}
            for pid, lv, sd in model.top_lifts(20)
        ],
        notes=notes,
    )

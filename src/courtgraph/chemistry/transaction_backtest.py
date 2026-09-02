"""Transaction backtest -- roster changes as natural experiments (contract T4).

For each clean cross-season team switch (player X: A -> B, first B season S),
fit rung 3 on every stint from a season **earlier than S** -- a model that has
never seen X on B -- then compare B's realised post-move lineup value (X on the
floor) against rung 3's additive prediction, which carries X's *transferred*
alpha from his pre-move history.

    Delta = possession-weighted (realised - predicted)  over X's post-move stints

* ``Delta ~ 0``  -> additive talent transfers cleanly (the null).
* ``Delta`` systematically nonzero -> a player's contribution is partly
  roster-specific (fit / chemistry that moves, or does not move, with him).

A **phantom** cohort (players who did not switch, given a fake cross-season
"move" from the same team) receives the identical computation and is the null
band: a real roster-fit effect must beat the phantom ``|Delta|``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline_ladder import _group_realized
from courtgraph.chemistry.features import FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge
from courtgraph.chemistry.stints import StintTable
from courtgraph.chemistry.transactions import (
    Transaction,
    find_transactions,
    leakage_safe_train,
    phantom_transactions,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TransactionDelta:
    player_id: int
    from_team_id: int
    to_team_id: int
    cutover_season: str
    n_stints: int
    possessions: int
    realized: float
    predicted: float
    delta: float  # realized - predicted (points / 100)
    transferred_alpha: float  # X's rung-3 offensive coef in the leakage-safe fit

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "from_team_id": self.from_team_id,
            "to_team_id": self.to_team_id,
            "cutover_season": self.cutover_season,
            "n_stints": self.n_stints,
            "possessions": self.possessions,
            "realized": self.realized,
            "predicted": self.predicted,
            "delta": self.delta,
            "transferred_alpha": self.transferred_alpha,
        }


@dataclass(frozen=True)
class TransactionBacktest:
    n_transactions: int
    n_phantom: int
    min_poss_each_side: int
    real: dict[str, float]  # summary stats of the real Delta cohort
    phantom: dict[str, float]
    real_vs_phantom_abs: dict[str, float]  # bootstrap on |Delta_real| - |Delta_phantom|
    alpha_regression: dict[str, float]  # Delta ~ transferred_alpha (shrinkage check)
    deltas: tuple[TransactionDelta, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_transactions": self.n_transactions,
            "n_phantom": self.n_phantom,
            "min_poss_each_side": self.min_poss_each_side,
            "real": dict(self.real),
            "phantom": dict(self.phantom),
            "real_vs_phantom_abs": dict(self.real_vs_phantom_abs),
            "alpha_regression": dict(self.alpha_regression),
            "deltas": [d.as_dict() for d in self.deltas],
            "notes": list(self.notes),
        }


def _deltas_for_cohort(
    table: StintTable,
    txns: list[Transaction],
    *,
    config: HierarchicalConfig | None,
) -> list[TransactionDelta]:
    """One rung-3 fit per cutover season (shared by every transaction landing in
    it), then a possession-weighted realised-minus-predicted per transaction."""

    by_season: dict[str, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.post_stint_ids:
            by_season[t.cutover_season].append(t)

    out: list[TransactionDelta] = []
    for season, group in by_season.items():
        train = leakage_safe_train(table, season)
        if len(train) < 500:
            continue
        space = FeatureSpace.from_training(train)
        model = HierarchicalRidge.fit(space.build(train), space, config=config)
        pidx = space.player_index()

        test_ids = {sid for t in group for sid in t.post_stint_ids}
        test_table = table.subset(test_ids)
        if len(test_table) == 0:
            continue
        pos = {sid: i for i, sid in enumerate(s.stint_id for s in test_table)}
        test_design = space.build(test_table)

        groups = {
            str(t.player_id): np.array(
                [pos[sid] for sid in t.post_stint_ids if sid in pos], dtype=np.int64
            )
            for t in group
        }
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        realized = _group_realized(
            test_design, {k: v.tolist() for k, v in groups.items()}
        )
        pred = model.group_predictive(test_design, groups)

        for t in group:
            key = str(t.player_id)
            if key not in groups:
                continue
            rows = groups[key]
            w = float(test_design.weight[rows].sum())
            r = float(realized[key])
            p = float(pred[key][0])
            alpha = (
                float(model.offense_coef[pidx[t.player_id]])
                if (t.player_id in pidx)
                else 0.0
            )
            out.append(
                TransactionDelta(
                    player_id=t.player_id,
                    from_team_id=t.from_team_id,
                    to_team_id=t.to_team_id,
                    cutover_season=season,
                    n_stints=len(rows),
                    possessions=int(round(w)),
                    realized=r,
                    predicted=p,
                    delta=r - p,
                    transferred_alpha=alpha,
                )
            )
    return out


def _summary(deltas: list[TransactionDelta]) -> dict[str, float]:
    d = np.array([x.delta for x in deltas], dtype=np.float64)
    if len(d) == 0:
        return {"n": 0.0}
    return {
        "n": float(len(d)),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "mean_abs": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "frac_gt_0": float((d > 0).mean()),
    }


def _bootstrap_abs_gap(
    real: list[TransactionDelta],
    phantom: list[TransactionDelta],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap CI on ``mean|Delta_real| - mean|Delta_phantom|`` -- positive and
    CI-excluding-0 means real moves scatter more than the phantom null."""

    a = np.abs(np.array([x.delta for x in real], dtype=np.float64))
    b = np.abs(np.array([x.delta for x in phantom], dtype=np.float64))
    if len(a) < 2 or len(b) < 2:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "frac_gt_0": 0.0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ra = a[rng.integers(0, len(a), len(a))].mean()
        rb = b[rng.integers(0, len(b), len(b))].mean()
        diffs[i] = ra - rb
    diffs.sort()
    return {
        "mean": float(a.mean() - b.mean()),
        "ci_lo": float(diffs[int(0.025 * n_boot)]),
        "ci_hi": float(diffs[min(int(0.975 * n_boot), n_boot - 1)]),
        "frac_gt_0": float((diffs > 0).mean()),
    }


def _alpha_regression(deltas: list[TransactionDelta]) -> dict[str, float]:
    if len(deltas) < 5:
        return {"slope": 0.0, "intercept": 0.0, "corr": 0.0}
    x = np.array([d.transferred_alpha for d in deltas], dtype=np.float64)
    y = np.array([d.delta for d in deltas], dtype=np.float64)
    if float(np.ptp(x)) == 0.0:
        return {"slope": 0.0, "intercept": float(y.mean()), "corr": 0.0}
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "corr": corr}


def run_backtest(
    table: StintTable,
    *,
    min_poss_each_side: int = 500,
    n_phantom: int | None = None,
    n_boot: int = 3000,
    seed: int = 0,
    config: HierarchicalConfig | None = None,
) -> TransactionBacktest:
    real_txns = find_transactions(table, min_poss_each_side=min_poss_each_side)
    phantom_txns = phantom_transactions(
        table,
        real_txns,
        min_poss_each_side=min_poss_each_side,
        seed=seed,
        n=n_phantom,
    )

    real_d = _deltas_for_cohort(table, real_txns, config=config)
    phantom_d = _deltas_for_cohort(table, phantom_txns, config=config)

    notes = (
        "Delta = possession-weighted (realised - rung-3 prediction) over the "
        "player's post-move stints on the new team; rung 3 fit on seasons "
        "strictly before the move, so X's alpha is transferred, never fit on "
        "the new roster.",
        "phantom cohort: non-movers given a fake same-team cross-season 'move'; "
        "identical computation. real |Delta| must exceed phantom |Delta| for a "
        "roster-fit effect.",
        "alpha_regression: Delta vs transferred alpha -- a nonzero slope means "
        "the gap partly tracks rung-3 shrinkage of the mover's coefficient, not "
        "roster fit.",
    )
    return TransactionBacktest(
        n_transactions=len(real_d),
        n_phantom=len(phantom_d),
        min_poss_each_side=min_poss_each_side,
        real=_summary(real_d),
        phantom=_summary(phantom_d),
        real_vs_phantom_abs=_bootstrap_abs_gap(
            real_d, phantom_d, n_boot=n_boot, seed=seed
        ),
        alpha_regression=_alpha_regression(real_d),
        deltas=tuple(real_d),
        notes=notes,
    )

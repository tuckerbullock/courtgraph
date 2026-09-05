"""Model-ladder rungs 0 and 1 -- the required predecessors to RAPM.

`RESEARCH_CONTRACT.md` §16 requires every rung 0-7 to exist and be reported
together. Rungs 2-5 have been implemented and evaluated; this fills the two
below them:

* **Rung 0 -- context-only mean.** Weighted least squares on the context
  columns alone (intercept, home, period, season, rest, ...), no player
  terms. Predicts the same value for every lineup in a given context.
  Must beat a constant mean (it can, on the chronological holdout, by
  tracking the season columns).

* **Rung 1 -- raw + empirical-Bayes shrunk lineup ratings.** Rung 0's
  prediction plus a per-exact-lineup residual, shrunk toward zero by a
  classic grouped-mean EB factor `B_L = tau2 / (tau2 + sigma2 / w_L)` where
  `w_L` is the lineup's training possessions. A lineup unseen in training
  gets rung 0's prediction (`B_L = 0`). This is "the value of direct lineup
  history": it can only help where a held-out lineup actually recurs
  (chronological, unseen-pair), and is identical to rung 0 on the
  unseen-lineup holdout by construction. Must beat rung 0.

Both are closed-form single-pass fits -- no iteration, no grid, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.stints import StintTable

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ContextMeanModel:
    """Rung 0: weighted least squares on the context columns only."""

    feature_space: FeatureSpace
    context_coef: FloatArray
    l2_context: float

    @classmethod
    def fit(
        cls,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        *,
        l2_context: float = 1e-3,
    ) -> ContextMeanModel:
        c = design.context
        w = design.weight
        cw = c * w[:, None]
        gram = c.T @ cw
        gram[np.diag_indices_from(gram)] += l2_context
        rhs = cw.T @ design.y
        coef = np.linalg.solve(gram, rhs)
        return cls(
            feature_space=feature_space,
            context_coef=np.asarray(coef, dtype=np.float64),
            l2_context=float(l2_context),
        )

    def predict(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(design.context @ self.context_coef, dtype=np.float64)


@dataclass(frozen=True)
class LineupMeanModel:
    """Rung 1: rung 0 + an empirical-Bayes-shrunk per-lineup residual."""

    rung0: ContextMeanModel
    shrunk_residual: dict[str, float]  # offense_lineup_id -> B_L * r_L
    tau2: float
    sigma2: float
    n_lineups: int

    @classmethod
    def fit(
        cls,
        table: StintTable,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        *,
        l2_context: float = 1e-3,
    ) -> LineupMeanModel:
        rung0 = ContextMeanModel.fit(design, feature_space, l2_context=l2_context)
        resid = design.y - rung0.predict(design)
        w = design.weight

        rows_by_lineup: dict[str, list[int]] = {}
        for i, stint in enumerate(table):
            rows_by_lineup.setdefault(stint.offense_lineup_id, []).append(i)

        lids = list(rows_by_lineup)
        r_l = np.empty(len(lids))
        w_l = np.empty(len(lids))
        within_sq = 0.0
        within_w = 0.0
        for k, lid in enumerate(lids):
            idx = np.asarray(rows_by_lineup[lid], dtype=np.int64)
            wr = w[idx]
            mean_r = float(np.average(resid[idx], weights=wr))
            r_l[k] = mean_r
            w_l[k] = float(wr.sum())
            within_sq += float((wr * (resid[idx] - mean_r) ** 2).sum())
            within_w += float(wr.sum())

        # sigma2: possession-scaled within-lineup noise. tau2: between-lineup
        # variance by method of moments (the observed spread of r_L minus the
        # sampling noise sigma2 / w_L it already contains), floored at 0.
        sigma2 = within_sq / within_w if within_w else float(np.var(resid))
        total_w = float(w_l.sum())
        grand_r = float(np.average(r_l, weights=w_l)) if total_w else 0.0
        observed_var = (
            float(np.average((r_l - grand_r) ** 2, weights=w_l)) if total_w else 0.0
        )
        mean_sampling = float(np.average(sigma2 / np.maximum(w_l, 1.0), weights=w_l))
        tau2 = max(0.0, observed_var - mean_sampling)

        shrunk: dict[str, float] = {}
        for k, lid in enumerate(lids):
            b = tau2 / (tau2 + sigma2 / max(w_l[k], 1.0)) if tau2 > 0 else 0.0
            shrunk[lid] = b * r_l[k]

        return cls(
            rung0=rung0,
            shrunk_residual=shrunk,
            tau2=float(tau2),
            sigma2=float(sigma2),
            n_lineups=len(lids),
        )

    def predict(self, table: StintTable, design: DesignMatrices) -> FloatArray:
        out = self.rung0.predict(design).copy()
        for i, stint in enumerate(table):
            adj = self.shrunk_residual.get(stint.offense_lineup_id)
            if adj is not None:
                out[i] += adj
        return out

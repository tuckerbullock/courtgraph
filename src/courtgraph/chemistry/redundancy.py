"""Candidate idea #3 -- skill redundancy / anti-synergy.

Instead of a free role-cluster-pair matrix (15 parameters), the interaction is
a handful of coefficients on engineered **concentration** features. For each
offensive role dimension ``d`` (usage, three-rate, rim-rate, ...), a lineup's
concentration on ``d`` is

    conc_d = (sum_i z_id)^2 - sum_i z_id^2  =  2 * sum_{i<j} z_id * z_jd

over its offensive players' standardized role vectors ``z``. It is positive
when the players align on ``d`` (all above or all below average -- "redundant")
and negative when they oppose.

    y_s ~ N( C_s.theta_c + sum_i alpha_i - sum_j beta_j + sum_d rho_d * conc_d,
             sigma^2 / w_s )      rho_d ~ N(0, tau_rho^2)

``rho_d < 0`` means concentrating skill ``d`` hurts (e.g. redundant
ball-handlers clash); ``rho_d > 0`` means it helps (e.g. shooting -> spacing).
D = 6 parameters, each backed by all 266k stints. Same weighted-Gaussian EM as
the other interaction models (dense extra block in
:mod:`courtgraph.chemistry._augmented_em`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry._augmented_em import fit_augmented_em
from courtgraph.chemistry.baseline import AdditiveDecomposition
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.features.role_clusters import RoleClustering

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

REDUNDANCY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RedundancyConfig:
    tau_c2: float = 1.0e6
    max_iters: int = 100
    tol: float = 1e-7


def raw_concentration(
    offense_index: IntArray, feature_space: FeatureSpace, clustering: RoleClustering
) -> FloatArray:
    """``(n, D)`` raw concentration features (before standardization)."""

    id_of = {pos: pid for pid, pos in feature_space.player_index().items()}
    d = len(clustering.features)
    p = feature_space.n_players
    # (p + 1, D) role vectors; row p is the "no role" fill of zeros
    table = np.zeros((p + 1, d), dtype=np.float64)
    have = np.zeros(p + 1, dtype=bool)
    for pos in range(p):
        vec = clustering.vector_of(id_of.get(pos, -1))
        if vec is not None:
            table[pos] = vec
            have[pos] = True

    idx = np.where(offense_index >= 0, offense_index, p)  # (n, 5)
    vecs = table[idx]  # (n, 5, D)
    present = have[idx][:, :, None]  # (n, 5, 1)
    vecs = np.where(present, vecs, 0.0)
    s1 = vecs.sum(axis=1)  # (n, D)
    s2 = (vecs**2).sum(axis=1)  # (n, D)
    return np.asarray(s1**2 - s2, dtype=np.float64)


@dataclass(frozen=True)
class RedundancyInteraction:
    feature_space: FeatureSpace
    clustering: RoleClustering
    context_coef: FloatArray
    offense_coef: FloatArray
    defense_coef: FloatArray
    rho: FloatArray  # (D,) concentration coefficients
    conc_mean: FloatArray
    conc_std: FloatArray
    sigma2: float
    tau_off2: float
    tau_def2: float
    tau_rho2: float
    tau_c2: float
    n_iters: int
    converged: bool
    final_loglik: float
    n_obs: int
    gram: FloatArray
    rhs: FloatArray
    chol: FloatArray

    @classmethod
    def fit(
        cls,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        clustering: RoleClustering,
        *,
        config: RedundancyConfig | None = None,
    ) -> RedundancyInteraction:
        cfg = config or RedundancyConfig()
        raw = raw_concentration(design.offense_index, feature_space, clustering)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std == 0.0] = 1.0
        conc = (raw - mean) / std

        core = fit_augmented_em(
            design,
            feature_space,
            None,
            conc.shape[1],
            extra_dense=conc,
            tau_c2=cfg.tau_c2,
            max_iters=cfg.max_iters,
            tol=cfg.tol,
            label="redundancy",
        )
        return cls(
            feature_space=feature_space,
            clustering=clustering,
            context_coef=core.context_coef,
            offense_coef=core.offense_coef,
            defense_coef=core.defense_coef,
            rho=core.extra_coef,
            conc_mean=mean,
            conc_std=std,
            sigma2=core.sigma2,
            tau_off2=core.tau_off2,
            tau_def2=core.tau_def2,
            tau_rho2=core.tau_extra2,
            tau_c2=core.tau_c2,
            n_iters=core.n_iters,
            converged=core.converged,
            final_loglik=core.final_loglik,
            n_obs=core.n_obs,
            gram=core.gram,
            rhs=core.rhs,
            chol=core.chol,
        )

    # -- prediction --------------------------------------------------------- #

    @property
    def alpha(self) -> float:
        idx = self.feature_space.context_columns.index("intercept")
        return float(self.context_coef[idx])

    def _conc(self, design: DesignMatrices) -> FloatArray:
        raw = raw_concentration(
            design.offense_index, self.feature_space, self.clustering
        )
        return np.asarray((raw - self.conc_mean) / self.conc_std, dtype=np.float64)

    def _gather(self, coef: FloatArray, index: IntArray) -> FloatArray:
        pad = np.concatenate([coef, [0.0]])
        return np.asarray(pad[np.where(index >= 0, index, len(coef))].sum(axis=1))

    def predict(self, design: DesignMatrices) -> FloatArray:
        ctx = design.context @ self.context_coef
        off = self._gather(self.offense_coef, design.offense_index)
        deff = self._gather(self.defense_coef, design.defense_index)
        red = self._conc(design) @ self.rho
        return np.asarray(ctx + off - deff + red, dtype=np.float64)

    def residuals(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(design.y - self.predict(design), dtype=np.float64)

    def _dim(self) -> int:
        return (
            self.feature_space.n_context
            + 2 * self.feature_space.n_players
            + int(self.rho.shape[0])
        )

    def _design_rows(self, design: DesignMatrices, rows: IntArray) -> FloatArray:
        c = self.feature_space.n_context
        p = self.feature_space.n_players
        out = np.zeros((len(rows), self._dim()))
        out[:, :c] = design.context[rows]
        for k in range(5):
            off_k = design.offense_index[rows, k]
            seen = off_k >= 0
            out[np.flatnonzero(seen), c + off_k[seen]] += 1.0
            def_k = design.defense_index[rows, k]
            seen = def_k >= 0
            out[np.flatnonzero(seen), c + p + def_k[seen]] -= 1.0
        out[:, c + 2 * p :] = self._conc(design)[rows]
        return out

    def group_predictive(
        self, design: DesignMatrices, groups: dict[str, NDArray[np.int64]]
    ) -> dict[str, tuple[float, float, float]]:
        keys = list(groups)
        g_mat = np.zeros((len(keys), self._dim()))
        totals = np.zeros(len(keys))
        for gi, key in enumerate(keys):
            rows = np.asarray(groups[key], dtype=np.int64)
            w = design.weight[rows]
            block = self._design_rows(design, rows)
            total = float(w.sum())
            g_mat[gi] = (block * w[:, None]).sum(axis=0) / total
            totals[gi] = total

        theta = np.concatenate(
            [self.context_coef, self.offense_coef, self.defense_coef, self.rho]
        )
        point = g_mat @ theta
        solved = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, g_mat.T))
        var_param = np.einsum("ij,ji->i", g_mat, solved)
        sd = np.sqrt(np.maximum(var_param, 0.0) + self.sigma2 / totals)
        return {
            key: (float(point[gi]), float(sd[gi]), float(totals[gi]))
            for gi, key in enumerate(keys)
        }

    def decompose_row(self, design: DesignMatrices, row: int) -> AdditiveDecomposition:
        intercept_idx = self.feature_space.context_columns.index("intercept")
        context_contrib = float(
            design.context[row] @ self.context_coef
            - design.context[row, intercept_idx] * self.context_coef[intercept_idx]
        )
        one = slice(row, row + 1)
        off = float(self._gather(self.offense_coef, design.offense_index[one])[0])
        deff = float(self._gather(self.defense_coef, design.defense_index[one])[0])
        red = float((self._conc(design)[one] @ self.rho)[0])
        return AdditiveDecomposition(
            talent=self.alpha + off - deff + red, context=context_contrib
        )

    def rho_by_feature(self) -> dict[str, float]:
        return dict(
            zip(
                self.clustering.features,
                (float(x) for x in self.rho),
                strict=True,
            )
        )

    def variance_components(self) -> dict[str, Any]:
        return {
            "sigma": math.sqrt(self.sigma2),
            "tau_off": math.sqrt(self.tau_off2),
            "tau_def": math.sqrt(self.tau_def2),
            "tau_rho": math.sqrt(self.tau_rho2),
            "sigma2": self.sigma2,
            "tau_off2": self.tau_off2,
            "tau_def2": self.tau_def2,
            "tau_rho2": self.tau_rho2,
            "tau_c2": self.tau_c2,
            "n_iters": self.n_iters,
            "converged": self.converged,
            "final_loglik": self.final_loglik,
        }

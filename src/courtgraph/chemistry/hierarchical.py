"""Rung 3 -- empirical-Bayes hierarchical player-impact model.

Same design as the rung-2 additive baseline
(:class:`~courtgraph.chemistry.baseline.AdditiveRidge`):

    y_s | theta  ~  N(A_s . theta,  sigma^2 / w_s)
    A = [context | +offense_onehot | -defense_onehot]

but the single CV-picked ``l2_player`` is replaced by **variance components
learned from the data** by EM:

    alpha_i  ~  N(0, tau_off^2)      beta_j  ~  N(0, tau_def^2)
    theta_c  ~  N(0, tau_c^2)        tau_c^2 large and fixed

so the shrinkage ``sigma^2 / tau_off^2`` (and the defensive one) is the
data-optimal partial pooling with no grid and no cross-validation rail, and it
is exposure-aware automatically -- a low-possession player's weak likelihood
lets the prior dominate. The Gaussian posterior gives a per-effect standard
deviation that propagates to a per-lineup predictive interval.

Empirical Bayes with a Laplace/Gaussian posterior is the permitted production
path (master plan section 14.7); the intervals are labelled *approximate*.
Players are static -- a dynamic / per-season layer is out of research cycle 1.
Pure NumPy, deterministic (no RNG in the fit).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import (
    AdditiveDecomposition,
    _gather_sum,
    _normal_equations,
    _split_theta,
)
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace

FloatArray = NDArray[np.float64]

HIERARCHICAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HierarchicalConfig:
    """EM settings for :meth:`HierarchicalRidge.fit`. Deterministic."""

    tau_c2: float = 1.0e6  # fixed, weak context prior
    max_iters: int = 200
    tol: float = 1e-7  # on max |delta log variance-component|
    init_tau_off2: float = 1.0
    init_tau_def2: float = 1.0


@dataclass(frozen=True)
class HierarchicalRidge:
    """A fitted empirical-Bayes hierarchical additive model.

    ``context_coef`` / ``offense_coef`` / ``defense_coef`` are the posterior
    means (mirrors :class:`AdditiveRidge` so the same downstream code works).
    ``chol`` is the Cholesky factor of the posterior precision ``M`` -- kept for
    :meth:`group_predictive`; it is an in-memory field, not serialized.
    """

    feature_space: FeatureSpace
    context_coef: FloatArray
    offense_coef: FloatArray
    defense_coef: FloatArray
    sigma2: float
    tau_off2: float
    tau_def2: float
    tau_c2: float
    n_iters: int
    converged: bool
    final_loglik: float
    n_obs: int
    gram: FloatArray  # (d, d) unpenalised weighted Gram -- sufficient statistic
    rhs: FloatArray  # (d,) A^T (w y)
    chol: FloatArray  # (d, d) lower Cholesky of M = gram/sigma2 + diag(Lambda)

    # -- fitting -----------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        *,
        config: HierarchicalConfig | None = None,
    ) -> HierarchicalRidge:
        cfg = config or HierarchicalConfig()
        space = feature_space
        n_context = space.n_context
        n_players = space.n_players
        dim = n_context + 2 * n_players
        o0, d0 = n_context, n_context + n_players

        gram, rhs = _normal_equations(design, space)
        w, y = design.weight, design.y
        n = design.n_rows
        y_w_y = float((w * y**2).sum())
        sum_log_w = float(np.log(w).sum())

        # precision of the (fixed) context prior, and the two learned ones
        inv_tau_c2 = 1.0 / cfg.tau_c2
        tau_off2 = float(cfg.init_tau_off2)
        tau_def2 = float(cfg.init_tau_def2)
        sigma2 = float(np.var(y)) or 1.0

        prev = np.array([math.log(tau_off2), math.log(tau_def2), math.log(sigma2)])
        last_loglik = -np.inf
        n_iters = 0
        converged = False
        chol = np.empty((dim, dim))
        mu = np.zeros(dim)

        for n_iters in range(1, cfg.max_iters + 1):
            lam = np.empty(dim)
            lam[:n_context] = inv_tau_c2
            lam[o0:d0] = 1.0 / tau_off2
            lam[d0:] = 1.0 / tau_def2

            m = gram / sigma2
            m[np.diag_indices(dim)] += lam
            chol = np.asarray(np.linalg.cholesky(m), dtype=np.float64)
            mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))
            # diag(M^-1) without forming the inverse: M^-1 = L^-T L^-1
            chol_inv = np.linalg.solve(chol, np.eye(dim))
            v_diag = np.einsum("ij,ij->j", chol_inv, chol_inv)

            mu_c, mu_a, mu_b = _split_theta(mu, space)
            new_tau_off2 = float(mu_a @ mu_a + v_diag[o0:d0].sum()) / n_players
            new_tau_def2 = float(mu_b @ mu_b + v_diag[d0:].sum()) / n_players

            y_hat = (
                design.context @ mu_c
                + _gather_sum(mu_a, design.offense_index)
                - _gather_sum(mu_b, design.defense_index)
            )
            rss = float((w * (y - y_hat) ** 2).sum())
            # tr(gram . V) with V = M^-1; gram = sigma2 (M - diag(lam)) so
            # tr(gram V) = sigma2 (dim - sum_k lam_k V_kk). Diagonal-only.
            trace_term = sigma2 * (dim - float((lam * v_diag).sum()))
            new_sigma2 = (rss + trace_term) / n

            # marginal log-likelihood (must not decrease)
            logdet_m = 2.0 * float(np.log(np.diag(chol)).sum())
            logdet_sigma = (
                n * math.log(sigma2) - sum_log_w - float(np.log(lam).sum()) + logdet_m
            )
            quad = y_w_y / sigma2 - float(rhs @ mu) / sigma2
            loglik = -0.5 * (n * math.log(2.0 * math.pi) + logdet_sigma + quad)
            if loglik + 1e-6 < last_loglik:
                warnings.warn(
                    f"hierarchical EM log-likelihood decreased at iter {n_iters}: "
                    f"{last_loglik:.6f} -> {loglik:.6f}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            last_loglik = loglik

            tau_off2, tau_def2, sigma2 = new_tau_off2, new_tau_def2, new_sigma2
            cur = np.array([math.log(tau_off2), math.log(tau_def2), math.log(sigma2)])
            if float(np.max(np.abs(cur - prev))) < cfg.tol:
                converged = True
                prev = cur
                break
            prev = cur

        # one final E-step at the converged variance components
        lam = np.empty(dim)
        lam[:n_context] = inv_tau_c2
        lam[o0:d0] = 1.0 / tau_off2
        lam[d0:] = 1.0 / tau_def2
        m = gram / sigma2
        m[np.diag_indices(dim)] += lam
        chol = np.asarray(np.linalg.cholesky(m), dtype=np.float64)
        mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))
        mu_c, mu_a, mu_b = _split_theta(mu, space)

        return cls(
            feature_space=space,
            context_coef=np.asarray(mu_c, dtype=np.float64),
            offense_coef=np.asarray(mu_a, dtype=np.float64),
            defense_coef=np.asarray(mu_b, dtype=np.float64),
            sigma2=float(sigma2),
            tau_off2=float(tau_off2),
            tau_def2=float(tau_def2),
            tau_c2=float(cfg.tau_c2),
            n_iters=int(n_iters),
            converged=bool(converged),
            final_loglik=float(last_loglik),
            n_obs=int(n),
            gram=gram,
            rhs=rhs,
            chol=chol,
        )

    # -- prediction ------------------------------------------------------------

    @property
    def alpha(self) -> float:
        idx = self.feature_space.context_columns.index("intercept")
        return float(self.context_coef[idx])

    def predict(self, design: DesignMatrices) -> FloatArray:
        ctx = design.context @ self.context_coef
        off = _gather_sum(self.offense_coef, design.offense_index)
        deff = _gather_sum(self.defense_coef, design.defense_index)
        return np.asarray(ctx + off - deff, dtype=np.float64)

    def residuals(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(design.y - self.predict(design), dtype=np.float64)

    def talent_of(self, player_id: int) -> tuple[float, float]:
        pos = self.feature_space.player_index().get(player_id)
        if pos is None:
            return 0.0, 0.0
        return float(self.offense_coef[pos]), float(self.defense_coef[pos])

    def decompose_row(self, design: DesignMatrices, row: int) -> AdditiveDecomposition:
        intercept_idx = self.feature_space.context_columns.index("intercept")
        context_contrib = float(
            design.context[row] @ self.context_coef
            - design.context[row, intercept_idx] * self.context_coef[intercept_idx]
        )
        one = slice(row, row + 1)
        off = float(_gather_sum(self.offense_coef, design.offense_index[one])[0])
        deff = float(_gather_sum(self.defense_coef, design.defense_index[one])[0])
        return AdditiveDecomposition(
            talent=self.alpha + off - deff, context=context_contrib
        )

    def variance_components(self) -> dict[str, Any]:
        return {
            "sigma": math.sqrt(self.sigma2),
            "tau_off": math.sqrt(self.tau_off2),
            "tau_def": math.sqrt(self.tau_def2),
            "sigma2": self.sigma2,
            "tau_off2": self.tau_off2,
            "tau_def2": self.tau_def2,
            "tau_c2": self.tau_c2,
            "shrinkage_off": self.sigma2 / self.tau_off2,
            "shrinkage_def": self.sigma2 / self.tau_def2,
            "n_iters": self.n_iters,
            "converged": self.converged,
            "final_loglik": self.final_loglik,
        }

    # -- uncertainty ---------------------------------------------------------

    def _design_rows(
        self, design: DesignMatrices, rows: NDArray[np.int64]
    ) -> FloatArray:
        """Dense (len(rows), d) design block: context, +1 per offense player,
        -1 per defense player. Unseen (-1) players contribute nothing."""

        n_context = self.feature_space.n_context
        n_players = self.feature_space.n_players
        out = np.zeros((len(rows), n_context + 2 * n_players), dtype=np.float64)
        out[:, :n_context] = design.context[rows]
        for k in range(5):
            off_k = design.offense_index[rows, k]
            seen = off_k >= 0
            out[np.flatnonzero(seen), n_context + off_k[seen]] += 1.0
            def_k = design.defense_index[rows, k]
            seen = def_k >= 0
            out[np.flatnonzero(seen), n_context + n_players + def_k[seen]] -= 1.0
        return out

    def group_predictive(
        self, design: DesignMatrices, groups: dict[str, NDArray[np.int64]]
    ) -> dict[str, tuple[float, float, float]]:
        """For each group (row indices into ``design``): the possession-weighted
        predicted lineup value, its predictive standard deviation (parametric
        posterior + outcome noise ``sigma^2 / sum_w``), and total possessions."""

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

        point = g_mat @ np.concatenate(
            [self.context_coef, self.offense_coef, self.defense_coef]
        )
        # Var_param = g' M^-1 g  for every group at once
        solved = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, g_mat.T))
        var_param = np.einsum("ij,ji->i", g_mat, solved)
        sd = np.sqrt(np.maximum(var_param, 0.0) + self.sigma2 / totals)
        return {
            key: (float(point[gi]), float(sd[gi]), float(totals[gi]))
            for gi, key in enumerate(keys)
        }

    def _dim(self) -> int:
        return self.feature_space.n_context + 2 * self.feature_space.n_players

    # -- serialization (small models only; rung 3 is eval-only for cycle 1) --

    def to_dict(self) -> dict[str, Any]:
        return {
            "hierarchical_schema_version": HIERARCHICAL_SCHEMA_VERSION,
            "feature_space": self.feature_space.to_dict(),
            "context_coef": self.context_coef.tolist(),
            "offense_coef": self.offense_coef.tolist(),
            "defense_coef": self.defense_coef.tolist(),
            "sigma2": self.sigma2,
            "tau_off2": self.tau_off2,
            "tau_def2": self.tau_def2,
            "tau_c2": self.tau_c2,
            "n_iters": self.n_iters,
            "converged": self.converged,
            "final_loglik": self.final_loglik,
            "n_obs": self.n_obs,
            "gram": self.gram.tolist(),
            "rhs": self.rhs.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HierarchicalRidge:
        if data.get("hierarchical_schema_version") != HIERARCHICAL_SCHEMA_VERSION:
            raise ValueError("unrecognized hierarchical artifact schema version")
        space = FeatureSpace.from_dict(data["feature_space"])
        dim = space.n_context + 2 * space.n_players
        o0, d0 = space.n_context, space.n_context + space.n_players
        gram = np.asarray(data["gram"], dtype=np.float64)
        sigma2 = float(data["sigma2"])
        lam = np.empty(dim)
        lam[:o0] = 1.0 / float(data["tau_c2"])
        lam[o0:d0] = 1.0 / float(data["tau_off2"])
        lam[d0:] = 1.0 / float(data["tau_def2"])
        m = gram / sigma2
        m[np.diag_indices(dim)] += lam
        return cls(
            feature_space=space,
            context_coef=np.asarray(data["context_coef"], dtype=np.float64),
            offense_coef=np.asarray(data["offense_coef"], dtype=np.float64),
            defense_coef=np.asarray(data["defense_coef"], dtype=np.float64),
            sigma2=sigma2,
            tau_off2=float(data["tau_off2"]),
            tau_def2=float(data["tau_def2"]),
            tau_c2=float(data["tau_c2"]),
            n_iters=int(data["n_iters"]),
            converged=bool(data["converged"]),
            final_loglik=float(data["final_loglik"]),
            n_obs=int(data["n_obs"]),
            gram=gram,
            rhs=np.asarray(data["rhs"], dtype=np.float64),
            chol=np.asarray(np.linalg.cholesky(m), dtype=np.float64),
        )

"""Master plan §45 Phase A -- pooled player-lift on lineup value.

Rungs 3-5 tested a **symmetric** interaction attached to a lineup's value and it
is not supported. "Does a good player make teammates better?" is a different
estimand: **asymmetric** and **pooled** -- one scalar per player, not one per
pair. Phase A adds that scalar to the rung-3 frame at the cheapest possible
fidelity (lineup value, not per-player production -- that is Phase B):

    mu_s = rung3(s) + sum_{i in off(s)} lambda_i * (A_off,s - alpha_i)

where ``A_off,s = sum_{i in off} alpha_i`` is total offensive talent on the
floor and ``alpha_i`` is player i's rung-3 offensive coefficient, so the lift
term rewards lineups where a high-``lambda`` player shares the court with
strong teammates. ``lambda_i ~ N(0, tau_lambda^2)``.

Fit is **two-stage** (§45.2): freeze ``alpha`` from rung 3, regress the rung-3
residual on the ``lambda_i * (A_off - alpha_i)`` design by ridge, with
``tau_lambda`` chosen by marginal likelihood. This is a rank-1
provision/need term with the receiver side pinned to observed talent -- rung 5's
general low-rank form already failed, so this has a negative prior and must
clear the same evidence bar (beat rung 3 out of sample **and** beat a
player-permutation placebo, calibrated and seed-stable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import _gather_sum
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PLAYER_LIFT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlayerLiftConfig:
    # tau_lambda^2 grid (points/100 per unit of centred teammate talent), searched
    # by marginal likelihood on the training residual. The floor (1e-5) doubles
    # as "no evidence for a nonzero lift variance" -- a valid Phase-A outcome.
    tau2_grid: tuple[float, ...] = (
        1e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
    )
    hierarchical: HierarchicalConfig | None = None


def _lift_design_columns(
    offense_index: IntArray, alpha: FloatArray, *, perm: IntArray | None = None
) -> tuple[IntArray, IntArray, FloatArray]:
    """Return ``(rows, cols, vals)`` COO triplets of the sparse lift design
    ``D`` (n x n_players): for stint ``s`` with offense player ``p`` in a slot,
    ``D[s, p] = A_off,s - alpha_p``. ``perm`` (a bijection over player rows)
    relabels the column identity -- the placebo control.

    ``A_off,s`` is a sum, so it is permutation-invariant; only the
    (player <-> teammate-talent) correspondence is broken by ``perm``."""

    _, k = offense_index.shape
    a_off = _gather_sum(alpha, offense_index)  # (n,)
    rows_list: list[IntArray] = []
    cols_list: list[IntArray] = []
    vals_list: list[FloatArray] = []
    for slot in range(k):
        p = offense_index[:, slot]
        seen = p >= 0
        idx = np.flatnonzero(seen)
        pv = p[seen]
        col = pv if perm is None else perm[pv]
        rows_list.append(idx.astype(np.int64))
        cols_list.append(col.astype(np.int64))
        vals_list.append(a_off[seen] - alpha[pv])
    return (
        np.concatenate(rows_list),
        np.concatenate(cols_list),
        np.concatenate(vals_list),
    )


def _lift_normal_equations(
    offense_index: IntArray,
    alpha: FloatArray,
    weight: FloatArray,
    resid: FloatArray,
    n_players: int,
    *,
    perm: IntArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """``D' W D`` (p x p) and ``D' W r`` (p,) for the 5-sparse lift design,
    accumulated over the 25 offense slot-pairs (fully vectorised)."""

    k = offense_index.shape[1]
    a_off = _gather_sum(alpha, offense_index)
    gram = np.zeros((n_players * n_players,), dtype=np.float64)
    rhs = np.zeros(n_players, dtype=np.float64)
    cols = []
    vals = []
    for slot in range(k):
        p = offense_index[:, slot]
        col = np.where(p >= 0, p if perm is None else perm[np.where(p >= 0, p, 0)], -1)
        val = np.where(p >= 0, a_off - alpha[np.where(p >= 0, p, 0)], 0.0)
        cols.append(col)
        vals.append(val)
    for a in range(k):
        ca, va = cols[a], vals[a]
        wv_a = weight * va
        oka = ca >= 0
        rhs += np.bincount(ca[oka], weights=(wv_a * resid)[oka], minlength=n_players)
        for b in range(k):
            cb, vb = cols[b], vals[b]
            ok = oka & (cb >= 0)
            lin = ca[ok] * n_players + cb[ok]
            gram += np.bincount(
                lin, weights=(wv_a * vb)[ok], minlength=n_players * n_players
            )
    return gram.reshape(n_players, n_players), rhs


@dataclass(frozen=True)
class PlayerLift:
    """A fitted Phase-A player-lift model (rung 3 + pooled lift scalars)."""

    feature_space: FeatureSpace
    rung3: HierarchicalRidge
    lambda_: FloatArray  # (n_players,) posterior mean lift scalar
    lambda_cov_diag: FloatArray  # (n_players,) posterior variance
    tau_lambda2: float
    sigma2: float
    loglik_by_tau2: dict[float, float]
    permuted: bool

    @classmethod
    def fit(
        cls,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        *,
        config: PlayerLiftConfig | None = None,
        seed: int = 0,
        permuted: bool = False,
    ) -> PlayerLift:
        cfg = config or PlayerLiftConfig()
        rung3 = HierarchicalRidge.fit(design, feature_space, config=cfg.hierarchical)
        alpha = rung3.offense_coef
        p = feature_space.n_players
        w = np.asarray(design.weight, dtype=np.float64)
        resid = np.asarray(design.y - rung3.predict(design), dtype=np.float64)

        perm = np.random.default_rng(seed + 1).permutation(p) if permuted else None
        gram, rhs = _lift_normal_equations(
            design.offense_index, alpha, w, resid, p, perm=perm
        )
        gram = 0.5 * (gram + gram.T)  # symmetrise (slot-pair loop is asymmetric)

        sigma2 = float(rung3.sigma2)
        # marginal-likelihood pick of tau_lambda^2
        loglik: dict[float, float] = {}
        best_tau2 = cfg.tau2_grid[0]
        best_ll = -np.inf
        for tau2 in cfg.tau2_grid:
            m = gram / sigma2 + np.eye(p) / tau2
            try:
                chol = np.linalg.cholesky(m)
            except np.linalg.LinAlgError:
                continue
            mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))
            # log p(r) for r ~ N(0, sigma2 W^-1 + tau2 D D'), via Woodbury and
            # dropping the terms constant across tau2 (n log 2pi, log|sigma2 W^-1|,
            # rss/sigma2):  log p = -0.5[ p log tau2 + log|M| - mu' M mu ]
            # with M = D'WD/sigma2 + I/tau2 and mu = M^-1 (D'Wr / sigma2).
            logdet_m = 2.0 * float(np.log(np.diag(chol)).sum())
            quad = float(mu @ (m @ mu))
            ll = -0.5 * (p * math.log(tau2) + logdet_m - quad)
            loglik[tau2] = ll
            if ll > best_ll:
                best_ll, best_tau2 = ll, tau2

        m = gram / sigma2 + np.eye(p) / best_tau2
        chol = np.linalg.cholesky(m)
        lam = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))
        inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(p)))
        return cls(
            feature_space=feature_space,
            rung3=rung3,
            lambda_=np.asarray(lam, dtype=np.float64),
            lambda_cov_diag=np.asarray(np.diag(inv), dtype=np.float64),
            tau_lambda2=float(best_tau2),
            sigma2=sigma2,
            loglik_by_tau2=loglik,
            permuted=permuted,
        )

    # -- prediction --------------------------------------------------------- #

    def _lift_term(self, design: DesignMatrices) -> FloatArray:
        rows, cols, vals = _lift_design_columns(
            design.offense_index, self.rung3.offense_coef
        )
        out = np.zeros(design.offense_index.shape[0], dtype=np.float64)
        np.add.at(out, rows, self.lambda_[cols] * vals)
        return out

    def predict(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(
            self.rung3.predict(design) + self._lift_term(design), dtype=np.float64
        )

    def residuals(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(design.y - self.predict(design), dtype=np.float64)

    def lift_of(self, player_id: int) -> tuple[float, float]:
        pos = self.feature_space.player_index().get(player_id)
        if pos is None:
            return 0.0, 0.0
        return float(self.lambda_[pos]), float(
            math.sqrt(max(self.lambda_cov_diag[pos], 0.0))
        )

    def group_predictive(
        self, design: DesignMatrices, groups: dict[str, NDArray[np.int64]]
    ) -> dict[str, tuple[float, float, float]]:
        """rung-3 group predictive plus the (frozen-alpha, independent-approx)
        lift term: point += weighted-mean lift contribution, variance +=
        g_lift' Cov(lambda) g_lift."""

        base = self.rung3.group_predictive(design, groups)
        lift_all = self._lift_term(design)
        # per-group lift design row g (p,) for the variance term
        out: dict[str, tuple[float, float, float]] = {}
        alpha = self.rung3.offense_coef
        p = self.feature_space.n_players
        for key, rows in groups.items():
            rows = np.asarray(rows, dtype=np.int64)
            w = design.weight[rows]
            total = float(w.sum())
            r, c, v = _lift_design_columns(design.offense_index[rows], alpha)
            g = np.zeros(p, dtype=np.float64)
            np.add.at(g, c, w[r] * v)
            g /= total
            point0, sd0, tot = base[key]
            point = point0 + float((lift_all[rows] * w).sum() / total)
            var_lift = float(g @ (self.lambda_cov_diag * g))
            sd = math.sqrt(sd0 * sd0 + var_lift)
            out[key] = (point, sd, tot)
        return out

    def variance_components(self) -> dict[str, Any]:
        vc = dict(self.rung3.variance_components())
        vc.update(
            {
                "tau_lambda": math.sqrt(self.tau_lambda2),
                "tau_lambda2": self.tau_lambda2,
                "lambda_abs_mean": float(np.abs(self.lambda_).mean()),
                "lambda_abs_max": float(np.abs(self.lambda_).max()),
                "permuted": self.permuted,
            }
        )
        return vc

    def top_lifts(self, n: int = 15) -> list[tuple[int, float, float]]:
        id_of = {pos: pid for pid, pos in self.feature_space.player_index().items()}
        order = np.argsort(-np.abs(self.lambda_))[:n]
        return [
            (
                int(id_of[int(pos)]),
                float(self.lambda_[pos]),
                float(math.sqrt(max(self.lambda_cov_diag[pos], 0.0))),
            )
            for pos in order
        ]

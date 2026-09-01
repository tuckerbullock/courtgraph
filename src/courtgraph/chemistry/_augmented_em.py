"""Shared core for the augmented empirical-Bayes RAPM models.

Both rung 4 (:mod:`courtgraph.chemistry.pair_interaction`) and the
role-conditioned interaction model (:mod:`courtgraph.chemistry.role_interaction`)
solve the same system: the rung-3 hierarchical frame

    y_s ~ N( C_s.theta_c + sum_i alpha_i - sum_j beta_j + sum_k gamma_{e(s,k)},
             sigma^2 / w_s )

plus one extra block of coefficients ``gamma`` indexed per stint by an
``(n, K)`` integer matrix ``extra_index`` (values in ``[-1, n_extra)``; ``-1``
contributes nothing). rung 4 fills that matrix with admitted-pair rows; the
role model fills it with role-cluster-pair bins. Everything else -- the sparse
normal-equations assembly and the 3-variance-component EM -- is identical, so
it lives here once.

Pure NumPy, deterministic.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import (
    _cross_ctx,
    _cross_gram,
    _cross_rhs,
    _gather_sum,
)
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AugmentedEMResult:
    """Posterior means and learned variance components for one fit."""

    context_coef: FloatArray
    offense_coef: FloatArray
    defense_coef: FloatArray
    extra_coef: FloatArray
    sigma2: float
    tau_off2: float
    tau_def2: float
    tau_extra2: float
    tau_c2: float
    n_iters: int
    converged: bool
    final_loglik: float
    n_obs: int
    gram: FloatArray
    rhs: FloatArray
    chol: FloatArray


def fit_augmented_em(
    design: DesignMatrices,
    feature_space: FeatureSpace,
    extra_index: IntArray,
    n_extra: int,
    *,
    tau_c2: float,
    max_iters: int,
    tol: float,
    label: str = "augmented",
) -> AugmentedEMResult:
    """Fit ``[context | +offense | -defense | +extra]`` by empirical-Bayes EM.

    ``extra_index`` is ``(n, K)`` with entries in ``[-1, n_extra)``. ``label``
    only names the RuntimeWarning raised on a non-monotone log-likelihood.
    """

    space = feature_space
    c = space.n_context
    p = space.n_players
    q = n_extra
    dim = c + 2 * p + q
    o0, d0, g0 = c, c + p, c + 2 * p

    w, y = design.weight, design.y
    n = design.n_rows
    context_weighted = design.context * w[:, None]

    gram = np.zeros((dim, dim), dtype=np.float64)
    gram[:c, :c] = context_weighted.T @ design.context
    g_oc = _cross_ctx(design.offense_index, context_weighted, p)
    g_dc = _cross_ctx(design.defense_index, context_weighted, p)
    g_gc = _cross_ctx(extra_index, context_weighted, q)
    gram[:c, o0:d0] = g_oc.T
    gram[o0:d0, :c] = g_oc
    gram[:c, d0:g0] = -g_dc.T
    gram[d0:g0, :c] = -g_dc
    gram[:c, g0:] = g_gc.T
    gram[g0:, :c] = g_gc
    gram[o0:d0, o0:d0] = _cross_gram(
        design.offense_index, design.offense_index, w, p, p
    )
    gram[d0:g0, d0:g0] = _cross_gram(
        design.defense_index, design.defense_index, w, p, p
    )
    gram[g0:, g0:] = _cross_gram(extra_index, extra_index, w, q, q)
    g_od = _cross_gram(design.offense_index, design.defense_index, w, p, p)
    gram[o0:d0, d0:g0] = -g_od
    gram[d0:g0, o0:d0] = -g_od.T
    g_og = _cross_gram(design.offense_index, extra_index, w, p, q)
    gram[o0:d0, g0:] = g_og
    gram[g0:, o0:d0] = g_og.T
    g_dg = _cross_gram(design.defense_index, extra_index, w, p, q)
    gram[d0:g0, g0:] = -g_dg
    gram[g0:, d0:g0] = -g_dg.T

    rhs = np.zeros(dim, dtype=np.float64)
    rhs[:c] = context_weighted.T @ y
    rhs[o0:d0] = _cross_rhs(design.offense_index, w * y, p)
    rhs[d0:g0] = -_cross_rhs(design.defense_index, w * y, p)
    rhs[g0:] = _cross_rhs(extra_index, w * y, q)

    y_w_y = float((w * y**2).sum())
    sum_log_w = float(np.log(w).sum())
    inv_tau_c2 = 1.0 / tau_c2

    tau_off2 = tau_def2 = tau_extra2 = 1.0
    sigma2 = float(np.var(y)) or 1.0
    prev = np.array(
        [
            math.log(tau_off2),
            math.log(tau_def2),
            math.log(tau_extra2),
            math.log(sigma2),
        ]
    )
    last_loglik = -np.inf
    n_iters = 0
    converged = False
    chol = np.empty((dim, dim))
    mu = np.zeros(dim)

    def _lam(t_off: float, t_def: float, t_extra: float) -> FloatArray:
        lam = np.empty(dim)
        lam[:c] = inv_tau_c2
        lam[o0:d0] = 1.0 / t_off
        lam[d0:g0] = 1.0 / t_def
        lam[g0:] = 1.0 / t_extra
        return lam

    for n_iters in range(1, max_iters + 1):
        lam = _lam(tau_off2, tau_def2, tau_extra2)
        m = gram / sigma2
        m[np.diag_indices(dim)] += lam
        chol = np.asarray(np.linalg.cholesky(m), dtype=np.float64)
        mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))
        chol_inv = np.linalg.solve(chol, np.eye(dim))
        v_diag = np.einsum("ij,ij->j", chol_inv, chol_inv)

        mu_a, mu_b, mu_g = mu[o0:d0], mu[d0:g0], mu[g0:]
        new_off = float(mu_a @ mu_a + v_diag[o0:d0].sum()) / p
        new_def = float(mu_b @ mu_b + v_diag[d0:g0].sum()) / p
        new_extra = float(mu_g @ mu_g + v_diag[g0:].sum()) / q if q else tau_extra2

        y_hat = (
            design.context @ mu[:c]
            + _gather_sum(mu_a, design.offense_index)
            - _gather_sum(mu_b, design.defense_index)
            + _gather_sum(mu_g, extra_index)
        )
        rss = float((w * (y - y_hat) ** 2).sum())
        trace_term = sigma2 * (dim - float((lam * v_diag).sum()))
        new_sigma2 = (rss + trace_term) / n

        logdet_m = 2.0 * float(np.log(np.diag(chol)).sum())
        logdet_sigma = (
            n * math.log(sigma2) - sum_log_w - float(np.log(lam).sum()) + logdet_m
        )
        quad = y_w_y / sigma2 - float(rhs @ mu) / sigma2
        loglik = -0.5 * (n * math.log(2.0 * math.pi) + logdet_sigma + quad)
        if loglik + 1e-6 < last_loglik:
            warnings.warn(
                f"{label} EM log-likelihood decreased at iter {n_iters}: "
                f"{last_loglik:.6f} -> {loglik:.6f}",
                RuntimeWarning,
                stacklevel=2,
            )
        last_loglik = loglik

        tau_off2, tau_def2, tau_extra2, sigma2 = (
            new_off,
            new_def,
            new_extra,
            new_sigma2,
        )
        cur = np.array(
            [
                math.log(tau_off2),
                math.log(tau_def2),
                math.log(tau_extra2),
                math.log(sigma2),
            ]
        )
        if float(np.max(np.abs(cur - prev))) < tol:
            converged = True
            prev = cur
            break
        prev = cur

    lam = _lam(tau_off2, tau_def2, tau_extra2)
    m = gram / sigma2
    m[np.diag_indices(dim)] += lam
    chol = np.asarray(np.linalg.cholesky(m), dtype=np.float64)
    mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))

    return AugmentedEMResult(
        context_coef=np.asarray(mu[:c], dtype=np.float64),
        offense_coef=np.asarray(mu[o0:d0], dtype=np.float64),
        defense_coef=np.asarray(mu[d0:g0], dtype=np.float64),
        extra_coef=np.asarray(mu[g0:], dtype=np.float64),
        sigma2=float(sigma2),
        tau_off2=float(tau_off2),
        tau_def2=float(tau_def2),
        tau_extra2=float(tau_extra2),
        tau_c2=float(tau_c2),
        n_iters=int(n_iters),
        converged=bool(converged),
        final_loglik=float(last_loglik),
        n_obs=int(n),
        gram=gram,
        rhs=rhs,
        chol=chol,
    )

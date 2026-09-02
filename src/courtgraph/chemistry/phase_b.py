"""Master plan §45 Phase B -- a direct per-player-production lift model.

Rungs 3-5 and Phase A tested interaction on the **lineup's** value and found
nothing. Phase B changes the outcome: each offensive player-stint's own
per-possession production, regressed on the pooled lift of the *other four*
players on the floor.

    prod_rate_{k,s} = mu + context_s . theta + base_k
                      + sum_{i in off(s), i != k} lift_i + noise

* ``base_k ~ N(0, tau_base^2)``  -- player k's own production level (EM-shrunk);
* ``lift_i ~ N(0, tau_lift^2)``  -- "the average per-100 bump a teammate's
  offense gets when player i is also on the floor", holding the receiver's own
  level and the context fixed.

Two-stage fit (freeze ``base`` + context, ridge the residual on the teammate
multi-hot design, ``tau_lift`` by marginal likelihood -- same structure as
:mod:`courtgraph.chemistry.player_lift`). ``credited(config)`` sets the outcome
(points, or points + ``assist_credit`` * assisted teammate points); Phase B
reports both. Placebo permutes the ``lift_i -> player`` map.

Supported (contract §45.4) only if ``lift_i`` improves held-out
teammate-production prediction over the base-only model **and** beats the
giver-shuffle placebo, calibrated and seed-stable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.stints import StintTable
from courtgraph.features.player_production import (
    PlayerStintProduction,
    ProductionConfig,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PHASE_B_SCHEMA_VERSION = 1

# stint context columns that plausibly shift everyone's production
_CONTEXT = ("home", "playoff", "garbage")


@dataclass(frozen=True)
class PhaseBConfig:
    assist_credit: float = 0.5
    base_l2_grid: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    lift_tau2_grid: tuple[float, ...] = (
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
        10.0,
    )
    min_receiver_possessions: int = 200


@dataclass(frozen=True)
class PhaseBDesign:
    y: FloatArray  # (n,) credited production per 100 possessions
    w: FloatArray  # (n,) possessions
    receiver: IntArray  # (n,) receiver player row
    teammates: IntArray  # (n, 4) the other four offensive player rows (-1 pad)
    context: FloatArray  # (n, C)
    player_ids: tuple[int, ...]
    context_names: tuple[str, ...]

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


def build_phase_b_design(
    table: StintTable,
    production: list[PlayerStintProduction],
    *,
    config: PhaseBConfig | None = None,
) -> PhaseBDesign:
    cfg = config or PhaseBConfig()
    prod_cfg = ProductionConfig(assist_credit=cfg.assist_credit)
    stint_by_id = {s.stint_id: s for s in table}

    # receiver exposure filter
    exposure: dict[int, int] = defaultdict(int)
    for r in production:
        exposure[r.player_id] += r.offensive_possessions
    keep_players = {p for p, e in exposure.items() if e >= cfg.min_receiver_possessions}

    players = sorted(keep_players)
    pidx = {p: i for i, p in enumerate(players)}

    y_list: list[float] = []
    w_list: list[float] = []
    rec_list: list[int] = []
    team_list: list[list[int]] = []
    ctx_list: list[list[float]] = []
    for r in production:
        if r.player_id not in pidx or r.offensive_possessions <= 0:
            continue
        stint = stint_by_id.get(r.stint_id)
        if stint is None:
            continue
        others = [
            pidx[p] for p in stint.offense_player_ids if p != r.player_id and p in pidx
        ]
        if not others:
            continue
        y_list.append(100.0 * r.credited(prod_cfg) / r.offensive_possessions)
        w_list.append(float(r.offensive_possessions))
        rec_list.append(pidx[r.player_id])
        team_list.append((others + [-1, -1, -1, -1])[:4])
        ctx_list.append(
            [
                1.0 if stint.home_offense else 0.0,
                1.0 if stint.playoff else 0.0,
                float(stint.garbage_time_weight),
            ]
        )

    ctx = np.asarray(ctx_list, dtype=np.float64).reshape(-1, len(_CONTEXT))
    # drop context columns that are constant (they alias the intercept and make
    # the stage-1 normal equations singular -- e.g. playoff is always 0 on RS)
    keep_ctx = [
        j
        for j in range(ctx.shape[1])
        if ctx.shape[0] and float(np.ptp(ctx[:, j])) > 0.0
    ]
    return PhaseBDesign(
        y=np.asarray(y_list, dtype=np.float64),
        w=np.asarray(w_list, dtype=np.float64),
        receiver=np.asarray(rec_list, dtype=np.int64),
        teammates=np.asarray(team_list, dtype=np.int64),
        context=ctx[:, keep_ctx] if keep_ctx else np.zeros((ctx.shape[0], 0)),
        player_ids=tuple(players),
        context_names=tuple(_CONTEXT[j] for j in keep_ctx),
    )


def _teammate_normal_equations(
    teammates: IntArray,
    w: FloatArray,
    resid: FloatArray,
    n_players: int,
    *,
    perm: IntArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """``D' W D`` (p x p) and ``D' W r`` (p,) for the 4-sparse teammate multi-hot
    design (each row: +1 for four teammate columns)."""

    k = teammates.shape[1]
    cols = []
    for slot in range(k):
        c = teammates[:, slot]
        cols.append(
            np.where(c >= 0, c if perm is None else perm[np.where(c >= 0, c, 0)], -1)
        )
    gram = np.zeros(n_players * n_players, dtype=np.float64)
    rhs = np.zeros(n_players, dtype=np.float64)
    for a in range(k):
        ca = cols[a]
        oka = ca >= 0
        rhs += np.bincount(ca[oka], weights=(w * resid)[oka], minlength=n_players)
        for b in range(k):
            cb = cols[b]
            ok = oka & (cb >= 0)
            lin = ca[ok] * n_players + cb[ok]
            gram += np.bincount(lin, weights=w[ok], minlength=n_players * n_players)
    return gram.reshape(n_players, n_players), rhs


@dataclass(frozen=True)
class PhaseBModel:
    design_players: tuple[int, ...]
    mu: float
    context_coef: FloatArray
    base: FloatArray  # (p,) EM-shrunk per-player production level
    base_l2: float
    tau_base2: float
    lift: FloatArray  # (p,) pooled giver effect
    lift_cov_diag: FloatArray
    tau_lift2: float
    sigma2: float
    permuted: bool
    loglik_by_tau2: dict[float, float]

    @classmethod
    def fit(
        cls,
        design: PhaseBDesign,
        *,
        config: PhaseBConfig | None = None,
        seed: int = 0,
        permuted: bool = False,
    ) -> PhaseBModel:
        cfg = config or PhaseBConfig()
        p = design.n_players
        n = len(design.y)
        w = design.w
        # --- stage 1: mu + context + base_k (one player per row) via ridge ------
        # normal equations for [1 | context | receiver one-hot]
        c = design.context.shape[1]
        dim = 1 + c + p
        # build A'WA sparsely: intercept + context dense, receiver one-hot sparse
        ac = np.concatenate([np.ones((n, 1)), design.context], axis=1)  # (n, 1+c)
        awc = ac * w[:, None]
        g_cc = ac.T @ awc  # (1+c, 1+c)
        # receiver block
        rec = design.receiver
        g_rr = np.bincount(rec, weights=w, minlength=p)  # diag of receiver'W receiver
        g_cr = np.zeros((1 + c, p))
        for j in range(1 + c):
            g_cr[j] = np.bincount(rec, weights=awc[:, j], minlength=p)
        rhs_c = awc.T @ design.y
        rhs_r = np.bincount(rec, weights=w * design.y, minlength=p)

        best = None
        for l2 in cfg.base_l2_grid:
            gram = np.zeros((dim, dim))
            gram[: 1 + c, : 1 + c] = g_cc
            gram[: 1 + c, 1 + c :] = g_cr
            gram[1 + c :, : 1 + c] = g_cr.T
            gram[1 + c :, 1 + c :] = np.diag(g_rr)
            pen = np.full(dim, 1e-6)  # keep the intercept/context block PD
            pen[1 + c :] = l2
            try:
                chol = np.linalg.cholesky(gram + np.diag(pen))
            except np.linalg.LinAlgError:
                continue
            rhs = np.concatenate([rhs_c, rhs_r])
            coef = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
            fitted = coef[0] + design.context @ coef[1 : 1 + c] + coef[1 + c :][rec]
            rss = float((w * (design.y - fitted) ** 2).sum())
            gcv = rss / max(n - dim, 1)  # crude; picks a sane l2
            if best is None or gcv < best[0]:
                best = (gcv, l2, coef)
        assert best is not None
        _, base_l2, coef = best
        mu = float(coef[0])
        ctx_coef = np.asarray(coef[1 : 1 + c], dtype=np.float64)
        base = np.asarray(coef[1 + c :], dtype=np.float64)
        stage1 = mu + design.context @ ctx_coef + base[rec]
        sigma2 = float((w * (design.y - stage1) ** 2).sum() / max(n - dim, 1))
        tau_base2 = float(base @ base / p) if p else 1.0

        # --- stage 2: teammate lift on the residual ----------------------------
        resid = design.y - stage1
        perm = np.random.default_rng(seed + 1).permutation(p) if permuted else None
        gram, rhs = _teammate_normal_equations(design.teammates, w, resid, p, perm=perm)
        gram = 0.5 * (gram + gram.T)

        loglik: dict[float, float] = {}
        best_tau2 = cfg.lift_tau2_grid[0]
        best_ll = -np.inf
        for tau2 in cfg.lift_tau2_grid:
            m = gram / sigma2 + np.eye(p) / tau2
            try:
                ch = np.linalg.cholesky(m)
            except np.linalg.LinAlgError:
                continue
            mu_l = np.linalg.solve(ch.T, np.linalg.solve(ch, rhs / sigma2))
            logdet = 2.0 * float(np.log(np.diag(ch)).sum())
            quad = float(mu_l @ (m @ mu_l))
            ll = -0.5 * (p * math.log(tau2) + logdet - quad)
            loglik[tau2] = ll
            if ll > best_ll:
                best_ll, best_tau2 = ll, tau2

        m = gram / sigma2 + np.eye(p) / best_tau2
        ch = np.linalg.cholesky(m)
        lift = np.linalg.solve(ch.T, np.linalg.solve(ch, rhs / sigma2))
        cov_diag = np.diag(np.linalg.solve(ch.T, np.linalg.solve(ch, np.eye(p))))
        return cls(
            design_players=design.player_ids,
            mu=mu,
            context_coef=ctx_coef,
            base=base,
            base_l2=base_l2,
            tau_base2=tau_base2,
            lift=np.asarray(lift, dtype=np.float64),
            lift_cov_diag=np.asarray(cov_diag, dtype=np.float64),
            tau_lift2=float(best_tau2),
            sigma2=sigma2,
            permuted=permuted,
            loglik_by_tau2=loglik,
        )

    # -- prediction on a design built with the SAME player vocabulary --------- #

    def predict(self, design: PhaseBDesign) -> FloatArray:
        base_only = (
            self.mu + design.context @ self.context_coef + self.base[design.receiver]
        )
        lift_term = np.zeros(len(design.y))
        for slot in range(design.teammates.shape[1]):
            c = design.teammates[:, slot]
            ok = c >= 0
            lift_term[ok] += self.lift[c[ok]]
        return np.asarray(base_only + lift_term, dtype=np.float64)

    def predict_base_only(self, design: PhaseBDesign) -> FloatArray:
        return np.asarray(
            self.mu + design.context @ self.context_coef + self.base[design.receiver],
            dtype=np.float64,
        )

    def variance_components(self) -> dict[str, Any]:
        return {
            "sigma": math.sqrt(self.sigma2),
            "tau_base": math.sqrt(max(self.tau_base2, 0.0)),
            "tau_lift": math.sqrt(self.tau_lift2),
            "tau_lift2": self.tau_lift2,
            "base_l2": self.base_l2,
            "lift_abs_mean": float(np.abs(self.lift).mean()),
            "lift_abs_max": float(np.abs(self.lift).max()),
            "permuted": self.permuted,
        }

    def top_lifts(self, n: int = 20) -> list[tuple[int, float, float]]:
        order = np.argsort(-np.abs(self.lift))[:n]
        return [
            (
                int(self.design_players[i]),
                float(self.lift[i]),
                float(math.sqrt(max(self.lift_cov_diag[i], 0.0))),
            )
            for i in order
        ]

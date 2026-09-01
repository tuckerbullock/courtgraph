"""Rung 4 -- explicit teammate-pair interaction RAPM (empirical Bayes).

Extends the rung-3 hierarchical model with an explicit term per **observed**
offensive teammate pair:

    y_s ~ N( C_s.theta_c + sum_i alpha_i - sum_j beta_j
             + sum_{(i,j) in offense pairs, admitted} gamma_ij ,  sigma^2 / w_s )

    alpha_i  ~ N(0, tau_off^2)   beta_j ~ N(0, tau_def^2)
    gamma_ij ~ N(0, tau_pair^2)  (offense pairs only for v1)
    theta_c  ~ N(0, tau_c^2)     tau_c^2 large and fixed

``(sigma^2, tau_off^2, tau_def^2, tau_pair^2)`` are learned by the same EM as
:class:`~courtgraph.chemistry.hierarchical.HierarchicalRidge`, now with a third
variance component.

A pair is only given a ``gamma_ij`` if it has ``>= min_co_stints`` shared
offensive stints in the training data (:class:`PairVocabulary`) -- master plan
section 15.3: do not score pairs the model cannot distinguish, and it keeps the
linear system tractable (~3-6k pairs on a real season, not ~20k). A lineup
containing a pair outside that vocabulary silently degrades to the additive
prediction for that pair term, which is the correct rung-4 behaviour: unseen /
under-observed pairs are rung 5's job (the low-rank factorization).

Static players, pure NumPy, deterministic. Offense pairs only; the symmetric
defensive pair term is a documented follow-up.
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
    _cross_ctx,
    _cross_gram,
    _cross_rhs,
    _gather_sum,
)
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.stints import StintTable, pair_id

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PAIR_SCHEMA_VERSION = 1
_SLOT_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(5) for b in range(a + 1, 5)
)  # the 10 unordered slot pairs among 5 offensive positions


@dataclass(frozen=True)
class PairHierarchicalConfig:
    tau_c2: float = 1.0e6
    max_iters: int = 100
    tol: float = 1e-7
    min_co_stints: int = 200


@dataclass(frozen=True)
class PairVocabulary:
    """The offensive teammate pairs a rung-4 model estimates a term for:
    every pair with at least ``min_co_stints`` shared offensive stints in the
    training data. Keyed by :func:`courtgraph.chemistry.stints.pair_id`."""

    pair_ids: tuple[str, ...]
    min_co_stints: int
    _index: dict[str, int]
    # Optional positional remap ``pair_ids[i] -> coefficient row``. ``None`` is
    # the identity. Used only by the placebo control in ``baseline_ladder``:
    # a non-injective override collapses distinct pairs onto shared rows,
    # breaking pair-specific signal while keeping the parameter count and total
    # pair exposure fixed.
    row_override: tuple[int, ...] | None = None

    @property
    def n_pairs(self) -> int:
        return len(self.pair_ids)

    def index_of(self, key: str) -> int:
        return self._index.get(key, -1)

    @classmethod
    def from_training(
        cls, table: StintTable, *, min_co_stints: int = 200
    ) -> PairVocabulary:
        counts: dict[str, int] = {}
        for stint in table:
            ids = stint.offense_player_ids
            for a, b in _SLOT_PAIRS:
                key = pair_id(ids[a], ids[b])
                counts[key] = counts.get(key, 0) + 1
        admitted = sorted(k for k, c in counts.items() if c >= min_co_stints)
        return cls(
            pair_ids=tuple(admitted),
            min_co_stints=min_co_stints,
            _index={k: i for i, k in enumerate(admitted)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {"pair_ids": list(self.pair_ids), "min_co_stints": self.min_co_stints}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairVocabulary:
        ids = tuple(str(k) for k in data["pair_ids"])
        return cls(
            pair_ids=ids,
            min_co_stints=int(data["min_co_stints"]),
            _index={k: i for i, k in enumerate(ids)},
        )


def build_offense_pair_index(
    offense_index: IntArray, feature_space: FeatureSpace, vocab: PairVocabulary
) -> IntArray:
    """``(n, 10)`` int array: for each stint, the admitted-pair row of each of
    its 10 offensive slot pairs, or ``-1`` if that pair is not admitted (or a
    slot holds an unseen player). Fully vectorized over rows."""

    pos_of = feature_space.player_index()  # player_id -> position
    n = offense_index.shape[0]
    p = feature_space.n_players
    # dense (n_players+1)^2 lookup: (pos_lo, pos_hi) -> admitted pair row
    lookup = np.full((p + 1) * (p + 1), -1, dtype=np.int64)
    override = vocab.row_override
    for row, key in enumerate(vocab.pair_ids):
        a_id, b_id = (int(x) for x in key.split("-"))
        ia, ib = pos_of.get(a_id, -1), pos_of.get(b_id, -1)
        if ia < 0 or ib < 0:
            continue
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        lookup[lo * (p + 1) + hi] = override[row] if override is not None else row

    out = np.full((n, 10), -1, dtype=np.int64)
    for k, (a, b) in enumerate(_SLOT_PAIRS):
        col_a, col_b = offense_index[:, a], offense_index[:, b]
        seen = (col_a >= 0) & (col_b >= 0)
        col_lo = np.where(col_a < col_b, col_a, col_b)
        col_hi = np.where(col_a < col_b, col_b, col_a)
        flat_key = np.where(seen, col_lo * (p + 1) + col_hi, 0)
        out[:, k] = np.where(seen, lookup[flat_key], -1)
    return out


@dataclass(frozen=True)
class PairHierarchicalRidge:
    feature_space: FeatureSpace
    vocab: PairVocabulary
    context_coef: FloatArray
    offense_coef: FloatArray
    defense_coef: FloatArray
    pair_coef: FloatArray
    sigma2: float
    tau_off2: float
    tau_def2: float
    tau_pair2: float
    tau_c2: float
    n_iters: int
    converged: bool
    final_loglik: float
    n_obs: int
    gram: FloatArray
    rhs: FloatArray
    chol: FloatArray

    # -- fitting -----------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        vocab: PairVocabulary,
        *,
        config: PairHierarchicalConfig | None = None,
    ) -> PairHierarchicalRidge:
        cfg = config or PairHierarchicalConfig()
        space = feature_space
        c = space.n_context
        p = space.n_players
        q = vocab.n_pairs
        dim = c + 2 * p + q
        o0, d0, g0 = c, c + p, c + 2 * p

        opair = build_offense_pair_index(design.offense_index, space, vocab)
        w, y = design.weight, design.y
        n = design.n_rows
        context_weighted = design.context * w[:, None]

        # normal equations for A = [context | +offense | -defense | +pairs]
        gram = np.zeros((dim, dim), dtype=np.float64)
        gram[:c, :c] = context_weighted.T @ design.context
        g_oc = _cross_ctx(design.offense_index, context_weighted, p)
        g_dc = _cross_ctx(design.defense_index, context_weighted, p)
        g_gc = _cross_ctx(opair, context_weighted, q)
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
        gram[g0:, g0:] = _cross_gram(opair, opair, w, q, q)
        g_od = _cross_gram(design.offense_index, design.defense_index, w, p, p)
        gram[o0:d0, d0:g0] = -g_od
        gram[d0:g0, o0:d0] = -g_od.T
        g_og = _cross_gram(design.offense_index, opair, w, p, q)
        gram[o0:d0, g0:] = g_og
        gram[g0:, o0:d0] = g_og.T
        g_dg = _cross_gram(design.defense_index, opair, w, p, q)
        gram[d0:g0, g0:] = -g_dg
        gram[g0:, d0:g0] = -g_dg.T

        rhs = np.zeros(dim, dtype=np.float64)
        rhs[:c] = context_weighted.T @ y
        rhs[o0:d0] = _cross_rhs(design.offense_index, w * y, p)
        rhs[d0:g0] = -_cross_rhs(design.defense_index, w * y, p)
        rhs[g0:] = _cross_rhs(opair, w * y, q)

        y_w_y = float((w * y**2).sum())
        sum_log_w = float(np.log(w).sum())
        inv_tau_c2 = 1.0 / cfg.tau_c2

        tau_off2 = tau_def2 = tau_pair2 = 1.0
        sigma2 = float(np.var(y)) or 1.0
        prev = np.array(
            [
                math.log(tau_off2),
                math.log(tau_def2),
                math.log(tau_pair2),
                math.log(sigma2),
            ]
        )
        last_loglik = -np.inf
        n_iters = 0
        converged = False
        chol = np.empty((dim, dim))
        mu = np.zeros(dim)

        def _lam(t_off: float, t_def: float, t_pair: float) -> FloatArray:
            lam = np.empty(dim)
            lam[:c] = inv_tau_c2
            lam[o0:d0] = 1.0 / t_off
            lam[d0:g0] = 1.0 / t_def
            lam[g0:] = 1.0 / t_pair
            return lam

        for n_iters in range(1, cfg.max_iters + 1):
            lam = _lam(tau_off2, tau_def2, tau_pair2)
            m = gram / sigma2
            m[np.diag_indices(dim)] += lam
            chol = np.asarray(np.linalg.cholesky(m), dtype=np.float64)
            mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))
            chol_inv = np.linalg.solve(chol, np.eye(dim))
            v_diag = np.einsum("ij,ij->j", chol_inv, chol_inv)

            mu_a, mu_b, mu_g = mu[o0:d0], mu[d0:g0], mu[g0:]
            new_off = float(mu_a @ mu_a + v_diag[o0:d0].sum()) / p
            new_def = float(mu_b @ mu_b + v_diag[d0:g0].sum()) / p
            new_pair = float(mu_g @ mu_g + v_diag[g0:].sum()) / q if q else tau_pair2

            y_hat = (
                design.context @ mu[:c]
                + _gather_sum(mu_a, design.offense_index)
                - _gather_sum(mu_b, design.defense_index)
                + _gather_sum(mu_g, opair)
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
                    f"pair EM log-likelihood decreased at iter {n_iters}: "
                    f"{last_loglik:.6f} -> {loglik:.6f}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            last_loglik = loglik

            tau_off2, tau_def2, tau_pair2, sigma2 = (
                new_off,
                new_def,
                new_pair,
                new_sigma2,
            )
            cur = np.array(
                [
                    math.log(tau_off2),
                    math.log(tau_def2),
                    math.log(tau_pair2),
                    math.log(sigma2),
                ]
            )
            if float(np.max(np.abs(cur - prev))) < cfg.tol:
                converged = True
                prev = cur
                break
            prev = cur

        lam = _lam(tau_off2, tau_def2, tau_pair2)
        m = gram / sigma2
        m[np.diag_indices(dim)] += lam
        chol = np.asarray(np.linalg.cholesky(m), dtype=np.float64)
        mu = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / sigma2))

        return cls(
            feature_space=space,
            vocab=vocab,
            context_coef=np.asarray(mu[:c], dtype=np.float64),
            offense_coef=np.asarray(mu[o0:d0], dtype=np.float64),
            defense_coef=np.asarray(mu[d0:g0], dtype=np.float64),
            pair_coef=np.asarray(mu[g0:], dtype=np.float64),
            sigma2=float(sigma2),
            tau_off2=float(tau_off2),
            tau_def2=float(tau_def2),
            tau_pair2=float(tau_pair2),
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

    def _opair(self, design: DesignMatrices) -> IntArray:
        return build_offense_pair_index(
            design.offense_index, self.feature_space, self.vocab
        )

    def predict(self, design: DesignMatrices) -> FloatArray:
        ctx = design.context @ self.context_coef
        off = _gather_sum(self.offense_coef, design.offense_index)
        deff = _gather_sum(self.defense_coef, design.defense_index)
        pair = _gather_sum(self.pair_coef, self._opair(design))
        return np.asarray(ctx + off - deff + pair, dtype=np.float64)

    def residuals(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(design.y - self.predict(design), dtype=np.float64)

    def talent_of(self, player_id: int) -> tuple[float, float]:
        pos = self.feature_space.player_index().get(player_id)
        if pos is None:
            return 0.0, 0.0
        return float(self.offense_coef[pos]), float(self.defense_coef[pos])

    def pair_surplus(self, player_a: int, player_b: int) -> float:
        """Estimated offensive surplus for the teammate pair; ``0.0`` for a pair
        outside the admitted vocabulary (rung 4 has no estimate there)."""

        row = self.vocab.index_of(pair_id(player_a, player_b))
        return 0.0 if row < 0 else float(self.pair_coef[row])

    def decompose_row(self, design: DesignMatrices, row: int) -> AdditiveDecomposition:
        intercept_idx = self.feature_space.context_columns.index("intercept")
        context_contrib = float(
            design.context[row] @ self.context_coef
            - design.context[row, intercept_idx] * self.context_coef[intercept_idx]
        )
        one = slice(row, row + 1)
        off = float(_gather_sum(self.offense_coef, design.offense_index[one])[0])
        deff = float(_gather_sum(self.defense_coef, design.defense_index[one])[0])
        pair = float(_gather_sum(self.pair_coef, self._opair(design)[one])[0])
        return AdditiveDecomposition(
            talent=self.alpha + off - deff + pair, context=context_contrib
        )

    def variance_components(self) -> dict[str, Any]:
        return {
            "sigma": math.sqrt(self.sigma2),
            "tau_off": math.sqrt(self.tau_off2),
            "tau_def": math.sqrt(self.tau_def2),
            "tau_pair": math.sqrt(self.tau_pair2),
            "sigma2": self.sigma2,
            "tau_off2": self.tau_off2,
            "tau_def2": self.tau_def2,
            "tau_pair2": self.tau_pair2,
            "tau_c2": self.tau_c2,
            "n_admitted_pairs": self.vocab.n_pairs,
            "min_co_stints": self.vocab.min_co_stints,
            "n_iters": self.n_iters,
            "converged": self.converged,
            "final_loglik": self.final_loglik,
        }

    # -- uncertainty ---------------------------------------------------------

    def _dim(self) -> int:
        return (
            self.feature_space.n_context
            + 2 * self.feature_space.n_players
            + self.vocab.n_pairs
        )

    def _design_rows(
        self, design: DesignMatrices, rows: NDArray[np.int64]
    ) -> FloatArray:
        c = self.feature_space.n_context
        p = self.feature_space.n_players
        q = self.vocab.n_pairs
        opair = self._opair(design)
        out = np.zeros((len(rows), c + 2 * p + q), dtype=np.float64)
        out[:, :c] = design.context[rows]
        for k in range(5):
            off_k = design.offense_index[rows, k]
            seen = off_k >= 0
            out[np.flatnonzero(seen), c + off_k[seen]] += 1.0
            def_k = design.defense_index[rows, k]
            seen = def_k >= 0
            out[np.flatnonzero(seen), c + p + def_k[seen]] -= 1.0
        for k in range(10):
            g_k = opair[rows, k]
            seen = g_k >= 0
            out[np.flatnonzero(seen), c + 2 * p + g_k[seen]] += 1.0
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
            [self.context_coef, self.offense_coef, self.defense_coef, self.pair_coef]
        )
        point = g_mat @ theta
        solved = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, g_mat.T))
        var_param = np.einsum("ij,ji->i", g_mat, solved)
        sd = np.sqrt(np.maximum(var_param, 0.0) + self.sigma2 / totals)
        return {
            key: (float(point[gi]), float(sd[gi]), float(totals[gi]))
            for gi, key in enumerate(keys)
        }

    # -- serialization (small models only; eval-only for cycle 1) --

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_schema_version": PAIR_SCHEMA_VERSION,
            "feature_space": self.feature_space.to_dict(),
            "vocab": self.vocab.to_dict(),
            "context_coef": self.context_coef.tolist(),
            "offense_coef": self.offense_coef.tolist(),
            "defense_coef": self.defense_coef.tolist(),
            "pair_coef": self.pair_coef.tolist(),
            "sigma2": self.sigma2,
            "tau_off2": self.tau_off2,
            "tau_def2": self.tau_def2,
            "tau_pair2": self.tau_pair2,
            "tau_c2": self.tau_c2,
            "n_iters": self.n_iters,
            "converged": self.converged,
            "final_loglik": self.final_loglik,
            "n_obs": self.n_obs,
            "gram": self.gram.tolist(),
            "rhs": self.rhs.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairHierarchicalRidge:
        if data.get("pair_schema_version") != PAIR_SCHEMA_VERSION:
            raise ValueError("unrecognized pair-interaction artifact schema version")
        space = FeatureSpace.from_dict(data["feature_space"])
        vocab = PairVocabulary.from_dict(data["vocab"])
        c, p, q = space.n_context, space.n_players, vocab.n_pairs
        dim = c + 2 * p + q
        gram = np.asarray(data["gram"], dtype=np.float64)
        sigma2 = float(data["sigma2"])
        lam = np.empty(dim)
        lam[:c] = 1.0 / float(data["tau_c2"])
        lam[c : c + p] = 1.0 / float(data["tau_off2"])
        lam[c + p : c + 2 * p] = 1.0 / float(data["tau_def2"])
        lam[c + 2 * p :] = 1.0 / float(data["tau_pair2"])
        m = gram / sigma2
        m[np.diag_indices(dim)] += lam
        return cls(
            feature_space=space,
            vocab=vocab,
            context_coef=np.asarray(data["context_coef"], dtype=np.float64),
            offense_coef=np.asarray(data["offense_coef"], dtype=np.float64),
            defense_coef=np.asarray(data["defense_coef"], dtype=np.float64),
            pair_coef=np.asarray(data["pair_coef"], dtype=np.float64),
            sigma2=sigma2,
            tau_off2=float(data["tau_off2"]),
            tau_def2=float(data["tau_def2"]),
            tau_pair2=float(data["tau_pair2"]),
            tau_c2=float(data["tau_c2"]),
            n_iters=int(data["n_iters"]),
            converged=bool(data["converged"]),
            final_loglik=float(data["final_loglik"]),
            n_obs=int(data["n_obs"]),
            gram=gram,
            rhs=np.asarray(data["rhs"], dtype=np.float64),
            chol=np.asarray(np.linalg.cholesky(m), dtype=np.float64),
        )

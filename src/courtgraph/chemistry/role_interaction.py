"""Role-conditioned interaction RAPM (candidate idea #1).

Instead of a free parameter per **observed** teammate pair (rung 4, keyed by
player identity, so most pairs are thin), this keys the offensive interaction
term on the pair of **role clusters** the two players belong to
(:mod:`courtgraph.features.role_clusters`):

    y_s ~ N( C_s.theta_c + sum_i alpha_i - sum_j beta_j
             + sum_{(i,j) in offense pairs} delta_{r(i), r(j)} , sigma^2 / w_s )

    delta_{a,b} ~ N(0, tau_role^2)   (symmetric; K*(K+1)/2 free parameters)

With K = 5 roles that is 15 interaction parameters pooled over every teammate
pair, each backed by thousands of stints -- a far better-powered test of
"do certain kinds of players fit / clash" than the ~2-3k thin per-identity
pairs. Same weighted-Gaussian EM as rungs 3/4 (shared core in
:mod:`courtgraph.chemistry._augmented_em`).

A player with no role (below the profile exposure floor) contributes no role
pairs for that stint -- the term degrades to additive, exactly as rung 4 does
for an unadmitted pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry._augmented_em import fit_augmented_em
from courtgraph.chemistry.baseline import AdditiveDecomposition, _gather_sum
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.features.role_clusters import RoleClustering

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ROLE_INTERACTION_SCHEMA_VERSION = 1
_SLOT_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(5) for b in range(a + 1, 5)
)


@dataclass(frozen=True)
class RoleInteractionConfig:
    tau_c2: float = 1.0e6
    max_iters: int = 100
    tol: float = 1e-7


def n_role_pairs(k: int) -> int:
    """Number of unordered role-cluster pairs (with repeats): ``K*(K+1)/2``."""

    return k * (k + 1) // 2


def role_pair_bin(role_a: int, role_b: int, k: int) -> int:
    """Upper-triangular index of the unordered role pair ``{a, b}``."""

    lo, hi = (role_a, role_b) if role_a <= role_b else (role_b, role_a)
    return lo * k - lo * (lo - 1) // 2 + (hi - lo)


def build_role_pair_index(
    offense_index: IntArray, feature_space: FeatureSpace, clustering: RoleClustering
) -> IntArray:
    """``(n, 10)`` int array: the role-pair bin of each offensive slot pair, or
    ``-1`` when a slot holds an unseen player or a player with no role."""

    id_of = {pos: pid for pid, pos in feature_space.player_index().items()}
    k = clustering.n_clusters
    n = offense_index.shape[0]
    role_at_pos = np.full(feature_space.n_players + 1, -1, dtype=np.int64)
    for pos in range(feature_space.n_players):
        role_at_pos[pos] = clustering.role_of(id_of.get(pos, -1))

    roles = np.where(offense_index >= 0, role_at_pos[offense_index], -1)  # (n, 5)
    out = np.full((n, 10), -1, dtype=np.int64)
    for slot, (a, b) in enumerate(_SLOT_PAIRS):
        ra, rb = roles[:, a], roles[:, b]
        ok = (ra >= 0) & (rb >= 0)
        lo = np.minimum(ra, rb)
        hi = np.maximum(ra, rb)
        binned = lo * k - lo * (lo - 1) // 2 + (hi - lo)
        out[:, slot] = np.where(ok, binned, -1)
    return out


@dataclass(frozen=True)
class RoleClusterInteraction:
    feature_space: FeatureSpace
    clustering: RoleClustering
    context_coef: FloatArray
    offense_coef: FloatArray
    defense_coef: FloatArray
    role_coef: FloatArray  # (K*(K+1)/2,)
    sigma2: float
    tau_off2: float
    tau_def2: float
    tau_role2: float
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
        config: RoleInteractionConfig | None = None,
    ) -> RoleClusterInteraction:
        cfg = config or RoleInteractionConfig()
        rindex = build_role_pair_index(design.offense_index, feature_space, clustering)
        core = fit_augmented_em(
            design,
            feature_space,
            rindex,
            n_role_pairs(clustering.n_clusters),
            tau_c2=cfg.tau_c2,
            max_iters=cfg.max_iters,
            tol=cfg.tol,
            label="role",
        )
        return cls(
            feature_space=feature_space,
            clustering=clustering,
            context_coef=core.context_coef,
            offense_coef=core.offense_coef,
            defense_coef=core.defense_coef,
            role_coef=core.extra_coef,
            sigma2=core.sigma2,
            tau_off2=core.tau_off2,
            tau_def2=core.tau_def2,
            tau_role2=core.tau_extra2,
            tau_c2=core.tau_c2,
            n_iters=core.n_iters,
            converged=core.converged,
            final_loglik=core.final_loglik,
            n_obs=core.n_obs,
            gram=core.gram,
            rhs=core.rhs,
            chol=core.chol,
        )

    # -- prediction ----------------------------------------------------------- #

    @property
    def alpha(self) -> float:
        idx = self.feature_space.context_columns.index("intercept")
        return float(self.context_coef[idx])

    def _rindex(self, design: DesignMatrices) -> IntArray:
        return build_role_pair_index(
            design.offense_index, self.feature_space, self.clustering
        )

    def predict(self, design: DesignMatrices) -> FloatArray:
        ctx = design.context @ self.context_coef
        off = _gather_sum(self.offense_coef, design.offense_index)
        deff = _gather_sum(self.defense_coef, design.defense_index)
        role = _gather_sum(self.role_coef, self._rindex(design))
        return np.asarray(ctx + off - deff + role, dtype=np.float64)

    def residuals(self, design: DesignMatrices) -> FloatArray:
        return np.asarray(design.y - self.predict(design), dtype=np.float64)

    def role_pair_effect(self, role_a: int, role_b: int) -> float:
        return float(
            self.role_coef[role_pair_bin(role_a, role_b, self.clustering.n_clusters)]
        )

    def _dim(self) -> int:
        return (
            self.feature_space.n_context
            + 2 * self.feature_space.n_players
            + int(self.role_coef.shape[0])
        )

    def _design_rows(self, design: DesignMatrices, rows: IntArray) -> FloatArray:
        c = self.feature_space.n_context
        p = self.feature_space.n_players
        out = np.zeros((len(rows), self._dim()))
        out[:, :c] = design.context[rows]
        rindex = self._rindex(design)
        for k in range(5):
            off_k = design.offense_index[rows, k]
            seen = off_k >= 0
            out[np.flatnonzero(seen), c + off_k[seen]] += 1.0
            def_k = design.defense_index[rows, k]
            seen = def_k >= 0
            out[np.flatnonzero(seen), c + p + def_k[seen]] -= 1.0
        for slot in range(10):
            g_k = rindex[rows, slot]
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
            [self.context_coef, self.offense_coef, self.defense_coef, self.role_coef]
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
        off = float(_gather_sum(self.offense_coef, design.offense_index[one])[0])
        deff = float(_gather_sum(self.defense_coef, design.defense_index[one])[0])
        role = float(_gather_sum(self.role_coef, self._rindex(design)[one])[0])
        return AdditiveDecomposition(
            talent=self.alpha + off - deff + role, context=context_contrib
        )

    def variance_components(self) -> dict[str, Any]:
        return {
            "sigma": math.sqrt(self.sigma2),
            "tau_off": math.sqrt(self.tau_off2),
            "tau_def": math.sqrt(self.tau_def2),
            "tau_role": math.sqrt(self.tau_role2),
            "sigma2": self.sigma2,
            "tau_off2": self.tau_off2,
            "tau_def2": self.tau_def2,
            "tau_role2": self.tau_role2,
            "tau_c2": self.tau_c2,
            "n_role_clusters": self.clustering.n_clusters,
            "n_role_pairs": int(self.role_coef.shape[0]),
            "n_iters": self.n_iters,
            "converged": self.converged,
            "final_loglik": self.final_loglik,
        }

    def role_pair_matrix(self) -> FloatArray:
        """The symmetric ``K x K`` matrix of fitted role-pair effects."""

        k = self.clustering.n_clusters
        out = np.zeros((k, k))
        for a in range(k):
            for b in range(a, k):
                out[a, b] = out[b, a] = self.role_pair_effect(a, b)
        return out

"""Additive ridge baseline (model ladder rung 2, separate O/D talent).

Fits, by weighted ridge regression on stints,

    ortg  ~  context . k  +  sum_{i in L_o} b_off[i]  -  sum_{j in L_d} b_def[j]

with possessions as the exposure weight. Both ``b_off`` and ``b_def`` are signed
larger-is-better (research contract 4). The player ridge strength is chosen by a
game-blocked internal search on the training data only -- no test leakage, no
magic constant.

This is the pure "sum of the parts" model: whatever error it leaves on
structurally unseen pairs and lineups is the room a real chemistry signal has
to improve on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.features import DesignMatrices, FeatureSpace

FloatArray = NDArray[np.float64]

DEFAULT_L2_GRID: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0)
_SELECTION_FOLDS = 3


@dataclass(frozen=True)
class AdditiveDecomposition:
    """Additive prediction split into talent and context (points per 100)."""

    talent: float
    context: float

    @property
    def total(self) -> float:
        return self.talent + self.context


@dataclass(frozen=True)
class AdditiveRidge:
    """A fitted additive ridge model over a fixed :class:`FeatureSpace`."""

    feature_space: FeatureSpace
    context_coef: FloatArray  # (n_context,)
    offense_coef: FloatArray  # (n_players,)  b_off, larger-is-better
    defense_coef: FloatArray  # (n_players,)  b_def, larger-is-better (pts prevented)
    l2_player: float
    l2_context: float

    # -- fitting ---------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        design: DesignMatrices,
        feature_space: FeatureSpace,
        *,
        l2_player: float | None = None,
        l2_context: float = 1e-3,
        l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    ) -> AdditiveRidge:
        if l2_player is None:
            l2_player = _select_l2_player(design, feature_space, l2_context, l2_grid)
        context_coef, offense_coef, defense_coef = _solve_ridge(
            design, feature_space, l2_player, l2_context
        )
        return cls(
            feature_space=feature_space,
            context_coef=context_coef,
            offense_coef=offense_coef,
            defense_coef=defense_coef,
            l2_player=float(l2_player),
            l2_context=float(l2_context),
        )

    # -- prediction ----------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Baseline offensive rating at the reference context (intercept)."""

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
        """(offensive, defensive) talent for one player; (0, 0) if unseen."""

        index = self.feature_space.player_index()
        pos = index.get(player_id)
        if pos is None:
            return 0.0, 0.0
        return float(self.offense_coef[pos]), float(self.defense_coef[pos])

    def decompose_row(self, design: DesignMatrices, row: int) -> AdditiveDecomposition:
        intercept_idx = self.feature_space.context_columns.index("intercept")
        context_contrib = float(
            design.context[row] @ self.context_coef
            - design.context[row, intercept_idx] * self.context_coef[intercept_idx]
        )
        off = float(
            _gather_sum(self.offense_coef, design.offense_index[row : row + 1])[0]
        )
        deff = float(
            _gather_sum(self.defense_coef, design.defense_index[row : row + 1])[0]
        )
        talent = self.alpha + off - deff
        return AdditiveDecomposition(talent=talent, context=context_contrib)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_space": self.feature_space.to_dict(),
            "context_coef": self.context_coef.tolist(),
            "offense_coef": self.offense_coef.tolist(),
            "defense_coef": self.defense_coef.tolist(),
            "l2_player": self.l2_player,
            "l2_context": self.l2_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdditiveRidge:
        return cls(
            feature_space=FeatureSpace.from_dict(data["feature_space"]),
            context_coef=np.asarray(data["context_coef"], dtype=np.float64),
            offense_coef=np.asarray(data["offense_coef"], dtype=np.float64),
            defense_coef=np.asarray(data["defense_coef"], dtype=np.float64),
            l2_player=float(data["l2_player"]),
            l2_context=float(data["l2_context"]),
        )


def _gather_sum(coef: FloatArray, index: NDArray[np.int64]) -> FloatArray:
    """Row sums of ``coef`` over the (n, 5) player indices; -1 contributes 0.

    Equivalent to ``onehot @ coef`` where the one-hot has no bit set for an
    unseen (-1) player -- a padded gather instead of an (n, n_players) matmul.
    """

    pad = np.append(coef, 0.0)
    gathered = pad[np.where(index >= 0, index, coef.shape[0])]
    return np.asarray(gathered.sum(axis=1), dtype=np.float64)


def _cross_gram(
    ix: NDArray[np.int64],
    jx: NDArray[np.int64],
    w: FloatArray,
    dim_i: int,
    dim_j: int,
) -> FloatArray:
    """``(dim_i, dim_j)`` block: entry (i, j) = sum of ``w`` over rows with i in
    ``ix`` and j in ``jx``. This is ``Ixᵀ (w ⊙) Jx`` for the two 0/1 one-hots,
    built by scattering the ``k_i · k_j`` index pairs per row -- O(n · k_i · k_j),
    never an ``(n, dim)`` intermediate. ``-1`` in either index contributes 0.
    """

    if dim_i == 0 or dim_j == 0:
        return np.zeros((dim_i, dim_j), dtype=np.float64)
    n, k_i = ix.shape
    k_j = jx.shape[1]
    bi = np.broadcast_to(ix[:, :, None], (n, k_i, k_j))
    bj = np.broadcast_to(jx[:, None, :], (n, k_i, k_j))
    ok = (bi >= 0) & (bj >= 0)
    lin = np.where(ok, bi * dim_j + bj, 0).ravel()
    weights = np.broadcast_to(w[:, None, None], (n, k_i, k_j)).astype(
        np.float64, copy=True
    )
    weights[~ok] = 0.0
    flat = np.bincount(lin, weights=weights.ravel(), minlength=dim_i * dim_j)
    return np.asarray(flat.reshape(dim_i, dim_j), dtype=np.float64)


def _cross_ctx(
    ix: NDArray[np.int64], context_weighted: FloatArray, dim_i: int
) -> FloatArray:
    """``(dim_i, n_context)`` block: row i = sum over rows containing i of
    ``weight ⊙ context[row]``. Column loop keeps the transient at O(n · k)."""

    k = ix.shape[1]
    flat = ix.ravel()
    seen = flat >= 0
    kept = flat[seen]
    out = np.zeros((dim_i, context_weighted.shape[1]), dtype=np.float64)
    for c in range(context_weighted.shape[1]):
        column = np.repeat(context_weighted[:, c], k)[seen]
        out[:, c] = np.bincount(kept, weights=column, minlength=dim_i)
    return out


def _cross_rhs(ix: NDArray[np.int64], row_values: FloatArray, dim_i: int) -> FloatArray:
    """``(dim_i,)`` vector: entry i = sum of ``row_values`` over rows containing i."""

    k = ix.shape[1]
    flat = ix.ravel()
    seen = flat >= 0
    return np.asarray(
        np.bincount(
            flat[seen], weights=np.repeat(row_values, k)[seen], minlength=dim_i
        ),
        dtype=np.float64,
    )


def _pair_gram(
    ix: NDArray[np.int64], jx: NDArray[np.int64], w: FloatArray, n_players: int
) -> FloatArray:
    return _cross_gram(ix, jx, w, n_players, n_players)


def _ctx_gram(
    ix: NDArray[np.int64], context_weighted: FloatArray, n_players: int
) -> FloatArray:
    return _cross_ctx(ix, context_weighted, n_players)


def _idx_rhs(
    ix: NDArray[np.int64], row_values: FloatArray, n_players: int
) -> FloatArray:
    return _cross_rhs(ix, row_values, n_players)


def _normal_equations(
    design: DesignMatrices, feature_space: FeatureSpace
) -> tuple[FloatArray, FloatArray]:
    """The weighted normal-equations Gram (no penalty) and rhs for the design
    ``A = [context | +offense_onehot | -defense_onehot]``, built blockwise
    without ever materializing a one-hot matrix."""

    n_players = feature_space.n_players
    n_context = feature_space.n_context
    off, deff = design.offense_index, design.defense_index
    w, y = design.weight, design.y
    context_weighted = design.context * w[:, None]

    g_cc = context_weighted.T @ design.context
    g_oc = _ctx_gram(off, context_weighted, n_players)
    g_dc = _ctx_gram(deff, context_weighted, n_players)
    g_oo = _pair_gram(off, off, w, n_players)
    g_dd = _pair_gram(deff, deff, w, n_players)
    g_od = _pair_gram(off, deff, w, n_players)  # OᵀWD; the -D sign is applied below

    dim = n_context + 2 * n_players
    o0, d0 = n_context, n_context + n_players
    gram = np.zeros((dim, dim), dtype=np.float64)
    gram[:n_context, :n_context] = g_cc
    gram[:n_context, o0:d0] = g_oc.T
    gram[o0:d0, :n_context] = g_oc
    gram[:n_context, d0:] = -g_dc.T
    gram[d0:, :n_context] = -g_dc
    gram[o0:d0, o0:d0] = g_oo
    gram[d0:, d0:] = g_dd
    gram[o0:d0, d0:] = -g_od
    gram[d0:, o0:d0] = -g_od.T

    rhs = np.zeros(dim, dtype=np.float64)
    rhs[:n_context] = context_weighted.T @ y
    rhs[o0:d0] = _idx_rhs(off, w * y, n_players)
    rhs[d0:] = -_idx_rhs(deff, w * y, n_players)
    return gram, rhs


def _split_theta(
    theta: FloatArray, feature_space: FeatureSpace
) -> tuple[FloatArray, FloatArray, FloatArray]:
    c, p = feature_space.n_context, feature_space.n_players
    return (
        np.asarray(theta[:c], dtype=np.float64),
        np.asarray(theta[c : c + p], dtype=np.float64),
        np.asarray(theta[c + p :], dtype=np.float64),
    )


def _solve_ridge(
    design: DesignMatrices,
    feature_space: FeatureSpace,
    l2_player: float,
    l2_context: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n_context = feature_space.n_context
    n_players = feature_space.n_players
    gram, rhs = _normal_equations(design, feature_space)
    penalty = np.concatenate(
        [
            np.full(n_context, l2_context),
            np.full(2 * n_players, l2_player),
        ]
    )
    theta = np.linalg.solve(gram + np.diag(penalty), rhs)
    return _split_theta(theta, feature_space)


def _select_l2_player(
    design: DesignMatrices,
    feature_space: FeatureSpace,
    l2_context: float,
    grid: tuple[float, ...],
) -> float:
    """Pick the ridge strength by a deterministic game-blocked k-fold search.

    The normal-equations Gram / rhs for each fold's training rows are built
    once; the grid values are then cheap re-solves of ``Gram + diag(penalty)``.
    """

    unique_games = sorted(set(design.game_ids))
    fold_of_game = {g: (i % _SELECTION_FOLDS) for i, g in enumerate(unique_games)}
    fold = np.array([fold_of_game[g] for g in design.game_ids])
    n_context = feature_space.n_context
    n_players = feature_space.n_players

    folds: list[tuple[FloatArray, FloatArray, DesignMatrices]] = []
    for f in range(_SELECTION_FOLDS):
        train = fold != f
        val = ~train
        if not val.any() or not train.any():
            continue
        gram, rhs = _normal_equations(_mask_design(design, train), feature_space)
        folds.append((gram, rhs, _mask_design(design, val)))

    best_l2, best_err = grid[0], np.inf
    for l2 in grid:
        penalty = np.concatenate(
            [np.full(n_context, l2_context), np.full(2 * n_players, l2)]
        )
        errors: list[float] = []
        for gram, rhs, val_design in folds:
            theta = np.linalg.solve(gram + np.diag(penalty), rhs)
            context_coef, offense_coef, defense_coef = _split_theta(
                theta, feature_space
            )
            model = AdditiveRidge(
                feature_space=feature_space,
                context_coef=context_coef,
                offense_coef=offense_coef,
                defense_coef=defense_coef,
                l2_player=float(l2),
                l2_context=l2_context,
            )
            resid = val_design.y - model.predict(val_design)
            errors.append(
                float(np.sqrt(np.average(resid**2, weights=val_design.weight)))
            )
        mean_err = float(np.mean(errors)) if errors else np.inf
        if mean_err < best_err:
            best_l2, best_err = l2, mean_err
    return best_l2


def _mask_design(design: DesignMatrices, mask: NDArray[np.bool_]) -> DesignMatrices:
    keep = np.flatnonzero(mask)
    return DesignMatrices(
        context=design.context[keep],
        offense_index=design.offense_index[keep],
        defense_index=design.defense_index[keep],
        y=design.y[keep],
        weight=design.weight[keep],
        game_ids=tuple(design.game_ids[i] for i in keep),
        stint_ids=tuple(design.stint_ids[i] for i in keep),
    )

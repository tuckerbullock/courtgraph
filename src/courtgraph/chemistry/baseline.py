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
        off = design.offense_onehot @ self.offense_coef
        deff = design.defense_onehot @ self.defense_coef
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
        off = float(design.offense_onehot[row] @ self.offense_coef)
        deff = float(design.defense_onehot[row] @ self.defense_coef)
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


def _assemble(design: DesignMatrices) -> FloatArray:
    """A = [context | +offense_onehot | -defense_onehot]."""

    return np.concatenate(
        [design.context, design.offense_onehot, -design.defense_onehot], axis=1
    )


def _solve_ridge(
    design: DesignMatrices,
    feature_space: FeatureSpace,
    l2_player: float,
    l2_context: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    a = _assemble(design)
    w = design.weight
    n_context = feature_space.n_context
    n_players = feature_space.n_players
    penalty = np.concatenate(
        [
            np.full(n_context, l2_context),
            np.full(n_players, l2_player),
            np.full(n_players, l2_player),
        ]
    )
    aw = a * w[:, None]
    gram = a.T @ aw + np.diag(penalty)
    rhs = aw.T @ design.y
    theta = np.linalg.solve(gram, rhs)
    context_coef = theta[:n_context]
    offense_coef = theta[n_context : n_context + n_players]
    defense_coef = theta[n_context + n_players :]
    return (
        np.asarray(context_coef, dtype=np.float64),
        np.asarray(offense_coef, dtype=np.float64),
        np.asarray(defense_coef, dtype=np.float64),
    )


def _select_l2_player(
    design: DesignMatrices,
    feature_space: FeatureSpace,
    l2_context: float,
    grid: tuple[float, ...],
) -> float:
    """Pick the ridge strength by a deterministic game-blocked k-fold search."""

    unique_games = sorted(set(design.game_ids))
    fold_of_game = {g: (i % _SELECTION_FOLDS) for i, g in enumerate(unique_games)}
    fold = np.array([fold_of_game[g] for g in design.game_ids])
    a = _assemble(design)
    aw = a * design.weight[:, None]
    n_context = feature_space.n_context
    n_players = feature_space.n_players
    penalty_players = np.concatenate(
        [np.full(n_context, l2_context), np.full(2 * n_players, 1.0)]
    )
    penalty_context = np.zeros(a.shape[1])
    penalty_context[:n_context] = l2_context

    best_l2, best_err = grid[0], np.inf
    for l2 in grid:
        penalty = penalty_context + l2 * (penalty_players > l2_context)
        errors: list[float] = []
        for f in range(_SELECTION_FOLDS):
            tr = fold != f
            va = ~tr
            if not va.any() or not tr.any():
                continue
            gram = a[tr].T @ aw[tr] + np.diag(penalty)
            theta = np.linalg.solve(gram, aw[tr].T @ design.y[tr])
            pred = a[va] @ theta
            resid = design.y[va] - pred
            errors.append(
                float(np.sqrt(np.average(resid**2, weights=design.weight[va])))
            )
        mean_err = float(np.mean(errors)) if errors else np.inf
        if mean_err < best_err:
            best_l2, best_err = l2, mean_err
    return best_l2


def _mask_design(design: DesignMatrices, mask: NDArray[np.bool_]) -> DesignMatrices:
    keep = np.flatnonzero(mask)
    return DesignMatrices(
        context=design.context[keep],
        offense_onehot=design.offense_onehot[keep],
        defense_onehot=design.defense_onehot[keep],
        offense_index=design.offense_index[keep],
        defense_index=design.defense_index[keep],
        y=design.y[keep],
        weight=design.weight[keep],
        game_ids=tuple(design.game_ids[i] for i in keep),
        stint_ids=tuple(design.stint_ids[i] for i in keep),
    )

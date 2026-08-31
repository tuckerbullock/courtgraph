"""Permutation-invariant low-rank player-embedding chemistry model.

Model ladder rung 5-6 (research contract 11): an explicit additive skip path
plus a low-rank *provision / need* interaction pathway,

    V(L_o, L_d, c) = alpha + sum_{i in L_o} b_off[i] - sum_{j in L_d} b_def[j]   (T)
                   + [ K(c) ]                                                    (K)
                   + [ C(L_o) - C_ref ]                                          (C)

    C(L_o) = ( sum_i p_i ) . ( sum_i n_i ) - sum_i p_i . n_i
           = sum_{i<j in L_o} ( p_i . n_j + p_j . n_i ) .

The interaction is a **sum over the offensive set**, so it is permutation
invariant by construction and defined for lineups and pairs never seen together
(only per-player ``p_i`` / ``n_i`` are needed). It is trained on the
**cross-fitted residual** of the additive baseline (research contract 13/27.4),
regularized toward zero, and zero-sum centered over a reference lineup
distribution so the reported ``C`` is a genuine surplus, not a level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
from courtgraph.chemistry.stints import Stint, StintTable, lineup_id, pair_id

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ChemistryConfig:
    """Hyper-parameters for :meth:`ChemistryModel.fit`. Deterministic given seed."""

    seed: int = 0
    rank: int = 3
    cross_fit_folds: int = 4
    als_sweeps: int = 20
    selection_folds: int = 3
    n_bootstrap: int = 8
    init_scale: float = 0.1
    interaction_l2_grid: tuple[float, ...] = (8.0, 25.0, 70.0, 200.0)
    reference_sample: int = 4000
    convergence_tol: float = 1e-7


@dataclass(frozen=True)
class LineupDecomposition:
    """A single lineup value split into the four contract quantities."""

    talent: float
    interaction: float
    context: float
    total: float
    offense_novelty: str
    unseen_offense_players: tuple[int, ...]
    unseen_defense_players: tuple[int, ...]
    interaction_lower: float = 0.0
    interaction_upper: float = 0.0
    prob_interaction_positive: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "talent": self.talent,
            "interaction": self.interaction,
            "interaction_lower": self.interaction_lower,
            "interaction_upper": self.interaction_upper,
            "prob_interaction_positive": self.prob_interaction_positive,
            "context": self.context,
            "total": self.total,
            "offense_novelty": self.offense_novelty,
            "unseen_offense_players": list(self.unseen_offense_players),
            "unseen_defense_players": list(self.unseen_defense_players),
        }


# --------------------------------------------------------------------------- #
# Low-rank interaction pathway
# --------------------------------------------------------------------------- #


@dataclass
class LowRankInteraction:
    """Provision / need embeddings fit by deterministic full-batch Adam.

    The pathway models a **standardized** target (the additive residual scaled to
    unit possession-weighted variance) so the L2 strength is scale-free; the
    fitted embeddings are mapped back to points per 100 by ``output_scale``.
    Row ``n_players`` of each matrix is the "unknown player" slot and stays at
    zero, so an unseen player contributes nothing to ``C``.
    """

    provision: FloatArray  # (n_players + 1, rank), standardized units
    need: FloatArray  # (n_players + 1, rank)
    output_scale: float  # standardized C -> points per 100
    l2: float
    sweeps: int
    final_delta: float

    @property
    def rank(self) -> int:
        return int(self.provision.shape[1])

    @property
    def n_players(self) -> int:
        return int(self.provision.shape[0] - 1)

    def _raw(self, offense_index: IntArray) -> FloatArray:
        idx = np.where(offense_index < 0, self.n_players, offense_index)
        p = self.provision[idx]  # (n, 5, rank)
        q = self.need[idx]
        p_sum = p.sum(axis=1)
        q_sum = q.sum(axis=1)
        cross = (p_sum * q_sum).sum(axis=1)
        diag = (p * q).sum(axis=(1, 2))
        return np.asarray(cross - diag, dtype=np.float64)

    def interaction(self, offense_index: IntArray) -> FloatArray:
        """Uncentered C(L_o) in points per 100 possessions."""

        return np.asarray(
            self._raw(offense_index) * self.output_scale, dtype=np.float64
        )

    def pair_surplus(self, pos_a: int, pos_b: int) -> float:
        """Same-team offensive pair surplus in points per 100 possessions."""

        pa, na = self.provision[pos_a], self.need[pos_a]
        pb, nb = self.provision[pos_b], self.need[pos_b]
        return float((pa @ nb + pb @ na) * self.output_scale)

    @classmethod
    def fit(
        cls,
        offense_index: IntArray,
        target: FloatArray,
        weight: FloatArray,
        n_players: int,
        *,
        rank: int,
        l2: float,
        seed: int,
        als_sweeps: int,
        init_scale: float,
        convergence_tol: float,
    ) -> LowRankInteraction:
        """Alternating ridge least squares.

        With ``need`` fixed, ``C(L_o) = sum_{i in L_o} provision[i] . (Nsum - N[i])``
        is linear in ``provision``, so each half-sweep is one ridge solve; the
        problem is symmetric in ``provision`` / ``need``. Closed-form, no learning
        rate, deterministic given the seed used only for the initial ``need``.
        """

        rng = np.random.default_rng(seed)
        # In training every offensive player is known, so indices are < n_players.
        idx = np.clip(offense_index, 0, n_players - 1).astype(np.int64)
        n = len(target)
        w = weight / weight.mean()
        target_mean = float(np.average(target, weights=weight))
        target_std = (
            float(np.sqrt(np.average((target - target_mean) ** 2, weights=weight)))
            or 1.0
        )
        t = (target - target_mean) / target_std

        width = n_players * rank
        buf = np.zeros((n, n_players, rank))  # reused across half-sweeps
        rows = np.arange(n)[:, None]
        eye_l2 = l2 * np.eye(width)
        wt = w * t

        provision = np.zeros((n_players, rank))
        need = rng.normal(0.0, init_scale, size=(n_players, rank))

        def half_step(other: FloatArray) -> FloatArray:
            other_g = other[idx]  # (n, 5, rank)
            coef = other_g.sum(axis=1)[:, None, :] - other_g
            buf.fill(0.0)
            buf[rows, idx, :] = coef
            flat = buf.reshape(n, width)
            gram = flat.T @ (w[:, None] * flat) + eye_l2
            rhs = flat.T @ wt
            return np.asarray(
                np.linalg.solve(gram, rhs).reshape(n_players, rank), dtype=np.float64
            )

        delta = np.inf
        sweep = 0
        while sweep < als_sweeps:
            sweep += 1
            new_provision = half_step(need)
            new_need = half_step(new_provision)
            scale = max(
                1.0, float(np.sqrt((new_provision**2).sum() + (new_need**2).sum()))
            )
            delta = float(
                np.sqrt(
                    ((new_provision - provision) ** 2).sum()
                    + ((new_need - need) ** 2).sum()
                )
                / scale
            )
            provision, need = new_provision, new_need
            if delta < convergence_tol:
                break

        pad_p = np.vstack([provision, np.zeros((1, rank))])
        pad_n = np.vstack([need, np.zeros((1, rank))])
        return cls(
            provision=pad_p,
            need=pad_n,
            output_scale=target_std,
            l2=float(l2),
            sweeps=int(sweep),
            final_delta=delta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provision": self.provision.tolist(),
            "need": self.need.tolist(),
            "output_scale": self.output_scale,
            "l2": self.l2,
            "sweeps": self.sweeps,
            "final_delta": self.final_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LowRankInteraction:
        return cls(
            provision=np.asarray(data["provision"], dtype=np.float64),
            need=np.asarray(data["need"], dtype=np.float64),
            output_scale=float(data["output_scale"]),
            l2=float(data["l2"]),
            sweeps=int(data["sweeps"]),
            final_delta=float(data["final_delta"]),
        )


# --------------------------------------------------------------------------- #
# Combined model
# --------------------------------------------------------------------------- #


@dataclass
class ChemistryModel:
    """Additive ridge skip path + low-rank interaction pathway."""

    feature_space: FeatureSpace
    additive: AdditiveRidge
    interaction: LowRankInteraction
    interaction_reference: float
    config: ChemistryConfig
    training_stints: int = 0
    training_possessions: int = 0
    seen_offense_lineups: frozenset[str] = field(default_factory=frozenset)
    seen_pairs: frozenset[str] = field(default_factory=frozenset)
    training_player_possessions: dict[int, int] = field(default_factory=dict)
    interaction_cv: dict[str, float] = field(default_factory=dict)
    interaction_ensemble: tuple[LowRankInteraction, ...] = ()
    ensemble_references: tuple[float, ...] = ()

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit(
        cls, table: StintTable, config: ChemistryConfig | None = None
    ) -> ChemistryModel:
        cfg = config or ChemistryConfig()
        table = table.sorted_chronologically()
        space = FeatureSpace.from_training(table)
        design = space.build(table)

        oof_residual = _cross_fitted_residual(design, space, cfg)
        interaction, interaction_cv = _select_interaction(
            design, oof_residual, space, cfg
        )
        additive = AdditiveRidge.fit(design, space)

        ref_index = _reference_index(space.n_players, cfg)
        reference = float(interaction.interaction(ref_index).mean())
        ensemble, ensemble_refs = _bootstrap_interactions(
            design, oof_residual, space, cfg, interaction.l2, ref_index
        )

        seen_lineups = frozenset(s.offense_lineup_id for s in table)
        seen_pairs = _seen_pairs(table)
        player_poss: dict[int, int] = {}
        for s in table:
            for pid in s.offense_player_ids:
                player_poss[pid] = player_poss.get(pid, 0) + s.offensive_possessions

        return cls(
            feature_space=space,
            additive=additive,
            interaction=interaction,
            interaction_reference=reference,
            config=cfg,
            training_stints=len(table),
            training_possessions=table.total_possessions(),
            seen_offense_lineups=seen_lineups,
            seen_pairs=seen_pairs,
            training_player_possessions=player_poss,
            interaction_cv=interaction_cv,
            interaction_ensemble=ensemble,
            ensemble_references=ensemble_refs,
        )

    # -- uncertainty ---------------------------------------------------

    @staticmethod
    def _offense_fully_known(offense_index: IntArray) -> FloatArray:
        """1.0 for rows whose whole offensive five is in the training vocabulary.

        Predictions for a lineup containing an unseen player are conservatively
        additive-only (interaction surplus C = 0) everywhere -- prediction,
        decomposition, interval, and serialized output.
        """

        return np.asarray(
            (np.asarray(offense_index) >= 0).all(axis=1), dtype=np.float64
        )

    def interaction_samples(self, offense_index: IntArray) -> FloatArray:
        """(n_replicates, n_rows) centered interaction predictions.

        Approximate block-bootstrap ensemble: games resampled with replacement,
        additive skip path and selected L2 held fixed. Not a calibrated
        posterior -- a 'how load-bearing / how stable is this?' band. Zeroed for
        rows with an unseen offensive player.
        """

        known = self._offense_fully_known(offense_index)
        if not self.interaction_ensemble:
            return self.interaction_component_from_index(offense_index)[None, :]
        rows = [
            (member.interaction(offense_index) - ref) * known
            for member, ref in zip(
                self.interaction_ensemble, self.ensemble_references, strict=True
            )
        ]
        return np.asarray(rows, dtype=np.float64)

    def interaction_component_from_index(self, offense_index: IntArray) -> FloatArray:
        raw = self.interaction.interaction(offense_index) - self.interaction_reference
        return np.asarray(
            raw * self._offense_fully_known(offense_index), dtype=np.float64
        )

    def interaction_interval(
        self, offense_ids: tuple[int, ...], quantiles: tuple[float, float] = (0.1, 0.9)
    ) -> dict[str, Any]:
        index = self.feature_space.player_index()
        if any(p not in index for p in offense_ids):
            point = 0.0
            return {
                "point": point,
                "lower": point,
                "upper": point,
                "prob_positive": 0.5,
                "method": "unseen-player-no-estimate",
            }
        idx = np.array([[index[p] for p in offense_ids]], dtype=np.int64)
        samples = self.interaction_samples(idx)[:, 0]
        point = float(self.interaction_component_from_index(idx)[0])
        # centre the ensemble spread on the point estimate so the interval and
        # P(C>0) are consistent with the reported point value.
        centred = samples - float(samples.mean()) + point
        return {
            "point": point,
            "lower": float(np.quantile(centred, quantiles[0])),
            "upper": float(np.quantile(centred, quantiles[1])),
            "prob_positive": float((centred > 0).mean()),
            "method": "approximate-block-bootstrap-ensemble",
        }

    # -- prediction ------------------------------------------------------

    def predict_additive(self, design: DesignMatrices) -> FloatArray:
        return self.additive.predict(design)

    def predict_total(self, design: DesignMatrices) -> FloatArray:
        add = self.additive.predict(design)
        inter = self.interaction_component(design)
        return np.asarray(add + inter, dtype=np.float64)

    def interaction_component(self, design: DesignMatrices) -> FloatArray:
        return self.interaction_component_from_index(design.offense_index)

    def decompose(
        self,
        offense_ids: tuple[int, ...],
        defense_ids: tuple[int, ...],
        context: dict[str, Any],
    ) -> LineupDecomposition:
        context = dict(context)
        labels = self.feature_space.season_labels
        if "season" not in context and labels:
            idx = int(context.get("season_index", len(labels) - 1))
            context["season"] = labels[min(max(idx, 0), len(labels) - 1)]
        stint = _reference_stint(offense_ids, defense_ids, context)
        design = self.feature_space.build(StintTable.from_stints([stint]))
        add = self.additive.decompose_row(design, 0)
        inter = float(self.interaction_component_from_index(design.offense_index)[0])
        index = self.feature_space.player_index()
        unseen_off = tuple(p for p in offense_ids if p not in index)
        unseen_def = tuple(p for p in defense_ids if p not in index)
        lid = lineup_id(offense_ids)
        if unseen_off:
            novelty = "unseen"
        elif lid in self.seen_offense_lineups:
            novelty = "seen"
        elif _pairs_all_seen(offense_ids, self.seen_pairs):
            novelty = "partially-seen"
        else:
            novelty = "unseen"
        interval = self.interaction_interval(offense_ids)
        return LineupDecomposition(
            talent=add.talent,
            interaction=inter,
            context=add.context,
            total=add.talent + add.context + inter,
            offense_novelty=novelty,
            unseen_offense_players=unseen_off,
            unseen_defense_players=unseen_def,
            interaction_lower=interval["lower"],
            interaction_upper=interval["upper"],
            prob_interaction_positive=interval["prob_positive"],
        )

    def lineup_support(self, offense_ids: tuple[int, ...]) -> dict[str, Any]:
        poss = [self.training_player_possessions.get(p, 0) for p in offense_ids]
        return {
            "min_player_possessions": int(min(poss)) if poss else 0,
            "median_player_possessions": int(np.median(poss)) if poss else 0,
            "lineup_seen": lineup_id(offense_ids) in self.seen_offense_lineups,
            "n_unseen_players": sum(
                1 for p in offense_ids if p not in self.feature_space.player_index()
            ),
        }

    # -- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "feature_space": self.feature_space.to_dict(),
            "additive": self.additive.to_dict(),
            "interaction": self.interaction.to_dict(),
            "interaction_reference": self.interaction_reference,
            "config": _config_to_dict(self.config),
            "training_stints": self.training_stints,
            "training_possessions": self.training_possessions,
            "seen_offense_lineups": sorted(self.seen_offense_lineups),
            "seen_pairs": sorted(self.seen_pairs),
            "training_player_possessions": {
                str(k): v for k, v in sorted(self.training_player_possessions.items())
            },
            "interaction_cv": dict(self.interaction_cv),
            "interaction_ensemble": [m.to_dict() for m in self.interaction_ensemble],
            "ensemble_references": list(self.ensemble_references),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChemistryModel:
        version = data.get("artifact_schema_version")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact_schema_version {version!r}; "
                f"this build reads {ARTIFACT_SCHEMA_VERSION}"
            )
        return cls(
            feature_space=FeatureSpace.from_dict(data["feature_space"]),
            additive=AdditiveRidge.from_dict(data["additive"]),
            interaction=LowRankInteraction.from_dict(data["interaction"]),
            interaction_reference=float(data["interaction_reference"]),
            config=_config_from_dict(data["config"]),
            training_stints=int(data.get("training_stints", 0)),
            training_possessions=int(data.get("training_possessions", 0)),
            seen_offense_lineups=frozenset(data.get("seen_offense_lineups", [])),
            seen_pairs=frozenset(data.get("seen_pairs", [])),
            training_player_possessions={
                int(k): int(v)
                for k, v in data.get("training_player_possessions", {}).items()
            },
            interaction_cv={
                k: float(v) for k, v in data.get("interaction_cv", {}).items()
            },
            interaction_ensemble=tuple(
                LowRankInteraction.from_dict(m)
                for m in data.get("interaction_ensemble", [])
            ),
            ensemble_references=tuple(
                float(x) for x in data.get("ensemble_references", [])
            ),
        )


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #


def _cross_fitted_residual(
    design: DesignMatrices, space: FeatureSpace, cfg: ChemistryConfig
) -> FloatArray:
    """Out-of-fold additive residual (research contract 27.4 cross-fitting)."""

    games = sorted(set(design.game_ids))
    fold_of_game = {g: (i % cfg.cross_fit_folds) for i, g in enumerate(games)}
    fold = np.array([fold_of_game[g] for g in design.game_ids])
    residual = np.zeros(design.n_rows, dtype=np.float64)
    for f in range(cfg.cross_fit_folds):
        train_mask = fold != f
        test_mask = ~train_mask
        if not test_mask.any() or not train_mask.any():
            residual[test_mask] = design.y[test_mask]
            continue
        sub = _mask(design, train_mask)
        model = AdditiveRidge.fit(sub, space)
        held = _mask(design, test_mask)
        residual[test_mask] = held.y - model.predict(held)
    return residual


def _select_interaction(
    design: DesignMatrices,
    oof_residual: FloatArray,
    space: FeatureSpace,
    cfg: ChemistryConfig,
) -> tuple[LowRankInteraction, dict[str, float]]:
    """Pick the interaction L2 by a game-blocked 3-fold search on the residual.

    The score is the *reduction* in weighted residual RMSE from adding the
    interaction term -- a value <= 0 means "no interaction helps", and the
    largest L2 (strongest shrinkage toward zero) is selected, so a genuinely
    signal-free dataset yields an interaction pathway that predicts ~0.
    """

    folds = max(2, cfg.selection_folds)
    games = sorted(set(design.game_ids))
    fold_of_game = {g: (i % folds) for i, g in enumerate(games)}
    fold = np.array([fold_of_game[g] for g in design.game_ids])

    def fit_l2(mask: NDArray[np.bool_], l2: float) -> LowRankInteraction:
        return LowRankInteraction.fit(
            design.offense_index[mask],
            oof_residual[mask],
            design.weight[mask],
            space.n_players,
            rank=cfg.rank,
            l2=l2,
            seed=cfg.seed,
            als_sweeps=cfg.als_sweeps,
            init_scale=cfg.init_scale,
            convergence_tol=cfg.convergence_tol,
        )

    # Score at the offensive-lineup group level: possession-weighted group-mean
    # residuals are far less noisy than single stints, which is where a small
    # chemistry signal is detectable at all (research contract 14, headline
    # metric = macro group error).
    lineup_key = np.array(
        [lineup_id(o) for o in _iter_offense_ids(design)], dtype=object
    )

    best_l2 = cfg.interaction_l2_grid[-1]
    best_gain = 0.0
    scores: dict[str, float] = {}
    for l2 in cfg.interaction_l2_grid:
        gains: list[float] = []
        for f in range(folds):
            tr = fold != f
            va = ~tr
            if not tr.any() or not va.any():
                continue
            model = fit_l2(tr, l2)
            pred = model.interaction(design.offense_index[va])
            gains.append(
                _group_rmse_gain(
                    lineup_key[va],
                    oof_residual[va],
                    pred,
                    design.weight[va],
                )
            )
        mean_gain = float(np.mean(gains)) if gains else 0.0
        scores[f"l2={l2:g}"] = mean_gain
        if mean_gain > best_gain:
            best_gain, best_l2 = mean_gain, l2

    final = fit_l2(np.ones(design.n_rows, dtype=bool), best_l2)
    scores["selected_l2"] = float(best_l2)
    scores["selected_cv_group_rmse_gain"] = float(best_gain)
    return final, scores


def _iter_offense_ids(design: DesignMatrices) -> list[tuple[int, ...]]:
    return [tuple(int(x) for x in row if x >= 0) for row in design.offense_index]


def _group_rmse_gain(
    keys: NDArray[Any],
    residual: FloatArray,
    prediction: FloatArray,
    weight: FloatArray,
    min_group: int = 2,
) -> float:
    """Weighted group-mean RMSE(residual) - RMSE(residual - prediction)."""

    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    r_s, p_s, w_s = residual[order], prediction[order], weight[order]
    base_sq: list[float] = []
    full_sq: list[float] = []
    weights: list[float] = []
    start = 0
    n = len(keys_s)
    for end in range(1, n + 1):
        if end == n or keys_s[end] != keys_s[start]:
            block = slice(start, end)
            if end - start >= min_group:
                gw = float(w_s[block].sum())
                r_mean = float(np.average(r_s[block], weights=w_s[block]))
                d_mean = float(np.average(r_s[block] - p_s[block], weights=w_s[block]))
                base_sq.append(gw * r_mean**2)
                full_sq.append(gw * d_mean**2)
                weights.append(gw)
            start = end
    if not weights:
        return 0.0
    total = sum(weights)
    base = (sum(base_sq) / total) ** 0.5
    full = (sum(full_sq) / total) ** 0.5
    return float(base - full)


def _reference_index(n_players: int, cfg: ChemistryConfig) -> IntArray:
    """Deterministic random offensive fives that define the zero-sum center."""

    rng = np.random.default_rng(cfg.seed + 991)
    return np.array(
        [
            rng.choice(n_players, size=5, replace=False)
            for _ in range(cfg.reference_sample)
        ],
        dtype=np.int64,
    )


def _bootstrap_interactions(
    design: DesignMatrices,
    oof_residual: FloatArray,
    space: FeatureSpace,
    cfg: ChemistryConfig,
    l2: float,
    ref_index: IntArray,
) -> tuple[tuple[LowRankInteraction, ...], tuple[float, ...]]:
    """Refit the interaction pathway on game-block bootstrap resamples."""

    if cfg.n_bootstrap <= 0:
        return (), ()
    games = np.array(design.game_ids)
    unique_games = np.array(sorted(set(design.game_ids)))
    rng = np.random.default_rng(cfg.seed + 4242)
    members: list[LowRankInteraction] = []
    refs: list[float] = []
    for b in range(cfg.n_bootstrap):
        drawn = rng.choice(unique_games, size=len(unique_games), replace=True)
        counts: dict[str, int] = {}
        for g in drawn:
            counts[g] = counts.get(g, 0) + 1
        multiplier = np.array([counts.get(g, 0) for g in games], dtype=np.float64)
        mask = multiplier > 0
        member = LowRankInteraction.fit(
            design.offense_index[mask],
            oof_residual[mask],
            design.weight[mask] * multiplier[mask],
            space.n_players,
            rank=cfg.rank,
            l2=l2,
            seed=cfg.seed + 1 + b,
            als_sweeps=cfg.als_sweeps,
            init_scale=cfg.init_scale,
            convergence_tol=cfg.convergence_tol,
        )
        members.append(member)
        refs.append(float(member.interaction(ref_index).mean()))
    return tuple(members), tuple(refs)


def _mask(design: DesignMatrices, mask: NDArray[np.bool_]) -> DesignMatrices:
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


def _seen_pairs(table: StintTable) -> frozenset[str]:
    pairs: set[str] = set()
    for stint in table:
        ids = stint.offense_player_ids
        for a in range(5):
            for b in range(a + 1, 5):
                pairs.add(pair_id(ids[a], ids[b]))
    return frozenset(pairs)


def _pairs_all_seen(offense_ids: tuple[int, ...], seen_pairs: frozenset[str]) -> bool:
    for a in range(len(offense_ids)):
        for b in range(a + 1, len(offense_ids)):
            if pair_id(offense_ids[a], offense_ids[b]) not in seen_pairs:
                return False
    return True


def _reference_stint(
    offense_ids: tuple[int, ...],
    defense_ids: tuple[int, ...],
    context: dict[str, Any],
) -> Stint:
    return Stint(
        stint_id="query",
        game_id="query",
        game_date=str(context.get("game_date", "2000-01-01")),
        season=str(context.get("season", "S1")),
        season_index=int(context.get("season_index", 0)),
        period=int(context.get("period", 1)),
        start_time_seconds=0.0,
        offense_team_id=1,
        defense_team_id=2,
        offense_player_ids=tuple(sorted(offense_ids)),  # type: ignore[arg-type]
        defense_player_ids=tuple(sorted(defense_ids)),  # type: ignore[arg-type]
        offensive_possessions=int(context.get("offensive_possessions", 100)),
        points_scored=int(context.get("points_scored", 100)),
        home_offense=bool(context.get("home_offense", True)),
        score_margin_offense=int(context.get("score_margin_offense", 0)),
        playoff=bool(context.get("playoff", False)),
        days_rest_offense=int(context.get("days_rest_offense", 1)),
        garbage_time_weight=float(context.get("garbage_time_weight", 1.0)),
        source="query",
    )


def _config_to_dict(cfg: ChemistryConfig) -> dict[str, Any]:
    return {
        "seed": cfg.seed,
        "rank": cfg.rank,
        "cross_fit_folds": cfg.cross_fit_folds,
        "als_sweeps": cfg.als_sweeps,
        "selection_folds": cfg.selection_folds,
        "n_bootstrap": cfg.n_bootstrap,
        "init_scale": cfg.init_scale,
        "interaction_l2_grid": list(cfg.interaction_l2_grid),
        "reference_sample": cfg.reference_sample,
        "convergence_tol": cfg.convergence_tol,
    }


def _config_from_dict(data: dict[str, Any]) -> ChemistryConfig:
    return ChemistryConfig(
        seed=int(data["seed"]),
        rank=int(data["rank"]),
        cross_fit_folds=int(data["cross_fit_folds"]),
        als_sweeps=int(data["als_sweeps"]),
        selection_folds=int(data.get("selection_folds", 3)),
        n_bootstrap=int(data.get("n_bootstrap", 8)),
        init_scale=float(data["init_scale"]),
        interaction_l2_grid=tuple(float(x) for x in data["interaction_l2_grid"]),
        reference_sample=int(data["reference_sample"]),
        convergence_tol=float(data["convergence_tol"]),
    )

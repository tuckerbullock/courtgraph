"""Leakage-safe evaluation: the additive baseline vs the full chemistry model.

For each holdout this reports, in points per 100 possessions:

* RMSE / MAE against the **realized** stint outcome, possession-weighted
  (micro) and equal-weight per held-out group (macro) -- the research-contract
  headline is the macro number on unseen / sparse groups;
* RMSE / MAE against the **known synthetic generative truth** when it is
  supplied (the model-recovery view -- master plan 33.3);
* an **approximate** block-bootstrap interval on the additive-minus-full error
  difference (games are the resampling block; clearly labeled approximate);
* support / exposure per group and a novelty class (seen / partially-seen /
  unseen).

Nothing here is specific to synthetic data except the optional ``truth`` hook,
so the same evaluation runs unchanged on real NBA stints (with ``truth=None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.chemistry_model import ChemistryConfig, ChemistryModel
from courtgraph.chemistry.features import DesignMatrices
from courtgraph.chemistry.splits import SplitManifest, novelty_of_lineup, verify_split
from courtgraph.chemistry.stints import StintTable, lineup_id, pair_id
from courtgraph.chemistry.synthetic import GroundTruth

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GroupRow:
    """One held-out group (lineup, pair, or the whole test set)."""

    group_id: str
    novelty: str
    test_stints: int
    test_possessions: int
    realized_value: float
    additive_prediction: float
    full_prediction: float
    truth_value: float | None
    interaction_mean: float
    interaction_sd: float
    prob_interaction_positive: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "novelty": self.novelty,
            "test_stints": self.test_stints,
            "test_possessions": self.test_possessions,
            "realized_value": self.realized_value,
            "additive_prediction": self.additive_prediction,
            "full_prediction": self.full_prediction,
            "truth_value": self.truth_value,
            "interaction_mean": self.interaction_mean,
            "interaction_sd": self.interaction_sd,
            "prob_interaction_positive": self.prob_interaction_positive,
        }


@dataclass(frozen=True)
class HoldoutResult:
    kind: str
    n_train_stints: int
    n_test_stints: int
    n_test_groups: int
    selected_interaction_l2: float
    leakage_violations: tuple[str, ...]
    metrics: dict[str, float]
    approximate_delta_interval: dict[str, float]
    groups: tuple[GroupRow, ...]

    @property
    def headline_improvement_pct(self) -> float:
        return self.metrics.get("headline_improvement_pct", 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_train_stints": self.n_train_stints,
            "n_test_stints": self.n_test_stints,
            "n_test_groups": self.n_test_groups,
            "selected_interaction_l2": self.selected_interaction_l2,
            "leakage_violations": list(self.leakage_violations),
            "metrics": dict(self.metrics),
            "approximate_delta_interval": dict(self.approximate_delta_interval),
            "groups": [g.as_dict() for g in self.groups],
        }


@dataclass(frozen=True)
class EvaluationSummary:
    dataset: dict[str, Any]
    holdouts: tuple[HoldoutResult, ...]
    decomposition_examples: tuple[dict[str, Any], ...] = ()
    recovery: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": dict(self.dataset),
            "holdouts": [h.as_dict() for h in self.holdouts],
            "decomposition_examples": [dict(d) for d in self.decomposition_examples],
            "recovery": dict(self.recovery),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #


def _rmse(
    pred: FloatArray, target: FloatArray, weight: FloatArray | None = None
) -> float:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.average(err**2, weights=weight)))


def _mae(
    pred: FloatArray, target: FloatArray, weight: FloatArray | None = None
) -> float:
    err = np.abs(
        np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )
    return float(np.average(err, weights=weight))


def _group_index(
    test_table: StintTable, manifest: SplitManifest
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    if manifest.kind == "unseen_lineup":
        for i, stint in enumerate(test_table):
            groups.setdefault(stint.offense_lineup_id, []).append(i)
    elif manifest.kind == "unseen_pair":
        held = set(manifest.held_out_pairs)
        for i, stint in enumerate(test_table):
            ids = stint.offense_player_ids
            for a in range(5):
                for b in range(a + 1, 5):
                    key = pair_id(ids[a], ids[b])
                    if key in held:
                        groups.setdefault(key, []).append(i)
    else:  # chronological -- bucket by season so "macro" is still meaningful
        for i, stint in enumerate(test_table):
            groups.setdefault(stint.season, []).append(i)
    return groups


# --------------------------------------------------------------------------- #
# Per-holdout evaluation
# --------------------------------------------------------------------------- #


def evaluate_holdout(
    table: StintTable,
    manifest: SplitManifest,
    *,
    config: ChemistryConfig | None = None,
    truth: GroundTruth | None = None,
    model: ChemistryModel | None = None,
    bootstrap_blocks: int = 250,
) -> HoldoutResult:
    violations = tuple(verify_split(table, manifest))
    train_table = manifest.train_table(table)
    test_table = manifest.test_table(table)
    if model is None:
        model = ChemistryModel.fit(train_table, config)
    space = model.feature_space
    test_design = space.build(test_table)

    additive = model.predict_additive(test_design)
    full = model.predict_total(test_design)
    realized = test_design.y
    weight = test_design.weight

    truth_values = _true_values(truth, test_table) if truth is not None else None

    point_interaction = model.interaction_component(test_design)
    samples = model.interaction_samples(test_design.offense_index)  # (B, n_test)
    # centre each bootstrap member on the point model so the group-level samples
    # below are consistent with the full prediction's interaction term.
    centred_members = samples - samples.mean(axis=0, keepdims=True) + point_interaction

    groups = _group_index(test_table, manifest)
    group_rows = _build_group_rows(
        test_table,
        train_table,
        manifest,
        groups,
        additive,
        full,
        realized,
        weight,
        truth_values,
        centred_members,
    )

    metrics = _holdout_metrics(
        additive, full, realized, weight, truth_values, group_rows
    )
    if truth is not None:
        true_interaction = _true_interaction(truth, test_table)
        pred_interaction = model.interaction_component(test_design)
        if pred_interaction.std() > 1e-9 and true_interaction.std() > 1e-9:
            metrics["interaction_recovery_corr"] = float(
                np.corrcoef(pred_interaction, true_interaction)[0, 1]
            )
        else:
            metrics["interaction_recovery_corr"] = 0.0
        metrics["predicted_interaction_sd"] = float(pred_interaction.std())
    delta_ci = _delta_interval(
        test_design,
        additive,
        full,
        realized,
        truth_values,
        blocks=bootstrap_blocks,
        seed=(config or ChemistryConfig()).seed,
    )

    return HoldoutResult(
        kind=manifest.kind,
        n_train_stints=len(train_table),
        n_test_stints=len(test_table),
        n_test_groups=len(group_rows),
        selected_interaction_l2=model.interaction.l2,
        leakage_violations=violations,
        metrics=metrics,
        approximate_delta_interval=delta_ci,
        groups=tuple(group_rows),
    )


def _true_values(truth: GroundTruth, test_table: StintTable) -> FloatArray:
    out = np.zeros(len(test_table))
    for i, stint in enumerate(test_table):
        context = stint.context_vector()
        context["season_index"] = float(stint.season_index)
        out[i] = truth.lineup_value(
            stint.offense_player_ids, stint.defense_player_ids, context
        )
    return out


def _true_interaction(truth: GroundTruth, test_table: StintTable) -> FloatArray:
    reference = truth.mean_lineup_interaction(2000)
    return np.array(
        [truth.lineup_interaction(s.offense_player_ids) - reference for s in test_table]
    )


def _build_group_rows(
    test_table: StintTable,
    train_table: StintTable,
    manifest: SplitManifest,
    groups: dict[str, list[int]],
    additive: FloatArray,
    full: FloatArray,
    realized: FloatArray,
    weight: FloatArray,
    truth_values: FloatArray | None,
    centred_members: FloatArray,  # (n_bootstrap, n_test) centred interaction
) -> list[GroupRow]:
    stints = list(test_table)
    rows: list[GroupRow] = []
    for group_id, idx_list in sorted(groups.items()):
        idx = np.array(idx_list)
        w = weight[idx]
        total_w = float(w.sum())
        example = stints[idx_list[0]]
        novelty = novelty_of_lineup(train_table, example.offense_player_ids)
        # group-level uncertainty: each bootstrap member's possession-weighted
        # group prediction first, then the mean / SD / P(C>0) across members.
        member_group_c = np.average(centred_members[:, idx], axis=1, weights=w)
        rows.append(
            GroupRow(
                group_id=group_id,
                novelty=novelty,
                test_stints=len(idx_list),
                test_possessions=int(total_w),
                realized_value=float(np.average(realized[idx], weights=w)),
                additive_prediction=float(np.average(additive[idx], weights=w)),
                full_prediction=float(np.average(full[idx], weights=w)),
                truth_value=(
                    float(np.average(truth_values[idx], weights=w))
                    if truth_values is not None
                    else None
                ),
                interaction_mean=float(member_group_c.mean()),
                interaction_sd=float(member_group_c.std()),
                prob_interaction_positive=float((member_group_c > 0).mean()),
            )
        )
    return rows


def _holdout_metrics(
    additive: FloatArray,
    full: FloatArray,
    realized: FloatArray,
    weight: FloatArray,
    truth_values: FloatArray | None,
    group_rows: list[GroupRow],
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "additive_rmse_realized_micro": _rmse(additive, realized, weight),
        "full_rmse_realized_micro": _rmse(full, realized, weight),
        "additive_mae_realized_micro": _mae(additive, realized, weight),
        "full_mae_realized_micro": _mae(full, realized, weight),
    }
    g_realized = np.array([g.realized_value for g in group_rows])
    g_add = np.array([g.additive_prediction for g in group_rows])
    g_full = np.array([g.full_prediction for g in group_rows])
    if len(group_rows):
        metrics["additive_rmse_realized_macro"] = _rmse(g_add, g_realized)
        metrics["full_rmse_realized_macro"] = _rmse(g_full, g_realized)
        metrics["additive_mae_realized_macro"] = _mae(g_add, g_realized)
        metrics["full_mae_realized_macro"] = _mae(g_full, g_realized)

    if truth_values is not None:
        metrics["additive_rmse_truth_micro"] = _rmse(additive, truth_values, weight)
        metrics["full_rmse_truth_micro"] = _rmse(full, truth_values, weight)
        metrics["additive_mae_truth_micro"] = _mae(additive, truth_values, weight)
        metrics["full_mae_truth_micro"] = _mae(full, truth_values, weight)
        if len(group_rows):
            g_truth = np.array([g.truth_value for g in group_rows], dtype=np.float64)
            metrics["additive_rmse_truth_macro"] = _rmse(g_add, g_truth)
            metrics["full_rmse_truth_macro"] = _rmse(g_full, g_truth)

    base_key = "rmse_truth_macro" if truth_values is not None else "rmse_realized_macro"
    add_v = metrics.get(f"additive_{base_key}")
    full_v = metrics.get(f"full_{base_key}")
    if add_v and full_v is not None and add_v > 0:
        metrics["headline_metric_is_truth"] = float(truth_values is not None)
        metrics["headline_improvement_pct"] = 100.0 * (1.0 - full_v / add_v)
    return metrics


def _delta_interval(
    test_design: DesignMatrices,
    additive: FloatArray,
    full: FloatArray,
    realized: FloatArray,
    truth_values: FloatArray | None,
    *,
    blocks: int,
    seed: int,
) -> dict[str, float]:
    """Approximate block-bootstrap CI on RMSE(additive) - RMSE(full)."""

    target = truth_values if truth_values is not None else realized
    games = np.array(test_design.game_ids)
    unique = np.array(sorted(set(test_design.game_ids)))
    if len(unique) < 2:
        return {"lower80": 0.0, "point": 0.0, "upper80": 0.0, "prob_full_better": 0.0}
    game_rows = {g: np.flatnonzero(games == g) for g in unique}
    rng = np.random.default_rng(seed + 7)
    diffs = np.zeros(blocks)
    add_err = (additive - target) ** 2
    full_err = (full - target) ** 2
    w = test_design.weight
    for b in range(blocks):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([game_rows[g] for g in drawn])
        rmse_add = np.sqrt(np.average(add_err[rows], weights=w[rows]))
        rmse_full = np.sqrt(np.average(full_err[rows], weights=w[rows]))
        diffs[b] = rmse_add - rmse_full
    point = float(
        np.sqrt(np.average(add_err, weights=w))
        - np.sqrt(np.average(full_err, weights=w))
    )
    return {
        "lower80": float(np.quantile(diffs, 0.1)),
        "point": point,
        "upper80": float(np.quantile(diffs, 0.9)),
        "prob_full_better": float((diffs > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Whole-suite evaluation
# --------------------------------------------------------------------------- #


def evaluate_suite(
    table: StintTable,
    manifests: dict[str, SplitManifest],
    *,
    config: ChemistryConfig | None = None,
    truth: GroundTruth | None = None,
    full_model: ChemistryModel | None = None,
) -> EvaluationSummary:
    holdouts = tuple(
        evaluate_holdout(table, manifests[k], config=config, truth=truth)
        for k in ("chronological", "unseen_pair", "unseen_lineup")
        if k in manifests
    )
    dataset = {
        "stints": len(table),
        "possessions": table.total_possessions(),
        "players": len(table.player_ids()),
        "seasons": list(table.season_order()),
        "synthetic": all(s.source == "synthetic" for s in table),
    }
    recovery: dict[str, float] = {}
    if truth is not None:
        model = full_model or ChemistryModel.fit(table, config)
        recovery = _recovery_metrics(model, truth)
    notes = (
        "Synthetic demonstration data. Chemistry is a small residual signal; the "
        "realized-outcome micro metric is dominated by possession noise, so the "
        "recoverable signal is visible mainly in the truth-referenced and macro "
        "(group-level) views.",
        "Bootstrap intervals are approximate: block resampling of games, point "
        "model additive fit held fixed.",
    )
    return EvaluationSummary(
        dataset=dataset, holdouts=holdouts, recovery=recovery, notes=notes
    )


def _recovery_metrics(model: ChemistryModel, truth: GroundTruth) -> dict[str, float]:
    """Parameter-level recovery of the known latent generative structure.

    Pair and lineup recovery are measured only over **well-supported** players
    (those with real training exposure); the low-rank pathway makes no claim
    about arbitrary never-co-observed groups at this data scale.
    """

    space = model.feature_space
    index = space.player_index()
    truth_index = {p: i for i, p in enumerate(truth.player_ids)}

    truth_off = np.array([truth.off_talent[truth_index[p]] for p in space.player_ids])
    model_off = np.array([model.additive.talent_of(p)[0] for p in space.player_ids])
    truth_def = np.array([truth.def_talent[truth_index[p]] for p in space.player_ids])
    model_def = np.array([model.additive.talent_of(p)[1] for p in space.player_ids])

    supported = [
        p
        for p in space.player_ids
        if model.training_player_possessions.get(p, 0) >= 400
    ]
    pairs = [
        (supported[a], supported[b])
        for a in range(len(supported))
        for b in range(a + 1, len(supported))
        if (a * 7 + b) % 11 == 0
    ]
    metrics = {
        "offensive_talent_corr": float(np.corrcoef(truth_off, model_off)[0, 1]),
        "defensive_talent_corr": float(np.corrcoef(truth_def, model_def)[0, 1]),
        "supported_players": len(supported),
    }
    if len(pairs) >= 8:
        truth_pair = np.array([truth.pair_surplus(a, b) for a, b in pairs])
        model_pair = np.array(
            [model.interaction.pair_surplus(index[a], index[b]) for a, b in pairs]
        )
        if model_pair.std() > 1e-9:
            metrics["pair_surplus_corr"] = float(
                np.corrcoef(truth_pair, model_pair)[0, 1]
            )
        else:
            metrics["pair_surplus_corr"] = 0.0
    return metrics


def decomposition_examples(
    model: ChemistryModel,
    truth: GroundTruth | None,
    train_table: StintTable,
    *,
    count: int = 6,
    seed: int = 3,
) -> list[dict[str, Any]]:
    """A few unseen offensive lineups with the model's T / C / K / total split."""

    ids = list(model.feature_space.player_ids)
    seen = {s.offense_lineup_id for s in train_table}
    rng = np.random.default_rng(seed)
    defense = tuple(sorted(int(ids[k]) for k in rng.choice(len(ids), 5, replace=False)))
    context = {
        "home_offense": True,
        "score_margin_offense": 0,
        "period": 2,
        "playoff": False,
        "days_rest_offense": 1,
        "garbage_time_weight": 1.0,
        "season_index": len(train_table.season_order()) - 1,
    }
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < count and attempts < count * 40:
        attempts += 1
        offense = tuple(
            sorted(int(ids[k]) for k in rng.choice(len(ids), 5, replace=False))
        )
        if lineup_id(offense) in seen or set(offense) & set(defense):
            continue
        decomp = model.decompose(offense, defense, context)
        support = model.lineup_support(offense)
        row: dict[str, Any] = {
            "offense": list(offense),
            "decomposition": decomp.as_dict(),
            "support": support,
        }
        if truth is not None:
            row["truth"] = truth.decomposition(
                offense, defense, context, truth.mean_lineup_interaction(1500)
            )
        out.append(row)
    return out

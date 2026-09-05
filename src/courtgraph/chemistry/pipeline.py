"""End-to-end wiring for the ``courtgraph`` chemistry commands.

Keeps the CLI thin: generate or load stints, build leakage-safe splits,
evaluate the additive baseline against the full model, fit and serialize a
model, and produce the decomposition for a single lineup.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from courtgraph.chemistry.baseline_ladder import LadderComparison
    from courtgraph.chemistry.confirm import ConfirmationResult
    from courtgraph.chemistry.mechanistic import MechanisticComparison
    from courtgraph.chemistry.phase_b_eval import PhaseBComparison
    from courtgraph.chemistry.player_lift_eval import PlayerLiftComparison
    from courtgraph.chemistry.redundancy_eval import RedundancyComparison
    from courtgraph.chemistry.role_eval import RoleComparison
    from courtgraph.chemistry.transaction_backtest import TransactionBacktest
    from courtgraph.chemistry.transport import TransportResult

from courtgraph.chemistry.artifact import load_model, save_model
from courtgraph.chemistry.chemistry_model import (
    ChemistryConfig,
    ChemistryModel,
    LineupDecomposition,
)
from courtgraph.chemistry.evaluate import (
    EvaluationSummary,
    decomposition_examples,
    evaluate_suite,
)
from courtgraph.chemistry.splits import make_all_splits
from courtgraph.chemistry.stints import LINEUP_SIZE, read_stints, write_stints
from courtgraph.chemistry.synthetic import GroundTruth, SyntheticConfig, generate

DEFAULT_CONTEXT: dict[str, Any] = {
    "home_offense": True,
    "score_margin_offense": 0,
    "period": 2,
    "playoff": False,
    "days_rest_offense": 1,
    "garbage_time_weight": 1.0,
    "season_index": 0,
}


@dataclass(frozen=True)
class DemoResult:
    summary: EvaluationSummary
    model_path: Path
    report_path: Path | None
    stints_path: Path
    examples: tuple[dict[str, Any], ...]

    def headline(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for holdout in self.summary.holdouts:
            rows.append(
                {
                    "holdout": holdout.kind,
                    "test_groups": holdout.n_test_groups,
                    "additive_macro_rmse": holdout.metrics.get(
                        "additive_rmse_truth_macro",
                        holdout.metrics.get("additive_rmse_realized_macro", 0.0),
                    ),
                    "full_macro_rmse": holdout.metrics.get(
                        "full_rmse_truth_macro",
                        holdout.metrics.get("full_rmse_realized_macro", 0.0),
                    ),
                    "improvement_pct": holdout.headline_improvement_pct,
                    "leakage_violations": len(holdout.leakage_violations),
                }
            )
        return rows


def run_demo(
    *,
    out_dir: str | Path,
    report_path: str | Path | None = None,
    seed: int = 20260830,
    n_boot: int = 8,
    synthetic_config: SyntheticConfig | None = None,
    chemistry_config: ChemistryConfig | None = None,
) -> DemoResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if n_boot < 0:
        raise ValueError("--bootstrap must be >= 0")
    syn_cfg = synthetic_config or SyntheticConfig(seed=seed)
    base_cfg = chemistry_config or ChemistryConfig(
        seed=0, rank=syn_cfg.embedding_rank + 1
    )
    # --bootstrap always sets the interaction-ensemble size.
    chem_cfg = replace(base_cfg, n_bootstrap=n_boot)
    table, truth = generate(syn_cfg)

    stints_path = out / "demo_stints.jsonl"
    write_stints(table, stints_path)

    splits = make_all_splits(table)
    for manifest in splits.values():
        manifest.write(out / f"split_{manifest.kind}.json")

    model = ChemistryModel.fit(table, chem_cfg)
    summary = evaluate_suite(
        table, splits, config=chem_cfg, truth=truth, full_model=model
    )

    model_path = save_model(
        model,
        out / "demo_model.json",
        metadata={
            "source": "synthetic-demo",
            "seed": seed,
            "synthetic_config": _syn_config_dict(syn_cfg),
            "note": (
                "Trained on synthetic demonstration stints. Not a basketball model."
            ),
        },
    )

    examples = tuple(decomposition_examples(model, truth, table, count=6))
    resolved_report: Path | None = None
    if report_path is not None:
        from courtgraph.chemistry.report import write_report

        resolved_report = write_report(
            Path(report_path),
            summary=summary,
            model=model,
            truth=truth,
            examples=examples,
            seed=seed,
        )

    return DemoResult(
        summary=summary,
        model_path=model_path,
        report_path=resolved_report,
        stints_path=stints_path,
        examples=examples,
    )


def fit_model_file(
    input_path: str | Path,
    model_out: str | Path,
    *,
    config: ChemistryConfig | None = None,
) -> tuple[ChemistryModel, Path]:
    table = read_stints(input_path)
    if len(table) < 50:
        raise ValueError(
            f"{input_path}: only {len(table)} stints; need a substantially larger "
            "table to fit a chemistry model"
        )
    model = ChemistryModel.fit(table, config)
    path = save_model(
        model,
        model_out,
        metadata={
            "source": str(input_path),
            "stints": len(table),
            "possessions": table.total_possessions(),
        },
    )
    return model, path


def evaluate_model_file(
    input_path: str | Path,
    *,
    config: ChemistryConfig | None = None,
) -> EvaluationSummary:
    table = read_stints(input_path)
    splits = make_all_splits(table)
    return evaluate_suite(table, splits, config=config, truth=None)


def run_baselines(
    input_path: str | Path,
    *,
    seed: int = 0,
    n_boot: int = 150,
    rung4: bool = False,
) -> LadderComparison:
    """Fit rung 2 (additive RAPM) and rung 3 (hierarchical EB) -- and, when
    ``rung4`` is set, rung 4 (explicit teammate-pair interaction) -- and compare
    point accuracy + interval calibration on the leakage-safe holdouts."""

    from courtgraph.chemistry.baseline_ladder import compare_rungs
    from courtgraph.chemistry.pair_interaction import PairHierarchicalConfig

    table = read_stints(input_path)
    if len(table) < 50:
        raise ValueError(
            f"{input_path}: only {len(table)} stints; need a substantially larger "
            "table to compare the baselines"
        )
    splits = make_all_splits(table)
    return compare_rungs(
        table,
        splits,
        seed=seed,
        n_boot=n_boot,
        rung4_config=PairHierarchicalConfig() if rung4 else None,
    )


def run_transport(
    train_path: str | Path,
    test_path: str | Path,
    *,
    seed: int = 0,
    n_boot: int = 150,
    rung4: bool = False,
) -> TransportResult:
    """Fit rungs 2/3 (and, when ``rung4`` is set, rung 4) on ``train_path`` -- the
    regular season -- and evaluate them on the held-out ``test_path`` -- the
    playoffs. See :mod:`courtgraph.chemistry.transport`."""

    from courtgraph.chemistry.pair_interaction import PairHierarchicalConfig
    from courtgraph.chemistry.transport import evaluate_transport

    train_table = read_stints(train_path)
    test_table = read_stints(test_path)
    if len(train_table) < 50:
        raise ValueError(
            f"{train_path}: only {len(train_table)} training stints; need a "
            "substantially larger table"
        )
    if len(test_table) < 10:
        raise ValueError(f"{test_path}: only {len(test_table)} test stints")
    return evaluate_transport(
        train_table,
        test_table,
        seed=seed,
        n_boot=n_boot,
        rung4_config=PairHierarchicalConfig() if rung4 else None,
    )


def run_roles(
    stints_path: str | Path,
    profiles_path: str | Path,
    *,
    n_clusters: int = 5,
    seed: int = 0,
    n_boot: int = 120,
) -> RoleComparison:
    """Fit role clusters from ``profiles_path``, then compare the
    role-conditioned interaction model against rungs 2/3 and a permuted-role
    placebo on the leakage-safe holdouts. See
    :mod:`courtgraph.chemistry.role_eval`."""

    from courtgraph.chemistry.role_eval import evaluate_role_interaction
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(
            f"{stints_path}: only {len(table)} stints; need a substantially "
            "larger table"
        )
    profiles = read_player_profiles(profiles_path)
    clustering = fit_role_clusters(profiles, n_clusters=n_clusters, seed=seed)
    splits = make_all_splits(table)
    return evaluate_role_interaction(
        table, splits, clustering, seed=seed, n_boot=n_boot
    )


def run_mechanistic(
    stints_path: str | Path,
    snapshot_dir: str | Path,
    profiles_path: str | Path,
    *,
    outcome: str = "pts_per_shot",
    min_fga: int = 3,
    n_clusters: int = 5,
    seed: int = 0,
    n_boot: int = 120,
) -> MechanisticComparison:
    """Attribute shots to stints, then compare rung 2 / rung 3 / role /
    permuted-role-placebo on a mechanistic outcome (shot quality or shot mix).
    See :mod:`courtgraph.chemistry.mechanistic`."""

    from courtgraph.chemistry.mechanistic import EVENT_OUTCOMES, evaluate_mechanistic
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters
    from courtgraph.features.stint_events import attribute_play_events
    from courtgraph.features.stint_shots import attribute_shots
    from courtgraph.ingest.snapshot import load_snapshot

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(f"{stints_path}: only {len(table)} stints; need more")
    snapshot = load_snapshot(snapshot_dir)
    attribution = (
        attribute_play_events(snapshot, table)
        if outcome in EVENT_OUTCOMES
        else attribute_shots(snapshot, table)
    )
    clustering = fit_role_clusters(
        read_player_profiles(profiles_path), n_clusters=n_clusters, seed=seed
    )
    return evaluate_mechanistic(
        table,
        attribution,
        clustering,
        outcome=outcome,
        min_fga=min_fga,
        seed=seed,
        n_boot=n_boot,
    )


def run_redundancy(
    stints_path: str | Path,
    profiles_path: str | Path,
    *,
    n_clusters: int = 5,
    seed: int = 0,
    n_boot: int = 120,
) -> RedundancyComparison:
    """Fit the concentration-feature (redundancy / anti-synergy) model and
    compare it against rungs 2/3 and a permuted-role placebo on the
    leakage-safe holdouts. See :mod:`courtgraph.chemistry.redundancy_eval`."""

    from courtgraph.chemistry.redundancy_eval import evaluate_redundancy
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(f"{stints_path}: only {len(table)} stints; need more")
    clustering = fit_role_clusters(
        read_player_profiles(profiles_path), n_clusters=n_clusters, seed=seed
    )
    splits = make_all_splits(table)
    return evaluate_redundancy(table, splits, clustering, seed=seed, n_boot=n_boot)


def run_player_lift(
    stints_path: str | Path,
    *,
    seed: int = 0,
    n_boot: int = 120,
    side: str = "offense",
) -> PlayerLiftComparison:
    """Master plan §45 Phase A -- fit one EM-shrunk lift scalar per player on
    the rung-3 frame and compare against rungs 2/3 and a player-permutation
    placebo on the leakage-safe holdouts. ``side="defense"`` runs the defensive
    analog (lift keyed on the defensive lineup).
    :mod:`courtgraph.chemistry.player_lift_eval`."""

    from courtgraph.chemistry.player_lift_eval import evaluate_player_lift

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(f"{stints_path}: only {len(table)} stints; need more")
    splits = make_all_splits(table)
    return evaluate_player_lift(table, splits, seed=seed, n_boot=n_boot, side=side)


def run_phase_b(
    stints_path: str | Path,
    production_path: str | Path,
    *,
    assist_credit: float = 0.5,
    seed: int = 0,
    n_boot: int = 3000,
) -> PhaseBComparison:
    """Master plan §45 Phase B -- the per-player-production lift model, base-only
    vs base + pooled lift vs a giver-shuffle placebo on a chronological holdout.
    :mod:`courtgraph.chemistry.phase_b_eval`."""

    from courtgraph.chemistry.phase_b import PhaseBConfig
    from courtgraph.chemistry.phase_b_eval import evaluate_phase_b
    from courtgraph.features.player_production import read_production

    table = read_stints(stints_path)
    production = read_production(production_path)
    if len(production) < 5000:
        raise ValueError(
            f"{production_path}: only {len(production)} rows; need a larger table"
        )
    return evaluate_phase_b(
        table,
        production,
        seed=seed,
        n_boot=n_boot,
        config=PhaseBConfig(assist_credit=assist_credit),
    )


def run_transaction_backtest(
    stints_path: str | Path,
    *,
    min_poss_each_side: int = 500,
    n_phantom: int | None = None,
    n_boot: int = 3000,
    seed: int = 0,
) -> TransactionBacktest:
    """Contract T4 -- clean cross-season team switches as natural experiments.
    :mod:`courtgraph.chemistry.transaction_backtest`."""

    from courtgraph.chemistry.transaction_backtest import run_backtest

    table = read_stints(stints_path)
    if len(table.season_order()) < 2:
        raise ValueError(f"{stints_path}: need >= 2 seasons for a transaction cohort")
    return run_backtest(
        table,
        min_poss_each_side=min_poss_each_side,
        n_phantom=n_phantom,
        n_boot=n_boot,
        seed=seed,
    )


def run_confirmation_file(
    stints_path: str | Path,
    profiles_path: str | Path,
    snapshot_dir: str | Path,
    *,
    k_values: tuple[int, ...] = (3, 5, 7),
    outcomes: tuple[str, ...] = ("three_share",),
    n_lineups: int = 120,
    n_boot: int = 2000,
    min_fga: int = 3,
) -> ConfirmationResult:
    """Re-run the three interaction positives with a wider ``unseen_lineup``
    holdout, a K sweep, and bootstrap CIs on the (baseline - model) delta.
    See :mod:`courtgraph.chemistry.confirm`."""

    from courtgraph.chemistry.confirm import run_confirmation
    from courtgraph.chemistry.mechanistic import EVENT_OUTCOMES
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters
    from courtgraph.features.stint_events import attribute_play_events
    from courtgraph.features.stint_shots import attribute_shots
    from courtgraph.ingest.snapshot import load_snapshot

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(f"{stints_path}: only {len(table)} stints; need more")
    profiles = read_player_profiles(profiles_path)
    clustering_by_k = {
        k: fit_role_clusters(profiles, n_clusters=k, seed=0) for k in k_values
    }
    snapshot = load_snapshot(snapshot_dir)
    needs_shots = any(o not in EVENT_OUTCOMES for o in outcomes)
    needs_events = any(o in EVENT_OUTCOMES for o in outcomes)
    attribution = attribute_shots(snapshot, table) if needs_shots else None
    event_attribution = attribute_play_events(snapshot, table) if needs_events else None
    return run_confirmation(
        table,
        clustering_by_k,
        attribution,
        event_attribution=event_attribution,
        outcomes=outcomes,
        n_lineups=n_lineups,
        n_boot=n_boot,
        min_fga=min_fga,
    )


@dataclass(frozen=True)
class PredictionResult:
    offense: tuple[int, ...]
    defense: tuple[int, ...]
    context: dict[str, Any]
    decomposition: LineupDecomposition
    support: dict[str, Any]
    interaction_interval: dict[str, Any]
    model_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "offense": list(self.offense),
            "defense": list(self.defense),
            "context": dict(self.context),
            "decomposition": self.decomposition.as_dict(),
            "support": dict(self.support),
            "interaction_interval": dict(self.interaction_interval),
            "model_metadata": dict(self.model_metadata),
        }


def predict_lineup(
    model_path: str | Path,
    offense: list[int],
    defense: list[int],
    *,
    context: dict[str, Any] | None = None,
) -> PredictionResult:
    if len(offense) != LINEUP_SIZE or len(defense) != LINEUP_SIZE:
        raise ValueError("offense and defense each need exactly 5 player ids")
    if len(set(offense)) != LINEUP_SIZE or len(set(defense)) != LINEUP_SIZE:
        raise ValueError("offense and defense player ids must be distinct")
    if set(offense) & set(defense):
        raise ValueError("a player cannot be on both offense and defense")
    model, meta = load_model(model_path)
    merged = {**DEFAULT_CONTEXT, **(context or {})}
    off_t = tuple(sorted(int(p) for p in offense))
    def_t = tuple(sorted(int(p) for p in defense))
    decomp = model.decompose(off_t, def_t, merged)
    return PredictionResult(
        offense=off_t,
        defense=def_t,
        context=merged,
        decomposition=decomp,
        support=model.lineup_support(off_t),
        interaction_interval=model.interaction_interval(off_t),
        model_metadata=meta,
    )


# z-scores for a Gaussian predictive interval (rung 3's posterior + noise SD).
_Z_80 = 1.2816
_Z_95 = 1.9600

RUNG3_NOTE = (
    "Additive talent + context only -- no interaction/chemistry term. "
    "Lineup chemistry is not a supported predictive effect on scoring "
    "(research cycle 1; see docs/RESEARCH_REPORT.md)."
)


@dataclass(frozen=True)
class Rung3PredictionResult:
    """A rung-3 (empirical-Bayes hierarchical additive) prediction for one
    5-vs-5 lineup. Deliberately has no interaction/chemistry field -- rung 3
    has none, and every symmetric/asymmetric interaction form tested on real
    data was null (`docs/RESEARCH_REPORT.md`)."""

    offense: tuple[int, ...]
    defense: tuple[int, ...]
    context: dict[str, Any]
    talent: float
    context_value: float
    total: float
    predictive_sd: float
    interval_80: tuple[float, float]
    interval_95: tuple[float, float]
    support: dict[str, Any]
    model_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "offense": list(self.offense),
            "defense": list(self.defense),
            "context": dict(self.context),
            "talent": self.talent,
            "context_value": self.context_value,
            "total": self.total,
            "predictive_sd": self.predictive_sd,
            "interval_80": list(self.interval_80),
            "interval_95": list(self.interval_95),
            "support": dict(self.support),
            "model_metadata": dict(self.model_metadata),
            "note": RUNG3_NOTE,
        }


def fit_rung3_file(
    input_path: str | Path,
    model_out: str | Path,
    *,
    config: Any | None = None,
) -> tuple[Any, Path]:
    """Fit rung 3 (empirical-Bayes hierarchical additive model, no interaction
    term) on a real stint file and persist it via
    :mod:`courtgraph.chemistry.rung3_artifact`."""

    from courtgraph.chemistry import rung3_artifact
    from courtgraph.chemistry.features import FeatureSpace
    from courtgraph.chemistry.hierarchical import HierarchicalConfig, HierarchicalRidge

    table = read_stints(input_path)
    if len(table) < 50:
        raise ValueError(
            f"{input_path}: only {len(table)} stints; need a substantially larger "
            "table to fit rung 3"
        )
    space = FeatureSpace.from_training(table)
    design = space.build(table)
    model = HierarchicalRidge.fit(
        design, space, config=config if isinstance(config, HierarchicalConfig) else None
    )

    possessions: dict[int, int] = {}
    for stint in table:
        for pid in stint.offense_player_ids:
            possessions[pid] = possessions.get(pid, 0) + stint.offensive_possessions

    path = rung3_artifact.save_model(
        model,
        model_out,
        metadata={
            "source": str(input_path),
            "stints": len(table),
            "possessions": table.total_possessions(),
            "training_player_possessions": {str(k): v for k, v in possessions.items()},
        },
    )
    return model, path


def predict_lineup_rung3(
    model_path: str | Path,
    offense: list[int],
    defense: list[int],
    *,
    context: dict[str, Any] | None = None,
) -> Rung3PredictionResult:
    """Score an arbitrary 5-vs-5 lineup with a fitted rung-3 model. Additive
    talent + context and a calibrated predictive interval only -- see
    :data:`RUNG3_NOTE`."""

    from courtgraph.chemistry import rung3_artifact

    model, meta = rung3_artifact.load_model(model_path)
    return _predict_lineup_rung3_with_model(model, meta, offense, defense, context)


def _predict_lineup_rung3_with_model(
    model: Any,
    meta: dict[str, Any],
    offense: list[int],
    defense: list[int],
    context: dict[str, Any] | None,
) -> Rung3PredictionResult:
    if len(offense) != LINEUP_SIZE or len(defense) != LINEUP_SIZE:
        raise ValueError("offense and defense each need exactly 5 player ids")
    if len(set(offense)) != LINEUP_SIZE or len(set(defense)) != LINEUP_SIZE:
        raise ValueError("offense and defense player ids must be distinct")
    if set(offense) & set(defense):
        raise ValueError("a player cannot be on both offense and defense")

    import numpy as np

    from courtgraph.chemistry.chemistry_model import _reference_stint
    from courtgraph.chemistry.stints import StintTable

    merged = {**DEFAULT_CONTEXT, **(context or {})}
    off_t = tuple(sorted(int(p) for p in offense))
    def_t = tuple(sorted(int(p) for p in defense))

    stint = _reference_stint(off_t, def_t, merged)
    design = model.feature_space.build(StintTable.from_stints([stint]))
    add = model.decompose_row(design, 0)
    point, sd, _w = model.group_predictive(
        design, {"lineup": np.array([0], dtype=np.int64)}
    )["lineup"]

    index = model.feature_space.player_index()
    unseen_off = tuple(p for p in off_t if p not in index)
    unseen_def = tuple(p for p in def_t if p not in index)
    poss_table = {
        int(k): int(v) for k, v in meta.get("training_player_possessions", {}).items()
    }
    off_poss = [poss_table.get(p, 0) for p in off_t]
    support = {
        "unseen_offense_players": list(unseen_off),
        "unseen_defense_players": list(unseen_def),
        "min_offense_player_possessions": min(off_poss) if off_poss else 0,
        "median_offense_player_possessions": (
            float(np.median(off_poss)) if off_poss else 0.0
        ),
    }

    # the full per-player training-possessions table (hundreds/thousands of
    # entries) is only an input to `support` above; keep it out of the
    # per-prediction metadata so a JSON result stays readable.
    small_meta = {k: v for k, v in meta.items() if k != "training_player_possessions"}

    return Rung3PredictionResult(
        offense=off_t,
        defense=def_t,
        context=merged,
        talent=add.talent,
        context_value=add.context,
        total=float(point),
        predictive_sd=float(sd),
        interval_80=(float(point - _Z_80 * sd), float(point + _Z_80 * sd)),
        interval_95=(float(point - _Z_95 * sd), float(point + _Z_95 * sd)),
        support=support,
        model_metadata=small_meta,
    )


def compare_lineups_rung3(
    model_path: str | Path,
    offense_a: list[int],
    offense_b: list[int],
    defense: list[int],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score two offensive fives against the same defense/context with a
    fitted rung-3 model, and report the point difference. See
    :data:`RUNG3_NOTE`."""

    from courtgraph.chemistry import rung3_artifact

    model, meta = rung3_artifact.load_model(model_path)
    return _compare_lineups_rung3_with_model(
        model, meta, offense_a, offense_b, defense, context
    )


def _compare_lineups_rung3_with_model(
    model: Any,
    meta: dict[str, Any],
    offense_a: list[int],
    offense_b: list[int],
    defense: list[int],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    a = _predict_lineup_rung3_with_model(model, meta, offense_a, defense, context)
    b = _predict_lineup_rung3_with_model(model, meta, offense_b, defense, context)
    return {
        "a": a.as_dict(),
        "b": b.as_dict(),
        "delta": {
            "talent": b.talent - a.talent,
            "context_value": b.context_value - a.context_value,
            "total": b.total - a.total,
        },
        "delta_note": (
            "Point difference only. Rung 3's per-lineup interval is not a "
            "calibrated interval on the A-vs-B difference (the two "
            "predictions' posterior errors are correlated)."
        ),
        "note": RUNG3_NOTE,
    }


def _syn_config_dict(cfg: SyntheticConfig) -> dict[str, Any]:
    return {
        "seed": cfg.seed,
        "n_players": cfg.n_players,
        "n_teams": cfg.n_teams,
        "n_seasons": cfg.n_seasons,
        "games_per_matchup": cfg.games_per_matchup,
        "stints_per_game": cfg.stints_per_game,
        "embedding_rank": cfg.embedding_rank,
        "interaction_scale": cfg.interaction_scale,
    }


def summarize_ground_truth(truth: GroundTruth) -> dict[str, Any]:
    return truth.as_dict()

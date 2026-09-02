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
    from courtgraph.chemistry.redundancy_eval import RedundancyComparison
    from courtgraph.chemistry.role_eval import RoleComparison
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

    from courtgraph.chemistry.mechanistic import evaluate_mechanistic
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters
    from courtgraph.features.stint_shots import attribute_shots
    from courtgraph.ingest.snapshot import load_snapshot

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(f"{stints_path}: only {len(table)} stints; need more")
    snapshot = load_snapshot(snapshot_dir)
    attribution = attribute_shots(snapshot, table)
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


def run_confirmation_file(
    stints_path: str | Path,
    profiles_path: str | Path,
    snapshot_dir: str | Path,
    *,
    k_values: tuple[int, ...] = (3, 5, 7),
    n_lineups: int = 120,
    n_boot: int = 2000,
) -> ConfirmationResult:
    """Re-run the three interaction positives with a wider ``unseen_lineup``
    holdout, a K sweep, and bootstrap CIs on the (baseline - model) delta.
    See :mod:`courtgraph.chemistry.confirm`."""

    from courtgraph.chemistry.confirm import run_confirmation
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters
    from courtgraph.features.stint_shots import attribute_shots
    from courtgraph.ingest.snapshot import load_snapshot

    table = read_stints(stints_path)
    if len(table) < 50:
        raise ValueError(f"{stints_path}: only {len(table)} stints; need more")
    profiles = read_player_profiles(profiles_path)
    clustering_by_k = {
        k: fit_role_clusters(profiles, n_clusters=k, seed=0) for k in k_values
    }
    attribution = attribute_shots(load_snapshot(snapshot_dir), table)
    return run_confirmation(
        table,
        clustering_by_k,
        attribution,
        n_lineups=n_lineups,
        n_boot=n_boot,
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

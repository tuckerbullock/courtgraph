"""CourtGraph command-line interface.

``doctor`` stays dependency-free (no NumPy import on that path). The modeling
commands -- ``demo``, ``fit``, ``predict`` -- lazily import the chemistry
package, which requires NumPy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from courtgraph.health import HealthReport, run_health_checks


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without performing any side effects."""

    parser = argparse.ArgumentParser(
        prog="courtgraph",
        description="CourtGraph research and lineup-chemistry tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="verify the Python runtime and required project files",
    )
    doctor.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project root to inspect (default: current directory)",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit a stable machine-readable result",
    )

    demo = subparsers.add_parser(
        "demo",
        help="run the full synthetic chemistry slice: data, splits, models, eval",
    )
    demo.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write a self-contained HTML report to this path",
    )
    demo.add_argument(
        "--out-dir",
        type=Path,
        default=Path("courtgraph_demo"),
        help="directory for generated stints, splits, and the model artifact",
    )
    demo.add_argument("--seed", type=int, default=20260830, help="master random seed")
    demo.add_argument(
        "--bootstrap",
        type=int,
        default=8,
        help="bootstrap replicates for approximate interaction uncertainty",
    )
    demo.add_argument("--json", action="store_true", help="print the summary as JSON")

    fit = subparsers.add_parser("fit", help="fit a chemistry model on a stint file")
    fit.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    fit.add_argument(
        "--model-out",
        type=Path,
        required=True,
        help="where to write the model artifact",
    )
    fit.add_argument("--seed", type=int, default=0, help="model random seed")
    fit.add_argument("--rank", type=int, default=3, help="interaction embedding rank")
    fit.add_argument(
        "--bootstrap",
        type=int,
        default=8,
        help="interaction bootstrap-ensemble size (0 to skip; large stint files "
        "are much faster with 0 and the holdout RMSEs are unaffected)",
    )
    fit.add_argument(
        "--evaluate",
        action="store_true",
        help="also run the leakage-safe holdout evaluation and print it",
    )
    fit.add_argument("--json", action="store_true", help="print result as JSON")

    baselines = subparsers.add_parser(
        "baselines",
        help="fit rung-2 (additive RAPM) and rung-3 (hierarchical EB) and "
        "compare point accuracy and interval calibration on the leakage-safe "
        "holdouts",
    )
    baselines.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    baselines.add_argument(
        "--bootstrap",
        type=int,
        default=150,
        help="rung-2 predictive-band resamples (0 to skip the band)",
    )
    baselines.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    baselines.add_argument(
        "--rung4",
        action="store_true",
        help="also fit rung 4 (explicit teammate-pair interaction; slower)",
    )
    baselines.add_argument("--json", action="store_true", help="print result as JSON")

    transport = subparsers.add_parser(
        "transport",
        help="fit rungs 2/3 (and --rung4) on one stint file (the regular season) "
        "and evaluate them on a held-out second file (the playoffs)",
    )
    transport.add_argument(
        "--train", type=Path, required=True, help="training stint file (regular season)"
    )
    transport.add_argument(
        "--test", type=Path, required=True, help="held-out stint file (playoffs)"
    )
    transport.add_argument(
        "--bootstrap",
        type=int,
        default=150,
        help="rung-2 predictive-band resamples (0 to skip the band)",
    )
    transport.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    transport.add_argument(
        "--rung4",
        action="store_true",
        help="also fit rung 4 (explicit teammate-pair interaction; slower)",
    )
    transport.add_argument("--json", action="store_true", help="print result as JSON")

    tmech = subparsers.add_parser(
        "transport-mechanistic",
        help="train the role-conditioned model on one file's mechanistic "
        "outcome (e.g. three_share) and evaluate it on a disjoint second file",
    )
    tmech.add_argument("--train", type=Path, required=True, help="training stint file")
    tmech.add_argument("--test", type=Path, required=True, help="held-out stint file")
    tmech.add_argument(
        "--train-snapshot", type=Path, required=True, help="snapshot for the train file"
    )
    tmech.add_argument(
        "--test-snapshot", type=Path, required=True, help="snapshot for the test file"
    )
    tmech.add_argument(
        "--profiles", type=Path, required=True, help="player_profiles.jsonl"
    )
    tmech.add_argument(
        "--outcome",
        choices=("three_share", "pts_per_shot", "rim_share"),
        default="three_share",
    )
    tmech.add_argument("--clusters", type=int, default=5)
    tmech.add_argument("--min-fga", type=int, default=3)
    tmech.add_argument("--boot", type=int, default=2000)
    tmech.add_argument("--seed", type=int, default=0)
    tmech.add_argument("--json", action="store_true", help="print result as JSON")

    roles = subparsers.add_parser(
        "roles",
        help="compare a role-conditioned interaction model (interaction keyed "
        "by role-cluster pair) against rungs 2/3 and a permuted-role placebo",
    )
    roles.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    roles.add_argument(
        "--profiles",
        type=Path,
        required=True,
        help="player_profiles.jsonl from `courtgraph player-features`",
    )
    roles.add_argument(
        "--clusters", type=int, default=5, help="number of role clusters (default 5)"
    )
    roles.add_argument(
        "--bootstrap",
        type=int,
        default=120,
        help="rung-2 predictive-band resamples (0 to skip)",
    )
    roles.add_argument(
        "--seed", type=int, default=0, help="clustering / bootstrap seed"
    )
    roles.add_argument("--json", action="store_true", help="print result as JSON")

    confirm = subparsers.add_parser(
        "confirm",
        help="better-powered re-run of the three interaction positives: wider "
        "unseen-lineup holdout, K sweep, bootstrap CIs on the model-vs-baseline "
        "delta",
    )
    confirm.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    confirm.add_argument(
        "--profiles", type=Path, required=True, help="player_profiles.jsonl"
    )
    confirm.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="snapshot directory (for shot attribution)",
    )
    confirm.add_argument(
        "--k", default="3,5,7", help="comma-separated role-cluster counts to sweep"
    )
    confirm.add_argument(
        "--outcomes",
        default="three_share",
        help="comma-separated mechanistic outcomes "
        "(three_share, pts_per_shot, rim_share)",
    )
    confirm.add_argument(
        "--lineups", type=int, default=120, help="unseen-lineup holdout groups"
    )
    confirm.add_argument(
        "--min-fga", type=int, default=3, help="drop stints below this many FGA"
    )
    confirm.add_argument(
        "--boot", type=int, default=2000, help="bootstrap resamples for the CI"
    )
    confirm.add_argument("--json", action="store_true", help="print result as JSON")

    redundancy = subparsers.add_parser(
        "redundancy",
        help="test skill redundancy / anti-synergy: coefficients on per-role "
        "concentration features, vs rungs 2/3 and a permuted-role placebo",
    )
    redundancy.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    redundancy.add_argument(
        "--profiles",
        type=Path,
        required=True,
        help="player_profiles.jsonl from `courtgraph player-features`",
    )
    redundancy.add_argument(
        "--clusters", type=int, default=5, help="role clusters (default 5)"
    )
    redundancy.add_argument(
        "--bootstrap", type=int, default=120, help="rung-2 band resamples"
    )
    redundancy.add_argument("--seed", type=int, default=0, help="seed")
    redundancy.add_argument("--json", action="store_true", help="print result as JSON")

    player_lift = subparsers.add_parser(
        "player-lift",
        help="master plan §45 Phase A: one EM-shrunk lift scalar per player on "
        "the rung-3 frame, vs rungs 2/3 and a player-permutation placebo",
    )
    player_lift.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    player_lift.add_argument(
        "--bootstrap", type=int, default=120, help="rung-2 band resamples"
    )
    player_lift.add_argument("--seed", type=int, default=0, help="seed")
    player_lift.add_argument(
        "--json", action="store_true", help="print result as JSON"
    )

    mechanistic = subparsers.add_parser(
        "mechanistic",
        help="test whether lineup composition shifts a mechanistic outcome "
        "(shot quality / shot mix) non-additively",
    )
    mechanistic.add_argument(
        "--input", type=Path, required=True, help="stint file (.jsonl/.json)"
    )
    mechanistic.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="snapshot directory the stints were built from (read-only)",
    )
    mechanistic.add_argument(
        "--profiles",
        type=Path,
        required=True,
        help="player_profiles.jsonl from `courtgraph player-features`",
    )
    mechanistic.add_argument(
        "--outcome",
        choices=("pts_per_shot", "rim_share", "three_share"),
        default="pts_per_shot",
        help="the mechanistic target (default pts_per_shot)",
    )
    mechanistic.add_argument(
        "--min-fga", type=int, default=3, help="drop stints below this many FGA"
    )
    mechanistic.add_argument(
        "--clusters", type=int, default=5, help="role clusters (default 5)"
    )
    mechanistic.add_argument(
        "--bootstrap", type=int, default=120, help="rung-2 band resamples"
    )
    mechanistic.add_argument("--seed", type=int, default=0, help="seed")
    mechanistic.add_argument("--json", action="store_true", help="print result as JSON")

    player_features = subparsers.add_parser(
        "player-features",
        help="derive per-(player, season) role/skill profiles from a snapshot "
        "and a stint file (usage, shot profile, playmaking, turnover rate)",
    )
    player_features.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="stats_nba_pbpstats/v1 snapshot directory (read-only)",
    )
    player_features.add_argument(
        "--stints",
        type=Path,
        required=True,
        help="stint file whose games / possessions set the exposure denominator",
    )
    player_features.add_argument(
        "--out", type=Path, required=True, help="write profiles.jsonl here"
    )
    player_features.add_argument(
        "--min-possessions",
        type=int,
        default=200,
        help="on-court offensive possessions below which per-possession rates "
        "are left null (default 200)",
    )
    player_features.add_argument(
        "--json", action="store_true", help="also print a summary as JSON"
    )

    ingest = subparsers.add_parser(
        "ingest",
        help="convert an offline NBA snapshot into validated stint records",
    )
    ingest.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="stats_nba_pbpstats/v1 snapshot directory (immutable; not modified)",
    )
    ingest.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for stints.jsonl, quarantine.jsonl, and manifest.json",
    )
    ingest.add_argument(
        "--allow-score-mismatch",
        action="store_true",
        help="emit a game whose derived final score != the official total "
        "(flagged in the manifest) instead of quarantining it",
    )
    ingest.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also write a readable self-contained HTML report to this path",
    )
    ingest.add_argument("--json", action="store_true", help="print result as JSON")

    shuf = subparsers.add_parser(
        "snapshot-from-shufinskiy",
        help="build a stats_nba_pbpstats/v1 snapshot from a local "
        "SRC-SHUFINSKIY archive of one or more seasons (local-dev-only; no network)",
    )
    shuf.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        action="append",
        help="directory holding the archive's nbastats_*.csv / datanba_*.csv / "
        "shotdetail_*.csv (one file per provider, or one per provider per "
        "season). Repeatable -- pass several season-range dirs to build one "
        "combined snapshot.",
    )
    shuf_selection = shuf.add_mutually_exclusive_group(required=True)
    shuf_selection.add_argument(
        "--game",
        action="append",
        metavar="GAME_ID",
        help="NBA game id (8- or 10-digit), repeatable",
    )
    shuf_selection.add_argument(
        "--all-games",
        action="store_true",
        help="include every game present in all three providers' archive inputs",
    )
    shuf.add_argument(
        "--out-dir", type=Path, required=True, help="snapshot directory to create"
    )
    shuf.add_argument("--json", action="store_true", help="print result as JSON")

    live = subparsers.add_parser(
        "fetch-live",
        help="optional rate-limited live acquisition from stats.nba.com "
        "(DATA_SOURCES.md §5.1; cache-and-freeze). Fills gaps the frozen "
        "archive cannot: boxscore period starters, the current season.",
    )
    live.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="content-addressed live cache (created if absent; reused)",
    )
    live_action = live.add_mutually_exclusive_group(required=True)
    live_action.add_argument(
        "--smoke-test",
        action="store_true",
        help="five cheap requests to confirm this machine can reach stats.nba.com",
    )
    live_action.add_argument(
        "--period-starters",
        nargs="+",
        metavar="GAME_ID",
        help="fetch boxscoretraditionalv2 for these games and write "
        "game_details/stats_boxscore_<gid>.json into --snapshot-dir",
    )
    live.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="snapshot to write fetched boxscores into (with --period-starters)",
    )
    live.add_argument("--json", action="store_true", help="print result as JSON")

    app = subparsers.add_parser(
        "app", help="open a local browser explorer and synthetic sandbox"
    )
    app.add_argument(
        "--port", type=int, default=8765, help="loopback port (default: 8765)"
    )
    app.add_argument(
        "--ingest-dir",
        type=Path,
        help="existing output with manifest.json and stints.jsonl",
    )
    app.add_argument(
        "--names", type=Path, help="optional display_names.json for that snapshot"
    )

    predict = subparsers.add_parser(
        "predict", help="decompose one lineup's predicted value with a fitted model"
    )
    predict.add_argument(
        "--model", type=Path, required=True, help="model artifact path"
    )
    predict.add_argument(
        "--offense", required=True, help="5 comma-separated offensive player ids"
    )
    predict.add_argument(
        "--defense", required=True, help="5 comma-separated defensive player ids"
    )
    predict.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a context field (repeatable), e.g. --context playoff=1",
    )
    predict.add_argument("--json", action="store_true", help="print result as JSON")
    return parser


def _render_human(result: HealthReport, output: TextIO) -> None:
    print(f"CourtGraph {result['courtgraph_version']}: {result['status']}", file=output)
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['name']}: {check['detail']}", file=output)


def _parse_ids(raw: str) -> list[int]:
    try:
        return [int(part) for part in raw.replace(" ", "").split(",") if part]
    except ValueError as exc:  # pragma: no cover - argparse-style error
        raise SystemExit(f"could not parse player ids from {raw!r}: {exc}") from exc


def _parse_context(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--context expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = _coerce(value.strip())
    return out


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _cmd_doctor(args: argparse.Namespace, stream: TextIO) -> int:
    result = run_health_checks(args.root)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True), file=stream)
    else:
        _render_human(result, stream)
    return 0 if result["status"] == "healthy" else 1


def _cmd_demo(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_demo

    result = run_demo(
        out_dir=args.out_dir,
        report_path=args.report,
        seed=args.seed,
        n_boot=args.bootstrap,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "model_path": str(result.model_path),
                    "report_path": (
                        str(result.report_path) if result.report_path else None
                    ),
                    "stints_path": str(result.stints_path),
                    "headline": result.headline(),
                    "recovery": result.summary.recovery,
                },
                indent=2,
                sort_keys=True,
            ),
            file=stream,
        )
        return 0
    d = result.summary.dataset
    print(
        f"CourtGraph demo (SYNTHETIC): {d['stints']} stints, {d['possessions']} "
        f"possessions, {d['players']} players, {len(d['seasons'])} seasons",
        file=stream,
    )
    print(
        f"{'holdout':<14}{'groups':>8}{'add macro':>12}{'full macro':>12}"
        f"{'improve':>10}{'leak':>7}",
        file=stream,
    )
    for row in result.headline():
        print(
            f"{row['holdout']:<14}{row['test_groups']:>8}"
            f"{row['additive_macro_rmse']:>12.2f}{row['full_macro_rmse']:>12.2f}"
            f"{row['improvement_pct']:>9.0f}%{row['leakage_violations']:>7}",
            file=stream,
        )
    if result.summary.recovery:
        rec = result.summary.recovery
        print(
            "recovery corr  talent off/def "
            f"{rec.get('offensive_talent_corr', 0):.2f}/"
            f"{rec.get('defensive_talent_corr', 0):.2f}  "
            f"pair surplus {rec.get('pair_surplus_corr', 0):.2f}",
            file=stream,
        )
        for h in result.summary.holdouts:
            corr = h.metrics.get("interaction_recovery_corr")
            if corr is not None:
                print(
                    f"  {h.kind:<14} test-set corr(predicted C, true C) = {corr:+.2f}",
                    file=stream,
                )
    print(f"model:  {result.model_path}", file=stream)
    print(f"stints: {result.stints_path}", file=stream)
    if result.report_path:
        print(f"report: {result.report_path}", file=stream)
    return 0


def _cmd_fit(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.chemistry_model import ChemistryConfig
    from courtgraph.chemistry.pipeline import evaluate_model_file, fit_model_file

    if args.bootstrap < 0:
        print("fit: --bootstrap must be >= 0", file=stream)
        return 2
    config = ChemistryConfig(seed=args.seed, rank=args.rank, n_bootstrap=args.bootstrap)
    model, path = fit_model_file(args.input, args.model_out, config=config)
    payload: dict[str, Any] = {
        "model_path": str(path),
        "training_stints": model.training_stints,
        "training_possessions": model.training_possessions,
        "interaction_l2": model.interaction.l2,
        "rank": model.config.rank,
    }
    if args.evaluate:
        summary = evaluate_model_file(args.input, config=config)
        payload["holdouts"] = [h.as_dict()["metrics"] for h in summary.holdouts]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=stream)
    else:
        print(f"fitted model -> {path}", file=stream)
        print(
            f"  {model.training_stints} stints, {model.training_possessions} "
            f"possessions, rank {model.config.rank}, interaction L2 "
            f"{model.interaction.l2:g}",
            file=stream,
        )
        if args.evaluate:
            for h in summary.holdouts:
                print(
                    f"  {h.kind:<14} improvement {h.headline_improvement_pct:+.0f}% "
                    f"(leakage {len(h.leakage_violations)})",
                    file=stream,
                )
    return 0


def _cmd_baselines(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_baselines

    if args.bootstrap < 0:
        print("baselines: --bootstrap must be >= 0", file=stream)
        return 2
    comparison = run_baselines(
        args.input, seed=args.seed, n_boot=args.bootstrap, rung4=args.rung4
    )
    if args.json:
        print(json.dumps(comparison.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    vc = comparison.variance_components
    conv = "converged" if vc["converged"] else "NOT converged"
    print(
        f"rung-3 hierarchical EB: sigma={vc['sigma']:.2f}  "
        f"tau_off={vc['tau_off']:.2f}  tau_def={vc['tau_def']:.2f}  "
        f"({vc['n_iters']} EM iters, {conv})",
        file=stream,
    )
    r4 = args.rung4
    print(
        f"  {'holdout':<14} {'groups':>6}  {'r2 macro':>9} {'r3 macro':>9}"
        + (f" {'r4 macro':>9}" if r4 else "")
        + f"  {'r3 cov50/80/95':>16}  {'r3 slope':>8}",
        file=stream,
    )
    for h in comparison.holdouts:
        cal = h.rung3_calibration
        line = (
            f"  {h.kind:<14} {h.n_groups:>6}  {h.rung2_macro_rmse:>9.3f} "
            f"{h.rung3_macro_rmse:>9.3f}"
        )
        if r4 and h.rung4_macro_rmse is not None:
            line += f" {h.rung4_macro_rmse:>9.3f}"
        elif r4:
            line += f" {'-':>9}"
        line += (
            f"  {cal['coverage_50']:>4.2f}/{cal['coverage_80']:.2f}/"
            f"{cal['coverage_95']:.2f}   {cal['slope']:>8.2f}"
        )
        print(line, file=stream)
        if r4 and h.rung4_pair_level is not None:
            pl = h.rung4_pair_level
            n_pg = int(pl.get("n_pair_groups", 0))
            if n_pg:
                print(
                    f"    seen-pairs (pair-level exit test, {n_pg} pairs): "
                    f"r2={pl['rung2_macro_rmse']:.3f}  "
                    f"r4={pl['rung4_macro_rmse']:.3f}  "
                    f"r4-placebo={pl['rung4_placebo_macro_rmse']:.3f}",
                    file=stream,
                )
            else:
                print(
                    "    seen-pairs: no admitted pair recurs in the test period",
                    file=stream,
                )
    return 0


def _cmd_transport(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_transport

    if args.bootstrap < 0:
        print("transport: --bootstrap must be >= 0", file=stream)
        return 2
    result = run_transport(
        args.train,
        args.test,
        seed=args.seed,
        n_boot=args.bootstrap,
        rung4=args.rung4,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    if result.leakage_violations:
        for v in result.leakage_violations:
            print(f"  LEAKAGE: {v}", file=stream)

    cov = result.coverage
    print(
        f"transport: train {result.n_train} stints -> test {result.n_test} stints; "
        f"{int(cov['test_players_unseen_in_train'])} of "
        f"{int(cov['test_players'])} test players unseen in train",
        file=stream,
    )
    print(
        f"  test lineups: {int(cov['test_stints_seen_lineup'])} seen / "
        f"{int(cov['test_stints_partially_seen_lineup'])} partially-seen / "
        f"{int(cov['test_stints_unseen_lineup'])} unseen (by stint)",
        file=stream,
    )
    if result.zeroed_context_columns:
        print(
            "  zeroed context columns unidentified in train: "
            + ", ".join(result.zeroed_context_columns),
            file=stream,
        )
    vc = result.variance_components
    conv = "converged" if vc["converged"] else "NOT converged"
    print(
        f"  rung-3 EB: sigma={vc['sigma']:.2f}  tau_off={vc['tau_off']:.2f}  "
        f"tau_def={vc['tau_def']:.2f}  ({vc['n_iters']} EM iters, {conv})",
        file=stream,
    )
    c3 = result.rung3_calibration
    r4_macro = (
        f"  r4={result.rung4_macro_rmse:.3f}"
        if result.rung4_macro_rmse is not None
        else ""
    )
    print(
        f"  playoff lineups ({result.n_lineup_groups} groups): "
        f"r2={result.rung2_macro_rmse:.3f}  r3={result.rung3_macro_rmse:.3f}"
        + r4_macro,
        file=stream,
    )
    print(
        f"    rung-3 calibration: cov {c3['coverage_50']:.2f}/"
        f"{c3['coverage_80']:.2f}/{c3['coverage_95']:.2f}  "
        f"z_mean={c3['z_mean']:.2f}  z_sd={c3['z_sd']:.2f}  slope={c3['slope']:.2f}",
        file=stream,
    )
    for nov, entry in result.by_novelty.items():
        if not entry.get("n_groups"):
            continue
        r4b = (
            f"  r4={entry['rung4_macro_rmse']:.3f}"
            if "rung4_macro_rmse" in entry
            else ""
        )
        print(
            f"    {nov:<15} {int(entry['n_groups']):>3} groups: "
            f"r2={entry['rung2_macro_rmse']:.3f}  r3={entry['rung3_macro_rmse']:.3f}"
            + r4b,
            file=stream,
        )
    if result.rung4_pair_level is not None:
        pl = result.rung4_pair_level
        n_pg = int(pl.get("n_pair_groups", 0))
        if n_pg:
            print(
                f"  playoff seen-pairs ({n_pg} pairs): "
                f"r2={pl['rung2_macro_rmse']:.3f}  r4={pl['rung4_macro_rmse']:.3f}  "
                f"r4-placebo={pl['rung4_placebo_macro_rmse']:.3f}",
                file=stream,
            )
    m = result.micro_rmse
    for label in ("all", "clutch", "non_clutch"):
        e = m[label]
        cols = "  ".join(
            f"{k.split('_')[0]}={e[k]:.2f}" for k in e if k.endswith("_micro_rmse")
        )
        print(
            f"  {label:<10} ({int(e['n_stints'])} stints, "
            f"{int(e['possessions'])} poss): {cols}",
            file=stream,
        )
    return 0


def _cmd_transport_mechanistic(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.mechanistic import transport_mechanistic
    from courtgraph.chemistry.stints import read_stints
    from courtgraph.features.player_season import read_player_profiles
    from courtgraph.features.role_clusters import fit_role_clusters
    from courtgraph.features.stint_shots import attribute_shots
    from courtgraph.ingest.snapshot import SnapshotError, load_snapshot

    if args.boot < 100 or args.clusters < 2 or args.min_fga < 1:
        print("transport-mechanistic: bad --boot / --clusters / --min-fga", file=stream)
        return 2
    try:
        train = read_stints(args.train)
        test = read_stints(args.test)
        train_attr = attribute_shots(load_snapshot(args.train_snapshot), train)
        test_attr = attribute_shots(load_snapshot(args.test_snapshot), test)
        clustering = fit_role_clusters(
            read_player_profiles(args.profiles), n_clusters=args.clusters, seed=0
        )
    except (SnapshotError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"transport-mechanistic: {exc}", file=stream)
        return 2

    result = transport_mechanistic(
        train,
        test,
        train_attr,
        test_attr,
        clustering,
        outcome=args.outcome,
        min_fga=args.min_fga,
        n_boot=args.boot,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True), file=stream)
        return 0
    d3 = result["delta_role_vs_rung3"]
    dp = result["delta_role_vs_placebo"]
    print(
        f"transport-mechanistic [{result['outcome']}]: {result['n_test_lineups']} "
        f"test lineups | rung3 {result['rung3_macro_rmse']:.4f}  role "
        f"{result['role_macro_rmse']:.4f}  placebo "
        f"{result['role_placebo_macro_rmse']:.4f}",
        file=stream,
    )
    print(
        f"  role vs rung3: {d3['mean']:+.4f} [{d3['ci_lo']:+.4f}, {d3['ci_hi']:+.4f}] "
        f"P={d3['frac_gt_0']:.2f}  | vs placebo: {dp['mean']:+.4f} "
        f"[{dp['ci_lo']:+.4f}, {dp['ci_hi']:+.4f}] P={dp['frac_gt_0']:.2f}",
        file=stream,
    )
    return 0


def _cmd_confirm(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_confirmation_file

    try:
        k_values = tuple(int(x) for x in str(args.k).split(","))
    except ValueError:
        print("confirm: --k must be comma-separated integers", file=stream)
        return 2
    outcomes = tuple(x.strip() for x in str(args.outcomes).split(",") if x.strip())
    if any(k < 2 for k in k_values) or args.boot < 100 or args.lineups < 10:
        print("confirm: bad --k / --boot / --lineups", file=stream)
        return 2
    try:
        result = run_confirmation_file(
            args.input,
            args.profiles,
            args.snapshot_dir,
            k_values=k_values,
            outcomes=outcomes,
            n_lineups=args.lineups,
            n_boot=args.boot,
            min_fga=args.min_fga,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"confirm: {exc}", file=stream)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    hg = result.holdout_groups
    print(
        f"confirm: K sweep {list(result.k_values)}, {result.n_boot} bootstrap "
        f"resamples; holdout groups " + ", ".join(f"{k}={v}" for k, v in hg.items()),
        file=stream,
    )
    print(
        f"  {'model':<16} {'K':>2} {'holdout':<14} {'outcome':<14} "
        f"{'delta vs r3 (95% CI)':<28} {'vs plc P>0':>10}",
        file=stream,
    )
    for r in result.rows:
        d = r.delta_vs_rung3
        print(
            f"  {r.model:<16} {r.k:>2} {r.holdout:<14} {r.outcome:<14} "
            f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"
            f"   {r.delta_vs_placebo['frac_gt_0']:>10.2f}",
            file=stream,
        )
    if result.mediation:
        m = result.mediation
        print(
            f"  mediation ({int(m['n_lineups'])} lineups): corr(role's "
            f"extra three_share, lineup's points surprise) = "
            f"{m['corr_d_three_share_vs_d_points']:+.3f}",
            file=stream,
        )
    return 0


def _cmd_redundancy(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_redundancy

    if args.bootstrap < 0 or args.clusters < 2:
        print("redundancy: bad --bootstrap / --clusters", file=stream)
        return 2
    try:
        result = run_redundancy(
            args.input,
            args.profiles,
            n_clusters=args.clusters,
            seed=args.seed,
            n_boot=args.bootstrap,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"redundancy: {exc}", file=stream)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    vc = result.variance_components
    print(
        f"redundancy: tau_rho={vc['tau_rho']:.4f}  sigma={vc['sigma']:.2f}",
        file=stream,
    )
    print(
        "  rho_d (concentration effect, points per 100; real / placebo):", file=stream
    )
    for feat, val in result.rho.items():
        print(
            f"    {feat:<16} {val:+.3f}   ({result.rho_placebo.get(feat, 0.0):+.3f})",
            file=stream,
        )
    print(
        f"  {'holdout':<14} {'r2':>8} {'r3':>8} {'redund':>8} {'red-plc':>9}",
        file=stream,
    )
    for h in result.holdouts:
        print(
            f"  {h.kind:<14} {h.rung2_macro_rmse:>8.3f} {h.rung3_macro_rmse:>8.3f} "
            f"{h.redundancy_macro_rmse:>8.3f} {h.redundancy_placebo_macro_rmse:>9.3f}",
            file=stream,
        )
    return 0


def _cmd_player_lift(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_player_lift

    if args.bootstrap < 0:
        print("player-lift: --bootstrap must be >= 0", file=stream)
        return 2
    try:
        result = run_player_lift(
            args.input, seed=args.seed, n_boot=args.bootstrap
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"player-lift: {exc}", file=stream)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    vc = result.variance_components
    print(
        f"player-lift: tau_lambda={vc['tau_lambda']:.4f}  "
        f"|lambda| mean={vc['lambda_abs_mean']:.4f} max={vc['lambda_abs_max']:.4f}",
        file=stream,
    )
    print(
        f"  {'holdout':<14} {'r2':>8} {'r3':>8} {'lift':>8} {'lift-plc':>9}  "
        f"{'tau_l':>7} {'tau_l-plc':>9}",
        file=stream,
    )
    for h in result.holdouts:
        print(
            f"  {h.kind:<14} {h.rung2_macro_rmse:>8.3f} {h.rung3_macro_rmse:>8.3f} "
            f"{h.lift_macro_rmse:>8.3f} {h.lift_placebo_macro_rmse:>9.3f}  "
            f"{h.tau_lambda:>7.4f} {h.tau_lambda_placebo:>9.4f}",
            file=stream,
        )
    print("  top |lift| players (id: lift ± sd):", file=stream)
    for t in result.top_lifts[:10]:
        print(
            f"    {int(t['player_id']):>8}: {t['lift']:+.3f} ± {t['sd']:.3f}",
            file=stream,
        )
    return 0


def _cmd_mechanistic(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_mechanistic

    if args.bootstrap < 0 or args.clusters < 2 or args.min_fga < 1:
        print("mechanistic: bad --bootstrap / --clusters / --min-fga", file=stream)
        return 2
    try:
        result = run_mechanistic(
            args.input,
            args.snapshot_dir,
            args.profiles,
            outcome=args.outcome,
            min_fga=args.min_fga,
            n_clusters=args.clusters,
            seed=args.seed,
            n_boot=args.bootstrap,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"mechanistic: {exc}", file=stream)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    print(
        f"mechanistic [{result.outcome}]: {result.n_stints_kept} stints "
        f"(>= {result.min_fga} FGA), mean {result.mean_outcome:.3f}, "
        f"shot match {result.shots_match_rate:.1%}",
        file=stream,
    )
    vc = result.role_variance_components
    print(f"  role tau={vc['tau_role']:.4f}  sigma={vc['sigma']:.2f}", file=stream)
    print(
        f"  {'holdout':<14} {'r2':>8} {'r3':>8} {'role':>8} {'role-plc':>9}",
        file=stream,
    )
    for h in result.holdouts:
        print(
            f"  {h.kind:<14} {h.rung2_macro_rmse:>8.4f} {h.rung3_macro_rmse:>8.4f} "
            f"{h.role_macro_rmse:>8.4f} {h.role_placebo_macro_rmse:>9.4f}",
            file=stream,
        )
    return 0


def _cmd_roles(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import run_roles

    if args.bootstrap < 0:
        print("roles: --bootstrap must be >= 0", file=stream)
        return 2
    if args.clusters < 2:
        print("roles: --clusters must be >= 2", file=stream)
        return 2
    try:
        result = run_roles(
            args.input,
            args.profiles,
            n_clusters=args.clusters,
            seed=args.seed,
            n_boot=args.bootstrap,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"roles: {exc}", file=stream)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0

    vc = result.role_variance_components
    print(
        f"roles: {result.n_clusters} clusters over {result.n_clustered_players} "
        f"players; tau_role={vc['tau_role']:.3f} "
        f"(tau_off={vc['tau_off']:.2f}, sigma={vc['sigma']:.1f})",
        file=stream,
    )
    print("  role-pair surplus matrix (points per 100):", file=stream)
    for a, row in enumerate(result.role_pair_matrix):
        cells = "  ".join(f"{v:+.2f}" for v in row)
        print(f"    c{a}: {cells}", file=stream)
    print(
        f"  {'holdout':<14} {'r2':>8} {'r3':>8} {'role':>8} {'role-plc':>9}",
        file=stream,
    )
    for h in result.holdouts:
        print(
            f"  {h.kind:<14} {h.rung2_macro_rmse:>8.3f} {h.rung3_macro_rmse:>8.3f} "
            f"{h.role_macro_rmse:>8.3f} {h.role_placebo_macro_rmse:>9.3f}",
            file=stream,
        )
    return 0


def _cmd_player_features(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.features.player_season import (
        build_from_paths,
        write_player_profiles,
    )
    from courtgraph.ingest.snapshot import SnapshotError

    if args.min_possessions < 0:
        print("player-features: --min-possessions must be >= 0", file=stream)
        return 2
    try:
        profiles = build_from_paths(
            args.snapshot_dir,
            args.stints,
            min_off_possessions=args.min_possessions,
        )
    except (SnapshotError, FileNotFoundError, OSError) as exc:
        print(f"player-features: {exc}", file=stream)
        return 2

    write_player_profiles(profiles, args.out)
    with_rates = sum(1 for p in profiles if p.usage is not None)
    seasons = sorted({p.season for p in profiles})
    if args.json:
        print(
            json.dumps(
                {
                    "profiles": len(profiles),
                    "with_rates": with_rates,
                    "seasons": seasons,
                    "out": str(args.out),
                },
                indent=2,
            ),
            file=stream,
        )
    else:
        print(
            f"player-features: {len(profiles)} (player, season) profiles "
            f"({with_rates} above the {args.min_possessions}-possession floor) "
            f"across {len(seasons)} season(s) -> {args.out}",
            file=stream,
        )
    return 0


def _cmd_ingest(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.ingest.pipeline import run_ingest
    from courtgraph.ingest.policy import IngestPolicy
    from courtgraph.ingest.snapshot import SnapshotError

    policy = IngestPolicy(allow_score_mismatch=bool(args.allow_score_mismatch))
    try:
        result = run_ingest(args.snapshot_dir, args.out_dir, policy=policy)
    except SnapshotError as exc:
        print(f"ingest: invalid snapshot: {exc}", file=stream)
        return 2

    report_path = None
    if args.report is not None:
        from courtgraph.ingest._paths import OutputPathError
        from courtgraph.ingest.report import write_report

        try:
            report_path = write_report(
                result.out_dir, args.report, snapshot_dir=args.snapshot_dir
            )
        except OutputPathError as exc:
            print(f"ingest: unsafe --report path: {exc}", file=stream)
            return 2

    if args.json:
        print(
            json.dumps(
                {
                    "stints_path": str(result.stints_path),
                    "manifest_path": str(result.manifest_path),
                    "quarantine_path": str(result.quarantine_path),
                    "stints_written": result.stints_written,
                    "games_accepted": result.games_accepted,
                    "games_quarantined": result.games_quarantined,
                    "possessions_excluded": result.possessions_excluded,
                    "report_path": str(report_path) if report_path else None,
                },
                indent=2,
                sort_keys=True,
            ),
            file=stream,
        )
    else:
        print(
            f"ingest (offline, file-only): {result.stints_written} stints "
            f"from {result.games_accepted} game(s); "
            f"{result.games_quarantined} game(s) quarantined, "
            f"{result.possessions_excluded} possession(s) excluded",
            file=stream,
        )
        print(f"stints:     {result.stints_path}", file=stream)
        print(f"quarantine: {result.quarantine_path}", file=stream)
        print(f"manifest:   {result.manifest_path}", file=stream)
        if report_path:
            print(f"report:     {report_path}", file=stream)
    return 0 if result.stints_written > 0 else 1


def _cmd_snapshot_from_shufinskiy(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.ingest._paths import OutputPathError
    from courtgraph.ingest.shufinskiy import ShufinskiyArchiveError, build_snapshot

    try:
        snap = build_snapshot(
            args.archive_dir, None if args.all_games else list(args.game), args.out_dir
        )
    except (ShufinskiyArchiveError, OutputPathError, FileNotFoundError) as exc:
        print(f"snapshot-from-shufinskiy: {exc}", file=stream)
        return 2

    payload = {
        "out_dir": str(snap.out_dir),
        "provenance": snap.provenance,
        "game_ids": list(snap.game_ids),
        "quarantine_expected": snap.quarantine_expected,
        "archive_coverage": snap.archive_coverage,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=stream)
    else:
        print(
            f"snapshot-from-shufinskiy: wrote {len(snap.game_ids)} of "
            f"{snap.archive_coverage['archive_games']} archive game(s) to "
            f"{snap.out_dir}  (local SRC-SHUFINSKIY archive; no network)",
            file=stream,
        )
        if len(snap.game_ids) <= 50:
            for gid in snap.game_ids:
                note = snap.quarantine_expected.get(gid)
                print(
                    f"  {gid}" + (f"  — will quarantine: {note}" if note else ""),
                    file=stream,
                )
        elif snap.quarantine_expected:
            print(
                f"  {len(snap.quarantine_expected)} game(s) expected to quarantine "
                "for missing context (see the ingest manifest for the breakdown)",
                file=stream,
            )
    return 0


def _cmd_fetch_live(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.ingest.live_fetch import (
        LiveAccessBlocked,
        LiveCache,
        LiveClient,
        smoke_test,
    )

    if args.smoke_test:
        try:
            summary = smoke_test(args.cache_dir)
        except LiveAccessBlocked as exc:
            print(f"fetch-live: BLOCKED — {exc}", file=stream)
            return 2
        if args.json:
            print(json.dumps(summary, indent=2), file=stream)
        else:
            print(
                f"fetch-live smoke test: {summary['ok']}/{summary['requests']} ok",
                file=stream,
            )
            for err in summary["errors"]:
                print(f"  {err}", file=stream)
        return 0 if summary["ok"] > 0 else 1

    # --period-starters
    if args.snapshot_dir is None:
        print("fetch-live: --period-starters requires --snapshot-dir", file=stream)
        return 2
    client = LiveClient(LiveCache(args.cache_dir))
    dest = Path(args.snapshot_dir) / "game_details"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    failed: list[str] = []
    for gid in args.period_starters:
        try:
            payload = client.fetch(
                "boxscoretraditionalv2", {"GameID": str(gid).zfill(10)}
            )
        except LiveAccessBlocked as exc:
            print(f"fetch-live: BLOCKED after {len(written)} — {exc}", file=stream)
            return 2
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append(f"{gid}: {exc}")
            continue
        out = dest / f"stats_boxscore_{str(gid).zfill(10)}.json"
        out.write_text(json.dumps(payload))
        written.append(out.name)
    if args.json:
        print(json.dumps({"written": written, "failed": failed}, indent=2), file=stream)
    else:
        print(f"fetch-live: wrote {len(written)} boxscore(s)", file=stream)
        for err in failed:
            print(f"  failed {err}", file=stream)
    return 0 if not failed else 1


def _cmd_predict(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.chemistry.pipeline import predict_lineup

    result = predict_lineup(
        args.model,
        _parse_ids(args.offense),
        _parse_ids(args.defense),
        context=_parse_context(args.context),
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=stream)
        return 0
    d = result.decomposition
    print(
        f"offense {list(result.offense)}  vs  defense {list(result.defense)}",
        file=stream,
    )
    print(f"  talent (T)       {d.talent:8.2f}", file=stream)
    print(
        f"  interaction (C)  {d.interaction:8.2f}   80% [{d.interaction_lower:+.2f}, "
        f"{d.interaction_upper:+.2f}]  P(C>0) {d.prob_interaction_positive:.2f}",
        file=stream,
    )
    print(f"  context (K)      {d.context:8.2f}", file=stream)
    print(
        f"  total value V    {d.total:8.2f}   points per 100 possessions", file=stream
    )
    print(
        f"  novelty {d.offense_novelty}  |  min player exposure "
        f"{result.support['min_player_possessions']} possessions"
        + (
            f"  |  unseen players {list(d.unseen_offense_players)}"
            if d.unseen_offense_players
            else ""
        ),
        file=stream,
    )
    return 0


def _cmd_app(args: argparse.Namespace, stream: TextIO) -> int:
    from courtgraph.app.server import serve

    return serve(args.port, args.ingest_dir, args.names, stream)


_COMMANDS = {
    "app": _cmd_app,
    "doctor": _cmd_doctor,
    "demo": _cmd_demo,
    "ingest": _cmd_ingest,
    "snapshot-from-shufinskiy": _cmd_snapshot_from_shufinskiy,
    "fetch-live": _cmd_fetch_live,
    "fit": _cmd_fit,
    "baselines": _cmd_baselines,
    "transport": _cmd_transport,
    "roles": _cmd_roles,
    "confirm": _cmd_confirm,
    "transport-mechanistic": _cmd_transport_mechanistic,
    "redundancy": _cmd_redundancy,
    "player-lift": _cmd_player_lift,
    "mechanistic": _cmd_mechanistic,
    "player-features": _cmd_player_features,
    "predict": _cmd_predict,
}


def main(argv: Sequence[str] | None = None, output: TextIO | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    args = build_parser().parse_args(argv)
    stream = output or sys.stdout
    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse guarantees a valid command
        raise AssertionError(f"Unhandled command: {args.command}")
    return handler(args, stream)

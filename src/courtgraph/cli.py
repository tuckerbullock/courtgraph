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
        help="directory holding the archive's nbastats*.csv / datanba*.csv / "
        "shotdetail*.csv (one file per provider, or one per provider per season)",
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
    "fit": _cmd_fit,
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

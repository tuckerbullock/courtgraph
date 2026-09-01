"""Leakage-safe splits + additive ridge RAPM baseline on a real stint file.

``courtgraph.chemistry.evaluate.evaluate_suite`` is the intended evaluation
entry point, but it also fits the low-rank interaction model, whose current
dense implementation does not scale to a full-season player pool (~1000
players): see ``docs/CURRENT_TASK.md``. Until that is reworked, this script
runs just the part that does scale -- the three leakage-safe holdouts and the
additive ridge baseline -- and reports the same macro (held-out-group) and
micro (stint) RMSE / MAE the suite would.

    uv run python scripts/eval_baseline.py path/to/stints.jsonl [--out summary.json]

Deterministic. No network. Reads the stint file only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from courtgraph.chemistry.baseline import AdditiveRidge
from courtgraph.chemistry.features import FeatureSpace
from courtgraph.chemistry.splits import SplitManifest, make_all_splits
from courtgraph.chemistry.stints import StintTable, lineup_id, pair_id, read_stints

_T0 = time.time()


def _log(message: str) -> None:
    print(f"[{time.time() - _T0:6.0f}s] {message}", flush=True)


def _groups(test_table: StintTable, manifest: SplitManifest) -> dict[str, list[int]]:
    """Bucket held-out stints exactly as ``evaluate._group_index`` does."""

    out: dict[str, list[int]] = defaultdict(list)
    if manifest.kind == "unseen_lineup":
        for i, stint in enumerate(test_table):
            out[lineup_id(stint.offense_player_ids)].append(i)
    elif manifest.kind == "unseen_pair":
        held = set(manifest.held_out_pairs)
        for i, stint in enumerate(test_table):
            ids = stint.offense_player_ids
            for a in range(5):
                for b in range(a + 1, 5):
                    key = pair_id(ids[a], ids[b])
                    if key in held:
                        out[key].append(i)
    else:
        for i, stint in enumerate(test_table):
            out[stint.season].append(i)
    return out


def _macro_rmse(
    groups: dict[str, list[int]],
    pred: np.ndarray,
    realized: np.ndarray,
    weight: np.ndarray,
) -> float:
    g_pred, g_real = [], []
    for idx_list in groups.values():
        idx = np.array(idx_list)
        w = weight[idx]
        g_pred.append(float(np.average(pred[idx], weights=w)))
        g_real.append(float(np.average(realized[idx], weights=w)))
    diff = np.array(g_pred) - np.array(g_real)
    return float(np.sqrt(np.mean(diff**2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stints", type=Path, help="stint file (.jsonl/.json)")
    parser.add_argument("--out", type=Path, help="also write the summary as JSON")
    args = parser.parse_args()

    table = read_stints(args.stints)
    _log(f"{len(table)} stints, {len(table.player_ids())} players")
    splits = make_all_splits(table)
    space = FeatureSpace.from_training(table)

    summary: dict[str, dict[str, float | int]] = {}
    for kind, manifest in splits.items():
        started = time.time()
        train = manifest.train_table(table)
        test = manifest.test_table(table)
        train_design = space.build(train)
        model = AdditiveRidge.fit(train_design, space)
        test_design = space.build(test)
        pred = model.predict(test_design)
        realized = test_design.y
        weight = test_design.weight
        mean_only = float(np.average(train_design.y, weights=train_design.weight))
        flat_pred = np.full_like(realized, mean_only)

        groups = _groups(test, manifest)
        macro_add = _macro_rmse(groups, pred, realized, weight)
        macro_mean = _macro_rmse(groups, flat_pred, realized, weight)
        micro_add = float(np.sqrt(np.average((pred - realized) ** 2, weights=weight)))
        micro_mean = float(
            np.sqrt(np.average((flat_pred - realized) ** 2, weights=weight))
        )
        summary[kind] = {
            "train_stints": len(train),
            "test_stints": len(test),
            "n_groups": len(groups),
            "l2_player": model.l2_player,
            "macro_rmse_additive": round(macro_add, 3),
            "macro_rmse_mean_only": round(macro_mean, 3),
            "macro_gain_pct": round(100 * (macro_mean - macro_add) / macro_mean, 1),
            "micro_rmse_additive": round(micro_add, 3),
            "micro_rmse_mean_only": round(micro_mean, 3),
        }
        _log(f"{kind}: {summary[kind]}  ({time.time() - started:.0f}s)")

    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

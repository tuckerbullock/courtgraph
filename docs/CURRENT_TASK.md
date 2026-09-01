# Current Task

Last updated: 2026-08-31

## State

Partly done — **leakage-safe splits + the additive ridge RAPM baseline ran on
the 266,518 real regular-season stints**; the low-rank chemistry model did
**not** — its current dense implementation does not scale to a full-season
player pool. Branch `task/real-stint-baseline` off `origin/main` (`0a2e30d`).

## What ran

### Splits (`courtgraph.chemistry.splits.make_all_splits`) — fine on real data

| holdout | train stints | test stints | leakage |
|---|---|---|---|
| chronological | 160,186 | 106,332 | none |
| unseen_pair | 245,905 | 20,613 (8 held pairs) | none |
| unseen_lineup | 262,752 | 3,766 (12 held lineups) | none |

### Additive ridge RAPM baseline vs predicting the league mean

Held-out **macro** RMSE (possession-weighted group means, the contract's
headline metric) and **micro** RMSE (per stint), points per 100 possessions:

| holdout | macro additive | macro mean-only | macro gain | micro additive | micro mean-only |
|---|---|---|---|---|---|
| chronological | 3.52 | **2.13** | **−65%** | 64.14 | 63.91 |
| unseen_pair | **3.74** | 6.37 | **+41%** | 59.63 | 59.94 |
| unseen_lineup | **4.35** | 8.39 | **+48%** | 48.69 | 49.52 |

- **unseen_pair / unseen_lineup:** the additive model beats the mean by ~40–48%
  at the group level — players' individual offense/defense estimates *do*
  generalize to teammate combinations never observed together. The expected,
  sane result, and the first evidence the pipeline works on real data.
- **chronological:** the baseline is *worse* than the league mean. Train is the
  early seasons, test is 2023-25; roster turnover and a shifted scoring
  environment mean coefficients fit on old data transfer poorly, and the
  "macro" bucket here is only **2 seasons** (an RMSE over two numbers). This is
  the hard split and the weak-signal case, not a bug.
- The ridge selection picked the **maximum** shrinkage (`l2_player=100`, grid
  top) on every split — the per-player effects are barely being fit. A wider
  grid and a proper hierarchical prior (contract rung 3) are the obvious next
  steps once the model scales.
- Stint-level (micro) RMSE is ~50–64 ppp100 everywhere — a 5–15-possession
  stint's rating is almost all possession noise, which is exactly why the
  contract's headline is the macro number.

Reproduce: `uv run python scripts/eval_baseline.py <stints.jsonl> --out summary.json`
(committed). Run against `data/nba_snapshots/rs_2020_2024/out/stints.jsonl`;
result saved locally as `data/nba_snapshots/rs_2020_2024/baseline_eval.json`
(gitignored). ~5 min per split.

## Why the low-rank chemistry model did not run

`ChemistryModel.fit` / `evaluate_suite` were built for the synthetic demo
(~17k stints, ~120 players). At **266k stints and 985 players** the dense
design blows up:

- `features.DesignMatrices` builds dense `(n_stints × n_players)` one-hots
  (~2 GB each here) and `baseline._solve_ridge` forms an
  `(n_players² · n_stints)` Gram — heavy but finishes in minutes.
- `chemistry_model.LowRankInteraction.fit` is the wall: its `half_step`
  allocates a dense `(n_stints × n_players × rank)` buffer (**~6 GB** at this
  size) and solves a `(players·rank)² ≈ 2955²` system over ~250k rows **per
  half-sweep** — ×2 per sweep, ×20 sweeps, ×3 selection folds, ×4 L2
  candidates, ×4 refits under `--evaluate`. That is ~10⁵–10⁶× the synthetic
  cost; one full fit did not complete in over an hour and was abandoned.

Each stint touches only 5 of 985 players, so the design is ~99.5% zeros. A
sparse formulation (sparse one-hots; accumulate the Gram as
`O(n · 25 · rank²)` instead of `O(n · (players·rank)²)`) should bring a full
fit back to minutes. **This is the next task.**

## Delivered on this branch (`017e197`)

- `courtgraph fit --bootstrap N` (default 8). Large stint files are far faster
  with `0`; the holdout RMSEs are unaffected (bootstrap only sizes the
  interaction uncertainty ensemble).
- `evaluate_suite` notes no longer hard-code "Synthetic demonstration data" —
  they say "Real reconstructed stints (no known ground truth)" for a
  non-synthetic table.
- `scripts/eval_baseline.py` — the scaling part of `evaluate_suite` (splits +
  additive baseline, macro + micro) as a standalone reproducible run.
- Tests: `fit --bootstrap 0 --evaluate`; negative `--bootstrap` rejected.
  158 → 160 tests, ruff / mypy / dependency-free path all pass.

## Next task (not started)

**Sparse-matrix rework of the chemistry model** so it fits at NBA scale:
`DesignMatrices` sparse one-hots; `AdditiveRidge._solve_ridge` and
`LowRankInteraction.fit` sparse Gram accumulation; numerical-equivalence tests
vs the current dense path on the small synthetic table; then re-run
`evaluate_suite` (or `courtgraph fit --evaluate --bootstrap 0`) on the 266k
stints and compare the full model to the additive baseline above and to the
contract's rung-2/3 references.

Then: wider `l2_player` grid + hierarchical prior; nullable `days_rest`
(schema v3) for the 68 season-opener quarantines; the 503 `network_required`
games. The 2024-25 playoffs archive stays held out for the transport test.

# Current Task

Last updated: 2026-09-01

## State

Done — **better-powered confirmation of the three interaction positives, run
on the 266k real regular-season stints.** Branch `task/confirmation-power` off
`main`. Committed; PR open.

**Result: one of the three survives. Role-conditioning significantly improves
prediction of a lineup's three-point-attempt share (`three_share`) — vs the
rung-3 baseline AND vs the permuted-role placebo, robust across K = 5 and
K = 7, on the widened 120-group unseen-lineup holdout. The aggregate
points/100 role effect is marginal and K = 5-dependent; the redundancy effect
does not survive.** All results preserved.

## What was built (`~confirmation-power`)

- **`make_all_splits`** threads `n_lineups` — the `unseen_lineup` holdout
  widens 60 → 120 groups. (`unseen_pair` stays at 40: the 15 %-of-stints
  exposure budget caps it; the bootstrap CI carries that uncertainty.)
- **`bootstrap_group_delta`** (`baseline_ladder.py`) — resamples the held-out
  group means with replacement (3,000×) and reports mean / 95 % CI /
  P(delta > 0) for `RMSE(baseline) − RMSE(model)`. CI excluding 0 = the
  improvement is not group-sampling noise.
- **`courtgraph confirm --input <stints> --profiles <profiles> --snapshot-dir
  <snap> [--k 3,5,7] [--lineups 120] [--boot N]`** — re-runs role, redundancy
  and mechanistic `three_share` on both structural holdouts across a K sweep,
  each with a CI vs rung 3 and vs the permuted-role placebo.
- 223 tests; ruff / mypy / dep-free clean.

## Result — 266k RS stints, K sweep {3, 5, 7}, 3,000 bootstrap resamples

`courtgraph confirm … --k 3,5,7 --lineups 120 --boot 3000`. Result:
`data/nba_snapshots/rs_2020_2024/chem_confirm_eval.json` (gitignored). Holdout
groups: unseen_lineup **120**, unseen_pair **40**.

### `three_share` — the confirmed positive

`RMSE(rung 3) − RMSE(role)` and `RMSE(placebo) − RMSE(role)` on the 120-group
unseen-lineup holdout (positive = role better):

| K | role RMSE | vs rung 3 (95 % CI) | vs placebo (95 % CI) |
|---|---|---|---|
| 3 | 0.0325 | +0.0006 [+0.0001, +0.0012] ✓ | +0.0003 [−0.0003, +0.0008] |
| **5** | **0.0321** | **+0.0010 [+0.0004, +0.0016] ✓** | **+0.0008 [+0.0002, +0.0014] ✓** |
| **7** | **0.0321** | **+0.0010 [+0.0003, +0.0017] ✓** | **+0.0011 [+0.0003, +0.0018] ✓** |

At K = 5 and K = 7 the improvement is significant against **both** the rung-3
baseline and the permuted-role placebo, and stable across the K sweep. Rung-3
RMSE on this outcome is ~0.033, so the effect is ~3 % of that — real but small.
On the 40-group unseen_pair holdout nothing reaches significance (P 0.6–0.9,
CI touches 0) — that holdout is too small.

### Role model on points/100 — marginal, K-fragile

| K | role RMSE | rung 3 RMSE | vs rung 3 (95 % CI) | P>0 |
|---|---|---|---|---|
| 3 | 6.062 | 6.061 | −0.002 [−0.071, +0.070] | 0.48 |
| **5** | **5.959** | 6.061 | **+0.101 [+0.00001, +0.203]** | 0.98 |
| 7 | 6.040 | 6.061 | +0.019 [−0.103, +0.140] | 0.62 |

At K = 5 (the pre-registered config from PR #20) the CI *just* excludes 0 and
P(delta > 0) ≈ 0.98 against both baseline and placebo — but K = 3 and K = 7
show nothing. The ~1–2 % points/100 role edge is **K = 5-dependent and not
robustly established**.

### Redundancy — not confirmed

`RMSE(rung 3) − RMSE(redundancy)` = +0.002, 95 % CI [−0.054, +0.051],
P = 0.55 on the 120-group holdout, unchanged across K (the model uses
continuous role vectors, which do not depend on K). The 0.5 % edge from PR #22
was group-sampling noise. The in-sample "all six ρ_d negative" remains a
descriptive observation, not a held-out predictive gain.

## Revised verdict

**On 266k real regular-season stints, the only non-additive lineup signal that
survives a properly-powered, bootstrap-CI, K-robust test is a small effect on
*shot selection*: role-conditioning predicts a lineup's three-point-attempt
share ~3 % better than additive talent, significant against baseline and
placebo. Lineup value in *points per 100* is not improved by any interaction
parameterisation at a level that survives proper power — the aggregate-scoring
non-additivity, if it exists, is below what 40–120 held-out group means can
resolve.**

Where lineups differ from the sum of their parts, it is in *how they shoot*,
not *how much they score*. `RESEARCH_CONTRACT.md` §17.1 (a significant macro
unseen-lineup improvement in the primary unit) remains **not met**.

## Candidate follow-up ideas — final status

1. Role/skill-conditioned interaction — done; points/100 effect **marginal,
   K-fragile** on proper power.
2. Mechanistic outcomes — done; **`three_share` confirmed** (significant, robust).
3. Skill redundancy — done; **not confirmed** on proper power.
4. Playoffs transport — done (null).
5. **Transaction backtest (T4) — NOT STARTED.** Needs a roster-change dataset.

Master-plan §45 player-lift also remains.

## Next action

Merge this branch. Then the user's call between candidate idea #5 (transaction
backtest — needs data acquisition), §45 player-lift, or writing the cycle-1
research report now that the interaction question has a defensible answer
(mostly null; one small shot-selection positive).

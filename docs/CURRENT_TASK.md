# Current Task

Last updated: 2026-09-01

## State

Done — **playoffs transport test (roadmap direction #3) built, tested, and run
on the 266k RS stints -> the held-out 2024-25 playoffs (3,325 stints).** Branch
`task/playoffs-transport` off `main` (`653c734`). Committed; PR open.

**Result: the interaction null holds under phase transport too -- a fifth
consecutive null.** But rung 3's empirical-Bayes predictive intervals
*transport* to the playoffs with near-nominal coverage, where rung 2's
bootstrap band does not. All results preserved.

## What was built (`af1d2d9`)

- **`src/courtgraph/chemistry/transport.py`** -- `evaluate_transport(train,
  test, ...)`: fits rungs 2/3(/4) on one stint table and evaluates on a
  disjoint second one. Distinct from `compare_rungs` (which partitions one
  table).
  - leakage gate: no shared game or stint reaches both sides; a coverage
    report (test players / pairs / lineup novelty vs. train);
  - macro RMSE + calibration over recurring playoff lineups, split by
    playoff-lineup **novelty** (`seen` / `partially-seen` / `unseen` vs. the
    regular season);
  - the rung-4 pair-level "seen pairs" test + placebo control, over
    RS-admitted pairs recurring in the playoffs;
  - possession-weighted error on **all / clutch / non-clutch** stints
    (clutch = one-possession game, 4th quarter or later).
- **The `playoff` context column is zeroed in the test design.** The regular
  season has no playoff rows, so that coefficient is pure prior; left in place
  it gives every playoff row a posterior variance of ~`tau_c2` (1e6) and the
  rung-3/4 predictive SD explodes to ~1000. Point predictions are unchanged
  (the coefficient is ~0). Reported in `zeroed_context_columns`.
- `calibration_line` now returns the uninformative line (slope 1, intercept 0)
  for < 2 groups or zero spread in the point predictions, instead of raising.
- `courtgraph transport --train ... --test ... [--rung4] [--bootstrap N]`.
- Validated on a two-phase synthetic with a shared, *transferable* planted pair
  effect: the pair-level test finds it (r4 0.47 vs placebo 0.66 vs additive
  1.39) and finds nothing when `tau_pair = 0`. 200 tests; ruff / mypy /
  dep-free clean.

## Result -- 266k RS -> 2024-25 playoffs

`courtgraph transport --train .../rs_2020_2024/out/stints.jsonl --test
.../all_2025_playoffs/out/stints.jsonl --bootstrap 120 --rung4`. Result:
`data/nba_snapshots/all_2025_playoffs/chem_transport_eval.json` (gitignored).

Clean transport: 0 shared games, all **210** playoff players seen in the RS,
**512 of 964** playoff offensive pairs admitted in the RS vocab, rung-3 EM
converged (78 iters).

### Point accuracy -- another interaction null

**Pair-level "seen pairs" test** (macro RMSE over 476 RS-admitted pairs
recurring in the playoffs, >= 5 playoff stints each):

| | macro RMSE |
|---|---|
| rung 2 (additive ridge) | 12.98 |
| rung 4 (explicit `gamma_ij`) | 13.22 |
| **rung 4, placebo pairs** | **13.24** |

Rung 4 is **worse** than additive and indistinguishable from its placebo. The
regular-season pair terms carry no transferable predictive value into the
playoffs.

**Macro RMSE over 157 recurring playoff lineups** (points per 100):

| bucket | groups | rung 2 | rung 3 | rung 4 |
|---|---|---|---|---|
| all playoff lineups | 157 | 23.92 | 24.00 | 24.23 |
| `seen` (five played together in RS) | 136 | 24.60 | 24.78 | 25.06 |
| `partially-seen` (pairs seen, five not) | 20 | 19.37 | 18.67 | **18.32** |
| `unseen` | 1 | -- | -- | -- |

Flat overall; rung 4 marginally worst on `seen`. The one place the interaction
models edge ahead is `partially-seen` (rung 4 ~5.5% below rung 2) -- but that
is 20 noisy group means, not a claim.

**Clutch** (302 one-possession 4th-quarter-or-later stints, possession-weighted
micro RMSE): rung 2 69.41, rung 3 69.33, rung 4 69.23. Non-clutch: 65.59 /
65.60 / 65.60. No clutch-specific interaction effect.

### Calibration -- rung 3's intervals transport, rung 2's do not

157 playoff lineup groups, coverage at 50 / 80 / 95 %:

| model | cov 50 / 80 / 95 | z_mean | z_sd | slope |
|---|---|---|---|---|
| rung 2 band | .30 / .53 / .75 | -0.16 | 1.69 | 0.73 |
| rung 3 EB | **.50 / .85 / .96** | -0.10 | 0.92 | 0.88 |
| rung 4 | .52 / .85 / .97 | -0.12 | 0.92 | 0.84 |

`z_mean` ~ 0 for all three: the RS-fit models' **average** playoff lineup
value is roughly unbiased -- offense and defense both tighten in the playoffs,
~cancelling in net rating, so there is no large systematic phase offset at the
lineup-value level. But rung 2's bootstrap band is badly overconfident
(`z_sd` 1.69; the 95% interval covers 75%), while **rung 3's empirical-Bayes
posterior stays near-nominal out of phase**. This mirrors the in-sample
structural-holdout finding and reinforces rung 3 as the reference baseline.

## Verdict against the contract

- The interaction question fails a fourth evaluation task: no interaction rung
  beats additive talent on held-out playoff prediction, and the pair terms are
  indistinguishable from a placebo. `RESEARCH_CONTRACT.md` sections 17.1 and 26
  ("successive models show no transferable interaction signal") both stand.
  **Fifth consecutive interaction null** (rung 5 low-rank; rung 3
  calibration-only; rung 4 in-sample; rung 4 pair-level; rung 4 transport).
- Rung 3's calibrated uncertainty **does** transport (near-nominal playoff
  coverage) -- a modest positive result for the EB model's interval quality.

## Roadmap -- the four directions (user: "do all of those")

1. **Report the "not supported" null -- NEXT.** Rungs 0-5 done on real data
   across four leakage-safe evaluation tasks (three in-sample holdouts + phase
   transport). Write it up as a standing findings document.
2. Better-powered pair evaluation -- DONE (`chem_rung4_pairlevel_eval.json`).
3. Playoffs transport -- DONE (this task).
4. A different interaction parameterization -- role/skill-conditioned pair
   effects (master plan section 21), **plus a new backlog item: quantify a
   player's effect on *improving their teammates'* individual production**
   (asymmetric "lift", not symmetric pair chemistry). To be appended to the
   backlog.

`ChemistryConfig` / `HierarchicalConfig` / `PairHierarchicalConfig` defaults
unchanged. No new data added; the playoff archive stays a held-out transport
target (it has now been used once, for this test, and should not enter
training).

## Next action

Merge `task/playoffs-transport` (the user asked). Then: write the interaction
null findings document (direction #1). Then: scope the "player lifts teammates"
backlog item.

# Current Task

Last updated: 2026-09-01

## State

Done — **the leakage-safe holdouts were widened, and the rung-2 vs rung-3
comparison re-run on the 266k real regular-season stints now has enough macro
groups to be conclusive.** Branch `task/wider-holdouts` off `origin/main`
(`f328a10`). Committed and pushed, PR open.

**Result: with properly-sized holdouts, rung 3 (hierarchical EB) beats rung 2
(additive RAPM) on calibration and stability on the two structural holdouts
`RESEARCH_CONTRACT.md` §17 cares about — clearing the rung-3 exit criterion.**
The earlier "rung 3 does not beat rung 2" conclusion was an artifact of the
too-small holdouts (8 pairs / 12 lineups / 2 seasons) — the §26 stop condition
flagged then, now resolved.

## What changed (`5d4364b`)

- **`make_unseen_pair_split`**: `n_pairs` 8 → 60; `max_test_fraction` 0.12 → 0.15;
  new `max_stints_per_pair_fraction` cap (default 1% of the table) so a handful
  of very-high-minute pairs cannot eat the budget and starve the group count.
  Real data: **8 → 40 held pairs**.
- **`make_unseen_lineup_split`**: `n_lineups` 12 → 60. Real data: **12 → 60**.
- **`evaluate._group_index`**: the chronological macro buckets by calendar month
  (`YYYY-MM`) instead of season. Real data: **2 → 13 groups**.
- All three splits stay outcome-blind and leakage-free (`verify_split`
  unchanged). Existing split / recovery tests unaffected; +2 new tests. 181
  tests pass.

## Result — rung 2 vs rung 3 on the 266k real stints (wider holdouts)

`courtgraph baselines --input …/rs_2020_2024/out/stints.jsonl --bootstrap 150`
(~18 min). Result: `data/nba_snapshots/rs_2020_2024/chem_rung3_eval_wide.json`
(gitignored). EM (78 iters, converged): `σ = 118.9`, `τ_off = 2.34`,
`τ_def = 1.83` ppp100.

**Held-out macro RMSE** (possession-weighted group means, points per 100):

| holdout | groups | rung 2 | rung 3 |
|---|---|---|---|
| chronological | 13 | 3.705 | **3.551** |
| unseen_pair | 40 | 19.568 | **19.203** |
| unseen_lineup | 60 | 5.383 | **5.256** |

Rung 3 is marginally better on point RMSE on all three. (unseen-pair's RMSE is
large because holding 40 representative pairs — not just the 8 highest-minute —
means the held-out group means carry real small-sample noise; the 8-pair run's
low RMSE was an artifact.)

**Calibration** — rung 3's Gaussian posterior vs rung 2's block-bootstrap band.
`z_sd` is the standardized-residual SD (ideal 1.0); coverage is nominal
50 / 80 / 95:

| holdout | rung 3 `z_sd` / slope / cov | rung 2 band `z_sd` / cov |
|---|---|---|
| chronological | 1.76 / −0.38 / 0.15·0.23·0.31 | 1.95 / 0.08·0.23·0.38 |
| unseen_pair | **1.06** / **1.00** / 0.45·0.78·0.93 | 1.45 / 0.48·0.73·0.88 |
| unseen_lineup | **1.04** / 0.73 / 0.40·0.70·**0.95** | 1.59 / 0.33·0.53·0.70 |

- **unseen_pair**: rung 3 is **well-calibrated** — `z_sd = 1.06`, calibration
  slope exactly `1.00`, coverage 0.45/0.78/0.93 vs nominal 0.50/0.80/0.95. Rung
  2's band is over-dispersed (`z_sd = 1.45`).
- **unseen_lineup**: rung 3 clearly beats the rung-2 band — `z_sd 1.04` vs
  `1.59`, and coverage far closer to nominal (0.95 vs 0.70 at the 95% level).
  Rung 3 still under-covers at 50/80 (intervals slightly narrow — no interaction
  term).
- **chronological**: **both models fail** — coverage ~0.15–0.38 against nominal,
  `z_mean ≈ 2.4` (systematic *under*-prediction of held-out month value). Static
  heavily-shrunk player effects cannot track league scoring-environment and
  roster drift across seasons. The contract (§15) asks only to *report* coverage
  under chronological shift; this is that report.

## Verdict against the contract

`RESEARCH_CONTRACT.md` §11: rung 3 must beat rung 2 "on calibration and
stability". **Met** on the two structural holdouts (unseen-pair, unseen-lineup)
— which are the ones §17's chemistry-usefulness thresholds are defined on —
with marginally better point RMSE as well. Both models fail the chronological
holdout under distribution shift; that is a shared limitation to carry forward,
not a rung-3 shortfall. §26's "split sizes too small" stop condition no longer
applies (13 / 40 / 60 groups).

**Rung 3 is now the established reference baseline** the chemistry claims (§17)
are measured against, and it is a well-calibrated one on the structural
holdouts.

## Next candidate tasks (not started)

1. **Rung 4 — explicit teammate-pair interactions**, evaluated against the
   rung-3 reference on the wider holdouts (§11: rung 4 must beat rung 2 on seen
   pairs; §17: rungs 5–7 must beat rung 4 on unseen-pair error).
2. Chronological-shift calibration: per-season player deviations (dynamic
   hierarchy is rung 8+ / out of cycle 1) or variance inflation / split-conformal
   for the shift holdout.
3. Recover the 840 quarantined games; nullable `days_rest` (schema v3); the
   2024-25 playoffs transport test.

`ChemistryConfig` defaults unchanged.

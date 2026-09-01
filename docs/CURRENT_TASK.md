# Current Task

Last updated: 2026-09-01

## State

Done — **model-ladder rung 3 (empirical-Bayes hierarchical player model) built,
tested, and run against rung 2 on the 266k real regular-season stints**. Branch
`task/rung3-hierarchical` off `origin/main` (`139ab01`). Committed and pushed,
PR open.

Rung 3 is now the reference baseline `RESEARCH_CONTRACT.md` §17 requires. **On
this data it does not cleanly beat rung 2 "on calibration and stability"**
(§11 exit criterion) — and the leakage-safe holdouts turn out to have too few
macro groups (2 / 8 / 12) for the calibration comparison to be conclusive, a
§26 stop condition. Both are reportable findings; both are preserved.

## What was built (`23dd20e`, `49cc608`)

- **`src/courtgraph/chemistry/hierarchical.py`** — `HierarchicalRidge`: the
  rung-2 additive design (`y_s ≈ Cθ_c + Σα_i − Σβ_j`) with the single CV-picked
  `l2_player` replaced by variance components `(σ², τ_off², τ_def²)` learned by
  EM. Reuses `baseline._normal_equations` for the `(n_context+2P)` weighted Gram
  (built once); each EM step re-solves `M = gram/σ² + diag(Λ)` by Cholesky and
  updates the components by the standard Gaussian-LMM EM (the residual-variance
  trace term is `tr(gram·V) = σ²(d − Σλ_k V_kk)`, diagonal-only). Marginal
  log-likelihood monitored for monotonicity. Static players, Normal prior, pure
  numpy, no RNG — deterministic. `group_predictive` propagates the Gaussian
  posterior (`g'M⁻¹g`) + outcome noise (`σ²/Σw`) to a per-group interval.
  Validated on well-specified synthetic: recovers the realised player-pool
  effect SD to ~2%, log-lik monotone, group coverage 0.50/0.81/0.96 for nominal
  50/80/95.
- **`src/courtgraph/chemistry/calibration.py`** — coverage @50/80/95, WLS
  calibration line (slope→1, intercept→0), standardized-residual moments,
  width-vs-error correlation. The contract's named calibration currency.
- **`src/courtgraph/chemistry/baseline_ladder.py`** + `courtgraph baselines` —
  per holdout, fit rung 2 + rung 3, bucket test rows with
  `evaluate._group_index`, report both models' macro/micro RMSE and interval
  calibration (rung 2 gets an approximate block-bootstrap-over-games band).
  `evaluate.py` / `ChemistryModel` and their tests are **untouched**.
- 179 tests (+15); ruff / mypy / dependency-free path clean.

## Result — rung 2 vs rung 3 on the 266k real stints

`courtgraph baselines --input …/rs_2020_2024/out/stints.jsonl --bootstrap 150`
(~15 min). Result JSON: `data/nba_snapshots/rs_2020_2024/chem_rung3_eval.json`
(gitignored).

**Variance components** (EM, 78 iters, converged): `σ = 118.9`,
`τ_off = 2.34`, `τ_def = 1.83` ppp100 — the additively-separable player-impact
signal is small. Implied shrinkage `σ²/τ_off² ≈ 2580` (vs rung-2's CV pick of
`l2_player = 100`).

**Held-out macro RMSE** (possession-weighted group means, points per 100):

| holdout | groups | rung 2 | rung 3 |
|---|---|---|---|
| chronological | 2 | 3.523 | **3.409** |
| unseen_pair | 8 | **3.738** | 5.545 |
| unseen_lineup | 12 | 4.349 | **4.058** |

Rung 3's data-driven heavy shrinkage helps chronological and unseen-lineup and
**badly hurts unseen-pair** — the same tension the earlier `l2_player`-grid
sweep found: the unseen-pair holdout's own optimum is *light* shrinkage
(`l2 ≈ 10`), everything else wants heavy, and **one learned variance component
cannot serve both.** Micro (stint) RMSE is within 0.35 ppp100 across all three.

**Calibration** (rung 3 posterior vs rung 2 bootstrap band):

| holdout | rung3 cov 50/80/95 | rung3 slope | rung2-band cov 50/80/95 |
|---|---|---|---|
| chronological | 0.00 / 0.00 / 0.00 | −0.32 | 0.00 / 0.50 / 0.50 |
| unseen_pair | 0.50 / 0.50 / 0.62 | 1.32 | 0.38 / 0.62 / 0.88 |
| unseen_lineup | 0.33 / 0.58 / **0.92** | 0.57 | 0.42 / 0.58 / 0.75 |

- **Both models under-cover the structural holdouts.** Rung 3's intervals are
  systematically too narrow (`mean_predictive_sd` 0.9–3.5 ppp100 while the
  realised spread is larger) — model misspecification (no interaction term,
  static players vs roster drift), not a bug.
- Rung 3's `z_mean > 0` on chronological (3.2) and unseen_pair (1.7): it
  **under-predicts** held-out group value — heavy shrinkage pulls the
  above-average groups (which survive the split budgets) toward the league mean.
- Rung 3's 95% coverage on unseen-lineup (0.92) is the one place it clearly
  beats the rung-2 band (0.75); its 50/80 are worse.
- **`chronological` has 2 groups and `unseen_pair` has 8** — coverage over 2–8
  points and a WLS line over 2 points are not meaningful. `make_unseen_pair_split`
  held only 8 pairs (its `max_test_fraction` budget + `min_test_stints` on 245k
  stints).

## Verdict against the contract

`RESEARCH_CONTRACT.md` §11: rung 3 must beat rung 2 "on calibration and
stability". **Not met on this data.** Rung 3 is better calibrated only at the
95% level on unseen-lineup; it is worse elsewhere and worse on unseen-pair point
error. §26 stop condition "split sizes are too small for reliable comparison"
**applies** — the macro group counts are 2 / 8 / 12.

The rung-3 machinery is a permanent asset (the contract needs it to exist), and
these numbers are the honest reference. Chemistry-usefulness claims (§17) are
still measured against rung 3, so the reference is recorded even though rung 3
did not out-calibrate rung 2 here.

## Next candidate tasks (not started — a genuine fork)

1. **Widen the leakage-safe holdouts** so the macro comparison has enough groups
   (raise `make_unseen_pair_split` / `make_unseen_lineup_split` budgets;
   consider rolling-origin chronological folds per §17). Prerequisite for *any*
   reliable rung-N calibration comparison.
2. **Rung 4 — explicit teammate-pair interactions.** The single-variance-
   component ceiling (unseen-pair wants light pooling, the rest heavy) is
   exactly what a per-pair term addresses.
3. A structure-aware rung-3 prior (per-pair or per-lineup variance inflation) /
   split-conformal intervals for the shift holdouts — deferred rung-3 follow-ups.

`ChemistryConfig` defaults unchanged. The 2024-25 playoffs archive is still held
out for the transport test.

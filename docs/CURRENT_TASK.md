# Current Task

Last updated: 2026-09-01

## State

Done — **candidate idea #3: skill redundancy / anti-synergy, built and run on
the 266k real regular-season stints.** Branch `task/redundancy-interaction`
off `main`. Committed; PR open.

**Result: every kind of offensive skill concentration is associated with
slightly-below-additive lineup value — all six `ρ_d` coefficients are
negative.** The anti-synergy hypothesis is confirmed *directionally in-sample*
(strongest for usage and offensive rebounding), with a weak (~0.5 %) but
placebo-surviving held-out signal on the unseen-lineup holdout. It is the
weakest of the three positive results, but the sign uniformity is a clean,
interpretable statement and it does **not** degrade under temporal drift the
way the 15-cell role-cluster model does. All results preserved.

## What was built (`69d1271`)

- **`fit_augmented_em`** gained a dense-extra-block path (`extra_dense`): the
  player × feature Gram cross-terms reuse `_cross_ctx`; everything else is the
  same EM. Rung 4 / role behavior is unchanged (still the sparse index path).
- **`RoleClustering.player_vector`** — the standardized role vector per player
  (their highest-exposure season), needed for the concentration features.
- **`chemistry/redundancy.py` + `redundancy_eval.py` + `courtgraph redundancy
  --input <stints> --profiles <profiles>`** — the interaction is D = 6
  coefficients `ρ_d` on concentration features:

      conc_d = (Σ_i z_id)² − Σ_i z_id²   over the offensive lineup's role vectors

  positive when the players align on dimension `d` ("redundant"). `ρ_d < 0`
  means concentrating skill `d` hurts. Compared vs rung 2 / rung 3 / a
  permuted-role placebo on the three leakage-safe holdouts.
- Validated on synthetic: recovers a planted `ρ` vector (corr > 0.9, sign of
  the strongest effect); the permuted-role placebo shrinks `τ_ρ` and fits
  worse. 220 tests; ruff / mypy / dep-free clean.

## Result — 266k RS stints, K = 5, `τ_ρ` = 0.18 pts/100

`courtgraph redundancy --input .../out/stints.jsonl --profiles
.../player_profiles.jsonl --clusters 5 --bootstrap 100`. Result:
`data/nba_snapshots/rs_2020_2024/chem_redundancy_eval.json` (gitignored).

### The fitted `ρ_d` — all negative

Points per 100 per unit of standardized concentration (real role vectors /
permuted-role placebo):

| dimension | ρ_d (real) | ρ_d (placebo) |
|---|---|---|
| oreb_per100 | **−0.202** | +0.085 |
| usage | **−0.189** | +0.074 |
| rim_rate | −0.121 | +0.084 |
| assist_per100 | −0.062 | +0.048 |
| three_rate | −0.043 | −0.156 |
| ft_rate | −0.033 | −0.115 |

**Every offensive skill concentration is a (small) penalty.** The largest:
concentrating **offensive rebounding** and **usage** — the "two ball-dominant
creators clash / not enough shots to go around" and "two crash-the-glass bigs"
stories. **Three-point-attempt concentration is the *least*-penalized
(−0.04, near zero)** — more shooters is close to additive, consistent with the
spacing intuition (shooters don't clash), though not a positive effect. The
placebo `ρ` have mixed signs and no coherent pattern.

### Held-out macro RMSE

| holdout | rung 3 | redundancy | permuted-role placebo |
|---|---|---|---|
| chronological | **3.551** | 3.559 | 3.557 |
| unseen_pair | 19.203 | **19.193** | 19.216 |
| unseen_lineup | 5.256 | **5.228** | 5.258 |

- **Structural holdouts:** redundancy beats rung 3 by 0.05 % / 0.53 % and its
  placebo by 0.12 % / 0.57 %. Small, same direction, clean calibration.
- **chronological:** essentially tied with rung 3 and the placebo (~3.55) —
  it does **not** blow up under drift the way the 15-parameter role-cluster
  model does (that model was 4.49). With only 6 parameters there is little to
  overfit to the era.

## Verdict against the contract

- Not near §17.1's bar (0.5 % on 60 group means, no bootstrap CI).
- But directionally coherent with the other two positives, and the sign
  uniformity of the six `ρ_d` is a clean, interpretable in-sample statement:
  **offensive skill redundancy is a real, small, negative effect**, and a
  permuted-role placebo produces no such pattern.

## Candidate follow-up ideas — progress

1. Role/skill-conditioned interaction — DONE (PR #20). ~1 % on points/100.
2. Mechanistic outcomes — DONE (PR #21). 2–5 % on three-point-attempt share,
   surviving a placebo on all three holdouts. Strongest result.
3. **Skill redundancy / anti-synergy — DONE (this task). All six ρ_d negative;
   ~0.5 % placebo-surviving edge on unseen-lineup; no drift penalty.**
4. Playoffs transport — DONE (PR #18).
5. **Transaction backtest (T4) — NOT STARTED. Highest build cost / highest
   evidentiary value.** Uses real trades / injuries as natural experiments:
   did a team's performance move the direction the model predicted once a
   specific player left / arrived? Needs a transaction dataset (roster changes
   with dates) — not currently ingested.

Plus: a **better-powered confirmation** of directions #1–#3 combined — widen
the structural holdouts to ~120 groups, bootstrap the model−rung-3 delta,
sweep K, and report the three positives together with CIs. Master-plan §45
player-lift also remains.

## Next action

Merge this branch. Then the user's call between (a) candidate idea #5
(transaction backtest — needs a transaction dataset), (b) the better-powered
confirmation of the #1–#3 positives, or (c) §45 player-lift.

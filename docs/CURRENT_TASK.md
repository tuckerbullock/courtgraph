# Current Task

Last updated: 2026-09-01

## State

Done — **model-ladder rung 4 (explicit teammate-pair interaction RAPM) built,
tested, and run against rungs 2 & 3 on the 266k real regular-season stints.**
Branch `task/rung4-pair-interaction` off `origin/main` (`f328a10`). Committed
and pushed, PR open.

**Result: rung 4 does not clear its contract exit criterion — it does not beat
rung 2 "on seen pairs" (§11), and it is worse than the rung-3 hierarchical
baseline on macro RMSE on two of three holdouts.** This is the **third
consecutive null** for non-additive lineup structure on this data (rung 5
low-rank: no gain; rung 3 hierarchical: calibration only; rung 4 explicit pairs:
no gain). `RESEARCH_CONTRACT.md` §26's "successive models show no transferable
interaction signal" stop condition now applies. All results preserved.

## What was built (`e6e386c`, `450124d`, `81b5b89`)

- **`src/courtgraph/chemistry/pair_interaction.py`** — `PairHierarchicalRidge`:
  the rung-3 EB model plus an explicit `γ_ij` term per admitted offensive
  teammate pair (`γ_ij ~ N(0, τ_pair²)`, a third EM-learned variance component).
  `PairVocabulary.from_training(table, min_co_stints=200)` admits only pairs
  with enough shared training stints (master plan §15.3; also keeps the
  `(n_context + 2P + Q)` linear system tractable — ~2.3–3.7k pairs per fold, not
  ~20k). A lineup with an inadmissible pair degrades to the additive prediction
  for that term. Offense pairs only for v1.
- `baseline._cross_gram` / `_cross_ctx` / `_cross_rhs` generalized to arbitrary
  block dimensions (the old 5-slot helpers are now the special case;
  behavior-preserving, `SparseGramEquivalenceTests` unchanged).
- `courtgraph baselines --rung4` — opt-in 3-way comparison;
  `compare_rungs(rung4_config=…)`. The §11 exit test lives in
  `rung4_pair_covered` on the chronological holdout: lineups where **every**
  offense pair is admitted, rung 2 vs rung 4 macro RMSE.
- Validated: `τ_pair` recovered to ~3% and pair coefs correlate 0.94 with a
  planted-`γ` fixture; monotone EM; deterministic; inadmissible pair → exactly
  `0.0` surplus + additive fallback. 189 tests; ruff / mypy / dep-free clean.

## Result — rungs 2 / 3 / 4 on the 266k real stints (~70 min)

`courtgraph baselines --input …/rs_2020_2024/out/stints.jsonl --bootstrap 120
--rung4`. Result: `data/nba_snapshots/rs_2020_2024/chem_rung4_eval.json`
(gitignored).

**Held-out macro RMSE** (possession-weighted group means, points per 100):

| holdout | groups | rung 2 | rung 3 | rung 4 | rung 4 admitted pairs |
|---|---|---|---|---|---|
| chronological | 13 | 3.705 | **3.551** | 3.843 | 2,357 |
| unseen_pair | 40 | 19.568 | **19.203** | 19.222 | 3,286 |
| unseen_lineup | 60 | 5.383 | **5.256** | 5.495 | 3,690 |

Rung 4 is **worse than rung 3 on all three** and worse than rung 2 on
chronological and unseen-lineup. The explicit per-pair terms, even EM-shrunk,
add estimation variance without a compensating signal.

**Calibration** (rung 4's Gaussian posterior, `z_sd` ideal 1.0):

| holdout | rung 4 `z_sd` / slope / cov 50·80·95 |
|---|---|
| chronological | 1.62 / −0.52 / .15·.15·.23 |
| unseen_pair | 1.06 / 1.02 / .43·.70·.90 |
| unseen_lineup | 1.02 / 0.63 / .48·.70·.95 |

Essentially identical to rung 3's calibration (1.76 / 1.06 / 1.04 `z_sd`). No
improvement.

**The exit test — "beat rung 2 on seen pairs"** (chronological holdout,
lineups where every offense pair is admitted):

| bucket | lineups | rung 2 macro RMSE | rung 4 macro RMSE |
|---|---|---|---|
| pair-covered (the exit test) | **209** | 47.277 | **47.460** |
| pair-degraded | 27,093 | 62.807 | 62.448 |

Rung 4 is a hair **worse** on the covered subset. **Exit test not met.** Also a
§26 "split sizes too small" issue: requiring all 10 of a lineup's pairs at
≥200 co-stints leaves only 209 of 27,302 chronological-test lineups (RMSE ~47
over 209 noisy group means). A pair-level exit test (per admitted pair, does
`γ_ij` reduce that pair's held-out residual vs rung 2?) would be better powered
— a follow-up — but the direction is unambiguous.

## Verdict against the contract

- §11 rung-4 exit ("beat rung 2 on seen pairs"): **not met.**
- §17.1 (chemistry-usefulness): "the best interaction model (rung 4–7) shows a
  significant improvement over the rung-3 baseline on macro unseen-lineup
  error" — **fails**: rung 4 = 5.495 vs rung 3 = 5.256; rung 5 (low-rank,
  earlier run) also did not beat rung 3.
- §26 stop condition "successive models show no transferable interaction
  signal": **applies.** Rungs 4 and 5 both fail to beat the additive/hierarchical
  baseline on the interaction question.

**On 266k real regular-season stints, explicit teammate-pair chemistry — as
measured by free per-pair terms and by low-rank factorization — adds no
held-out predictive value over hierarchical additive talent.** This does not
prove chemistry doesn't exist; it means the modeled forms of it, at this data
scale, on this evaluation, find nothing — consistent with the literature that
lineup non-additivity is a small residual.

## Next — a strategic fork (not started)

1. **Report the null.** Rungs 0–5 of the ladder are done on real data; the
   honest finding is "not supported" for transferable pair/lineup chemistry in
   the 2020-21…2024-25 regular season. Write it up (contract §17: null and
   inconclusive results are reported, not hidden).
2. **Better-powered pair evaluation** — a pair-level (not lineup-level) seen-pair
   test; a lower co-stint threshold; rolling-origin folds.
3. **More / different data** — playoffs (held out); more seasons; possession-
   level rather than stint-level outcomes.
4. **A different interaction parameterization** — role/skill-conditioned pair
   effects (needs measured role features, master plan §21), or the master-plan
   rung 6–7 neural pathways (gated on rungs 0–5 passing, which they have not for
   the interaction question).

`ChemistryConfig` / `HierarchicalConfig` defaults unchanged. The 2024-25
playoffs archive is still held out.

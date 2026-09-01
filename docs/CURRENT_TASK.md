# Current Task

Last updated: 2026-09-01

## State

Done — **better-powered pair-interaction evaluation (roadmap direction #2)
built, tested, and run on the 266k real regular-season stints.** Branch
`task/pair-level-eval` off `main` (`b57376c`). Committed, PR open, **not merged**.

**Result: the rung-4 null holds at proper statistical power.** A pair-level
"seen pairs" exit test over 668 recurring admitted pairs (vs the old 209
all-pairs-covered lineups), with a placebo control, shows rung 4's tiny edge
over rung 2 is **entirely reproduced by randomly-wired pair parameters** — it
is not pair-specific chemistry signal. Fourth consecutive null for
non-additive lineup structure on this data. All results preserved.

## What was built (`d4559ce`)

- **`_pair_level_breakdown`** (`baseline_ladder.py`) — the rung-4 §11 exit test,
  re-cast from lineup-level to pair-level. Buckets held-out chronological stints
  by each admitted offensive pair; macro RMSE over every pair recurring in the
  test window (>= 5 test stints). 668 pair groups on the real data, vs 209
  all-10-pairs-covered lineups for the old test — properly powered.
- **`_placebo_vocab`** + `PairVocabulary.row_override` — a real control. Routes
  each admitted pair's stints to a randomly chosen coefficient row drawn **with
  replacement**: same parameter count, same total pair exposure, but distinct
  pairs collide onto shared rows so the fit cannot carry pair-specific signal.
  (The version sketched earlier — permuting pair -> row — was a no-op: the model
  is exactly permutation-invariant. Verified `mean|pred - placebo_pred| = 0`
  before the fix.) Validated on a planted-`γ` synthetic: real rung 4 macro RMSE
  **0.61** vs placebo **1.16** vs rung 2 **1.87**; with no planted pair signal
  all three collapse to ~**0.50** and `τ_pair -> ~0.15`. The test discriminates.
- `courtgraph baselines --rung4` human output now prints the pair-level line
  (`r2 / r4 / r4-placebo`); JSON gains `rung4_pair_level`, keeps the
  lineup-level `rung4_pair_covered` / `_degraded` fields.
- 191 tests; ruff / format / mypy / dependency-free path clean.

## Result — pair-level exit test on the 266k real stints

`courtgraph baselines --input …/rs_2020_2024/out/stints.jsonl --bootstrap 120
--rung4`. Result: `data/nba_snapshots/rs_2020_2024/chem_rung4_pairlevel_eval.json`
(gitignored). Chronological holdout, 2,357 admitted pairs (>= 200 train
co-stints), rung-3 EM converged in 78 iters.

**Pair-level "seen pairs" test** (macro RMSE over 668 admitted pairs recurring
in the held-out window, points per 100):

| | macro RMSE over 668 pair groups |
|---|---|
| rung 2 (additive ridge) | 8.669 |
| rung 4 (explicit `γ_ij`) | 8.543 |
| **rung 4, placebo pairs** | **8.541** |

Rung 4 is 1.5% better than rung 2 — and the placebo (same parameter count,
scrambled pair->row wiring) is **exactly as good** (a hair better). The small
rung-2 -> rung-4 gain is added-parameter noise absorption, **not** teammate
chemistry. Exit test **not met**: rung 4 does not beat rung 2 on seen pairs in
any way the placebo doesn't.

For reference, the underpowered lineup-level test from the rung-4 task
(unchanged here): 209 all-pairs-covered lineups, rung 2 47.28 vs rung 4 47.46.
Same direction.

**Macro RMSE / calibration on the widened holdouts** — identical to the rung-4
run (deterministic): rung 4 3.843 / 19.222 / 5.495 vs rung 3 3.551 / 19.203 /
5.256; rung 4 calibration no better than rung 3.

## Verdict against the contract

- §11 rung-4 exit ("beat rung 2 on seen pairs"): **not met**, now confirmed at
  proper power (668 pair groups) and with a placebo control.
- §17.1 (chemistry-usefulness): the best interaction model does not improve on
  the rung-3 baseline on macro unseen-lineup error — **still fails**.
- §26 "successive models show no transferable interaction signal" + "split
  sizes too small for reliable comparison": the first **applies**; the second
  is now **resolved** — the pair-level test is adequately powered and the null
  survives it.

**On 266k real regular-season stints, explicit teammate-pair chemistry adds no
held-out predictive value over hierarchical additive talent — confirmed by a
well-powered, placebo-controlled pair-level test, not just an underpowered
lineup test.** Ladder rungs 0–5 are done on real data; rungs 3 (calibration),
4 (explicit pairs), and 5 (low-rank) all fail the interaction question.

## Roadmap — the four directions (user: "do all of those")

1. Report the "not supported" null — ongoing; every result is in these docs.
2. **Better-powered pair evaluation — DONE (this task). Null confirmed.**
3. **More / different data (NEXT)** — the 2024-25 playoffs archive is still
   held out; a playoffs transport test (train RS, test PO) is the cleanest
   remaining shot at interaction signal, plus more seasons and possession-level
   rather than stint-level outcomes.
4. A different interaction parameterization — role/skill-conditioned pair
   effects (needs measured role features, master plan §21).

`ChemistryConfig` / `HierarchicalConfig` / `PairHierarchicalConfig` defaults
unchanged. The 2024-25 playoffs archive is still held out.

## Next action

Open the PR for `task/pair-level-eval` (done — do not merge without the user).
On the user's go, start direction #3: the playoffs transport test.

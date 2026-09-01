# Interaction findings — is lineup chemistry predictively real?

Last updated: 2026-09-01

This is a standing findings document for the north-star question of research
cycle 1 (`RESEARCH_CONTRACT.md` §2):

> Can we estimate how NBA players will fit together before we have observed that
> exact combination on the court?

decomposed as `lineup value = individual talent + player interactions +
context`. It records what the model ladder has and has not shown on real NBA
data, in the contract's own language: a claim is **supported**, **not
supported**, or **inconclusive** (§7), and a well-characterised null is a valid
outcome (§7, §26).

## Verdict

**On 266,518 real regular-season stints (2020-21 … 2024-25) and the held-out
2024-25 playoffs, transferable teammate-pair / lineup chemistry — as measured
by every model form on the ladder through rung 5 — is NOT SUPPORTED.** No
interaction rung improves held-out prediction over hierarchical additive
talent on any of four leakage-safe evaluation tasks, and the explicit per-pair
terms are statistically indistinguishable from a placebo with the same
parameter count.

This is **not** a causal claim that no two players affect each other. It is a
predictive statement about the modelled forms of interaction at this data
scale, on this evaluation. The distinctions the contract requires (§8) are
kept: what follows separates observed facts, adjusted associations, and
predictions.

## What was tried

All models share the rung-2/3 weighted-Gaussian RAPM frame — separate
offensive and defensive per-player talent, ridge / empirical-Bayes shrinkage,
context columns — and add a different interaction term.

| rung | interaction form | generalises to unseen pairs? |
|---|---|---|
| 3 — hierarchical EB (`hierarchical.py`) | none; EM-learned variance components on additive talent | n/a (reference baseline) |
| 4 — explicit pairs (`pair_interaction.py`) | free `γ_ij ~ N(0, τ_pair²)` per offensive pair with ≥ 200 training co-stints | no — falls back to additive |
| 5 — low-rank (`chemistry_model.py`) | `γ_ij ≈ u_i · v_j` (provision × need), rank 3, alternating ridge on the cross-fitted residual | yes — by construction |

Rungs 0–2 (context mean, EB-shrunk lineup mean, additive ridge RAPM) are the
required predecessors and are established. Rungs 6–7 (neural) stay gated by
§26: they begin only after rungs 0–5 pass for the target task, which they have
not.

## Evaluation tasks

All outcome-blind, leakage-checked (`splits.verify_split`,
`transport.evaluate_transport`'s gate), macro-averaged over held-out groups.

1. **chronological** — train early games, test later ones by `game_date`
   (13 calendar-month groups).
2. **unseen_pair** — 40 teammate pairs with every co-play removed from
   training, each player kept individually observed.
3. **unseen_lineup** — 60 exact five-man sets with every training appearance
   removed.
4. **playoffs transport** — train the full regular season, test the held-out
   2024-25 playoffs (157 recurring lineup groups; 0 shared games; all 210
   playoff players seen in the regular season).

Plus a **pair-level "seen pairs" test**: bucket held-out stints by each
admitted pair, macro RMSE over every pair recurring in the test window
(668 pairs in-sample, 476 in the playoffs), against a **placebo** — a rung-4
fit whose pair→coefficient wiring is scrambled (same parameter count, same
exposure, no pair-specific signal). A real pair effect must beat its placebo.

## Results — held-out macro RMSE (points per 100 possessions)

| task | groups | rung 2 | rung 3 | rung 4 | rung 5 |
|---|---|---|---|---|---|
| chronological | 13 | 3.71 | **3.55** | 3.84 | ≈ rung 2 |
| unseen_pair | 40 | 19.57 | **19.20** | 19.22 | −0.08 % vs r2 |
| unseen_lineup | 60 | 5.38 | **5.26** | 5.49 | −8.4 % vs r2 |
| playoffs transport | 157 | **23.92** | 24.00 | 24.23 | not run |

Rung 3 (calibration, below) is the best point model on the structural
holdouts; rung 2 is marginally best on transport. **Neither interaction rung
(4 or 5) beats the additive/hierarchical baseline anywhere.**

### Pair-level "seen pairs" test + placebo

| context | pair groups | rung 2 | rung 4 | rung 4 placebo |
|---|---|---|---|---|
| in-sample (chronological) | 668 | 8.67 | 8.54 | **8.54** |
| playoffs transport | 476 | 12.98 | 13.22 | **13.24** |

Rung 4's edge over rung 2 in-sample (1.5 %) is exactly matched by the placebo;
in the playoffs rung 4 is worse than rung 2. **The per-pair terms carry no
pair-specific signal** — they act as extra regularised parameters that absorb
additive misfit.

### Calibration — the one positive result

Rung 3's empirical-Bayes predictive intervals are well-calibrated where rung
2's block-bootstrap band is not, and this **transports to the playoffs**:

| task | model | cov 50/80/95 | z_sd |
|---|---|---|---|
| unseen_pair | rung 2 / rung 3 | .48·.72·.88 / **.45·.78·.93** | 1.45 / **1.06** |
| unseen_lineup | rung 2 / rung 3 | .33·.53·.70 / **.40·.70·.95** | 1.59 / **1.04** |
| playoffs | rung 2 / rung 3 | .30·.53·.75 / **.50·.85·.96** | 1.69 / **0.92** |

`z_mean ≈ 0` on the playoffs for all rungs: the regular-season model's
*average* playoff lineup value is roughly unbiased — offense and defense both
tighten in the playoffs, ~cancelling in net rating. So rung 3 is the
established reference baseline, with calibrated uncertainty that holds out of
phase. Both models fail the chronological holdout's mean (systematic
under-prediction under era/roster drift; a shared, documented limitation).

## What this establishes — and what it does not

**Supported:**

- Hierarchical additive talent (rung 3) is a hard baseline: calibrated,
  seed-stable, and its intervals transport to the playoffs.
- Free per-pair chemistry terms and low-rank provision/need factorisation add
  **no** held-out predictive value over it, on four evaluation tasks, and the
  per-pair terms do not beat a parameter-matched placebo.

**Not established / out of scope — the null does not rule these out:**

- **Talent absorption.** If "makes teammates better" is a stable individual
  trait, it is already inside a player's additive coefficient. These models
  cannot separate "no interaction" from "interaction collinear with average
  individual impact." → the *player-lift* backlog item.
- **No features.** Every model uses bare player indicators. Role / skill
  complementarity (spacing, playmaker-finisher, rim coverage) would only show
  as a pair term if that exact pair recurs; parameterised by role it might
  generalise (master plan §21).
- **Noise floor.** Single-stint outcome SD is ≈ 119 pts/100; additive talent
  SD is ≈ 2.3. Pair effects have far less exposure per parameter. Effects
  below roughly 0.5 pts/100 would need many more seasons to resolve.
- **Dynamic chemistry** (develops over a season) — contract §27, out of cycle 1.
- **Offense-only.** Rung 4/5 model offensive pairs only; defensive `γ_ij^def`
  is a documented follow-up.

## What would change the verdict

A role-conditioned interaction model that beats rung 3 out of sample; a direct
per-player "lifts teammates' individual production" estimate (needs per-player
on-court production, a data extension); possession-level rather than
stint-level outcomes; substantially more seasons.

## Reproduce

```bash
courtgraph baselines --input data/nba_snapshots/rs_2020_2024/out/stints.jsonl --bootstrap 120 --rung4 --json
courtgraph transport \
  --train data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --test  data/nba_snapshots/all_2025_playoffs/out/stints.jsonl \
  --bootstrap 120 --rung4 --json
```

Result JSONs are gitignored under `data/nba_snapshots/`. Rung-5 numbers:
`courtgraph fit --evaluate` (see `docs/PROJECT_STATUS.md`).

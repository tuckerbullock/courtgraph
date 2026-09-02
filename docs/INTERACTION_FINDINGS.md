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
by the identity-keyed model forms on the ladder through rung 5 — is NOT
SUPPORTED.** Rungs 4 (explicit per-pair) and 5 (low-rank) do not improve
held-out prediction over hierarchical additive talent on any of four
leakage-safe evaluation tasks, and the explicit per-pair terms are
statistically indistinguishable from a placebo with the same parameter count.

**One small positive survives a properly-powered test: shot selection.**
Three later parameterisations (role-conditioned interaction, mechanistic
outcomes, skill redundancy) each showed a ~1–5 % edge on 40–60 group means. A
better-powered re-run (`courtgraph confirm`, 2026-09-01 — `unseen_lineup`
widened to 120 groups, a K sweep, and a 3,000-resample bootstrap CI on the
`RMSE(baseline) − RMSE(model)` delta) keeps **one** of them:

- **`three_share`** — role-conditioning predicts a lineup's
  three-point-attempt share ~3 % better than rung 3, **significant** (95 % CI
  excludes 0, P ≈ 1.0) against **both** the baseline and the permuted-role
  placebo, at K = 5 and K = 7, stable across the K sweep. The fitted role-pair
  matrix reads like the spacing mechanism: two movement shooters take more
  threes than additive predicts, a rim-running big fewer.
- **role on points/100** — marginal: the CI *just* excludes 0 at K = 5 (the
  pre-registered config) but K = 3 and K = 7 show nothing. K-fragile, not
  robustly established.
- **redundancy** — the held-out edge does not survive (CI [−0.05, +0.05]). The
  in-sample "all six concentration coefficients negative" stays a descriptive
  observation.

So: where lineups differ from the sum of their parts, it is in **how they
shoot**, not **how much they score**. Lineup value in points per 100 is not
improved by any interaction parameterisation at a level that survives proper
power. `RESEARCH_CONTRACT.md` §17.1 remains **not met**. See "Mechanistic
outcomes" and "Confirmation" below.

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

## Role-conditioned interaction — the weak positive

`courtgraph roles` (2026-09-01). The interaction term is keyed by the pair of
**role clusters** the two offensive players belong to, not by player identity.
Role clusters come from deterministic k-means over a standardised offensive
profile (usage, three-rate, rim-rate, assist/100, ft-rate, oreb/100) derived
from the play-by-play + shot chart (`courtgraph player-features`). With K = 5
that is 15 pooled interaction parameters, each backed by thousands of stints,
versus rung 4's ~2,357 thin per-identity pairs. The clustering is fit once on
the full profile set (role features only, never lineup value — outcome-blind)
and reused per fold. Placebo: a **permuted-role** fit (same cluster sizes,
players' role labels shuffled).

The five clusters on the real data are recognisable — movement shooter
(3-rate .62), rim-running big (rim .67), balanced wing, pass-first playmaker
(ast/100 9.1), high-usage lead creator (usage .34).

Held-out macro RMSE (points per 100):

| holdout | groups | rung 3 | **role** | permuted-role placebo |
|---|---|---|---|---|
| chronological | 13 | **3.55** | 4.49 | 3.58 |
| unseen_pair | 40 | 19.20 | **19.07** | 19.21 |
| unseen_lineup | 60 | 5.26 | **5.19** | 5.27 |

On the two **structural** holdouts role beats rung 3 by 0.7 % / 1.3 % and
beats its placebo by 0.7 % / 1.4 %, with clean calibration (z_sd ≈ 1.0). This
is the **first** interaction parameterisation to beat both baseline and
placebo out of sample. But: it is ~1 % on 40–60 group means, with no bootstrap
CI on the delta; and on the **chronological** holdout role is clearly *worse*
than rung 3 and its placebo (4.49; calibration z_mean 3.3) — under era/roster
drift the role terms hurt.

The fitted 5×5 role-pair surplus matrix (τ_role ≈ 1.0 pts/100, in-sample) is
interpretable: the high-usage creator paired with a rim-running big (+1.82) or
a shooter (+1.40) shows the largest surplus, while two ball-dominant creators
(+0.79) is the smallest such pairing — "star + complementary piece" beats
"star + star". This is a plausible story, but the held-out numbers above, not
the in-sample matrix, are the evidence.

**Status after the better-powered confirmation (see below): MARGINAL /
K-fragile.** On the 120-group `unseen_lineup` holdout the K = 5 improvement is
+0.10 pts/100 with a 95 % bootstrap CI that *just* excludes 0
([+0.00001, +0.20]) — but K = 3 and K = 7 show nothing. Not robustly
established.

## Mechanistic outcomes — the same signal, more clearly

`courtgraph mechanistic` (2026-09-01). Points/100 is a noisy target (σ ≈ 119).
Shots are attributed to stints by a time-window join (99.98 % matched) and the
outcome is swapped for a mechanical quantity: `pts_per_shot` (an eFG proxy),
`rim_share`, or `three_share`. Same rung 2 / 3 / role / permuted-role-placebo
comparison.

Held-out macro RMSE, role vs its placebo (positive = role better):

| outcome | chronological | unseen_pair | unseen_lineup |
|---|---|---|---|
| **three_share** | **+4.7 %** | **+1.9 %** | **+2.5 %** |
| pts_per_shot | −13 % | **+1.9 %** | **+2.1 %** |
| rim_share | **+3.2 %** | −0.3 % | **+1.4 %** |

`three_share` is the strongest result of the investigation: role-conditioning
beats the placebo on **all three** holdouts, including the chronological one
that every other test fails — shot-selection tendencies are more era-stable
than scoring efficiency. `pts_per_shot` beats the placebo by ~2 % on the two
structural holdouts (a bit stronger than the ~1 % on points/100).

The fitted `three_share` role-pair matrix is unambiguous about the mechanism:

- **two movement shooters together → the lineup takes *more* threes than the
  sum of their individual rates (+0.013 share);**
- **a rim-running big with anyone → *fewer* threes than additive predicts
  (−0.004 to −0.012).**

That is the spacing intuition, with the sign it predicts, surviving a placebo
on every holdout. It is not the contract's primary unit (points per 100), so
it does not clear §17.1 by itself. **The better-powered confirmation (below)
keeps this one:** on the 120-group holdout `RMSE(rung 3) − RMSE(role)` for
`three_share` is +0.001 with a 95 % bootstrap CI that excludes 0 against both
the baseline and the placebo, at K = 5 and K = 7. It is the one interaction
result that is properly established — small, and confined to shot selection.

## Skill redundancy — a coherent in-sample sign, a faint held-out edge

`courtgraph redundancy` (2026-09-01). Instead of a 15-cell role-pair matrix,
the interaction is D = 6 coefficients `ρ_d` on lineup *concentration* features:
`conc_d = (Σ zᵢ_d)² − Σ zᵢ_d²` over the offensive players' standardized role
vectors, positive when they double up on skill `d`. `ρ_d < 0` means
concentrating `d` hurts.

On the 266k stints **all six `ρ_d` are negative** — every kind of offensive
skill redundancy is a small penalty, largest for **offensive rebounding
(−0.20)** and **usage (−0.19)** ("two ball-dominant creators clash"), and
smallest for **three-point-attempt rate (−0.04, ~0)** — more shooters is close
to additive. The permuted-role placebo's `ρ` have mixed signs and no pattern.
Held out, redundancy beat its placebo by 0.1 % / 0.6 % on the two structural
holdouts on 60 group means. **The better-powered confirmation (below) does not
keep it:** on the 120-group holdout `RMSE(rung 3) − RMSE(redundancy)` = +0.002
with a 95 % CI of [−0.05, +0.05], P = 0.55. The 0.5 % edge was
group-sampling noise. The in-sample sign uniformity stands as a descriptive
observation; it does not translate to held-out predictive gain.

## Confirmation — bootstrap CIs and a K sweep

`courtgraph confirm` (2026-09-01). The three positives above each rested on
40–60 group means with no confidence interval. This re-runs all three with the
`unseen_lineup` holdout widened to **120 groups**, a **K sweep** {3, 5, 7},
and a **3,000-resample bootstrap CI** on `RMSE(baseline) − RMSE(model)` over
the held-out group means (vs both the rung-3 baseline and the permuted-role
placebo). `unseen_pair` stays at 40 groups (exposure budget) and reaches
significance for nothing.

On the 120-group `unseen_lineup` holdout:

| model | outcome | vs rung 3 (95 % CI) | vs placebo (95 % CI) | robust across K? |
|---|---|---|---|---|
| role (K = 5) | points/100 | +0.10 [+0.00001, +0.20] | +0.10 [+0.001, +0.21] | **no** — K 3/7 null |
| **role (K = 5, 7)** | **three_share** | **+0.001 [+0.0004, +0.0016]** | **+0.0008 [+0.0002, +0.0014]** | **yes** |
| redundancy | points/100 | +0.002 [−0.05, +0.05] | −0.005 [−0.06, +0.04] | n/a (K-independent) |

**Only `three_share` survives.** Role-conditioning improves prediction of a
lineup's three-point-attempt share by ~3 % of the rung-3 RMSE, with a CI that
excludes 0 against both baseline and placebo, at K = 5 and K = 7. Everything
about lineup *scoring* (points/100) is marginal at best.

## What this establishes — and what it does not

**Supported:**

- Hierarchical additive talent (rung 3) is a hard baseline: calibrated,
  seed-stable, and its intervals transport to the playoffs.
- Free per-pair chemistry terms and low-rank provision/need factorisation add
  **no** held-out predictive value over it, on four evaluation tasks, and the
  per-pair terms do not beat a parameter-matched placebo.
- **One small non-additivity in shot selection.** Role-conditioning predicts a
  lineup's three-point-attempt share ~3 % better than additive, significant
  against baseline and placebo on a 120-group holdout, robust across K. Two
  movement shooters take more threes together than additive predicts; a
  rim-running big fewer. This does **not** extend to points per 100 (marginal,
  K-fragile) — lineups differ from the sum of their parts in *how they shoot*,
  not *how much they score*.

**Not established / out of scope:**

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

A direct per-player "lifts teammates' individual production" estimate (needs
per-player on-court production, a data extension); a transaction backtest
(roster changes as natural experiments — the contract's T4); possession-level
outcomes; substantially more seasons (the confirmed `three_share` effect is
real but ~3 % of a small RMSE — points/100 would need many more seasons to
resolve a comparable effect).

## Reproduce

```bash
courtgraph baselines --input data/nba_snapshots/rs_2020_2024/out/stints.jsonl --bootstrap 120 --rung4 --json
courtgraph transport \
  --train data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --test  data/nba_snapshots/all_2025_playoffs/out/stints.jsonl \
  --bootstrap 120 --rung4 --json
courtgraph player-features \
  --snapshot-dir data/nba_snapshots/rs_2020_2024/snap \
  --stints data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --out data/nba_snapshots/rs_2020_2024/player_profiles.jsonl
courtgraph roles \
  --input data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --profiles data/nba_snapshots/rs_2020_2024/player_profiles.jsonl \
  --clusters 5 --bootstrap 120 --json
courtgraph mechanistic \
  --input data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --snapshot-dir data/nba_snapshots/rs_2020_2024/snap \
  --profiles data/nba_snapshots/rs_2020_2024/player_profiles.jsonl \
  --outcome three_share --clusters 5 --bootstrap 100 --json
courtgraph redundancy \
  --input data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --profiles data/nba_snapshots/rs_2020_2024/player_profiles.jsonl \
  --clusters 5 --bootstrap 100 --json
courtgraph confirm \
  --input data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --profiles data/nba_snapshots/rs_2020_2024/player_profiles.jsonl \
  --snapshot-dir data/nba_snapshots/rs_2020_2024/snap \
  --k 3,5,7 --lineups 120 --boot 3000 --json
```

Result JSONs are gitignored under `data/nba_snapshots/`. Rung-5 numbers:
`courtgraph fit --evaluate` (see `docs/PROJECT_STATUS.md`).

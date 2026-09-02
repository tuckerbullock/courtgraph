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

**On real regular-season stints (originally 266,518 for 2020-21 … 2024-25;
re-confirmed 2026-09-02 on 536,974 for 2016-17 … 2024-25) and the held-out
2024-25 playoffs, transferable teammate-pair / lineup chemistry — as measured
by the identity-keyed model forms on the ladder through rung 5 — is NOT
SUPPORTED.** Rungs 4 (explicit per-pair) and 5 (low-rank) do not improve
held-out prediction over hierarchical additive talent on any of four
leakage-safe evaluation tasks, and the explicit per-pair terms are
statistically indistinguishable from a placebo with the same parameter count.
Doubling the dataset did not change this (see "At 2× the data" below).

**One small positive survives a properly-powered test: shot selection.**
Three later parameterisations (role-conditioned interaction, mechanistic
outcomes, skill redundancy) each showed a ~1–5 % edge on 40–60 group means. A
better-powered re-run (`courtgraph confirm`, 2026-09-01 — `unseen_lineup`
widened to 120 groups, a K sweep, and a 3,000-resample bootstrap CI on the
`RMSE(baseline) − RMSE(model)` delta) keeps **one** of them:

- **`three_share`** — role-conditioning predicts a lineup's
  three-point-attempt share ~3 % better than rung 3, **significant** (95 % CI
  excludes 0) against the baseline across the K sweep (3–10). The fitted
  role-pair matrix reads like the spacing mechanism: two movement shooters take
  more threes than additive predicts, a rim-running big fewer. **But
  hardening (2026-09-01) downgrades it**: it beats the permuted-role placebo at
  only 2 of 6 K values, it does **not** transport to the held-out playoffs
  (Δ ≈ 0, P(Δ>0) = 0.53), and its shot-mix shift is uncorrelated with scoring
  (mediation r = 0.03). A small in-distribution regularity, not a value effect.
- **role on points/100** — was marginal (CI *just* excluded 0 at K = 5 on
  266k); on the doubled 297k dataset the CI now spans 0 at every K. Not
  established.
- **redundancy** — the held-out edge does not survive (CI [−0.05, +0.06]). The
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
| **role (K = 5, 7)** | **three_share** | **+0.001 [+0.0004, +0.0016]** | **+0.0008 [+0.0002, +0.0014]** | vs rung 3 yes; vs placebo K 5/8 only (see Hardening) |
| redundancy | points/100 | +0.002 [−0.05, +0.05] | −0.005 [−0.06, +0.04] | n/a (K-independent) |

**Only `three_share` survives.** Role-conditioning improves prediction of a
lineup's three-point-attempt share by ~3 % of the rung-3 RMSE, with a CI that
excludes 0 against both baseline and placebo, at K = 5 and K = 7 in this run.
Everything about lineup *scoring* (points/100) is marginal at best. The
Hardening subsection below extends the K sweep and adds two harder tests.

### Hardening — wider K sweep, playoffs transport, mediation

`courtgraph confirm --k 3,4,5,6,8,10 --outcomes three_share,pts_per_shot,rim_share
--boot 3000` plus `courtgraph transport-mechanistic` (RS → held-out 2024-25
playoffs), 2026-09-01. Three tests, and `three_share` **weakens** on all three:

1. **Wider K sweep (3–10).** vs **rung 3**, `three_share` role-conditioning
   beats additive at K = 3, 4, 5, 6, 8 (95 % CI excludes 0) and is marginal at
   K = 10 — robust. But vs the **permuted-role placebo** the CI excludes 0 at
   **only K = 5 and K = 8** (K = 3, 4, 6, 10: CI spans 0, P(Δ>0) 0.83–0.94).
   The "it is the roles, not just the extra parameters" claim holds at 2 of 6 K
   values, not as a rule.
2. **Playoffs transport: null.** Trained on the regular season, evaluated on
   the held-out playoffs (65 recurring lineups), the role model's `three_share`
   RMSE edge over rung 3 is +0.00005 [−0.0011, +0.0012], P(Δ>0) = 0.53 — no
   effect. The regularity is regular-season, in-distribution only.
3. **Mediation ≈ 0.** Over the 120 held-out unseen lineups, the correlation
   between the role model's incremental `three_share` prediction and the
   lineup's *scoring* surprise (realized points/100 − rung 3) is **0.03**. The
   shot-mix shift the model captures (mean |Δ| ≈ 0.3 pp of 3PA share) does not
   move points.

`pts_per_shot` and `rim_share`: null on every holdout and both K sweeps
(`rim_share` role-conditioning is slightly *worse* than additive); neither
transports.

**Revised read:** `three_share` is a small, real, *in-distribution* regularity
in how role-redundant lineups distribute their shot attempts. It is not a value
effect (mediation ≈ 0), it does not survive the placebo at most K, and it does
not transport to a new competitive context (the playoffs). It is the strongest
non-additivity the ladder has found and it is still well short of
`RESEARCH_CONTRACT.md` §17.1.

### At 2× the data (2026-09-02)

`task/data-acquisition` doubled the regular-season set: a second
(`data.nba.com`) reconstruction surface recovered 1,523 previously-quarantined
games, and 2016-17 → 2019-20 was ingested. **RS 266,518 → 536,974 stints**;
the 2020-21 → 2024-25 window alone went 266,518 → 297,404 (every prior stint a
strict subset).

Re-running `baselines` and `confirm` on the enlarged data **sharpens the
negative**:

- **rung 3 vs rung 2** — unchanged conclusion at both 297k and 537k: rung 3
  wins on `chronological` (6.47 vs 6.63) and `unseen_pair` (18.90 vs 19.78);
  `unseen_lineup` is a wash. Variance components move < 5 % (τ_off ≈ 2.2,
  σ ≈ 118) — **the noise floor is structural, not sample-size-limited**, so
  more seasons will not on their own resolve a sub-0.5 pts/100 pair effect.
  `chronological` calibration is still broken and slightly *worse* over the
  longer 2016 → 2024 span (z_sd 2.48).
- **role on points/100** — the K = 5 marginal edge from the first confirmation
  is **gone**: on 297k the 120-group `unseen_lineup` delta is +0.06
  [−0.02, +0.15], CI now spans 0 (was barely excluding it). No interaction
  form improves lineup *scoring* at a level that survives proper power.
- **`three_share`** — **holds**, essentially unchanged: beats rung 3 across
  K {3, 5, 7} (95 % CI excludes 0, ~2 % of RMSE), beats the placebo clearly at
  K = 3 and borderline at K = 5/7, mediation with scoring = **−0.01**. More
  data neither strengthened nor killed it — a robust, tiny, value-neutral
  shot-distribution regularity.
- **redundancy** — still null (CI [−0.05, +0.06]).

So the doubled dataset confirms the hardened verdict and removes the last
marginal points/100 signal. `RESEARCH_CONTRACT.md` §17.1 remains **not met**.

## What this establishes — and what it does not

**Supported:**

- Hierarchical additive talent (rung 3) is a hard baseline: calibrated,
  seed-stable, and its intervals transport to the playoffs.
- Free per-pair chemistry terms and low-rank provision/need factorisation add
  **no** held-out predictive value over it, on four evaluation tasks, and the
  per-pair terms do not beat a parameter-matched placebo.
- **One small non-additivity in shot selection.** Role-conditioning predicts a
  lineup's three-point-attempt share ~3 % better than additive on a 120-group
  in-distribution holdout, significant against the baseline across the K sweep
  (3–10). Two movement shooters take more threes together than additive
  predicts; a rim-running big fewer. Hardening bounds it tightly: it beats the
  permuted-role placebo at only 2 of 6 K, does **not** transport to the
  held-out playoffs (Δ ≈ 0), and its shot-mix shift is uncorrelated with
  scoring (mediation r = 0.03). It does **not** extend to points per 100 —
  lineups differ from the sum of their parts in *how they shoot*, not *how much
  they score*, and even that difference is in-distribution and value-neutral.

**Not established / out of scope:**

- **Talent absorption.** If "makes teammates better" is a stable individual
  trait, it is already inside a player's additive coefficient. These models
  cannot separate "no interaction" from "interaction collinear with average
  individual impact." → the *player-lift* item (§45; Phase A below, Phase B
  and the transaction backtest pending).
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

## §45 Phase A — pooled player-lift on lineup value (2026-09-02)

`courtgraph player-lift`. One EM-shrunk scalar per player, `λ_i ~ N(0, τ_λ²)`,
added to the rung-3 frame as `Σ_{i∈off} λ_i·(A_off,s − α_i)` — a lift term that
rewards lineups where a high-`λ` player shares the court with strong teammates.
Two-stage fit (α frozen from rung 3, ridge the residual, `τ_λ` by marginal
likelihood); placebo permutes the `λ_i → player` map (bijection, same exposure).

**Another null, as §45.2 predicted.** On both 297k (2020-24) and 537k
(8-season):

| holdout | rung 3 | lift | placebo | τ_λ real / placebo |
|---|---|---|---|---|
| chronological | 6.47 | 6.43 | 6.45 | 0.032 / **0.032** |
| unseen_pair | 18.90 | 18.92 | 18.90 | 0.032 / **0.032** |
| unseen_lineup | 4.60 | 4.57 | 4.59 | 0.017 / **0.017** |

- The full-fit marginal likelihood picks `τ_λ² = 1e-5` (the grid floor):
  **no evidence for a nonzero lift variance.** |λ_i| ≤ 0.0003 pts/100 per unit
  of teammate-talent surplus.
- Per fold, the permutation placebo recovers the **exact same** `τ_λ` — the
  variance the model finds is not player-specific; it is the pair-terms story
  again (parameter noise absorption, not an asymmetric lift).
- Held-out RMSE moves < 0.5 % vs rung 3 and is matched by the placebo.

"A player's average bump to lineups with strong teammates," measured on lineup
*value*, does not exist beyond additive talent at this scale.

### Phase B — direct lift on teammate individual production (2026-09-02)

`courtgraph phase-b`. The keystone: a per-(player, stint) production ingest
(`courtgraph player-production` — every made FG / FT / assist attributed to a
stint **and** a player, 99.2 % event match on the real data, validated against
known stars) feeds a model whose outcome is each offensive player-stint's own
**credited production per 100** — not the lineup's net rating. The design is
`μ + context + base_k` (the receiver's own EM-shrunk level) plus the pooled
lift of the receiver's four teammates, `lift_i ~ N(0, τ_lift²)`. Reported at
`assist_credit` 0.0 (points only) and 0.5.

On the 297k data, 441 held-out receivers (chronological split):

| outcome | lift vs **base-only** | lift vs giver-shuffle placebo |
|---|---|---|
| points only | +0.01 [−0.12, +0.13] | +0.63 [+0.44, +0.82] |
| points + 0.5·assists | −0.08 [−0.26, +0.11] | +0.91 [+0.66, +1.16] |

**Another null.** The lift model does **not** beat the base-only model — adding
"who the receiver's teammates were" does not improve out-of-sample prediction
of his individual production. It does beat the *placebo*, but that only says
real teammate assignments are less harmful than random ones, not that the lift
terms help. The model fits large lift coefficients in sample (|lift| up to
4.6 pts/100 — negative for high-usage bigs like Giannis / Embiid, i.e. usage
cannibalisation collinear with their own base) but they do not generalise.

**This closes the last open estimand.** A player's effect on teammates —
whether measured on lineup value (rungs 3–5, Phase A), across roster changes
(transaction backtest), or on teammates' individual production (Phase B) — is
not a transferable quantity beyond additive talent, at this data scale, on
these evaluations. Unseen giver-receiver and transaction-cohort Phase-B checks
are documented follow-ups but the base-only comparison already settles it.

## Transaction backtest — roster changes as natural experiments (2026-09-02)

`courtgraph transaction-backtest` (contract T4). The cohort is derived from the
stint data itself: **585 clean cross-season team switches** (player's team of
record changes between consecutive seasons, ≥ 500 offensive possessions each
side, no split seasons). For each switch A → B (first B season S), rung 3 is fit
on **seasons strictly before S** — a model that has never seen the player on B —
and `Δ = possession-weighted (realized − predicted)` over the player's post-move
stints on B, with the player's `α` *transferred* from his pre-move history. A
**phantom** cohort (1,200 non-movers given a fake same-team "move") gets the
identical computation.

If a player's value is partly roster-specific, movers' lineups should scatter
from the additive prediction *more* than non-movers'. **They do not:**

| cohort | n | mean Δ | mean \|Δ\| | RMSE |
|---|---|---|---|---|
| real switches | 585 | +3.32 | 4.69 | 5.69 |
| phantom (non-movers) | 1,200 | +4.27 | 4.99 | 6.02 |

- **mean \|Δ\| real − phantom = −0.30, 95 % CI [−0.63, +0.03]** — includes 0,
  leans *negative*. Movers are, if anything, marginally **more** predictable
  from additive talent than players who stayed put.
- The large positive mean Δ in **both** cohorts (~+3–4 pts/100) is the
  stale-model bias (the leakage-safe model is trained on older seasons, and
  league scoring rose 2016 → 2024) — it cancels in the real-vs-phantom contrast.
- `Δ` vs the transferred `α`: slope −0.07, corr −0.02 — the gap does **not**
  track shrinkage of the mover's coefficient.

**A player's lineup-value contribution transfers across a team change as
cleanly as a non-mover's stays put.** This is the best-powered test in the
project (585 vs the 40–60 group means the interaction models were limited to)
and it is a clean null: additive talent is sufficient across roster changes.
Mid-season trades (fuzzier cutover) are a documented follow-up.

## What would change the verdict

A direct per-player "lifts teammates' individual production" estimate (needs
per-player on-court production, a data extension — §45 Phase B); possession-level
outcomes; the defensive side (`matchups` surface, now acquired). The transaction
backtest (contract T4) has been run — a clean null (above).

**Not** more seasons alone: the 2026-09-02 doubling to 537k stints left the
variance components < 5 % changed and removed rather than sharpened the
marginal points/100 signal — the per-parameter noise floor for pair effects is
structural at this scale, so resolving a sub-0.5 pts/100 effect needs a
different estimand, not more of the same rows.

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
  --k 3,4,5,6,8,10 --outcomes three_share,pts_per_shot,rim_share \
  --lineups 120 --boot 3000 --json
courtgraph transport-mechanistic \
  --train data/nba_snapshots/rs_2020_2024/out/stints.jsonl \
  --test  data/nba_snapshots/all_2025_playoffs/out/stints.jsonl \
  --train-snapshot data/nba_snapshots/rs_2020_2024/snap \
  --test-snapshot  data/nba_snapshots/all_2025_playoffs/snap \
  --profiles data/nba_snapshots/rs_2020_2024/player_profiles.jsonl \
  --outcome three_share --clusters 5 --boot 3000 --json
```

Result JSONs are gitignored under `data/nba_snapshots/`. Rung-5 numbers:
`courtgraph fit --evaluate` (see `docs/PROJECT_STATUS.md`).

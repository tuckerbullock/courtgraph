# CourtGraph — Cycle 1 Research Report

Last updated: 2026-09-04

This is the standalone summary of research cycle 1
(`RESEARCH_CONTRACT.md`), written for a reader who has not followed the
project day to day. `docs/INTERACTION_FINDINGS.md` is the full, chronological
lab notebook this report distills; `docs/CURRENT_TASK.md` and
`docs/PROJECT_STATUS.md` carry the engineering history.

## 1. The question

> Can we estimate how NBA players will fit together before we have observed
> that exact combination on the court?

The working decomposition:

```text
lineup value = individual talent + player interactions + context
```

Primary unit: **offensive points per 100 possessions**, measured at the
**stint** level — a maximal span of consecutive possessions with the same 10
players on the court (5 offense, 5 defense), the finest grain at which a
specific five-man combination's value can be observed directly.

"Chemistry" here means something narrow and falsifiable: does knowing the
*identities* of the players sharing the court predict lineup value beyond
what their individual talents predict on their own? It is a predictive
question, not a causal one — nothing here establishes that any two players do
or do not affect each other; it establishes what a set of well-specified
models can and cannot extract from the data at this scale.

## 2. Data

**Pipeline:** raw NBA play-by-play (a pinned, immutable GitHub archive) →
`pbpstats` file-only possession reconstruction (two independent surfaces:
`stats.nba.com`'s feed as primary, `data.nba.com`'s feed as an offline
fallback) → CourtGraph's own validation layer (five players per side,
possession alternation, final-score reconciliation against an independent box
score) → a SHA-256 provenance manifest. Any game that fails validation is
**quarantined**, never patched or guessed — a fail-closed policy, so every
stint in the modeled data is one whose possession count, lineup, and score
have been independently checked.

**Scale:**

| RS window | games in | accepted | quarantined | stints |
|---|---|---|---|---|
| 2016-17 … 2019-20 | 4,746 | 4,556 | 190 (4.0%) | 239,570 |
| 2020-21 … 2024-25 | 5,998 | 5,760 | 238 (4.0%) | 297,404 |
| **8-season total** | 10,744 | **10,316** | 428 | **536,974** |

Plus the full 2024-25 playoffs (3,325 stints, 157 recurring lineups, 0 game
overlap with the regular season) — **held out of every training run**, used
only to test whether conclusions transport to a new competitive context.

The dataset was doubled mid-cycle (266,518 → 536,974 stints) by adding four
earlier seasons and a second possession-reconstruction surface that recovered
1,523 previously-quarantined games. Every stint from the original,
smaller dataset survived as a strict subset of the larger one — the doubling
was purely additive, not a re-definition.

## 3. Model ladder

Each rung adds one form of structure to a shared frame (weighted-Gaussian
RAPM: separate offensive/defensive per-player talent, context columns for
pace/rest/home-court, always fit and evaluated leakage-safe):

| rung | what it adds | verdict |
|---|---|---|
| 0 | context mean only | required predecessor, established |
| 2 | ridge-regularized additive talent (RAPM) | established; beats league mean 40-48% macro |
| 3 | empirical-Bayes hierarchical player model — variance components learned by EM, calibrated predictive intervals | **the reference baseline** — see §6 |
| 4 | explicit pairwise interaction term `γ_ij` per admitted teammate pair | tested, null |
| 5 | low-rank player embeddings, `γ_ij ≈ u_i·v_j` | tested, null |
| 6-7 | neural / permutation-invariant encoders | gated — never reached, because rungs 4-5 didn't clear the bar to justify them |

Rung 3 is not just "the next rung" — it is the model every later interaction
form is compared against, because it is the first one whose *uncertainty* is
trustworthy (§6), not only its point prediction.

**Rung 3 fitting, briefly.** Individual talent coefficients `α_i` are ridge
estimates as in rung 2. What rung 3 adds is that the ridge strength itself —
and the width of each player's predictive interval — is not a hand-tuned
constant. Two variance components (the spread of talent across players,
`τ_off²`, and the game-to-game noise around a lineup's true value, `σ²`) are
estimated from the data by **expectation-maximization**: alternate between
(E) computing each player's posterior mean and variance given the current
`τ_off², σ²`, and (M) re-estimating `τ_off², σ²` from those posteriors, until
convergence. This is the standard **empirical-Bayes** recipe — "let the data
tell you how much to shrink" — rather than picking a shrinkage strength by
cross-validated trial and error. The payoff is a *calibrated* predictive
interval per lineup, not just a point estimate, which is what makes the §6(a)
transport result meaningful: a rung that was only tuned for point accuracy
would have no interval to test for calibration in the first place.

## 4. Evaluation design

Every result below is evaluated on some subset of five leakage-safe
holdouts, chosen because each guards against a different way a model could
look good without actually generalizing:

1. **Chronological** — train on earlier games, test on later ones. Guards
   against the model just memorizing the era's average scoring level.
2. **Unseen-pair** — every co-stint of a held-out teammate pair is removed
   from training, though each player is individually observed elsewhere.
   Guards against a pair-specific term secretly re-deriving individual talent.
3. **Unseen-lineup** — every appearance of an exact five-man combination is
   removed from training. The direct test of the north-star question.
4. **Regular-season → playoff transport** — train on the full regular season,
   test on the held-out 2024-25 playoffs. Guards against a result that is an
   artifact of the regular-season sample rather than something that holds in
   a genuinely new context.
5. **Cross-season transaction backtest** — treat real player trades as
   natural experiments: fit only on seasons *before* a player's move, predict
   his new team's lineups, compare the prediction error to a "phantom" cohort
   of players who didn't move. The best-powered test in the project (585
   real cases).

On top of the holdouts: **permutation-placebo controls** (re-fit the same
model with the pair-to-parameter or role-to-player mapping randomly
scrambled — a result must beat this to be more than "extra free parameters
soaking up noise"), **3,000-sample block-bootstrap confidence intervals** on
every headline delta, and **calibration diagnostics** (does the model's
stated uncertainty match its actual error?).

## 5. Results — every estimand tested for "interaction," in order

Summary first, then each estimand's actual numbers below it.

| # | estimand | model / test | result |
|---|---|---|---|
| 1 | Symmetric pairwise interaction | rung 4 (explicit `γ_ij`) | **Null.** Doesn't beat rung 2/3 on any of 4 holdouts; indistinguishable from a parameter-matched placebo on a well-powered 668-pair in-sample test and a 476-pair playoff test. |
| 2 | Symmetric interaction, generalized form | rung 5 (low-rank embeddings) | **Null.** −0.08% to −8.4% vs. additive baseline across holdouts. |
| 3 | Role-conditioned interaction (points/100) | `courtgraph roles`, K-sweep confirmed | **Marginal → null.** A ~1% edge on 40-60 group means shrank to a CI spanning 0 once the dataset doubled. |
| 4 | Role-conditioned interaction (shot selection) | `three_share` outcome | **Small positive, bounded.** See §6. |
| 5 | Skill redundancy (lineup concentration on one skill) | `courtgraph redundancy` | **Null.** Held-out CI [−0.05, +0.06]. |
| 6 | Pooled asymmetric player lift on lineup value | `courtgraph player-lift` (§45 Phase A) | **Null.** Full-fit variance hits the grid floor; placebo recovers the identical variance per fold. |
| 7 | Player value across roster changes | Transaction backtest (T4) | **Null, best-powered.** Movers' lineups scatter from the additive prediction *no more* than non-movers' (mean \|Δ\| gap −0.30, 95% CI [−0.63, +0.03]). |
| 8 | Player lift on a teammate's *individual* production | `courtgraph player-production` + `phase-b` (§45 Phase B) | **Null.** Lift terms don't beat a base-only (receiver's own level) model out of sample. |
| 9 | Defensive-side pooled lift | `player-lift --side defense` | **Null.** Makes held-out prediction *worse* than additive; placebo matches exactly. |
| 10 | Role-conditioned interaction (turnover rate) | `turnover_rate` outcome | **Null.** Beats the placebo at only 1 of 6 K values swept — chance-level. |
| 11 | Role-conditioned interaction (assist rate) | `assist_rate` outcome | **Null.** Never beats the placebo at any of 6 K values swept. |

Eleven estimands: ten of eleven come back null, including every test that
measures lineup **scoring**. One test that measures **shot selection**
specifically (`three_share`) comes back positive — narrowly (§6) — and it
remains the only mechanistic outcome (of five tried) that clears a
properly-powered bar; `turnover_rate` and `assist_rate` were added later in
the same evaluation family and did not replicate the pattern (§5.9).

### 5.1 — Estimands 1-2: symmetric pairwise interaction (rungs 4-5)

Held-out macro RMSE, points per 100 possessions (lower is better):

| task | groups | rung 2 | rung 3 | rung 4 | rung 5 |
|---|---|---|---|---|---|
| chronological | 13 | 3.71 | **3.55** | 3.84 | ≈ rung 2 |
| unseen-pair | 40 | 19.57 | **19.20** | 19.22 | −0.08% vs r2 |
| unseen-lineup | 60 | 5.38 | **5.26** | 5.49 | −8.4% vs r2 |
| playoffs transport | 157 | **23.92** | 24.00 | 24.23 | not run |

Neither interaction rung beats the additive/hierarchical baseline anywhere.
Rung 4's per-pair terms were also checked at a better-powered
"seen-pairs" resolution — bucket held-out stints by which of the ~2,357
admitted pairs they contain, and compare to a placebo whose pair→coefficient
wiring is scrambled but has the identical parameter count and exposure:

| context | pair groups | rung 2 | rung 4 | rung 4 placebo |
|---|---|---|---|---|
| in-sample | 668 | 8.67 | 8.54 | **8.54** |
| playoffs | 476 | 12.98 | 13.22 | **13.24** |

Rung 4's 1.5% in-sample edge over rung 2 is exactly matched by the placebo,
and in the playoffs rung 4 is worse than rung 2 outright. The per-pair terms
carry no pair-specific signal — they behave as extra regularized parameters
absorbing additive misfit, nothing more.

### 5.2 — Estimand 3: role-conditioned interaction on points/100

`courtgraph roles` keys the interaction on the pair of **role clusters**
(K=5, deterministic k-means over usage/three-rate/rim-rate/assist-rate/
ft-rate/oreb-rate, fit once, outcome-blind) rather than player identity —
15 pooled parameters instead of rung 4's ~2,357 thin per-identity ones.
Placebo: role labels shuffled, cluster sizes preserved.

| holdout | groups | rung 3 | role | permuted-role placebo |
|---|---|---|---|---|
| chronological | 13 | **3.55** | 4.49 | 3.58 |
| unseen-pair | 40 | 19.20 | **19.07** | 19.21 |
| unseen-lineup | 60 | 5.26 | **5.19** | 5.27 |

On the two structural holdouts (unseen-pair, unseen-lineup) role beat both
baseline and placebo by ~1%, with clean calibration — the first interaction
form to do so. On the widened 120-group confirmation this shrank to a CI
that *just* excluded 0 ([+0.00001, +0.20] pts/100 at K=5), and after the
dataset doubled to 537k stints the CI opened back up to span 0
([−0.02, +0.15]). More data did not sharpen this signal — it erased it.

### 5.3 — Estimand 4: role-conditioned interaction on shot selection

Same role model, different outcome — instead of points/100, predict a
mechanical quantity directly: `pts_per_shot` (an eFG proxy), `rim_share`, or
`three_share`, each attributed to stints by a 99.98%-matched time-window
join. Held-out macro RMSE, role vs. its placebo (positive = role better):

| outcome | chronological | unseen-pair | unseen-lineup |
|---|---|---|---|
| **three_share** | **+4.7%** | **+1.9%** | **+2.5%** |
| pts_per_shot | −13% | **+1.9%** | **+2.1%** |
| rim_share | **+3.2%** | −0.3% | **+1.4%** |

`three_share` is the only outcome that beats the placebo on **all three**
holdouts, including the chronological one every other test in this report
fails — shot-selection tendencies are apparently more era-stable than
scoring efficiency. Full confirmation and hardening numbers are in §6(b).

### 5.4 — Estimand 5: skill redundancy

`courtgraph redundancy` replaces the 15-cell role-pair matrix with 6
coefficients `ρ_d` on lineup *concentration* — how much the five offensive
players double up on one skill dimension `d` (usage, rebounding, shooting,
etc.), via `conc_d = (Σ zᵢ_d)² − Σ zᵢ_d²` over standardized role vectors.

In-sample, **all six `ρ_d` are negative** — every kind of skill redundancy is
a small penalty, largest for offensive rebounding (−0.20) and usage (−0.19,
"two ball-dominant creators clash"), smallest for three-point rate (−0.04,
~0 — more shooters is close to additive). That sign pattern is coherent and
the permuted-role placebo's signs are mixed with no pattern. But held out,
on the widened 120-group confirmation, `RMSE(rung 3) − RMSE(redundancy)` =
+0.002 with 95% CI [−0.05, +0.05] (P = 0.55) — the small in-sample edge was
group-sampling noise. The sign-uniformity stands as a descriptive
observation about the fitted coefficients; it does not translate to
held-out predictive gain.

### 5.5 — Estimand 6: pooled asymmetric player lift on lineup value

`courtgraph player-lift` (§45 Phase A). One EM-shrunk scalar `λ_i` per
player, added on top of rung 3 as a lift term that should reward lineups
where a high-`λ` player shares the floor with strong teammates. Placebo
permutes the `λ_i → player` assignment.

| holdout | rung 3 | lift | placebo | τ_λ real / placebo |
|---|---|---|---|---|
| chronological | 6.47 | 6.43 | 6.45 | 0.032 / **0.032** |
| unseen-pair | 18.90 | 18.92 | 18.90 | 0.032 / **0.032** |
| unseen-lineup | 4.60 | 4.57 | 4.59 | 0.017 / **0.017** |

The full-fit marginal likelihood picks `τ_λ² = 1e-5` — the grid floor,
i.e. no evidence of nonzero lift variance at all (|λ_i| ≤ 0.0003 pts/100 per
unit of teammate-talent surplus). Per fold, the permutation placebo recovers
the **exact same** `τ_λ` as the real fit — whatever tiny variance the model
finds is not player-specific.

### 5.6 — Estimand 7: player value across roster changes (transaction backtest)

`courtgraph transaction-backtest` (contract T4). Cohort: **585 clean
cross-season team switches** (a player's team of record changes between
consecutive seasons, ≥500 offensive possessions logged on each side, no
split seasons). For each switch, rung 3 is fit only on seasons strictly
before the move — the model has never seen the player on his new team — and
`Δ = realized − predicted` is computed over his post-move stints, with his
`α` transferred from pre-move history. A **phantom** cohort of 1,200
non-movers gets the identical computation with a fake same-team "move" date.

| cohort | n | mean Δ | mean \|Δ\| | RMSE |
|---|---|---|---|---|
| real switches | 585 | +3.32 | 4.69 | 5.69 |
| phantom (non-movers) | 1,200 | +4.27 | 4.99 | 6.02 |

If a player's value were partly roster-specific, movers should scatter from
the additive prediction *more* than non-movers. They do not: mean |Δ| real
minus phantom = **−0.30**, 95% CI **[−0.63, +0.03]** — the CI includes 0 and
leans negative, meaning movers are if anything marginally *more* predictable
from additive talent alone than players who stayed put. (The large positive
mean Δ in both cohorts is a shared, expected artifact — the leakage-safe
model is always trained on older, lower-scoring seasons — and cancels out in
the real-vs-phantom contrast that's the actual test.) This is the
best-powered test in the project — 585 real cases vs. the 40-120 group means
the interaction models were limited to.

### 5.7 — Estimand 8: player lift on a teammate's individual production

`courtgraph player-production` + `phase-b` (§45 Phase B). A new per-(player,
stint) production ingest attributes every made FG/FT/assist to a stint
**and** a specific player (99.2% event match on real data, validated against
known stars' known box-score totals). The outcome switches from lineup net
rating to each offensive player-stint's own credited production per 100.
Model: `μ + context + base_k` (the receiver's own EM-shrunk level) plus the
pooled lift of his four teammates. Reported at `assist_credit` 0.0 (points
only) and 0.5.

On 297k stints, 441 held-out receivers (chronological split):

| outcome | lift vs. base-only | lift vs. giver-shuffle placebo |
|---|---|---|
| points only | +0.01 [−0.12, +0.13] | +0.63 [+0.44, +0.82] |
| points + 0.5·assists | −0.08 [−0.26, +0.11] | +0.91 [+0.66, +1.16] |

The lift model does not beat the base-only model — knowing who a receiver's
teammates were does not improve prediction of his individual production.
(It does beat the placebo, but that only shows real teammate assignments are
less harmful than random ones — not that the lift terms carry real signal.)
In-sample the model fits large lift coefficients (up to 4.6 pts/100,
strongly negative for high-usage bigs — usage cannibalization collinear with
their own base rate) but they don't generalize out of sample.

### 5.8 — Estimand 9: defensive-side pooled lift

`courtgraph player-lift --side defense` — the same Phase-A lift design keyed
on the defensive lineup instead of offensive. On both the 297k and 537k
datasets the defensive lift terms make held-out prediction *worse* than rung
3 (e.g. 4.72 vs. 4.60 on unseen-lineup), and the placebo recovers the
identical `τ_λ` per fold. A defensive per-pair `γ_ij^def` model and a deep
dive on the `matchups` (who-guarded-whom) surface remain documented
follow-ups, but the pooled result points the same direction as everything
else in this report.

### 5.9 — Estimands 10-11: turnover rate and assist rate

`three_share` is the one mechanistic outcome (of `pts_per_shot`, `rim_share`,
`three_share` — §5.3) that survived hardening. The natural next step was to
run the two remaining ball-security candidates — **turnover rate**
(turnovers per offensive possession) and **assist rate** (share of made
shots that were assisted) — through the identical role-conditioned /
permuted-role-placebo / K-sweep {3,4,5,6,8,10} / 3,000-resample bootstrap-CI
treatment. Both required a new stint-level attribution (a time-window join
of raw play-by-play turnover and assist events onto stints, mirroring the
shot-chart join `three_share` uses) — 98.7% event match rate on the 297k
(2020-24) dataset.

| outcome | holdout | K values where CI excludes 0 vs. rung 3 | vs. permuted-role placebo |
|---|---|---|---|
| `turnover_rate` | unseen-lineup (120 groups) | 1 of 6 (K=6) | 1 of 6 (K=6) |
| `turnover_rate` | unseen-pair (42 groups) | 0 of 6 | 0 of 6 |
| `assist_rate` | unseen-lineup (120 groups) | 1 of 6 (K=4) | **0 of 6** |
| `assist_rate` | unseen-pair (42 groups) | 0 of 6 | 0 of 6 |

**Both null.** A single K value out of six crossing zero on one holdout is
within the range six independent comparisons at a nominal 95% threshold
would produce by chance alone — contrast with `three_share`, which beat the
baseline at 5 of 6 K and beat the placebo at 2 of 6 (§6(b)), a materially
stronger and more holdout-consistent pattern. `assist_rate` never beats its
placebo at any K on either holdout — the weakest result of any mechanistic
outcome tried. Neither outcome was tested for playoff transport or mediation,
since neither showed a held-out edge worth explaining.

This closes the mechanistic-outcome ladder as scoped: five outcomes tried,
one small established non-additivity, four clean nulls.

## 6. The two results worth keeping

**(a) Rung 3's calibration transports — the actual headline finding.**
Point-prediction accuracy was never the differentiator between rung 2 and
rung 3; their *uncertainty* was. Rung 3's predictive intervals are close to
nominal coverage where rung 2's are not, and — the part that matters — this
holds up when tested on the held-out playoffs, a context the model never
trained on:

| task | model | coverage (50/80/95%) | standardized-residual SD |
|---|---|---|---|
| unseen-pair | rung 2 / rung 3 | .48·.72·.88 / **.45·.78·.93** | 1.45 / **1.06** |
| unseen-lineup | rung 2 / rung 3 | .33·.53·.70 / **.40·.70·.95** | 1.59 / **1.04** |
| playoffs | rung 2 / rung 3 | .30·.53·.75 / **.50·.85·.96** | 1.69 / **0.92** |

A standardized-residual SD near 1.0 means the model's stated uncertainty
matches its actual error — you can trust its confidence intervals, not just
its point estimate. That property surviving a shift from regular season to
playoffs is the strongest single result in the project.

**(b) One non-additivity survives, and it's in shot mix, not scoring.**
Role-conditioning (clustering players into 5 role types by usage, shot
profile, and playmaking, then letting the interaction term depend on the
*pair of roles* rather than the *pair of identities*) predicts a lineup's
three-point-attempt share about 3% better than the additive baseline. This
holds across a K-sweep (3-10 role clusters) against the additive baseline,
and the fitted role-pair matrix tells a coherent, mechanistically sensible
story: two movement shooters on the floor together take *more* threes than
the sum of their individual rates predicts; a rim-running big drags the
lineup's three-rate down regardless of who else is out there — textbook
floor-spacing.

The fitted role-pair matrix (5 clusters — movement shooter, rim-running big,
balanced wing, pass-first playmaker, high-usage lead creator) is
interpretable on its own terms:

- Two movement shooters together: **+0.013** three-point-attempt share vs.
  additive.
- A rim-running big paired with anyone: **−0.004 to −0.012** three-point
  share vs. additive.
- On points/100 (the weaker, non-surviving signal): a high-usage creator
  paired with a rim-running big (+1.82) or a shooter (+1.40) shows the
  largest in-sample surplus; two ball-dominant creators together (+0.79) the
  smallest — "star + complementary piece" beats "star + star," in sample.

**Confirmation — widened to a 120-group holdout, K-sweep {3,4,5,6,8,10}, a
3,000-resample bootstrap CI on `RMSE(baseline) − RMSE(model)`:**

| model | outcome | vs. rung 3 (95% CI) | vs. placebo (95% CI) | robust across K? |
|---|---|---|---|---|
| role (K=5) | points/100 | +0.10 [+0.00001, +0.20] | +0.10 [+0.001, +0.21] | **no** — K3/K7 null |
| role (K=5,7) | three_share | **+0.001 [+0.0004, +0.0016]** | **+0.0008 [+0.0002, +0.0014]** | vs. rung 3 yes; vs. placebo only K5/K8 |
| redundancy | points/100 | +0.002 [−0.05, +0.05] | −0.005 [−0.06, +0.04] | n/a |

Only `three_share` survives this pass. But it is bounded on every axis that
would make it a genuine "chemistry" result — three further, harder checks
(`--k 3,4,5,6,8,10`, playoff transport, mediation) each weaken it further:

1. **Wider K-sweep.** Vs. rung 3, `three_share` beats additive at K =
   3,4,5,6,8 (CI excludes 0) and is marginal at K=10 — this part is robust.
   But vs. the **permuted-role placebo**, the CI excludes 0 at only **K=5 and
   K=8** of the six values tested; at K=3,4,6,10 the CI spans 0 (P(Δ>0)
   0.83-0.94). "It's the roles, not just extra parameters" holds at 2 of 6 K
   values, not as a rule.
2. **Playoff transport: null.** Trained on the regular season, evaluated on
   the held-out playoffs (65 recurring lineups), the role model's
   `three_share` edge over rung 3 is +0.00005 [−0.0011, +0.0012], P(Δ>0) =
   0.53 — indistinguishable from no effect. The regularity is
   regular-season, in-distribution only.
3. **Mediation ≈ 0.** Across the 120 held-out lineups, the correlation
   between the role model's incremental `three_share` prediction and the
   lineup's actual *scoring* surprise (realized − rung-3-predicted
   points/100) is **0.03** (−0.01 after the dataset doubled). The shot-mix
   shift the model captures (mean |Δ| ≈ 0.3 percentage points of 3PA share)
   simply does not move points.

`pts_per_shot` and `rim_share` are null on every holdout and both K-sweeps
(`rim_share` role-conditioning is slightly *worse* than additive); neither
transports.

**After the dataset doubled to 537k stints (2026-09-02):** `three_share`
holds essentially unchanged — beats rung 3 across K {3,5,7} (95% CI excludes
0, ~2% of RMSE), beats the placebo clearly at K=3 and only borderline at
K=5/7, mediation with scoring = −0.01. More data neither strengthened nor
killed it. The marginal points/100 role signal, by contrast, did not
survive the doubling (§5.2).

**Read together:** lineups differ from the sum of their parts in how they
choose shots, not in how much they score, and even that difference doesn't
survive a change of context. `RESEARCH_CONTRACT.md` §17.1 (a significant
primary-unit improvement) is not met.

## 7. What the null does and does not establish

**Does not establish:** that no two players affect each other on the court.
Three specific gaps limit the claim:

- **Talent absorption.** If "makes teammates better" is a stable trait of a
  player, it may already be baked into his individual additive coefficient —
  these models cannot cleanly separate "no interaction exists" from
  "interaction is collinear with average individual impact." The player-lift
  tests (estimands 6, 8, 9 above) were built specifically to probe this and
  came back null too, but they don't fully close the gap.
- **No player features beyond bare identity/role.** Every symmetric-pair
  model uses raw player indicators. A richer feature-conditioned form
  (e.g. spacing-specific, not just role-cluster-specific) might generalize
  differently — role-conditioning is the one form tried, and the one that
  found something.
- **Dynamic chemistry** (interactions that build or decay over a season) is
  explicitly out of scope for this cycle.

**Does establish, with real statistical weight:** at this data scale, using
identity- or role-keyed interaction terms, and measured on scoring
efficiency specifically, transferable lineup chemistry is not a supported
predictive quantity across five different evaluation designs and nine
distinct estimands.

## 8. Threats to validity

- **Chronological holdout calibration is still broken** for every model —
  systematic under-prediction under era/roster drift (league scoring rose
  2016→2024), and this got slightly worse, not better, over the longer span.
  This is a documented, shared limitation, not something specific to any one
  rung.
- **Quarantine:** 4.0% of games are excluded per season window as
  unreconstructable rather than guessed at. This is fail-closed by design,
  but it does mean the dataset isn't a literal 100% sample.
- **"More data would fix it" is directly tested and rejected.** Doubling the
  dataset moved the model's learned variance components by under 5% and
  *removed* rather than sharpened the one marginal points/100 signal that
  had existed at the smaller scale. The noise floor here (single-stint
  outcome SD ≈ 119 pts/100 vs. additive talent SD ≈ 2.3) is structural — a
  sub-0.5 pts/100 pair effect, if one exists, needs a different kind of
  estimand to resolve, not more seasons of the same kind of row.

## 9. Engineering

The research above rests on infrastructure decisions that matter for whether
any of it should be trusted:

- **Fail-closed ingestion.** Possession reconstruction runs against two
  independent raw NBA feeds; any game whose reconstructed score doesn't
  reconcile against an independent box-score total, whose lineups aren't
  exactly five-per-side, or whose possession count can't be validated is
  **quarantined**, never patched, imputed, or silently dropped from the
  denominator. The pipeline emits a full audit manifest (per-game pass/fail
  reason, SHA-256 of every input file) rather than a bare stint count.
- **Leakage safety is enforced by code, not convention.** Every holdout split
  is passed through a `verify_split` gate that re-derives the forbidden
  overlaps (e.g. re-checks that a held-out pair truly never co-occurs in
  training) rather than trusting the split-construction logic to have gotten
  it right.
- **No network at model-fit time.** The possession reconstruction, the model
  fitting, and the evaluation are all offline by construction — an
  `offline_guard` turns any accidental network attempt during ingest into a
  hard quarantine rather than a silent live fetch that would break
  reproducibility.
- **Numerical scale.** The chemistry model's linear algebra accumulates its
  Gram matrix via sparse `np.bincount` scatter over the 5 players per stint
  rather than dense matrix multiplication — numerically identical to the
  dense computation to ~1e-13, but the difference between a fit that runs in
  ~16 minutes on 537k stints and one that doesn't finish.
- **Dependency discipline.** `courtgraph` itself ships with zero runtime
  dependencies (a `courtgraph doctor` health check must import nothing
  third-party); `numpy` and `pbpstats` are pinned exactly and lazily
  imported only where actually needed, so the dependency-free path stays
  testable in isolation.
- **Test coverage and static checks.** ~250 unit tests (`unittest`,
  standard library), `mypy --strict`, `ruff` lint + format, run in CI on
  Python 3.11 and 3.13, plus a separate dependency-free CI leg. Every fixed
  bug carries a regression test; every model has deterministic,
  seed-independent fixtures for its core numerical identities (e.g. the
  additive-talent-plus-context decomposition round-trips exactly).

## 10. Glossary

| term | meaning |
|---|---|
| **stint** | A maximal span of consecutive possessions with the same 10 players (5 per side) on the court — the finest unit at which a specific lineup's value is directly observable. |
| **RAPM** | Regularized Adjusted Plus-Minus — estimate each player's individual effect on scoring by regression across every lineup he's been part of, with ridge regularization to control for the huge number of players relative to data. |
| **empirical Bayes** | Estimate the *prior* (how much players vary, how noisy a single stint is) from the data itself via EM, rather than fixing it by hand or by cross-validated search — yields calibrated, not just point, predictions. |
| **EM (expectation-maximization)** | An iterative fitting algorithm: alternately compute the best current estimate of hidden quantities (here, each player's posterior talent) given the variance parameters, then re-estimate the variance parameters given those, until convergence. |
| **leakage-safe holdout** | An evaluation split constructed so that no information about the test cases (a pair, a lineup, a future date) is available during training — the only way to know if a model has learned something general rather than memorized. |
| **permutation placebo** | Re-fit the same model with a randomized identity/role mapping, keeping everything else (parameter count, exposure) identical — a result only counts if it beats this control, since otherwise "extra free parameters absorbing noise" is a simpler explanation. |
| **block bootstrap** | Resample held-out groups (not individual rows) with replacement, thousands of times, to build a confidence interval on a summary statistic — group-level resampling because rows within a group aren't independent. |
| **calibration** | Whether a model's *stated* uncertainty matches its *actual* error — e.g. do its nominal 80% intervals really contain the true value 80% of the time? A model can have great point accuracy and terrible calibration, or vice versa. |
| **transport (test)** | Evaluate a model trained on one context (regular season) on a genuinely different one (playoffs) it never saw — the strongest test that a finding isn't an artifact of the training distribution. |
| **mediation** | Whether an effect on one measured quantity (shot selection) actually flows through to the outcome that matters (points) — tested here as a simple correlation between the model's shot-mix prediction and actual scoring surprise. |

## 11. Conclusion

Cycle 1 set out to test whether NBA lineup chemistry, defined as an
identity- or role-keyed non-additive term on lineup scoring, is a
predictively supported quantity. Across eleven estimands, five leakage-safe
evaluation designs, permutation-placebo controls, and a dataset that was
doubled mid-cycle specifically to rule out "underpowered" as an explanation
— it is not. What the ladder found instead is a well-calibrated additive
baseline whose uncertainty is trustworthy even in a new competitive context,
and one small, real, in-distribution regularity in *how* — not how much —
role-diverse lineups shoot; two further mechanistic outcomes tried in the
same family (turnover rate, assist rate) did not replicate that regularity,
closing the mechanistic-outcome ladder as scoped (§5.9).

**What would change this answer:** a feature-conditioned interaction form
richer than the 5-cluster role model; a possession-level (rather than
stint-level) outcome that isn't swamped by the ≈119-pt/100 noise floor; or a
genuinely different estimand for talent absorption than the three tried
here. "More seasons of the same data" is specifically ruled out as the lever
by §8.

Two directions remain open for cycle 2: the product side (§44 / issue #8,
building something on the established additive + calibrated-uncertainty
model), or another item from the project's work queue (`docs/CURRENT_TASK.md`)
— the defensive-side role/redundancy extension, remaining model-ladder gaps,
or the nullable `days_rest` schema fix.

## 12. Appendix — reproduce

```bash
courtgraph baselines --input <stints.jsonl> --bootstrap 120 --rung4 --json
courtgraph transport --train <rs_stints.jsonl> --test <playoff_stints.jsonl> --bootstrap 120 --rung4 --json
courtgraph player-features --snapshot-dir <snap> --stints <stints.jsonl> --out <profiles.jsonl>
courtgraph roles --input <stints.jsonl> --profiles <profiles.jsonl> --clusters 5 --bootstrap 120 --json
courtgraph mechanistic --input <stints.jsonl> --snapshot-dir <snap> --profiles <profiles.jsonl> --outcome three_share --clusters 5 --bootstrap 100 --json
courtgraph redundancy --input <stints.jsonl> --profiles <profiles.jsonl> --clusters 5 --bootstrap 100 --json
courtgraph confirm --input <stints.jsonl> --profiles <profiles.jsonl> --snapshot-dir <snap> --k 3,4,5,6,8,10 --outcomes three_share,pts_per_shot,rim_share --lineups 120 --boot 3000 --json
courtgraph transport-mechanistic --train <rs_stints.jsonl> --test <playoff_stints.jsonl> --train-snapshot <rs_snap> --test-snapshot <playoff_snap> --profiles <profiles.jsonl> --outcome three_share --clusters 5 --boot 3000 --json
courtgraph player-lift --input <stints.jsonl> --json
courtgraph player-lift --input <stints.jsonl> --side defense --json
courtgraph player-production --snapshot-dir <snap> --stints <stints.jsonl> --out <production.jsonl>
courtgraph phase-b --input <stints.jsonl> --production <production.jsonl> --assist-credit 0.5 --json
courtgraph transaction-backtest --input <stints.jsonl> --json
courtgraph mechanistic --input <stints.jsonl> --snapshot-dir <snap> --profiles <profiles.jsonl> --outcome turnover_rate --min-fga 1 --clusters 5 --bootstrap 100 --json
courtgraph confirm --input <stints.jsonl> --profiles <profiles.jsonl> --snapshot-dir <snap> --k 3,4,5,6,8,10 --outcomes turnover_rate,assist_rate --min-fga 1 --lineups 120 --boot 3000 --json
```

Note `--min-fga 1` on the last two commands: the default of 3 is tuned for
field-goal-attempt exposure and silently drops ~46% of stints when applied
to `turnover_rate`'s possession-count denominator (median offensive
possessions per stint is also 3).

Full numbers, dates, and the chronological narrative of how each result was
found and hardened are in [`INTERACTION_FINDINGS.md`](INTERACTION_FINDINGS.md).
The falsifiable specification these tests are judged against is
[`RESEARCH_CONTRACT.md`](../RESEARCH_CONTRACT.md).

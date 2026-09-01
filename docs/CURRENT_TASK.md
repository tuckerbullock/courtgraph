# Current Task

Last updated: 2026-09-01

## State

Done — **candidate idea #1: role-conditioned interaction, built and run on the
266k real regular-season stints.** Branch `task/player-season-features` off
`main`. Committed; PR open.

**Result: the first non-null.** On the two structural holdouts (unseen-pair,
unseen-lineup) the role-conditioned model beats both the rung-3 hierarchical
baseline and its own permuted-role placebo, by ~1%. That is small — at the
edge of what 40–60 held-out group means can resolve, with no formal
significance test yet — and it degrades under temporal drift. But it is the
first interaction parameterisation to beat baseline **and** placebo on
held-out data. All results preserved.

## What was built

- **`courtgraph player-features`** (`603e1da`) — per-(player, season) role/skill
  profiles from the raw play-by-play + shot-chart payloads already in the
  snapshot (no new download): usage, assist / turnover / rebound / steal /
  block per 100, shot profile (three / rim / corner-three / free-throw share).
  3,137 profiles, 2,355 above the 200-possession exposure floor.
- **`fit_augmented_em`** (`60f00d8`) — the rung-3 EB frame + one extra
  coefficient block, extracted from rung 4 (behavior-preserving) so the role
  model reuses it.
- **`RoleClusterInteraction` + `courtgraph roles`** (`6991cde`) — deterministic
  k-means (k-means++, stable ids) over the standardised offensive-role vector
  (usage, three-rate, rim-rate, assist/100, ft-rate, oreb/100) → K role
  clusters; the offensive interaction term is keyed by **role-cluster pair**
  (K(K+1)/2 = 15 pooled parameters, each backed by thousands of stints)
  instead of ~2,357 thin per-identity pairs. `courtgraph roles` compares rung
  2 vs rung 3 vs role vs a **permuted-role placebo** on the three leakage-safe
  holdouts. Clustering is fit once on the full profile set (role features
  only, outcome-blind) and reused per fold.
- Validated on synthetic: recovers a planted role-pair matrix (corr > 0.85)
  and τ_role (within 30%); the placebo collapses τ_role and fits worse.
  210 tests; ruff / mypy / dep-free clean.

## Result — 266k RS stints, K = 5 clusters

`courtgraph roles --input .../rs_2020_2024/out/stints.jsonl --profiles
.../player_profiles.jsonl --clusters 5 --bootstrap 120`. Result:
`data/nba_snapshots/rs_2020_2024/chem_roles_eval.json` (gitignored).

### The clusters (782 clustered players) — real archetypes

| cluster | usage | 3-rate | rim-rate | ast/100 | ft-rate | reads as |
|---|---|---|---|---|---|---|
| c0 | .20 | **.62** | .18 | 3.8 | .14 | movement / spot-up shooter |
| c1 | .20 | .06 | **.67** | 3.5 | .38 | rim-running interior big |
| c2 | .22 | .36 | .38 | 3.6 | .26 | balanced wing / forward |
| c3 | .23 | .40 | .24 | **9.1** | .19 | pass-first playmaking guard |
| c4 | **.34** | .35 | .26 | 8.1 | .30 | high-usage lead creator |

### Held-out macro RMSE (points per 100)

| holdout | groups | rung 2 | rung 3 | **role** | role placebo |
|---|---|---|---|---|---|
| chronological | 13 | 3.70 | **3.55** | 4.49 | 3.58 |
| unseen_pair | 40 | 19.57 | 19.20 | **19.07** | 19.21 |
| unseen_lineup | 60 | 5.38 | 5.26 | **5.19** | 5.27 |

- **Structural holdouts:** role beats rung 3 by 0.7 % / 1.3 %, and beats its
  permuted-role placebo by 0.7 % / 1.4 %. Small, consistent direction, clean
  calibration (unseen_pair z_sd 1.05 cov .55/.82/.92; unseen_lineup z_sd 1.03
  cov .43/.70/.95).
- **chronological:** role is **worse** than rung 3 and its placebo (4.49),
  with bad calibration (z_mean 3.3, slope −0.98). Under era/roster drift the
  role terms hurt. Rung 2/3 also fail this holdout — a shared limitation — but
  role makes it worse.

### The role-pair surplus matrix (τ_role = 1.02 pts/100)

Offensive surplus for a lineup pair of a cluster-a and a cluster-b player, all
positive (0.07 … 1.82):

```
        c0     c1     c2     c3     c4
c0    0.56   0.52   0.07   0.70   1.40
c1    0.52   1.03   0.61   0.69   1.82
c2    0.07   0.61   0.74   0.55   1.39
c3    0.70   0.69   0.55   1.00   1.20
c4    1.40   1.82   1.39   1.20   0.79
```

The structure is interpretable: **the high-usage lead creator (c4) paired with
a rim-running big (c1, +1.82) or a shooter (c0, +1.40) shows the largest
surplus** — "star + complementary piece." Two ball-dominant creators (c4+c4,
+0.79) is the *lowest* c4 pairing — a mild redundancy penalty. This matches
the "spacing / fit" intuition, but it is an in-sample fit; the held-out
numbers above are what test whether it generalises, and they say "a little".

## Verdict against the contract

- `RESEARCH_CONTRACT.md` §17.1 ("significant improvement over the rung-3
  baseline on macro unseen-lineup error"): **not met** — 5.19 vs 5.26 is 1.3 %,
  with no bootstrap CI on the delta and only 60 group means. Suggestive, not
  established.
- §26 "successive models show no transferable interaction signal": this is the
  first model that **does** show a (small) transferable signal surviving a
  placebo. The stop condition is weakened, not cleared.

**Honest summary: role-conditioning is the first interaction parameterisation
to beat both the additive/hierarchical baseline and a matched placebo on
held-out structural holdouts, by ~1 %. Promising, and it warrants a
better-powered follow-up (more holdout groups, bootstrap CIs on the
role-minus-rung-3 delta, a K sweep). It does not yet clear the contract's
usefulness bar, and it does not survive temporal drift.**

## Candidate follow-up ideas — progress

1. **Role/skill-conditioned interaction — DONE (this task). First non-null;
   needs a better-powered confirmation.**
2. **Mechanistic outcome variables (NEXT — user asked for #2).** Predict a
   shot-quality / shot-mix / turnover-rate outcome instead of points/100.
   Needs shot & event data attributed to stints — a data-availability check
   and likely an ingest extension (`player-features` proved the raw zones /
   events are in the snapshot).
3. Anti-synergy / redundancy feature — can reuse the role machinery.
4. Playoffs transport — DONE (PR #18).
5. Transaction backtest (T4) — highest cost / highest evidentiary value.

Plus master-plan §45 player-lift (Phase A overlaps this task's EM core).

## Next action

Merge this branch. Then start idea #2 (mechanistic outcomes): first a
data-availability check on shot→stint attribution, then scope the ingest
extension or standalone derivation, then the model + leakage-safe evaluation.

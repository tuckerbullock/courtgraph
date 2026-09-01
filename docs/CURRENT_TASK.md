# Current Task

Last updated: 2026-09-01

## State

Done — **candidate idea #2: mechanistic outcomes, built and run on the 266k
real regular-season stints for all three outcomes.** Branch
`task/mechanistic-outcomes` off `main`. Committed; PR open.

**Result: the mechanistic outcomes carry the role-fit signal more cleanly than
aggregate points/100 — and `three_share` (share of shots from three) is the
strongest and most interpretable signal of the whole investigation.** The
role-conditioned model beats rung 3 and its permuted-role placebo on **all
three** holdouts for `three_share` (including chronological, where every prior
test failed), by 2–5 %, with a role-pair matrix that reads exactly like the
spacing mechanism. Still 13–60 group means with no bootstrap CI — suggestive
and consistent, not established. All results preserved.

## What was built (`8a3fdc7`)

- **`features/stint_shots.py`** — `attribute_shots` places every shot-chart
  shot in its stint by a per-`(period, offense-team)` time-window join: each
  offense stint owns `[start_time_seconds, next_start_or_period_end)`. No
  re-ingest, no new download. On the real data **99.98 % of shots matched**;
  the 0.02 % outside every window are dropped, never guessed.
- **`chemistry/mechanistic.py` + `courtgraph mechanistic --input <stints>
  --snapshot-dir <snap> --profiles <profiles> --outcome {pts_per_shot,
  rim_share, three_share}`** — swaps the design's outcome for the shot quantity
  (weight = FGA), then runs the same rung 2 / rung 3 / role-conditioned /
  permuted-role-placebo comparison as `courtgraph roles` on the three
  leakage-safe holdouts.
- Validated on synthetic (planted role effect on points-per-shot; the
  unmatched OT shot is dropped; the `min_fga` filter). 215 tests; ruff / mypy /
  dep-free clean.

## Result — 266k RS stints, K = 5 clusters, 142,885 stints with ≥3 FGA

`courtgraph mechanistic … --outcome <o> --clusters 5 --bootstrap 100`. Results:
`data/nba_snapshots/rs_2020_2024/chem_mechanistic_{pts_per_shot,rim_share,three_share}.json`
(gitignored). Held-out macro RMSE — role vs rung 3, and role vs its placebo:

| outcome | holdout | rung 3 | **role** | placebo | role vs r3 | role vs plc |
|---|---|---|---|---|---|---|
| **three_share** | chronological | 0.0301 | **0.0289** | 0.0303 | **+3.9 %** | **+4.7 %** |
| (τ_role 0.0076) | unseen_pair | 0.0564 | **0.0551** | 0.0562 | **+2.3 %** | **+1.9 %** |
| | unseen_lineup | 0.0309 | **0.0301** | 0.0309 | **+2.7 %** | **+2.5 %** |
| **pts_per_shot** | chronological | 0.0199 | 0.0234 | 0.0207 | −17.7 % | −13.2 % |
| (τ_role 0.0063) | unseen_pair | 0.1819 | **0.1783** | 0.1818 | **+2.0 %** | **+1.9 %** |
| | unseen_lineup | 0.0467 | **0.0458** | 0.0468 | **+2.1 %** | **+2.1 %** |
| **rim_share** | chronological | 0.0257 | **0.0249** | 0.0257 | **+3.1 %** | **+3.2 %** |
| (τ_role 0.0036) | unseen_pair | 0.0719 | 0.0721 | 0.0719 | −0.3 % | −0.3 % |
| | unseen_lineup | 0.0271 | **0.0267** | 0.0271 | **+1.4 %** | **+1.4 %** |

- **`three_share`** — role beats rung 3 **and** its placebo on **all three**
  holdouts, 2–5 %. It is the first result to survive the chronological
  holdout: shot-selection tendencies are more era-stable than scoring
  efficiency.
- **`pts_per_shot`** (an eFG proxy) — role beats both by ~2 % on the two
  structural holdouts (slightly stronger than the ~1 % on points/100), but
  fails chronological like the points/100 role model.
- **`rim_share`** — helps chronological and unseen_lineup, neutral on
  unseen_pair. The weakest of the three.

### The `three_share` role-pair matrix — the spacing mechanism, explicitly

Non-additive shift in the lineup's three-point-attempt share for a pair of a
cluster-a and a cluster-b player:

```
        c0(shoot) c1(rim big) c2(wing) c3(playmk) c4(creator)
c0(shoot)  +.0129   −.0044     +.0009    +.0058     +.0051
c1(rim big)−.0044   −.0047     −.0112    −.0121     −.0117
c2(wing)   +.0009   −.0112     −.0061    −.0011     −.0033
c3(playmk) +.0058   −.0121     −.0011    +.0095     +.0022
c4(creator)+.0051   −.0117     −.0033    +.0022     −.0014
```

- **Two movement shooters together (+.013): the lineup takes *more* threes
  than the sum of their individual rates.** Shooting reinforces.
- **A rim-running big with anyone (−.004 … −.012): the lineup takes *fewer*
  threes than additive predicts.** A big on the floor pulls the offense
  inside beyond what his own low three rate accounts for.

This is the "spacing" intuition, measured mechanistically, with the sign it
predicts, and surviving a permuted-role placebo on every holdout.

## Verdict against the contract

- The mechanistic outcomes are **not** the contract's primary unit (points per
  100, §5), so a win here does not by itself clear §17.1 — it is *supporting*
  evidence that a real, small, role-dependent non-additivity exists in how
  lineups shoot.
- Combined with the ~1–2 % role edge on points/100 and pts/shot, the picture
  is consistent: **role-conditioned lineup non-additivity is real but small,
  concentrated in shot selection / spacing, and at the current data scale sits
  at the edge of what 40–60 group means resolve.** §26's stop condition is
  weakened further; the effect wants a better-powered confirmation, not
  abandonment.

## Candidate follow-up ideas — progress

1. Role/skill-conditioned interaction — DONE (PR #20). First non-null, ~1 % on
   points/100.
2. **Mechanistic outcomes — DONE (this task). Strongest, most interpretable
   signal: role-conditioning shifts `three_share` by 2–5 %, surviving a
   placebo on all three holdouts.**
3. **Anti-synergy / redundancy (NEXT).** Instead of a 15-cell role-pair
   matrix, a handful of coefficients on engineered *concentration* features —
   per role dimension `d`, the lineup's `(Σ z_d)² − Σ z_d²`. Directional
   hypothesis: `ρ_usage < 0` (redundant creators clash), `ρ_three > 0`
   (shooting concentration = spacing). 6 parameters, each on all 266k stints;
   shuffled-role placebo. Reuses the role-vector machinery (needs per-player
   continuous role vectors added to `RoleClustering`).
4. Playoffs transport — DONE (PR #18).
5. Transaction backtest (T4) — highest cost / highest evidentiary value.

Plus: a better-powered confirmation of the role result (wider holdouts,
bootstrap the role−rung-3 delta, K sweep); master-plan §45 player-lift.

## Next action

Merge this branch. Then start idea #3 (redundancy / anti-synergy): add
per-player continuous role vectors to `RoleClustering`, add a dense-extra-block
path to `fit_augmented_em`, build `redundancy.py` + `courtgraph redundancy`,
leakage-safe evaluation + placebo, report the `ρ_d` coefficients.

# Current Task

Last updated: 2026-09-01

## State

Active — **candidate idea #1: role/skill-conditioned interaction terms** (see
the candidate list below). Branch `task/role-conditioned-interaction` off
`main`.

Prior state (done): roadmap directions #1–#3 complete. The interaction null is
written up (`docs/INTERACTION_FINDINGS.md`, PR #19). The playoffs transport
test is merged (PR #18). The master plan's §45 player-lift item is scoped.
Five consecutive interaction nulls; rung 3 (hierarchical EB) is the reference
baseline.

## What was written (PR #19)

- **`docs/INTERACTION_FINDINGS.md`** — the standing findings document for the
  north-star question. Verdict: **transferable teammate-pair / lineup
  chemistry is NOT SUPPORTED** on 266k real regular-season stints + the
  held-out 2024-25 playoffs, across four leakage-safe evaluation tasks
  (chronological, unseen-pair, unseen-lineup, playoffs transport). No
  interaction rung (4 explicit pairs, 5 low-rank) beats hierarchical additive
  talent (rung 3); the per-pair terms are indistinguishable from a
  parameter-matched placebo in-sample (668 pair groups) and in the playoffs
  (476). The one positive: rung 3's EB intervals transport at near-nominal
  coverage. The document separates what this establishes from what it leaves
  open — talent absorption, no role/skill features, the ~119 pts/100
  single-stint noise floor, dynamic chemistry (out of cycle 1), offense-only.
- **`docs/MASTER_PLAN.md` §45** — the player-lift backlog item (asymmetric
  giver→teammate effect; Phase A a pooled lift scalar on lineup value, Phase B
  a per-player on-court production model with a transaction-cohort test).

## Candidate follow-up ideas (2026-09-01)

Recorded by the user after four interaction nulls. The user has directed that
these be implemented; work them in the suggested priority order, one at a time,
recording the result of each before starting the next.

1. **Role/skill-conditioned interaction terms (ACTIVE).** The rung-4 pair term
   is keyed by raw player identity, so most admitted pairs have thin co-stint
   counts and the test is underpowered by construction even with placebo
   controls. Conditioning the interaction term on measured role/skill features
   (usage rate, 3PT rate, assist rate, defensive role/position) instead of
   identity pools evidence across many pairs sharing a profile (e.g. "two
   low-usage shooters" vs "two ball-dominant guards"), a materially more
   powerful test than per-identity pairs. Master plan §21.5 sketches role
   modeling; this task defines the role features, wires them into a
   `PairHierarchicalRidge`-style model, and re-runs the same leakage-safe +
   placebo-controlled evaluation used for rung 4.
2. **Mechanistic outcome variables instead of points-per-100.** Points/100 is a
   highly aggregated, noisy target that may wash out real but small mechanical
   effects. Candidates: shot-quality / shot-location shift (spacing), turnover
   rate, assist rate, defensive-matchup redundancy. Needs a source-data
   availability check first (shot-location and matchup data are not yet in the
   ingested schema).
3. **Test for anti-synergy (redundancy), not just positive synergy.** Skill
   redundancy (two non-shooters, two ball-dominant creators, overlapping
   defensive assignments) is a more mechanically direct effect and may be
   easier to detect. Reuse the rung-4/5 machinery with a redundancy feature as
   the interaction predictor.
4. **Playoffs transport test — DONE** (PR #18). Fifth consecutive interaction
   null; rung-4 pair terms did not transport RS -> playoffs and were
   indistinguishable from their placebo. Positive side result: rung 3's EB
   intervals stayed near-nominal out of phase.
5. **Transaction backtest (T4 in the research contract) as a causal check.**
   Real trades/injuries as natural experiments — did team performance move the
   way the model predicted once a specific player left/arrived. Causal-flavored,
   the project's eventual gold-standard evidence bar; highest build cost.

Master-plan §45 player-lift (asymmetric "makes teammates better") sits
alongside these — Phase A overlaps idea #1's machinery and can be folded in.

Order: **#1 (active) → #2 → #3 → #5**, plus §45 where it fits.

## Next action

Implement candidate idea #1. First: audit what role/skill features are
derivable from the existing ingested stint schema + snapshots without new
downloads (usage, shooting, playmaking proxies), present the feature set and
model plan, then build and evaluate.

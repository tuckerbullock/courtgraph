# Current Task

Last updated: 2026-09-01

## Active

**Task 1 of an autonomous work queue (user: "do all of those things ... just
keep going"): harden the one confirmed interaction result, `three_share`.**
Branch `task/harden-three-share`.

The `confirm` run established that role-conditioning predicts a lineup's
three-point-attempt share ~3 % better than additive (95 % CI excludes 0 vs
baseline and placebo, K-robust). It is the only interaction positive that
survives proper power. Before building on it, pressure-test it:

- **a. Playoffs transport** — train the role `three_share` model on the RS,
  test on the held-out 2024-25 playoffs. Does the shot-selection
  non-additivity transport?
- **b. Mediation** — does the predicted shot-mix shift move *points*? For the
  held-out lineups, correlate (role's incremental `three_share` prediction)
  with (the lineup's scoring surprise vs rung 3). If a lineup that "should
  shoot more threes" scores no better, the effect is real but not useful.
- **c. `confirm` for the other two shot outcomes** — `pts_per_shot` and
  `rim_share` deserve the same bootstrap-CI + K-sweep treatment (only
  `three_share` got it).
- **d. Robustness** — wider K sweep (up to ~10); sensitivity to the role
  feature set and the `min_fga` cut.

## The work queue (after this task, in order)

2. **Recover the 840 quarantined RS games** (503 `network_required`, 170
   pbpstats back-to-back, 93 score-reconciliation). ~15 % more data; the
   confirmation showed the models are power-limited.
3. **Nullable `days_rest` schema v3** — the 68 season-opener quarantines.
4. **Per-player production ingest** — a pass emitting per-player on-court
   offensive production from the play-by-play. Unlocks 5, 7, and the
   defensive side.
5. **§45 player-lift** — Phase A (pooled lift scalar on lineup value), then
   Phase B (per-player production model + transaction-cohort test).
6. **Defensive-side extension** — roles / redundancy / mechanistic on the
   defensive lineup (all current work is offense-only).
7. **Turnover-rate / assist-rate mechanistic outcomes** — with the full
   `confirm` treatment.
8. **Candidate #5 — transaction backtest (T4)** — real trades/injuries as
   natural experiments. Needs a roster-change dataset (acquisition).
9. **Model-ladder gaps** — rung 1 (EB-shrunk lineup mean) explicit; rung 5
   low-rank re-run under the bootstrap-CI regime.
10. **Contract deliverables** — the "strong" unseen-pair holdout (first-ever
    partnership in the test window); seed-stability across more seeds; the
    cycle-1 research report; the decision log.
11. **More seasons** — 2016-17 → 2019-20 (data-quality gated); 2025-26 when
    published.
12. **Product (§44 / issue #8)** — lineup finder, roster optimizer, real-data
    model serving in the app, the seven §44 appendix capabilities.

Each item lands as its own focused branch + PR. Preserve every result
whatever it is (contract §17). `ChemistryConfig` / `HierarchicalConfig` /
`RoleInteractionConfig` / `RedundancyConfig` defaults unchanged unless a task
explicitly changes them.

## Next action

Build task 1: mechanistic transport path, mediation analysis, `confirm`
`--outcomes` support, robustness sweeps. Run on the real data. Write up.

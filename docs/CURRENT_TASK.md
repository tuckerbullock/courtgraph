# Current Task

Last updated: 2026-09-01

## Active

**Task 1 of an autonomous work queue (user: "do all of those things ... just
keep going"): harden the one confirmed interaction result, `three_share`.**
Branch `task/harden-three-share`. PR #24. **Code + docs complete; result in.**

### Result (2026-09-01)

Hardening **downgrades `three_share`**. It is a small, real, *in-distribution*
regularity in how role-redundant lineups distribute shot attempts — not a
value effect, and it does not generalise to a new context.

- **a. Playoffs transport: NULL.** RS-trained role model, tested on the
  held-out 2024-25 playoffs (65 recurring lineups): Δ RMSE vs rung 3 =
  +0.00005 [−0.0011, +0.0012], P(Δ>0) = 0.53. `pts_per_shot` transport also
  null.
- **b. Mediation ≈ 0.** Over the 120 held-out unseen lineups, corr( role's
  incremental `three_share` prediction , lineup scoring surprise vs rung 3 )
  = **0.03**. Mean |Δ three_share| ≈ 0.3 pp. The shot-mix shift does not move
  points.
- **c. Other shot outcomes: null.** `pts_per_shot` and `rim_share` clear
  neither baseline nor placebo on any holdout / K (`rim_share` role model is
  slightly *worse* than additive).
- **d. Wider K sweep (3–10).** vs **rung 3**: `three_share` role model beats
  additive at K = 3,4,5,6,8 (CI excludes 0), marginal at K = 10 — robust. vs
  **placebo**: CI excludes 0 at **K = 5 and K = 8 only** (K 3,4,6,10 span 0,
  P 0.83–0.94). K-fragile against the placebo.

Verdict: strongest non-additivity the ladder has found, still well short of
`RESEARCH_CONTRACT.md` §17.1. Documented in `docs/INTERACTION_FINDINGS.md`
("Confirmation → Hardening"). Result JSONs gitignored under
`data/nba_snapshots/rs_2020_2024/` (`chem_confirm_hardened.json`,
`chem_tmech_three_share.json`, `chem_tmech_pts_per_shot.json`).

### Next action for task 1

Merge PR #24 (CI green, MERGEABLE). Then start task 2.

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

Merge PR #24, then begin task 2 (recover the ~840 quarantined RS games) on a
fresh branch. Quarantine scoping already done: `quarantine.jsonl` holds 79,135
rows but 78,007 are per-possession `split_lineup_possession`; the per-game
reasons are 503 `network_required`, 207 `unknown_team`, 171
`pbpstats_reconstruction_failed`, 93 `score_reconciliation_failed`, 81
`possession_alternation_failed`, 68 `missing_context`, 5 `ambiguous_scoring`.
Confirm the per-game total against `manifest.json` (not `quarantine.jsonl`)
first.

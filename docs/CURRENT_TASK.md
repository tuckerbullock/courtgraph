# Current Task

Last updated: 2026-09-01

## State

Done — **the interaction null is written up (roadmap direction #1), and the
"player-lift" research item is scoped into the master plan.** Branch
`task/interaction-null-writeup` off `main`. Docs only; no code change.

Roadmap directions #1–#3 are complete. #4 (a different interaction
parameterisation) is now the master plan's **§45 player-lift** item plus the
existing §21 role-conditioned work.

## What was written

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
- **`docs/MASTER_PLAN.md` §45** — the player-lift backlog item, appended after
  §44 (per the backlog rule; no roadmap reorder):
  - **why it is distinct** from the failed symmetric `γ_ij` — asymmetric
    (giver/receiver), pooled across teammates, and (Phase B) measured on a
    teammate's *individual* production, which a lineup-value model cannot
    separate from the giver's own talent;
  - **Phase A** (~1 day): one EM-shrunk `λ_i ~ N(0, τ_λ²)` per player,
    `+ Σ_i λ_i·(A_off − α_i)` on rung 3; four existing holdouts + a
    λ-shuffle placebo. A pinned rank-1 variant of rung 5 (which failed), so
    it must clear the same bar. Expected: another null, recorded either way.
  - **Phase B** (~1–2 weeks): attribute per-player offensive production per
    stint from the play-by-play already in the snapshots (no new download),
    fit `prod_{k,s} = base_k + Σ_{i≠k} lift_i + context`; output a per-player
    `lift_i` in points/100 with a calibrated interval. Decisive test: the
    **transaction cohort** — does `lift_i` predict a traded player's new
    teammates' production shift?
  - **evidence bar** for "supported" (contract §17, adapted): beats a
    receiver-only baseline out of sample, beats the giver-shuffle placebo,
    stays calibrated and seed-stable, shows a non-trivial transaction signal.

- `README.md` links the findings document; `docs/PROJECT_STATUS.md` updated.

## Verification

Docs only. `uv run ruff check .` / `ruff format --check .` unaffected (no
`.py` changes). Markdown reviewed for internal link correctness.

## Next action

Merge this branch. Then start **§45 Phase A** — the pooled lift scalar on
lineup value — as the next task (expected to confirm the null, but cheap and
a differently-shaped test; a clean null here further constrains where
chemistry could be hiding). Phase B is the larger, more valuable follow-up and
needs the per-player production ingest extension.

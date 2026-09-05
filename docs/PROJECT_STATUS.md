# Project Status

Last updated: 2026-09-05 (product side — real-data rung-3 lineup predictor, CLI + app)

## Current phase

Real regular-season data is in and was **doubled** on 2026-09-02
(`task/data-acquisition`):

- **Earlier seasons pulled** from the same pinned SRC-SHUFINSKIY archive: RS
  2016-17 → 2019-20 (+ playoffs 2016 → 2023; `cdnnba`/`nbastatsv3`/`matchups`
  surfaces for 2020-25). `scripts/fetch_shufinskiy.py`, GitHub raw only.
- **A second possession-reconstruction surface** (snapshot format `v2`): when
  the playbyplayv2 (`stats_nba`) path needs a network call, the game is retried
  with pbpstats' `data_nba` provider (period starters from the pbp walk, no
  network). Purely additive — every stint that reconstructed before still does.

Result, `courtgraph ingest` over the 8-season RS archive:

| RS window | games in | accepted | quarantined | stints | recovered via `data_nba` |
|---|---|---|---|---|---|
| 2016-17 … 2019-20 | 4,746 | 4,556 | 190 (4.0 %) | 239,570 | 903 games |
| 2020-21 … 2024-25 | 5,998 | 5,760 | 238 (4.0 %) | 297,404 | 620 games |
| **total** | 10,744 | **10,316** | 428 | **536,974** | 1,523 games |

Was: 5,158 accepted / 266,518 stints / 14 % quarantine (2020-24 only, v1). All
266,518 prior stints are a strict subset of the new 297,404 for that window
(regression-clean). Remaining quarantines are now mostly
`score_reconciliation_failed` (162, data.nba.com running score ≠ official —
fail-closed) and season-opener `missing_context` (143, → Task 3). The 2024-25
playoffs archive is untouched and held out for the transport test
(`DATA_SOURCES.md` §6). stats.nba.com is unreachable from this machine, so the
new `courtgraph fetch-live` path (built, tested) is deferred to the user for
2025-26 and the residual gaps.

This is observational data only — **no model has been fit or evaluated on it
yet**. The score check is within-NBA (stats reconstruction vs data.nba.com
feed), not an independent lineage. No demonstrated betting edge exists.

## Completed

- **Model ladder rung 4 — explicit teammate-pair interaction RAPM**
  (`src/courtgraph/chemistry/pair_interaction.py`, `courtgraph baselines
  --rung4`): the rung-3 EB model plus an explicit `γ_ij` term per admitted
  offensive teammate pair (learned `τ_pair²`); pairs admitted by a training
  co-stint threshold (master plan §15.3). Offense pairs only for v1. **On the
  real data it does not beat rungs 2 or 3** — see *In progress*.
- **Model ladder rung 3 — empirical-Bayes hierarchical player model**
  (`src/courtgraph/chemistry/hierarchical.py`, `courtgraph baselines`): variance
  components learned by EM (not a CV scalar), per-lineup Gaussian predictive
  intervals, and a calibration module (`calibration.py`) with the contract's
  coverage / calibration-line / width-vs-error diagnostics. Standalone —
  `ChemistryModel` untouched. On the widened leakage-safe holdouts it
  out-calibrates rung 2 on the two structural holdouts and is the reference the
  chemistry claims are judged against (see *In progress*).
- **Widened leakage-safe holdouts.** `make_unseen_pair_split` / `_lineup_split`
  default to 60 held groups (from 8 / 12), with a per-pair size cap; the
  chronological macro buckets by calendar month. On the real data the macro
  group counts go 8 / 12 / 2 → 40 / 60 / 13, enough for a conclusive rung-N
  comparison. Still outcome-blind and leakage-free.
- **Chemistry model scales to a real player pool.** The additive baseline and
  the low-rank interaction fit now accumulate every Gram / rhs by `np.bincount`
  scatter over the 5 players per stint instead of dense `(n × n_players)`
  matmuls — numerically identical to the old code (~1e-13), no new dependency.
  `courtgraph fit --evaluate` on 266k stints / 985 players runs in ~16 min.
- **Real regular-season ingestion (2016-17 → 2024-25, 8 seasons).** The
  SRC-SHUFINSKIY importer (multi-`--archive-dir`; a second `data_nba`
  reconstruction surface; `CONVERTER_VERSION` `cg-shufinskiy/4`; snapshot
  format `v2`), run end to end: `scripts/fetch_shufinskiy.py` (GitHub raw,
  pinned commit) → `snapshot-from-shufinskiy --all-games` → `ingest` →
  **536,974 stints** in the versioned `courtgraph.chemistry.stints` format.
  Per-window totals and the quarantine breakdown are in the table above and in
  `docs/CURRENT_TASK.md`.
- Local browser app (`src/courtgraph/app/`, `courtgraph app`), with no new
  dependencies: explicit read-only ingest inputs, verified stint checksum and
  per-game exposure, game/team/player/sample filters, observed lineup rates,
  evidence and quarantine panels, and source provenance.
- Separate deterministic synthetic model and A/B five-player builder, shared
  opposing lineup/context, talent + interaction + context decomposition,
  individual training support, and approximate interaction intervals.
- Loopback-only HTTP server with fixed asset routes, origin/host checks,
  bounded requests, and no filesystem browsing, uploads, or external assets.
- App tests cover weighted aggregation, input immutability, data/manifest
  mismatches, lineup validation, model parity, a non-zero sandbox interaction
  surplus, empty states, and HTTP boundaries. Automated verification is recorded
  in `docs/CURRENT_TASK.md`.
- Whole-archive local ingestion with explicit coverage accounting. The current
  archive produces 3,325 stints and 10,852 accepted possessions across 62 games;
  the UI exposes all 22 failed/incomplete games and their recorded reason.

- Defined the north-star research question.
- Created the full research, modeling, evaluation, engineering, and publication blueprint.
- Defined the model ladder and milestone exit criteria.
- Defined leakage-safe unseen-lineup, unseen-pair, temporal, and transaction evaluations.
- Defined the initial product and API concepts.
- Added the initial Python package skeleton and versioned CLI entry point.
- Added a dependency-free `courtgraph doctor` command with human and JSON output.
- Added bootstrap tests for runtime, repository layout, CLI output, and failure behavior.
- Added shared `AGENTS.md` instructions for coding agents working in the repository.
- Added a concise Claude Code entry point that imports the shared instructions.
- Added `docs/CURRENT_TASK.md` for durable single-task state and cross-agent handoffs.
- Adopted `uv` with a committed `uv.lock` for a locked, reproducible environment (local default Python 3.13).
- Added Ruff (lint + format) and mypy (`strict`) configuration; introduced typed health-report structures so `mypy --strict` passes.
- Added GitHub Actions CI (`.github/workflows/ci.yml`) verifying the lockfile, a clean install, `courtgraph doctor`, unit tests, compilation, Ruff lint, Ruff format, and mypy on Python 3.11 and 3.13. Run #1 passed both legs; actions are pinned to full commit SHAs and the workflow runs with `permissions: contents: read`.
- Added `RESEARCH_CONTRACT.md` — the binding, falsifiable scientific specification for research cycle 1 (units, primary units, model ladder rungs 0–7, four leakage-safe evaluation tasks, calibration and uncertainty standards, six-part evidence bar, permitted/prohibited claims). Merged via PR #2 (`83555f8`).
- Added `DATA_SOURCES.md` and the small NBA data-access & schema pilot (`pilot/`). Merged via PRs #3, #4.
- Added the **synthetic lineup-chemistry vertical slice** (`src/courtgraph/chemistry/`, `numpy` pinned):
  - `stints.py` — a versioned (schema v2) stint-data format: offensive five, defensive five, possessions, points, an explicit ISO `game_date`, season/time, and context; validation and JSON/JSONL IO.
  - `synthetic.py` — a deterministic generator with known additive talent and a low-rank provision/need chemistry structure; a `with_no_interaction()` variant for the no-signal control.
  - `splits.py` — leakage-safe holdouts: chronological (ordered by `game_date`, never `game_id`); **structurally** unseen teammate pairs (every co-play removed, each player kept individually observed — not the contract's "strong" variant); unseen exact lineups. A `verify_split` gate re-derives the forbidden overlaps.
  - `baseline.py` — additive ridge RAPM baseline (separate offensive/defensive talent, game-blocked ridge selection).
  - `chemistry_model.py` — a permutation-invariant low-rank player-embedding model: additive skip path + provision/need interaction fit by alternating ridge on the cross-fitted residual, zero-sum centered; predicts unseen combinations and decomposes each lineup value into talent, interaction surplus, context, and total. A lineup containing any unseen offensive player is predicted **additive-only** (C = 0) everywhere.
  - `evaluate.py` — RMSE/MAE (micro and macro) for the additive baseline vs the full model on all three holdouts, plus exposure/novelty per group and a group-level approximate block-bootstrap interval (ensemble size set by `--bootstrap`).
  - `report.py` — a self-contained HTML report; `artifact.py` — versioned model serialization.
  - CLI: `courtgraph demo [--report PATH]`, `courtgraph fit`, `courtgraph predict`.
  - `tests/test_chemistry_*.py` — deterministic tests for the decomposition identity, permutation invariance, serialization round-trips, leakage-safe splits (including caught leaks), CLI behaviour, additive-talent recovery, and recovery of a real interaction signal beyond the additive baseline (with a matched no-signal control).
- Added the **offline NBA snapshot -> stint importer** (`src/courtgraph/ingest/`, `pbpstats==1.3.11` pinned in an `ingest` dependency group, lazily imported):
  - `snapshot.py` — the documented `stats_nba_pbpstats/v1` layout (raw `stats.nba.com` `playbyplayv2` / `shotchartdetail` files + a `courtgraph_snapshot.json` metadata index); per-file hashes; a throwaway working copy so the snapshot stays immutable.
  - `possessions.py` — `pbpstats` in **file-only mode** as the reconstruction tool, wrapped in an offline guard that turns any network attempt into a quarantine.
  - `validate.py` — CourtGraph's own checks: five per side, possession alternation, exact final-score reconciliation against the independent box-score total, and per-possession exclusion (empty / split-lineup / ambiguous scoring). Never fabricates a value to satisfy the schema.
  - `stints.py` / `pipeline.py` — emits `courtgraph.chemistry.stints` records (non-contiguous same-lineup spells never merged), plus `quarantine.jsonl` and a full `manifest.json` audit trail. `courtgraph ingest --snapshot-dir PATH --out-dir DIR`.
  - `tests/test_nba_*` — hand-authored, NBA-shaped fixtures parsed by real `pbpstats`; covers ordinary play, offensive rebounds, free throws + technical, substitutions, overtime, malformed inputs, reconciliation failure, missing files, immutability, and no-network proof.

## In progress

- **Real-NBA model validation.** The chemistry model's linear algebra was
  reworked to sparse `np.bincount` Gram accumulation (numerically identical to
  the dense path to ~1e-13; `courtgraph fit --evaluate` on 266k stints now runs
  in ~16 min instead of never finishing). On the real regular-season data:
  - The additive ridge RAPM baseline beats the league mean by ~40–48% (macro)
    on the unseen-pair / unseen-lineup holdouts; loses on chronological
    (distribution shift, two test seasons).
  - The **rank-5 low-rank interaction model does not beat the additive
    baseline** — nil on chronological, −0.08% on unseen-pair, −8.4% on
    unseen-lineup (macro RMSE). A genuine null / unfavourable result.
  - **Rung 3 (empirical-Bayes hierarchical player model)** is built, and after
    widening the leakage-safe holdouts (8 → 40 pairs, 12 → 60 lineups, 2 → 13
    chronological months) it **beats rung 2 on calibration and stability** on
    the two structural holdouts — near-ideal standardized-residual SD (1.06 /
    1.04 vs rung 2's 1.45 / 1.59), calibration slope ~1 on unseen-pair, coverage
    close to nominal, and marginally better point RMSE everywhere. It clears the
    contract's rung-3 exit criterion and is now the reference baseline the
    chemistry claims are judged against. Both models fail the chronological
    holdout (systematic under-prediction under era / roster drift). Numbers in
    `docs/CURRENT_TASK.md`.
  - **Rung 4 (explicit teammate-pair interaction RAPM)** — the rung-3 model
    plus a free `γ_ij` term per admitted offensive pair — **does not beat rungs
    2 or 3** on the widened holdouts: macro RMSE 3.843 / 19.222 / 5.495
    (chronological / unseen-pair / unseen-lineup) vs rung 3's 3.551 / 19.203 /
    5.256, and its calibration is no better. The §11 exit test ("beat rung 2 on
    seen pairs") is not met. Numbers in `docs/CURRENT_TASK.md`.
  - **Better-powered pair evaluation (roadmap direction #2)** — a pair-level
    "seen pairs" exit test (macro RMSE over 668 admitted pairs recurring in the
    held-out window, vs the old 209 all-pairs-covered lineups) with a
    placebo-pair control. **The rung-4 null holds at proper power**: rung 2
    8.669 → rung 4 8.543, but the placebo (same parameter count, scrambled
    pair→row wiring) is 8.541 — the tiny gain is added-parameter noise
    absorption, not chemistry. §26's "split sizes too small" concern is now
    resolved; the null survives an adequately powered test. Fourth consecutive
    null for non-additive lineup structure on this data.
  - **Playoffs transport test (roadmap direction #3)** — train rungs 2/3/4 on
    the 266k RS stints, evaluate on the held-out 2024-25 playoffs (3,325
    stints, all 210 players seen in the RS, 0 game overlap). **The interaction
    null holds under phase transport**: pair-level test rung 2 12.98 → rung 4
    13.22, placebo 13.24; macro over 157 playoff lineups flat (23.9 / 24.0 /
    24.2); no clutch-specific effect. Fifth consecutive interaction null. But
    **rung 3's EB intervals transport** — near-nominal playoff coverage
    (.50 / .85 / .96) where rung 2's bootstrap band badly under-covers
    (.30 / .53 / .75). `z_mean ≈ 0` for all three (no large systematic phase
    offset at the lineup-value level). `courtgraph transport`; see
    `docs/CURRENT_TASK.md`.
  - **The "not supported" finding is written up** (roadmap direction #1):
    [`docs/INTERACTION_FINDINGS.md`](INTERACTION_FINDINGS.md) is the standing
    record — the verdict, the models, the four evaluation tasks, the results,
    and what the null does and does not establish (talent absorption, no
    features, the noise floor, dynamic effects out of scope).
  - **Role-conditioned interaction — the first non-null.** `courtgraph
    player-features` derives per-(player, season) role/skill profiles from the
    snapshot's play-by-play + shot chart (no new data); `courtgraph roles`
    keys the interaction on **role-cluster pair** (K=5 → 15 pooled params) and
    compares against rung 3 and a permuted-role placebo. On the two structural
    holdouts role beats rung 3 by 0.7 % / 1.3 % **and** beats its placebo by
    0.7 % / 1.4 % — small (40–60 group means, no CI yet) and it degrades under
    temporal drift, but the first form to beat baseline and placebo out of
    sample. In-sample role-pair matrix: "star + complementary piece" (+1.4 to
    +1.8) beats "star + star" (+0.8). `docs/CURRENT_TASK.md` has the numbers.
  - Confirmation + hardening done (`courtgraph confirm`,
    `transport-mechanistic`): only `three_share` beats the baseline, and even
    that fails the placebo at most K, fails to transport to the playoffs, and
    is decoupled from scoring (mediation r = 0.03). See
    `docs/INTERACTION_FINDINGS.md` "Confirmation → Hardening".
  - **Data doubled (2026-09-02):** RS dataset 266,518 → 536,974 stints (8
    seasons; +1,523 games recovered by the `data_nba` surface). Model-ladder /
    `confirm` re-run at the new scale sharpened the negative.
  - **§45 Phase A — null (2026-09-02, `courtgraph player-lift`).** A pooled
    lift scalar per player on lineup value is indistinguishable from zero and
    from a player-scrambled placebo (identical `τ_λ` per fold; full-fit `τ_λ²`
    at the grid floor).
  - **Transaction backtest — null (2026-09-02, `courtgraph transaction-backtest`).**
    585 clean cross-season team switches vs 1,200 phantom non-movers. Movers'
    lineups scatter from the additive prediction no more than non-movers'
    (mean |Δ| gap −0.30 [−0.63, +0.03]). A player's lineup-value contribution
    transfers across a team change as cleanly as a non-mover's stays put — the
    best-powered test in the project, and a clean null.
  - **§45 Phase B — null (2026-09-02, `courtgraph player-production` +
    `phase-b`).** A per-(player, stint) production ingest (99.2 % event match,
    validated vs known stars) feeds a lift model on teammates' *individual*
    production. The lift terms do not beat a base-only (receiver's own level)
    model out of sample (points-only +0.01 [−0.12, +0.13]). **Closes the last
    open estimand** — a player's effect on teammates is not transferable beyond
    additive talent, whether measured on lineup value, roster changes, or
    individual production.
  - **Defensive-side pooled lift — null (2026-09-02, `player-lift --side
    defense`).** On 297k and 537k the defensive lift terms make held-out
    prediction *worse* than rung 3; placebo recovers the identical `τ_λ`.
  - **The interaction arc is complete.** Symmetric pairs (rungs 4–5), pooled
    lift on lineup value (Phase A), across roster changes (transaction
    backtest), on individual production (Phase B), defensive side — **every
    estimand null.**
  - **Real-data lineup predictor (2026-09-05, `task/rung3-lineup-predictor`).**
    The product-side first slice: `courtgraph fit-rung3` / `predict-rung3`
    persist a fitted rung-3 model and score an arbitrary 5-vs-5 lineup of
    real observed players — additive talent + context + a calibrated
    interval, no interaction/chemistry field anywhere in the result type.
    Wired into the local app (`Observations.player_pool`/`predict`, new
    `/api/player-pool` and `/api/predict-real` endpoints, a new frontend
    panel) so it's something a user can touch, not just a CLI artifact.
    Issue #8's "chemistry surplus" ranking stays explicitly unbuildable —
    chemistry isn't a supported predictive effect. Next: another item from
    the work queue in `docs/CURRENT_TASK.md`, or more of issue #8 that
    doesn't depend on chemistry (real dated rosters, league-wide search).

## Not started
- Nullable `days_rest` (stint schema v3) for the season-opener `missing_context`
  quarantines (143 across the 8 RS seasons). The `network_required` and
  back-to-back quarantines are largely resolved by the `data_nba` surface; the
  ~92 residual `network_required` and 162 `score_reconciliation_failed` need
  a live fetch or a manual reconciliation-target review.
- 2025-26 regular season — needs `courtgraph fetch-live` (stats.nba.com
  unreachable here) or a `cdnnba`→pbp importer path; the `cdnnba`/`nbastatsv3`
  archives for it are pulled and waiting.
- Promotion of 2016-17 → 2019-20 to "binding" coverage — gated on the
  data-quality checks (final-score reconciliation, player-seconds vs box score)
  now that the era is ingested.
- Re-running the model ladder / `confirm` on the doubled RS dataset (in
  progress this task).
- The contract's independent-parser gate and multi-game reconciliation gate; minute/lineup-minute reconciliation.
- Model-ladder rungs 1 and 7; calibrated Bayesian uncertainty.
- Mid-season-trade transaction cohort (the cross-season T4 backtest is done —
  null); the contract's full six-part evidence bar.
- Real-NBA predictive lineup recommendations, dated complete-roster generation, and the broader product backlog. The local observational/synthetic app is implemented.

## Current verification

Run from the repository root (requires `uv`):

```bash
uv sync --locked
uv run courtgraph doctor
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Dependency-free path (`courtgraph doctor` imports no third-party package; the
chemistry tests skip when `numpy` is absent):

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Exercise the vertical slice:

```bash
uv run courtgraph demo --report demo_report.html --out-dir courtgraph_demo
```

The current implementation passes 260 unit tests, Ruff, mypy over 74 source
files, and JavaScript syntax validation. The multi-season pipeline was run end
to end on the 8-season regular-season archive (2016-17 → 2024-25): 10,744
games with all three inputs, **10,316 accepted, 428 quarantined (4.0 %),
536,974 stints**; 1,523 of those games were reconstructed via the second
(`data_nba`) surface. The full chemistry model was fit + evaluated on the
earlier 266k real stints in ~16 min after the sparse rework.

On the default synthetic dataset (17k stints, deterministic) the low-rank model
beats the additive baseline on macro held-out lineup value (vs the known truth)
by roughly 25–30% on the unseen-lineup and unseen-pair holdouts and by ~0% on
the chronological holdout (chemistry is a small residual; the improvement
concentrates in the group-level and truth-referenced views). The matched
no-signal control produces no improvement.

## Next verifiable outcome

Ladder rungs 0–5 have been fit and evaluated on the 266k real regular-season
stints across **four** leakage-safe evaluation tasks — the three widened
in-sample holdouts (chronological / unseen-pair / unseen-lineup) and the
playoffs phase-transport test (`docs/CURRENT_TASK.md`). Rung 3 (hierarchical
EB) is the reference baseline: it out-calibrates rung 2 on the structural
holdouts, and its predictive intervals transport to the playoffs at
near-nominal coverage. **No interaction rung beats it on any task**: rung 5
(low-rank) and rung 4 (explicit pairs) both fail to improve held-out
prediction; the rung-4 pair terms are indistinguishable from a placebo on a
well-powered pair-level test in-sample (668 pair groups) and in the playoffs
(476 pair groups). The null is written up in
[`docs/INTERACTION_FINDINGS.md`](INTERACTION_FINDINGS.md).

**One small non-additivity survives a properly-powered test — in shot
selection, not scoring.** Three later parameterisations (`courtgraph roles`,
`mechanistic`, `redundancy`) each showed a ~1–5 % edge on 40–60 group means. A
better-powered re-run (`courtgraph confirm` — `unseen_lineup` widened to 120
groups, a K sweep, a 3,000-resample bootstrap CI on the
`RMSE(baseline) − RMSE(model)` delta) keeps **one**: role-conditioning
predicts a lineup's **three-point-attempt share** ~3 % better than rung 3,
95 % CI excluding 0 against the baseline across the K sweep (3–10). The
**points/100** role effect is marginal (CI just excludes 0 at K = 5 only,
K-fragile); **redundancy** does not survive (CI [−0.05, +0.05]).

**Hardening (2026-09-01, `courtgraph confirm` wide-K + `transport-mechanistic`)
downgrades `three_share`**: it beats the permuted-role placebo at only K = 5
and K = 8 of 6; it does **not** transport to the held-out 2024-25 playoffs
(Δ RMSE vs rung 3 ≈ 0, P(Δ>0) = 0.53); and its predicted shot-mix shift is
uncorrelated with scoring (mediation r = 0.03). `pts_per_shot` / `rim_share`:
null everywhere. So it is a small, real, *in-distribution* shot-distribution
regularity, not a value effect. `RESEARCH_CONTRACT.md` §17.1 (a significant
primary-unit improvement) remains **not met**. Remaining: recover the
quarantined games (power), candidate idea #5 (transaction backtest), §45
player-lift, or the cycle-1 research report. Neural rungs 6–7 stay gated.

## Governing document

The [master plan](MASTER_PLAN.md) is the living operating blueprint. Material changes should be recorded in a decision log rather than silently replacing earlier reasoning.

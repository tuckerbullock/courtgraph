# Project Status

Last updated: 2026-09-01 (pair-level rung-4 evaluation)

## Current phase

Real regular-season data is in. `courtgraph ingest` has been run once over a
local, gitignored **five-season regular-season archive (2020-21 → 2024-25)**:
5,998 games with all three provider inputs, **5,158 accepted → 266,518 stints,
941,897 accepted possessions**, 30 teams, 985 players. 840 games (14%) are
quarantined, the majority (`network_required`, 503) because `pbpstats` wants a
box-score call the offline guard blocks. The 2024-25 playoffs archive is
untouched and held out for the transport test (`DATA_SOURCES.md` §6).

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
- **Real regular-season ingestion (2020-21 → 2024-25).** The SRC-SHUFINSKIY
  importer, generalized to multi-season archives (glob file discovery; rest
  days bounded to the same season; provenance hashes every consumed CSV;
  `CONVERTER_VERSION` `cg-shufinskiy/3`), run end to end: acquire (local,
  gitignored) → `snapshot-from-shufinskiy --all-games` → `ingest` → 266,518
  stints in the versioned `courtgraph.chemistry.stints` format. Coverage,
  per-season splits, and the quarantine breakdown are in `docs/CURRENT_TASK.md`.
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
  - Next (roadmap direction #3): more / different data — a playoffs transport
    test (train RS, test the held-out 2024-25 playoffs), more seasons, or
    possession-level outcomes. See `docs/CURRENT_TASK.md`.

## Not started
- Recovering the 840 quarantined regular-season games (503 `network_required`,
  170 pbpstats back-to-back exceptions, 93 score-reconciliation failures);
  nullable `days_rest` (stint schema v3) for the 68 season-opener quarantines.
- 2025-26 regular season (no play-by-play from this source yet) and
  2016-17 → 2019-20 (contract-gated on data-quality checks).
- The contract's independent-parser gate and multi-game reconciliation gate; minute/lineup-minute reconciliation.
- Model-ladder rungs 1 and 7; calibrated Bayesian uncertainty.
- Transaction backtest (T4); the contract's full six-part evidence bar.
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

The current implementation passes 190 unit tests, Ruff, mypy over 53 source
files, and JavaScript syntax validation. The multi-season pipeline was run end
to end (`snapshot-from-shufinskiy --all-games` → `ingest` → `courtgraph app
--ingest-dir`) on the local five-season regular-season archive: 6,000 games
found, 5,998 with all three inputs, 5,158 accepted, 840 quarantined, two missing
the feed. The full chemistry model was fit + evaluated on the 266k real stints
in ~16 min after the sparse rework (was: never finished).

On the default synthetic dataset (17k stints, deterministic) the low-rank model
beats the additive baseline on macro held-out lineup value (vs the known truth)
by roughly 25–30% on the unseen-lineup and unseen-pair holdouts and by ~0% on
the chronological holdout (chemistry is a small residual; the improvement
concentrates in the group-level and truth-referenced views). The matched
no-signal control produces no improvement.

## Next verifiable outcome

Ladder rungs 0–5 have been fit and evaluated on the 266k real regular-season
stints against the widened holdouts (`docs/CURRENT_TASK.md`). Rung 3
(hierarchical EB) is the reference baseline and out-calibrates rung 2 on the
structural holdouts. **Neither interaction rung beats it**: rung 5 (low-rank)
and rung 4 (explicit pairs) both fail to improve held-out prediction. The
rung-4 null now survives a well-powered, placebo-controlled pair-level exit
test (668 pair groups), so `RESEARCH_CONTRACT.md` §26's "successive models show
no transferable interaction signal" applies and its "split sizes too small"
caveat is resolved. Remaining shots (`docs/CURRENT_TASK.md`): a playoffs
transport test and more / different data (direction #3); a role-conditioned
interaction parameterization (direction #4). Neural rungs 6–7 stay gated — the
interaction question has not passed at rungs 0–5.

## Governing document

The [master plan](MASTER_PLAN.md) is the living operating blueprint. Material changes should be recorded in a decision log rather than silently replacing earlier reasoning.

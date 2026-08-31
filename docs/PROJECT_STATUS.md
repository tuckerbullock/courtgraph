# Project Status

Last updated: 2026-08-30 (synthetic chemistry vertical slice)

## Current phase

Vertical-slice prototype — one full path through the product on **synthetic**
data (model ladder rungs 0/2 and 5/6 of `RESEARCH_CONTRACT.md` §11), with
leakage-safe evaluation. Real NBA data is not yet ingested.

## Completed

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

## Not started

- Real data acquisition, possession/stint reconstruction (`DATA_SOURCES.md` §8).
- Model-ladder rungs 1, 3, 4, and 7; calibrated Bayesian uncertainty.
- Transaction backtest (T4); the contract's full six-part evidence bar.
- Dashboard or API implementation.

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

On the default synthetic dataset (17k stints, deterministic) the low-rank model
beats the additive baseline on macro held-out lineup value (vs the known truth)
by roughly 25–30% on the unseen-lineup and unseen-pair holdouts and by ~0% on
the chronological holdout (chemistry is a small residual; the improvement
concentrates in the group-level and truth-referenced views). The matched
no-signal control produces no improvement.

## Next verifiable outcome

Replace `synthetic.py` with a real NBA possession/stint source (`DATA_SOURCES.md`
§8) emitted in the `courtgraph.chemistry.stints` format, then re-run the same
splits, baseline, model, and evaluation on real data and compare to the
contract's rung-2/3 reference baselines.

## Governing document

The [master plan](MASTER_PLAN.md) is the living operating blueprint. Material changes should be recorded in a decision log rather than silently replacing earlier reasoning.

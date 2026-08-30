# Project Status

Last updated: 2026-08-29 (research contract merged; data-source decision in review)

## Current phase

Stage 0 — Research contract and repository foundation.

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

## In progress

- **Data-source registry and source-selection decision** (`DATA_SOURCES.md`, branch `task/data-sources`, uncommitted). Engineering assessment covering NBA raw surfaces, client/tool separation, provisional coverage windows (dev 2023-24…2025-26; cycle 2020-21…2025-26), an interim within-NBA validation stack, a manually curated transaction cohort, a conservative access policy, and a restricted public-release scope. Pending Codex re-review; not yet committed or merged.

## Not started

- Data acquisition
- Possession or stint reconstruction
- RAPM implementation
- Chemistry models
- Dashboard or API implementation

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

Dependency-free path (no `uv`):

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Next verifiable outcome

Complete Codex re-review of `DATA_SOURCES.md`, then commit and merge it. The
single task after that is the small data pilot it specifies (`DATA_SOURCES.md`
§8) or possession-rule work (master plan §7); neither begins until activated.

## Governing document

The [master plan](MASTER_PLAN.md) is the living operating blueprint. Material changes should be recorded in a decision log rather than silently replacing earlier reasoning.

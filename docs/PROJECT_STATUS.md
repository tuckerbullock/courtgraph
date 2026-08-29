# Project Status

Last updated: 2026-08-29 (dev environment + CI)

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
- Adopted `uv` with a `uv.lock` for a locked, reproducible environment (local default Python 3.13).
- Added Ruff (lint + format) and mypy (`strict`) configuration; introduced typed health-report structures so `mypy --strict` passes.

## In progress

- Locked development environment and automated CI. Implementation and local
  verification are done under actual CPython 3.11.16 and 3.13.15; `uv.lock` and
  `.github/workflows/ci.yml` are staged. Commit, push, and the first GitHub
  Actions run are still pending. See `docs/CURRENT_TASK.md`.

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

Commit and push `task/dev-environment-ci` and confirm both CI matrix legs pass.
The next single task after that is to create `RESEARCH_CONTRACT.md` (a concise
research contract). `DATA_SOURCES.md` and the first architecture decision record
are separate later tasks.

## Governing document

The [master plan](MASTER_PLAN.md) is the living operating blueprint. Material changes should be recorded in a decision log rather than silently replacing earlier reasoning.

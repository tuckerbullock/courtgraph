# Project Status

Last updated: 2026-08-29

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

## Not started

- Data acquisition
- Possession or stint reconstruction
- RAPM implementation
- Chemistry models
- Dashboard or API implementation

## Current verification

Run from the repository root:

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Next verifiable outcome

Create and lock the reproducible development environment, then formalize the concise research contract before evaluating data-source access and terms.

## Governing document

The [master plan](MASTER_PLAN.md) is the living operating blueprint. Material changes should be recorded in a decision log rather than silently replacing earlier reasoning.

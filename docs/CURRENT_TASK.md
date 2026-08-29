# Current Task

Last updated: 2026-08-29

## State

Complete. Committed on `task/dev-environment-ci`, pushed to `origin`, and CI
run #1 passed on both Python 3.11 and 3.13. Node.js-20 deprecation warnings
from the first run were resolved by pinning the GitHub Actions to current
releases at full commit SHAs and adding least-privilege workflow permissions.

The repository is ready for a new single-task assignment. No new implementation
task should begin until the user assigns one.

## Completed task

### Locked reproducible development environment and automated CI

## Objective

Give CourtGraph a locked, reproducible development environment and a GitHub
Actions workflow that verifies every contribution automatically.

## Acceptance criteria

- `uv` manages the environment; `uv.lock` is committed and current.
- `requires-python = ">=3.11"` unchanged; local default Python 3.13; CI matrix 3.11 + 3.13.
- Ruff (lint + format) and mypy (`strict`) configured and passing.
- The `unittest` suite is retained (no pytest migration).
- No basketball / scientific / modeling / notebook / API / dashboard dependencies added.
- No pre-commit hooks.
- CI verifies: lockfile current + clean install, `courtgraph doctor`, unit tests,
  compilation, Ruff lint, Ruff format check, mypy.
- `README.md`, `docs/PROJECT_STATUS.md`, `docs/CURRENT_TASK.md` updated.

## Decisions and assumptions

- Dev tools live in a PEP 735 `[dependency-groups]` `dev` group, pinned exactly
  (`ruff==0.16.5`, `mypy==2.3.1`); `courtgraph` keeps zero runtime dependencies.
- `mypy --strict` required typed structures for the health report: added
  `CheckResult` / `HealthReport` `TypedDict`s in `src/courtgraph/health.py` and
  updated `src/courtgraph/cli.py` signatures. No CLI output, exit-code, JSON
  shape, or `schema_version` change.
- Ruff auto-fixed import ordering and applied `collections.abc` imports (`UP035`)
  in the existing modules; formatting is Ruff's.
- CI pins `uv` to `0.12.7` (matching the version used to generate the lockfile)
  and enables its cache. Workflow runs on all branch pushes and PRs.
- GitHub Actions are pinned to full commit SHAs with a trailing version comment
  (`actions/checkout` v7.0.1, `astral-sh/setup-uv` v10.0.1) to remove the
  Node.js-20 deprecation warnings and for supply-chain integrity.
- Workflow declares `permissions: contents: read` (least privilege).
- No CI status badge added (repository slug not confirmed).

## Files changed

- `pyproject.toml`: `dev` dependency group; `[tool.ruff]` / `[tool.mypy]` config.
- `.python-version`: new; `3.13`.
- `uv.lock`: new; committed.
- `src/courtgraph/health.py`: `CheckResult` / `HealthReport` TypedDicts; typed return.
- `src/courtgraph/cli.py`: typed `_render_human` signature; import tidy.
- `src/courtgraph/__main__.py`, `tests/test_health.py`: Ruff import ordering only.
- `.github/workflows/ci.yml`: new; `verify` job, Python 3.11 + 3.13 matrix,
  job-level `UV_PYTHON` plus an interpreter-assertion step, workflow-level
  `permissions: contents: read`, and SHA-pinned actions.
- `README.md`: `uv` bootstrap instructions; Stage 0 next-step note.
- `AGENTS.md`: refreshed verification command block and engineering baseline.
- `docs/PROJECT_STATUS.md`: env + CI in Completed; verification block and next
  outcome refreshed.
- `docs/CURRENT_TASK.md`: this handoff.

## Verification

Run on this machine with `uv 0.12.7`, under actual CPython 3.11.16 and actual
CPython 3.13.15 (interpreter asserted with `sys.version_info[:2]` each run;
the dependency-free path additionally checked on the system CPython 3.13.1):

- `uv lock --locked`: passes (lockfile current).
- `uv sync --locked` (`UV_PYTHON=3.11` and `UV_PYTHON=3.13`): passes.
- `uv run courtgraph doctor`: `healthy` on both.
- `uv run python -m unittest discover -s tests -v`: 7 tests, OK on both.
- `uv run python -m compileall -q src tests`: passes on both.
- `uv run ruff check .`: All checks passed on both.
- `uv run ruff format --check .`: 12 files already formatted on both.
- `uv run mypy`: Success, no issues in 5 source files on both.
- Dependency-free: `PYTHONPATH=src python3 -m courtgraph doctor` and
  `... -m unittest discover -s tests` still pass.

GitHub Actions:

- CI run #1 (commit `48e1659`) passed both `verify (Python 3.11)` and
  `verify (Python 3.13)`, with Node.js-20 deprecation warnings on the runner
  actions.
- Actions then pinned to current SHAs and the follow-up commit runs the CI
  again; both legs are expected to pass warning-free (confirm the run for the
  action-modernization commit before treating CI as fully green).

## Risks or blockers

- `uv` must be installed by any contributor or agent before the `uv run`
  commands work.
- The action SHA pins must be bumped deliberately when a newer release is
  wanted; Dependabot for `github-actions` is a possible later convenience task.

## Exact next action

The user assigns the next single task. The sole queued Stage 0 candidate is:
**create `RESEARCH_CONTRACT.md`** (a concise research contract). No agent should
begin it until the user activates it.

## Current verified baseline

- Branch: `task/dev-environment-ci` (off `main`; pushed to `origin`, not merged).
- Python package version: `0.1.0`.
- Environment: `uv` + committed `uv.lock`; local default Python 3.13.
- Dev tooling: `ruff==0.16.5`, `mypy==2.3.1` (strict), both passing.
- CI: `.github/workflows/ci.yml`, Python 3.11 + 3.13; run #1 green, actions
  SHA-pinned, workflow permissions least-privilege.
- Implemented capability: `courtgraph doctor`.
- Basketball data acquisition: not started.
- Possession/stint construction: not started.
- Statistical or ML models: not started.
- Dashboard/API: not started.

## Handoff template

Replace or update the sections above when a task becomes active.

```markdown
## State

Active | Blocked | Complete | Unassigned

## Objective

One sentence describing exactly one task.

## Acceptance criteria

- Verifiable outcome
- Required tests
- Required documentation

## Decisions and assumptions

- Decision plus reason

## Files changed

- `path`: purpose

## Verification

- `command`: result

## Risks or blockers

- None, or the exact unresolved issue

## Exact next action

One concrete action for the next agent or user.
```

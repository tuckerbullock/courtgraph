# Current Task

Last updated: 2026-08-29

## State

Implementation and local verification complete. Commit, push, and the first
GitHub Actions run remain pending. No new implementation task should begin until
those are done and the user assigns the next task.

## Active task

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
- No CI status badge added (repository slug not confirmed).

## Files changed

- `pyproject.toml`: `dev` dependency group; `[tool.ruff]` / `[tool.mypy]` config.
- `.python-version`: new; `3.13`.
- `uv.lock`: new; staged, commit pending.
- `src/courtgraph/health.py`: `CheckResult` / `HealthReport` TypedDicts; typed return.
- `src/courtgraph/cli.py`: typed `_render_human` signature; import tidy.
- `src/courtgraph/__main__.py`, `tests/test_health.py`: Ruff import ordering only.
- `.github/workflows/ci.yml`: new; `verify` job, Python 3.11 + 3.13 matrix,
  job-level `UV_PYTHON` plus an interpreter-assertion step.
- `README.md`: `uv` bootstrap instructions; Stage 0 next-step note.
- `AGENTS.md`: refreshed verification command block and engineering baseline.
- `docs/PROJECT_STATUS.md`: env/tooling in Completed, CI in a new "In progress"
  section; verification block refreshed.
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

## Risks or blockers

- Commit and push have not happened; the branch is local only.
- CI has not been executed on GitHub yet. The workflow mirrors the locally
  verified commands and now forces the matrix interpreter via job-level
  `UV_PYTHON` plus an assertion step, but the first real run must still be
  confirmed after push.
- `uv` must be installed by any contributor or agent before the `uv run`
  commands work.

## Exact next action

1. Commit the staged changes on `task/dev-environment-ci` (logical commits).
2. Push the branch and confirm both CI matrix legs (3.11 and 3.13) pass.
3. Then update this file's State to `Complete`.

After that, the user assigns the next single task. The sole queued Stage 0
candidate is: **create `RESEARCH_CONTRACT.md`** (a concise research contract).
`DATA_SOURCES.md` and the first architecture decision record are separate future
tasks. No agent should begin any of them until the user activates it.

## Current verified baseline

- Branch: `task/dev-environment-ci` (off `main`; not committed, not pushed).
- Python package version: `0.1.0`.
- Environment: `uv` + committed `uv.lock`; local default Python 3.13.
- Dev tooling: `ruff==0.16.5`, `mypy==2.3.1` (strict), both passing.
- CI: `.github/workflows/ci.yml`, Python 3.11 + 3.13, not yet run on GitHub.
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

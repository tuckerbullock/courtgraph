# Current Task

Last updated: 2026-08-29

## State

No implementation task is active. The repository is ready for a new single-task assignment.

## Most recently completed task

### Cross-agent context and handoff layer

Objective:

> Make project context transferable between Codex, Claude Code, and other repository-aware coding agents without relying on chat history.

Completed:

- Added `AGENTS.md` as the vendor-neutral source of project working rules.
- Added `CLAUDE.md` as Claude Code’s concise entry point importing the shared rules.
- Added this file as the durable current-task and handoff ledger.
- Kept the full master plan out of automatically loaded agent context.

Acceptance criteria:

- Shared instructions identify the project purpose, sources of truth, one-task workflow, verification commands, and research-integrity constraints.
- Claude Code is directed to import the shared instructions and read only relevant master-plan sections.
- The handoff ledger can distinguish active, completed, blocked, and unassigned states.
- Existing CourtGraph health and test commands still pass.

## Current verified baseline

- Branch: `main`
- Python package version: `0.1.0`
- Implemented capability: `courtgraph doctor`
- Basketball data acquisition: not started
- Possession/stint construction: not started
- Statistical or ML models: not started
- Dashboard/API: not started

Verification commands:

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

## Next candidate task

Create a locked, reproducible development environment and automated GitHub Actions verification. This is a candidate only; no agent should begin it until the user activates it.

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

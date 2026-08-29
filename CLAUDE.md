@AGENTS.md

# Claude Code entry point

At the beginning of every CourtGraph session:

1. Read `README.md`, `docs/PROJECT_STATUS.md`, and `docs/CURRENT_TASK.md`.
2. Inspect `git status --short --branch` and recent commits.
3. Read only the sections of `docs/MASTER_PLAN.md` relevant to the current task.
4. Before editing, present the task scope, acceptance criteria, likely files, and verification plan.

Use plan mode for new or materially changed tasks. Work on only the active task and stop after its implementation, tests, documentation, and handoff are complete. Do not begin the next roadmap item automatically.

Use `/context` when necessary to confirm this file and `AGENTS.md` were loaded. Keep personal machine-specific instructions in `CLAUDE.local.md`, which is ignored by Git.

# Current Task

Last updated: 2026-08-29

## State

Active — `RESEARCH_CONTRACT.md` drafted and revised; **pending independent
(Codex) review**. Not committed, not pushed. No other task begins until the
contract is reviewed and committed.

## Objective

Create a root-level `RESEARCH_CONTRACT.md` that turns the north-star question into
a binding, falsifiable scientific specification for research cycle 1. All binding
decisions live in that document (see its §28 decision table and §29 deferred
questions); they are not duplicated here.

## Files changed

- `RESEARCH_CONTRACT.md` — new; 29 sections; the binding contract for cycle 1.
- `docs/CURRENT_TASK.md` — this handoff.

No code, `pyproject.toml`, `uv.lock`, `.github/`, `README.md`,
`docs/PROJECT_STATUS.md`, or data-source files changed.

## Verification

Documentation-only task; proportional checks on `task/research-contract`:

- Structure: sections 1–29 present and sequential; headings render; LaTeX
  delimiters balanced; no tabs / trailing whitespace.
- Reference: `docs/MASTER_PLAN.md` link resolves from repo root.
- `git diff --check`: clean.
- `uv run courtgraph doctor`: `healthy` (`RESEARCH_CONTRACT.md` is not a
  required-path check).
- `git status`: only `RESEARCH_CONTRACT.md` (new) and `docs/CURRENT_TASK.md`
  (modified).

## Open review concerns

- Real methodological commitments to confirm against `docs/MASTER_PLAN.md`:
  points-per-100 units with defensive β subtracted (§4); possession + stint
  units (§6); model ladder rungs 0–7 incl. neural embeddings and Deep Sets
  (§11); four evaluation tasks (§12); six-part evidence bar (§17).
- §8 counterfactual estimands (pair surplus, marginal/replacement value,
  dependency) rewritten to average over complete lineups under a declared
  reference distribution and use same-size 5-for-5 contrasts — confirm sound.
- Ten questions deferred to the data-source task (§29); contract not fully
  closed until resolved by amendment.

## Next action

Codex reviews `RESEARCH_CONTRACT.md`. On approval: commit on
`task/research-contract` (off `main`, which has the merged env/CI work at
`2e92424`), open a PR into `main`. Next single task after that is
`DATA_SOURCES.md`; do not begin until the user activates it.

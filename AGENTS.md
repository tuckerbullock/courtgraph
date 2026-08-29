# CourtGraph Agent Instructions

These instructions apply to every coding or research agent working in this repository. Tool-specific instruction files should import or defer to this file instead of duplicating it.

## Project purpose

CourtGraph is a research-grade system for learning and evaluating NBA lineup chemistry.

North-star question:

> Can we estimate how NBA players will fit together before we have observed that exact combination on the court?

The intended decomposition is:

```text
lineup value = individual talent + player interactions + context
```

Chemistry is a model-dependent predictive quantity. Do not present it as proven causation.

## Sources of truth

Read these in order when beginning work:

1. `README.md` — project overview and current commands.
2. `docs/PROJECT_STATUS.md` — completed and unstarted capabilities.
3. `docs/CURRENT_TASK.md` — the only active task and its handoff state.
4. Relevant sections of `docs/MASTER_PLAN.md` — detailed research and engineering blueprint.

Do not load or summarize the entire master plan unless the task genuinely requires it. Read the table of contents, then inspect only relevant sections.

## One-task rule

- Work on exactly one explicitly defined task at a time.
- Before editing, state the task, acceptance criteria, likely files, and verification plan.
- Do not begin the next task automatically after completing the current one.
- Do not add adjacent features merely because they are convenient.
- A task may add capability, evidence, reliability, or usability; novelty alone is not an improvement.
- If the active task is unclear or conflicts with `docs/CURRENT_TASK.md`, stop and resolve the discrepancy.

## Start-of-task protocol

1. Run `git status --short --branch` and inspect recent commits.
2. Preserve all existing user work and unrelated changes.
3. Read the sources of truth listed above.
4. Inspect existing implementation and tests before proposing new structure.
5. Confirm the task can be completed without hidden future data, undocumented assumptions, or destructive operations.
6. Use a focused branch for substantial changes unless the user explicitly chooses another workflow.

Never reset, discard, overwrite, or reformat unrelated work. Never force-push unless the user explicitly requests it and the exact consequences are understood.

## Current engineering baseline

- Supported Python: 3.11 and newer; local default 3.13.
- Environment manager: `uv`, with a committed `uv.lock`. Dev tools (`ruff`, `mypy`) live in the `dev` dependency group; `courtgraph` itself has no runtime dependencies.
- Package layout: `src/courtgraph/`.
- Tests currently use the Python standard library `unittest`.
- No basketball data or modeling implementation exists yet.
- Third-party dependencies must earn their inclusion and be pinned exactly in `pyproject.toml` and `uv.lock`.
- CI (`.github/workflows/ci.yml`) runs every verification command below on Python 3.11 and 3.13.

Current verification commands:

```bash
uv sync --locked
uv run courtgraph doctor
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The dependency-free path must keep working even as the scientific environment grows:

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Engineering standards

- Prefer clear typed interfaces, deterministic behavior, and small composable modules.
- Keep reusable logic out of notebooks.
- Add tests for success, failure, and important edge cases.
- Every fixed bug must receive a regression test.
- Avoid premature distributed systems, abstractions, or model complexity.
- Do not silently change schemas, metric definitions, units, signs, or reference populations.
- Store configuration explicitly; do not hide research choices in code constants.
- Never commit secrets, credentials, raw private data, large model artifacts, or local environment files.
- Keep generated data and artifacts out of Git unless the repository policy explicitly permits a small fixture.

## Research integrity

- Strong additive and shrinkage baselines must precede advanced interaction models.
- Random possession splits are diagnostics, not proof of roster-construction generalization.
- Historical predictions may use only information available at their stated cutoff.
- Keep exact unseen-lineup, unseen-pair, chronological, and transaction holdouts leakage-safe.
- Separate observed facts, adjusted associations, predictions, and causal claims.
- Report uncertainty, exposure, model version, data cutoff, and support status with predictions.
- Preserve failed, null, and unfavorable experiments.
- Do not select examples, trades, or lineups after seeing favorable outcomes.
- Raw source inputs are immutable; corrections and exclusions require an audit trail.
- Physical or provider differences in possession definitions must be documented and tested, not hidden.

## Model progression

Follow the evidence-gated ladder in the master plan:

```text
trusted possessions and stints
→ descriptive and shrinkage baselines
→ RAPM and hierarchical impact
→ explicit pair interactions
→ low-rank complementarity and embeddings
→ permutation-invariant lineup encoders
→ attention, graph, or hypergraph models only if justified
```

An advanced model proceeds only if it improves a preregistered task, maintains acceptable calibration, is stable across seeds, and justifies its compute and interpretability costs.

## Completion standard

A task is complete only when:

- the scoped capability or document is finished;
- relevant tests pass;
- failure behavior and important edge cases are checked;
- documentation and `docs/PROJECT_STATUS.md` are updated when state changes;
- `docs/CURRENT_TASK.md` contains a concise handoff and names no unverified success;
- the final diff contains no unrelated changes;
- exact verification commands and results are reported.

Commit and push only when requested by the user or explicitly included in the active task. Use focused commit messages and never claim a remote update succeeded without verifying it.

## Agent handoffs

Before switching agents or ending an unfinished task, update `docs/CURRENT_TASK.md` with:

- objective and status;
- acceptance criteria;
- decisions and assumptions;
- files changed;
- verification already run;
- unresolved risks or blockers;
- the exact next action.

Commit and push the handoff when the next agent will work from GitHub. Uncommitted local state and chat history do not transfer through the repository.

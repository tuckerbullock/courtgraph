# CourtGraph

CourtGraph is a research-grade project for learning and evaluating NBA lineup chemistry.

Its central question is:

> Can we estimate how NBA players will fit together before we have observed that exact combination on the court?

The project will separate lineup value into individual talent, player interactions, and context; quantify uncertainty; and evaluate whether the resulting chemistry signal generalizes to unseen lineups, unseen teammate pairs, future seasons, and post-transaction environments.

## Current status

**Stage 0: project foundation. No basketball data or modeling code has been started.**

The repository currently contains the complete operating blueprint for the project:

- [Master research and engineering plan](docs/MASTER_PLAN.md)
- [Current project status](docs/PROJECT_STATUS.md)
- [Active-task and agent handoff](docs/CURRENT_TASK.md)
- [Contributing and research-integrity rules](CONTRIBUTING.md)
- [Shared coding-agent instructions](AGENTS.md)

`CLAUDE.md` imports the shared instructions so Claude Code and other agents operate from the same project standards rather than relying on separate chat histories.

The first implemented capability is a dependency-free project health check that verifies the supported Python runtime and required governing files.

## Development bootstrap

CourtGraph supports Python 3.11 and newer and uses [`uv`](https://docs.astral.sh/uv/)
to manage a locked, reproducible development environment. The local default
interpreter is Python 3.13; GitHub Actions CI runs the checks below on both
3.11 and 3.13, and the first run passed on both.

Install `uv` (see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/)),
then, from the repository root, create the environment from the committed lockfile:

```bash
uv sync
```

Run the checks that CI runs:

```bash
uv run courtgraph doctor
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

For machine-readable health output:

```bash
uv run courtgraph doctor --json
```

The only third-party packages are the developer tools `ruff` and `mypy`
(pinned in the `dev` dependency group). The `courtgraph` package itself has no
runtime dependencies, so the health check and test suite also run without `uv`:

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The package declares a standard `courtgraph` console entry point in `pyproject.toml`.

## Planned research progression

```text
Trusted possession data
        ↓
RAPM and hierarchical baselines
        ↓
Explicit pair interactions
        ↓
Low-rank complementarity and player embeddings
        ↓
Permutation-invariant lineup encoders
        ↓
Unseen-pair and unseen-lineup evaluation
        ↓
Historical transaction backtests
        ↓
Evidence-aware roster construction product
```

Advanced neural, graph, or hypergraph models will be used only after the data pipeline and statistical baselines pass their stated quality gates.

## Research standard

CourtGraph is designed around four questions:

1. What does chemistry mean statistically?
2. How do we know it is not merely talent, opponent quality, lineup luck, or coaching context?
3. Can it predict fit before players share the floor?
4. Does it remain useful under chronological and realistic roster-decision evaluation?

The project will treat chemistry as a model-dependent predictive quantity, not as proven causation.

## Next milestone

The locked `uv` environment and GitHub Actions CI are in place, with the first CI run passing on Python 3.11 and 3.13. The next single task is a concise research contract (`RESEARCH_CONTRACT.md`). The first major research milestone remains a trustworthy two-season possession/stint dataset followed by a leakage-safe ridge RAPM baseline.

## License

No open-source license has been selected yet. Until one is added, all rights are reserved.

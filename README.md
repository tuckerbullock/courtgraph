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
- [Contributing and research-integrity rules](CONTRIBUTING.md)

The first implemented capability is a dependency-free project health check that verifies the supported Python runtime and required governing files.

## Development bootstrap

CourtGraph supports Python 3.11 and newer. From the repository root, run the health check directly from the source checkout:

```bash
PYTHONPATH=src python3 -m courtgraph doctor
```

For machine-readable output:

```bash
PYTHONPATH=src python3 -m courtgraph doctor --json
```

Run the current test suite without installing third-party dependencies:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The package declares a standard `courtgraph` console entry point in `pyproject.toml`. A locked development environment and automated CI are separate upcoming foundation tasks.

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

Complete the reproducible development environment and research contract before beginning data acquisition. The first major research milestone remains a trustworthy two-season possession/stint dataset followed by a leakage-safe ridge RAPM baseline.

## License

No open-source license has been selected yet. Until one is added, all rights are reserved.

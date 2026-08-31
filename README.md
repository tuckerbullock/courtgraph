# CourtGraph

CourtGraph is a research-grade project for learning and evaluating NBA lineup chemistry.

Its central question is:

> Can we estimate how NBA players will fit together before we have observed that exact combination on the court?

The project will separate lineup value into individual talent, player interactions, and context; quantify uncertainty; and evaluate whether the resulting chemistry signal generalizes to unseen lineups, unseen teammate pairs, future seasons, and post-transaction environments.

## Current status

**One complete vertical slice of the product runs end to end on synthetic data.**
`courtgraph demo` generates deterministic synthetic stints with a known talent /
chemistry / context structure, builds leakage-safe chronological, unseen-pair,
and unseen-lineup holdouts, fits an additive ridge baseline and a
permutation-invariant low-rank player-embedding model, and reports how well each
predicts held-out lineup value along with the talent / interaction / context
decomposition and approximate uncertainty. The model, data, split, and artifact
interfaces are shaped so real NBA stints replace the synthetic ones later
without touching the models or the evaluation.

Governing documents:

- [Master research and engineering plan](docs/MASTER_PLAN.md)
- [Current project status](docs/PROJECT_STATUS.md)
- [Active-task and agent handoff](docs/CURRENT_TASK.md)
- [Contributing and research-integrity rules](CONTRIBUTING.md)
- [Shared coding-agent instructions](AGENTS.md)

`CLAUDE.md` imports the shared instructions so Claude Code and other agents operate from the same project standards rather than relying on separate chat histories.

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

## Chemistry commands

All three run entirely on synthetic demonstration data — no network, no NBA
sources. They are deterministic given their seed.

```bash
# generate synthetic stints, fit both models, evaluate all three holdouts,
# and (optionally) write a self-contained HTML report. Takes ~1 minute.
uv run courtgraph demo --report demo_report.html --out-dir courtgraph_demo

# fit a model on a stint file in the versioned format (courtgraph demo writes one)
uv run courtgraph fit --input courtgraph_demo/demo_stints.jsonl \
    --model-out model.json --rank 3

# decompose one lineup's predicted value: talent (T) + interaction (C) + context (K)
uv run courtgraph predict --model model.json \
    --offense 1001,1002,1003,1004,1005 --defense 1006,1007,1008,1009,1010 \
    --context playoff=1
```

`--json` on any of them emits a stable machine-readable result.

## Offline NBA ingestion (`courtgraph ingest`)

`courtgraph ingest` converts a stored, immutable snapshot of `stats.nba.com`
responses into stint records in the same versioned format the chemistry
commands consume. It uses `pbpstats` **in file-only mode** purely as a
possession / lineup reconstruction tool, never contacts the network, never
modifies the snapshot, and **quarantines rather than fabricates** when an input
is ambiguous or incomplete.

```bash
uv run courtgraph ingest --snapshot-dir path/to/snapshot --out-dir ingest_out
uv run courtgraph fit --input ingest_out/stints.jsonl --model-out model.json
```

`--out-dir` receives `stints.jsonl`, `quarantine.jsonl` (dropped possessions /
quarantined games with reasons), and `manifest.json` (input hashes, tool and
policy versions, per-game reconciliation). The snapshot and destination are
validated before anything is written: an `--out-dir` that resolves into the
snapshot, is a non-directory, or holds a symlinked output file is rejected. The
run adds a `.gitignore` covering the generated data (preserving any the caller
already placed there). The snapshot layout (`stats_nba_pbpstats/v1`) is
documented in `src/courtgraph/ingest/snapshot.py`. Real snapshots, working
copies, derived stint rows, and manifests are gitignored (`DATA_SOURCES.md`
§1/§5.1); only hand-authored fixtures under `tests/` are committed.

## Runtime dependency

The one runtime dependency is `numpy` (pinned exactly in `pyproject.toml` and
`uv.lock`); the developer tools are `ruff` and `mypy` (in the `dev` group).
`pbpstats` (pinned) is an `ingest` dependency group, imported lazily only by
`courtgraph ingest`. The `courtgraph doctor` health check imports no
third-party package, so the dependency-free path still works:

```bash
PYTHONPATH=src python3 -m courtgraph doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v   # chemistry tests skip if numpy is absent
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

The synthetic vertical slice runs end to end, and `courtgraph ingest` provides
a fixture-tested offline adapter from stored `stats.nba.com` snapshots into the
same stint format. The next step is to obtain a permitted real snapshot
(`DATA_SOURCES.md` §8), run it through `ingest`, and re-run the same splits,
models, and evaluation on real data. Real-data validation (multi-game
reconciliation, an independent parser, the six-part evidence bar) is still
pending.

## License

No open-source license has been selected yet. Until one is added, all rights are reserved.

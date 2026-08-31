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

`--report PATH` also writes one self-contained HTML report (teams, lineups,
score check, exclusions). For a local demonstration on real playoff games,
`courtgraph snapshot-from-shufinskiy --archive-dir DIR --all-games --out-dir
DIR` builds every game that has the three required inputs in a local
`SRC-SHUFINSKIY` archive. Use repeatable `--game GID` instead for a selected
batch. This is the `DATA_SOURCES.md` §1 local-dev-only fallback; no network is
used. The converter records the consumed
CSV hashes and pinned source commit in `provenance.json`, and gitignores its
whole output. Its score check uses an operator `official_totals.json` when
present, otherwise the data.nba.com game feed (labelled per game) — a second
NBA surface, not an independent provider. One postseason is not evidence of
predictive accuracy.

## Local browser app (`courtgraph app`)

Run a private app on your computer using the existing Python environment:

```bash
uv run courtgraph app
# Open http://127.0.0.1:8765. Ctrl+C stops the server.
```

The **Lineup sandbox** trains a small deterministic model in memory from
synthetic data. Build two five-player lineups, choose a shared opposing five
and context, and compare offensive value, additive talent, interaction
surplus, support, and approximate interaction uncertainty. These are fictional
players, not NBA predictions. Startup fits the model once; it does not download
or save data.

To enable the **Game explorer**, point it at an existing ingest output:

```bash
uv run courtgraph app --ingest-dir path/to/ingest_out \
    --names path/to/snapshot/display_names.json
```

The current local 2025 playoff archive can be rebuilt and opened with:

```bash
uv run courtgraph snapshot-from-shufinskiy \
    --archive-dir data/nba_snapshots/_shufinskiy_src --all-games \
    --out-dir data/nba_snapshots/all_2025_playoffs/snap
uv run courtgraph ingest \
    --snapshot-dir data/nba_snapshots/all_2025_playoffs/snap \
    --out-dir data/nba_snapshots/all_2025_playoffs/out
uv run courtgraph app \
    --ingest-dir data/nba_snapshots/all_2025_playoffs/out \
    --names data/nba_snapshots/all_2025_playoffs/snap/display_names.json
```

That local archive has 84 games: 83 have all three required source inputs, 62
currently pass reconstruction, 21 are quarantined with recorded reasons, and
one lacks a required source feed. The explorer shows all 84 in its coverage and
game controls rather than silently hiding failed or incomplete games.

`--names` is optional. The explorer checks the stint checksum and game exposure
against the manifest before loading. Filter by game, offensive team, player,
and minimum possessions; inspect observed lineup scoring, sample sizes,
source provenance, score reconciliation, and quarantined games. Rates are
**raw offensive points per 100 accepted possessions**, with no garbage-time
weighting, adjustment, or chemistry prediction. The minimum-sample filter
changes displayed lineup rows, not the selection totals. Loaded participants
are not a complete team roster.

The server binds only to `127.0.0.1`; `--port` defaults to 8765 (use another port
if occupied). It serves only its bundled assets and these explicitly loaded
records, rejects foreign origins/hosts, and offers no file browsing or uploads.
No accounts, external assets, telemetry, NBA model training, or betting tools.
Real-data and synthetic calculation paths remain separate. This is a local
research app, not a production web server or evidence of predictive accuracy.

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

The local browser app now connects the 2025 playoff observational explorer and
the separate synthetic lineup sandbox. The next scientific milestone remains
full-season, multi-season permitted data validation and a chronological
real-NBA baseline comparison. The local explorer does not establish real-NBA
predictive accuracy;
independent-parser, minute reconciliation, and the full evidence gates remain
pending. Additional product ideas remain in the backlog, not active work.

## License

No open-source license has been selected yet. Until one is added, all rights are reserved.

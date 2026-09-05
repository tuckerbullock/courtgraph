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
- [Interaction findings — is lineup chemistry predictively real?](docs/INTERACTION_FINDINGS.md)
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
score check, exclusions). `courtgraph snapshot-from-shufinskiy --archive-dir DIR
--all-games --out-dir DIR` builds every game that has all three required
provider inputs in a local `SRC-SHUFINSKIY` archive; the archive may hold a
single postseason (`nbastats_po_2024.csv` …) or several regular seasons
(`nbastats_2020.csv` … `nbastats_2024.csv` — every `nbastats*.csv` /
`datanba*.csv` / `shotdetail*.csv` is read). Use repeatable `--game GID` for a
selected batch. This is the `DATA_SOURCES.md` §1 local-dev-only fallback; no
network is used. The converter records every consumed CSV's hash and the pinned
source commit in `provenance.json`, counts rest days only against a team's
earlier same-season game, and gitignores its whole output. Its score check uses
an operator `official_totals.json` when present, otherwise the data.nba.com game
feed (labelled per game) — a second NBA surface, not an independent provider.

The five regular seasons 2020-21 → 2024-25 have been ingested locally this way
(5,158 / 5,998 games → 266,518 stints); the data stays gitignored. A
within-NBA score check is not evidence of predictive accuracy.

## Interaction research

`courtgraph baselines` compares the model-ladder rungs (2 additive / 3
hierarchical EB / 4 explicit pairs) on the leakage-safe holdouts;
`courtgraph transport` trains on one stint file and evaluates on a disjoint
one (regular season → held-out playoffs); `courtgraph player-features` derives
per-(player, season) role/skill profiles from a snapshot; `courtgraph roles`,
`courtgraph mechanistic` and `courtgraph redundancy` fit role-conditioned
interaction models against those profiles (on points/100, on a shot-based
outcome, and on skill-concentration features respectively); and
`courtgraph confirm` re-runs them with a wider holdout, a K sweep, and
bootstrap confidence intervals on the improvement.

What these have and have not shown on the 266k real regular-season stints:
identity-keyed teammate chemistry is **not supported**, and of three
role-conditioned positives only one survives a properly-powered test — lineups
predictably shift their **three-point-attempt share** (not their scoring) away
from the sum of the individuals' tendencies. The standing record is
[`docs/INTERACTION_FINDINGS.md`](docs/INTERACTION_FINDINGS.md); a standalone
capstone write-up for a reader who hasn't followed it day to day is
[`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md).

## Real-lineup prediction (`fit-rung3` / `predict-rung3`)

Since chemistry isn't supported, the one thing worth predicting *from* is
rung 3 itself — the calibrated additive model. `fit-rung3` persists a fitted
rung-3 model (its own artifact format, distinct from the synthetic
`ChemistryModel` artifact `fit`/`predict` use); `predict-rung3` scores an
arbitrary 5-vs-5 lineup from real observed players — additive talent +
context and a Gaussian predictive interval, with **no interaction/chemistry
field anywhere in the result**:

```bash
uv run courtgraph fit-rung3 --input ingest_out/stints.jsonl --model-out rung3.json
uv run courtgraph predict-rung3 --model rung3.json \
    --offense 1001,1002,1003,1004,1005 --defense 1006,1007,1008,1009,1010
```

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

A local archive can be rebuilt and opened with (here, the five-season
regular-season archive):

```bash
uv run courtgraph snapshot-from-shufinskiy \
    --archive-dir data/nba_snapshots/_shufinskiy_rs_2020_2024 --all-games \
    --out-dir data/nba_snapshots/rs_2020_2024/snap
uv run courtgraph ingest \
    --snapshot-dir data/nba_snapshots/rs_2020_2024/snap \
    --out-dir data/nba_snapshots/rs_2020_2024/out
uv run courtgraph app \
    --ingest-dir data/nba_snapshots/rs_2020_2024/out \
    --names data/nba_snapshots/rs_2020_2024/snap/display_names.json
```

That archive holds 6,000 regular-season games (2020-21 → 2024-25); 5,998 have
all three provider inputs, 5,158 pass reconstruction (266,518 stints), 840 are
quarantined with recorded reasons, and two lack the data.nba.com feed. The
explorer shows every game in its coverage and game controls rather than silently
hiding failed or incomplete games. (The 2024-25 playoffs archive
`_shufinskiy_src` still works the same way and is kept separate.)

`--names` is optional. The explorer checks the stint checksum and game exposure
against the manifest before loading. Filter by game, offensive team, player,
and minimum possessions; inspect observed lineup scoring, sample sizes,
source provenance, score reconciliation, and quarantined games. Rates are
**raw offensive points per 100 accepted possessions**, with no garbage-time
weighting, adjustment, or chemistry prediction. The minimum-sample filter
changes displayed lineup rows, not the selection totals. Loaded participants
are not a complete team roster.

The explorer also has a **Predict a real lineup** panel: pick an offense
team and a defense team, pick five players each from the players observed on
that team in the loaded window (explicitly labeled as inferred exposure, not
an official roster), and get the same rung-3 additive prediction
`predict-rung3` produces — fit once, lazily, on first use, and cached for the
life of the server process. No chemistry/interaction claim anywhere in the
result.

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

Five regular NBA seasons (2020-21 → 2024-25) have been ingested into 266,518
real stints. The next scientific milestone is to run the leakage-safe splits,
the additive ridge RAPM baseline, and the low-rank chemistry model on that data
and compare held-out prediction to the contract's rung-2/3 references. Nothing
about the observational data or the within-NBA score check establishes
predictive accuracy; the independent-parser gate, minute reconciliation, and the
full evidence bar remain pending. Additional product ideas remain in the
backlog, not active work.

## License

No open-source license has been selected yet. Until one is added, all rights are reserved.

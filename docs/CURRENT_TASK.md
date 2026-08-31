# Current Task

Last updated: 2026-08-30

## State

Done — **synthetic lineup-chemistry vertical slice**, revised twice after Codex
review and **approved by Codex**. One complete path through the product runs end
to end on synthetic data. Branch `task/chemistry-mvp` off `origin/main`
(`589b2e6`); committed, pushed, and opened as a pull request to `main`.

Round 3 (this pass) fixed only the unseen-pair budget: `max_test_fraction` is
now a hard upper bound for the first selected pair too (Codex reproduced
`max_test_fraction=0` yielding 112 test stints), and an impossible budget now
raises `UnsatisfiableSplitError` instead of returning an unusable split. The
other four round-2 fixes are unchanged.

Prior work merged to `main`: dev environment + CI (#1), research contract (#2),
data-source registry (#3), data-access & schema pilot (#4). The uncommitted
`task/schema-contract` worktree is untouched.

## Codex-review revisions (round 2)

1. **Chronological order no longer trusts `game_id`.** Stint schema bumped to
   **v2** with a required ISO `game_date`; `chronological_key` orders by
   `game_date` (then `game_id` only as a within-day tiebreak). v1 records are
   rejected on load. Tested with deliberately reversed game IDs.
2. **Unseen-pair split is "structural", not "strong/trade-like".** The claim is
   removed. A pair is admitted only if **each player keeps
   `min_solo_train_stints` offensive training stints without the partner**;
   `verify_split` fails if a held player is absent from training offense. The
   `max_test_fraction` budget is a hard upper bound on the held-out size for
   **every** candidate, the first included; when no pair fits it,
   `make_unseen_pair_split` raises `UnsatisfiableSplitError` rather than return
   an unusable split. Regression tests added (oversized first candidate, and the
   zero-budget failure).
3. **Unseen offensive player → additive-only (C = 0) everywhere.** A single
   helper (`_offense_fully_known`) zeroes the interaction term in
   `predict_total`, `interaction_component`, `interaction_samples`,
   `decompose`, and `interaction_interval`, so the point value, decomposition,
   interval, and serialized CLI output all agree. Tested across all surfaces.
4. **`--bootstrap N` is wired through.** `run_demo` sets
   `ChemistryConfig.n_bootstrap` from `--bootstrap`; `--bootstrap 0` produces
   exactly zero ensemble members (verified in the saved artifact). Exact-size
   tests for 0 and positive N.
5. **Group uncertainty is computed at the group level.** Each bootstrap
   member's possession-weighted group prediction is formed first, then the
   group mean / SD / `P(C>0)` come from those group-level samples (not from
   averaging row-level statistics).

## Objective

Build one runnable vertical slice: a versioned stint format; a deterministic
synthetic stint generator with known talent and latent compatibility;
leakage-safe chronological / unseen-pair / unseen-lineup holdouts; an additive
ridge baseline; a permutation-invariant low-rank player-embedding model that
predicts unseen combinations and separates talent / interaction surplus /
context / total; evaluation (RMSE/MAE vs the baseline for all three holdouts,
support/exposure, approximate uncertainty); working `courtgraph demo`, `fit`,
`predict` commands; a self-contained HTML report from `demo --report PATH`;
deterministic tests. No web server, no live NBA sources.

## Outcome

- **`src/courtgraph/chemistry/`** (8 modules, `numpy==2.3.5` pinned + `uv.lock`
  updated):
  - `stints.py` — versioned (`SCHEMA_VERSION = 2`) `Stint` / `StintTable`:
    offensive five, defensive five, `offensive_possessions`, `points_scored`, an
    explicit ISO `game_date`, season/time, and context fields; validation;
    `.jsonl` / `.json` IO that round-trips exactly. No NumPy dependency (keeps
    `courtgraph doctor` clean).
  - `synthetic.py` — deterministic generator. Known `alpha`, per-player
    offensive/defensive talent, and a rank-`d` provision/need structure giving
    `C_true(L_o) = sum_{i<j} (p_i·n_j + p_j·n_i)`. `with_no_interaction()` gives
    a matched no-signal control. Returns the table **and** its `GroundTruth`.
  - `splits.py` — `make_chronological_split` (orders by `game_date`, cuts on a
    game boundary), `make_unseen_pair_split` (**structurally** unseen pairs:
    every training co-play removed, each player kept individually observed,
    every candidate held to a hard `max_test_fraction` upper bound, raising
    `UnsatisfiableSplitError` if none fits), `make_unseen_lineup_split` (every
    exact-set training stint removed). `verify_split` re-derives the forbidden
    overlaps — including that each held player is still in training offense —
    and returns violations. The leakage gate.
  - `baseline.py` — `AdditiveRidge`: weighted ridge RAPM, separate
    offensive/defensive talent (both signed larger-is-better), ridge strength by
    game-blocked CV. The "sum of the parts" model.
  - `chemistry_model.py` — `ChemistryModel`: additive skip path +
    `LowRankInteraction` (provision/need embeddings fit by **alternating ridge
    least squares** on the **cross-fitted** additive residual, L2 toward zero,
    zero-sum centered over a reference lineup distribution). `C(L_o)` is a sum
    over the offensive set → permutation invariant by construction and defined
    for never-co-observed pairs/lineups (but forced to **C = 0** whenever any
    offensive player is unseen). `decompose()` returns talent / interaction /
    context / total; a block-bootstrap ensemble (size = `--bootstrap`, may be 0)
    gives an approximate interaction interval and P(C>0). Versioned artifact
    (`ARTIFACT_SCHEMA_VERSION = 2`).
  - `evaluate.py` — additive vs full RMSE/MAE, micro (possession-weighted) and
    macro (per held-out group), against realized outcomes and — for synthetic
    data — the known truth; approximate game-block-bootstrap interval on the
    RMSE difference; **group-level** interaction mean / SD / P(C>0) (each
    bootstrap member's possession-weighted group prediction first, then the
    across-member statistics); per-group exposure and novelty class;
    parameter-recovery correlations.
  - `report.py` — self-contained HTML (inline CSS + inline SVG, no JS, no
    external assets), banner-labeled synthetic. `artifact.py`, `pipeline.py`.
- **CLI** (`src/courtgraph/cli.py`): `courtgraph demo [--report PATH]
  [--out-dir DIR] [--seed N] [--json]`, `courtgraph fit --input ... --model-out
  ... [--rank N] [--evaluate]`, `courtgraph predict --model ... --offense ...
  --defense ... [--context k=v]`. `doctor` still imports no third-party package.
- **Tests** (`tests/test_chemistry_*.py`, 71 cases): decomposition identity
  (T+C+K == V exactly), permutation invariance over offense and defense,
  serialization round-trip (fit → save → load → identical predictions),
  leakage-safe splits **and** that `verify_split` catches an injected leaked
  game / pair co-play / exact lineup / held player missing from training,
  chronological order with misleading `game_id`s, unseen-player additive-only
  behaviour across every surface, exact `--bootstrap` ensemble sizes, CLI exit
  codes and output, additive-talent recovery, and recovery of a real
  interaction signal beyond the additive baseline with a matched no-signal
  control that shows no spurious improvement.

### What the demo shows (default synthetic dataset, deterministic)

17,424 stints, 120 players, 3 seasons. Macro held-out lineup-value RMSE
(vs the known truth), additive → full:

| holdout | additive | full | improvement |
|---|---|---|---|
| chronological | 3.32 | 3.32 | ~0% |
| unseen_pair | ~2.4 | ~1.7 | ~29% |
| unseen_lineup | 3.55 | 2.60 | ~27% |

Talent recovery corr 0.88 (off) / 0.96 (def). Test-set corr(predicted C, true C)
~0.28 (unseen_pair), ~0.29 (unseen_lineup), ~0 (chronological). The no-signal
control shows ~0% improvement and a near-zero interaction pathway. (The
structural unseen-pair split keeps each held player individually observed, so
the additive baseline is already accurate there — the full model still improves
on it.)

## Honest limitations (does it jump ahead / skip anything?)

- **Chronological holdout shows no chemistry benefit** at demo scale — reported
  as-is. Chemistry is a small residual; the improvement lives in the group-level
  and truth-referenced metrics, and the realized-outcome micro RMSE barely
  moves. This matches `RESEARCH_CONTRACT.md` §14 but means the "headline" number
  in the report is the truth-referenced macro one, only available because the
  data is synthetic.
- **Individual pair-surplus recovery is weak** (corr ~0.16 over supported
  pairs); the model is reliable at the lineup-aggregate level, not per pair, at
  this data size.
- **Unseen-pair is not the contract's "strong" variant** — it is structural
  (co-play removed) but both players stay individually observed and their
  earlier partnerships elsewhere are not excluded. The strong/trade variant
  is a later task.
- **Uncertainty is explicitly approximate**: a block-bootstrap ensemble of the
  interaction pathway with the additive fit and selected L2 held fixed. Not a
  calibrated Bayesian posterior (contract §16 wants calibrated intervals — a
  later rung).
- **Model ladder**: only rungs 0/2 (additive) and ~5/6 (low-rank embeddings)
  are built. Rungs 1, 3, 4, 7, calibration gates, and the T4 transaction
  backtest are not — consistent with a *slice*, not the full cycle.
- **Two-stage residual fit**, not a jointly identified hierarchical model; the
  additive skip path can absorb some mean-field chemistry, so reported `C` is a
  conservative surplus.

## Verification (final run before commit — all pass)

```
uv lock --locked ; uv sync --locked                # Resolved / checked 9 packages
uv run courtgraph doctor                            # CourtGraph 0.2.0: healthy
uv run python -m unittest discover -s tests -v      # Ran 71 tests ... OK (~22s)
uv run python -m compileall -q src tests            # OK
uv run ruff check .                                 # All checks passed!
uv run ruff format --check .                        # 37 files already formatted
uv run mypy                                         # Success: no issues in 23 source files
PYTHONPATH=src python3 -m courtgraph doctor         # healthy (dependency-free path)
PYTHONPATH=src python3 -m unittest discover -s tests # Ran 71 tests ... OK
uv run courtgraph demo --bootstrap 0 --out-dir <dir> --report <dir>/report.html
```

`demo --bootstrap 0`: 17,424 synthetic stints; macro RMSE additive→full
chrono 3.32→3.32, unseen_pair 2.42→1.71 (29%), unseen_lineup 3.55→2.60 (27%);
**0 leakage violations** on all three holdouts; self-contained HTML report
written; saved artifact has **0 interaction-ensemble members and 0 ensemble
references** (`config.n_bootstrap = 0`).

## Files changed (branch `task/chemistry-mvp`)

- `src/courtgraph/chemistry/` — new package (8 modules + `__init__`).
- `src/courtgraph/cli.py` — `demo` / `fit` / `predict` subcommands (lazy imports).
- `src/courtgraph/__init__.py` — version 0.1.0 → 0.2.0.
- `tests/test_chemistry_*.py`, `tests/_chemistry_support.py` — new.
- `tests/test_health.py` — one assertion loosened for the version bump.
- `pyproject.toml`, `uv.lock` — `numpy==2.3.5` runtime dependency.
- `README.md`, `docs/PROJECT_STATUS.md`, `docs/CURRENT_TASK.md` — updated to
  describe code that runs.

No web server. No change to `DATA_SOURCES.md`, `RESEARCH_CONTRACT.md`,
`docs/MASTER_PLAN.md`, `.github/`, `pilot/`.

## Next action

Codex approved the slice; `task/chemistry-mvp` is committed, pushed, and has an
open PR to `main`. Merge the PR. The next single task (do not start until
activated): swap the synthetic generator for a real NBA stint source per
`DATA_SOURCES.md` §8, fed through the same stint format, and re-run the
evaluation on real data.

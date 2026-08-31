# Current Task

Last updated: 2026-08-31

## State

Done — **offline NBA snapshot -> stint importer** (`courtgraph ingest`),
**approved by Codex** after three review rounds. Branch `task/nba-stint-import`
off `origin/main` (`e56dc8c`, PR #5 merged); committed, pushed, and opened as a
pull request to `main`.

Prior work merged to `main`: dev env + CI (#1), research contract (#2),
data-source registry (#3), data-access & schema pilot (#4), synthetic
chemistry vertical slice (#5). The parked `task/schema-contract` worktree is
untouched.

**Fixture-only.** No real NBA snapshot has been ingested — none was provided
and none was fetched. The adapter and all 38 new tests run on hand-authored,
`pbpstats`-parsed fixtures. Real-data validation (a permitted snapshot through
`courtgraph ingest`, multi-game reconciliation, an independent parser) is the
next task and is **pending**.

## Codex-review revisions (round 1)

1. **Snapshot deletion.** `run_ingest` now resolves `--snapshot-dir` /
   `--out-dir` (following symlinks) and raises `SnapshotError` if either
   contains the other; the `pbpstats` working copy is a private
   `tempfile.mkdtemp()` directory, discarded after the run. `stage_working_copy`
   never deletes anything.
2. **Output gitignore.** `run_ingest` writes `<out-dir>/.gitignore` (`*`) so any
   output directory is self-protecting; the working copy is never near a repo.
   Verified with `git check-ignore`.
3. **Returning-player split.** `is_split_lineup` now tracks every ten-player set
   from the first live event onward — a player leaving, another playing live,
   and the first returning before the shot is excluded.
4. **Excluded-gap continuity.** Each possession carries a `sequence_index`;
   `possessions_to_stints` breaks a stint run whenever the accepted possessions
   are not consecutive, so spells on either side of an excluded possession are
   never merged even when their lineups match.
5. **Incomplete reconciliation.** `missing_context` requires an official final
   score for **both** teams; a one-sided total quarantines the game as
   `missing_context` even with `allow_score_mismatch`.
6. **Override provenance.** All `overrides/*.json` are hashed; each game's
   `input_files` includes them and the manifest records a
   `corrections.correction_set_id`.

## Codex-review revisions (round 2)

Output-safety hardening only. `run_ingest` now validates the snapshot
(`load_snapshot`) **and** the destination (`_validate_destination`) before
creating a directory or writing a byte — an `--out-dir` that resolves into the
snapshot, exists as a non-directory, or holds a symlinked generated file is
rejected with nothing created or mutated. An existing `<out-dir>/.gitignore` is
**preserved** (only `/stints.jsonl` / `/quarantine.jsonl` / `/manifest.json`
are appended when not already covered); it is written as `*` only when absent.
Every generated-file write goes through `_writable()`, which refuses to follow
a symlink. Regressions: existing-`.gitignore` preservation, invalid-input
non-mutation, rejected-nested-output non-mutation, and an output symlink
targeting snapshot metadata (three of these run on the dependency-free path).

## Codex-review revisions (round 3)

`_ensure_output_gitignore` only. It no longer infers protection from a bare
`*` or an exact rule (either can be undone by a later `!` negation). Instead it
keeps a marker-delimited block (`# BEGIN courtgraph ingest …`) of anchored
exclusions (`/stints.jsonl` / `/quarantine.jsonl` / `/manifest.json`) and
always (re)writes it at the **end** of the file, so Git's last-match-wins rule
makes the generated data files ignored regardless of earlier negations. The
block is rewritten in place, so repeated runs do not grow the file. Regressions
(via `git check-ignore`, dependency-free): `*` then negations, exact rules then
negations — all three data files stay ignored, unrelated caller rules survive —
and a re-run leaves the file byte-identical.

## Objective

Add an executable, fixture-tested adapter that converts a stored, immutable
snapshot of `stats.nba.com` responses into validated
`courtgraph.chemistry.stints` records, using `pbpstats` in **file-only mode**
as a possession/lineup reconstruction tool. No live sources, no bulk
acquisition, no new models.

## Outcome

- **`src/courtgraph/ingest/`** (7 modules) + `pbpstats==1.3.11` pinned in a new
  `ingest` dependency group (included by `dev`, so CI installs it; lazily
  imported so `courtgraph doctor` stays third-party-free):
  - `snapshot.py` — the one documented layout `stats_nba_pbpstats/v1`: raw
    `playbyplayv2` + `shotchartdetail` files consumed by `pbpstats` unchanged,
    plus a `courtgraph_snapshot.json` carrying the non-play-by-play metadata
    (date, teams, rest days, official period/final scores). Structural
    validation, per-file SHA-256, and `stage_working_copy()` — `pbpstats`
    parses a throwaway copy, never the snapshot.
  - `policy.py` — `IngestPolicy` (`policy_version = "cg-ingest-policy/1"`):
    every research choice named with a documented default — exact-final-score
    reconciliation, the deterministic garbage-time rule, split-lineup handling,
    the ambiguous-scoring bound.
  - `possessions.py` — the sole `pbpstats` boundary. `reconstruct_game()`
    returns plain typed `PossessionView`s. A hard `offline_guard()` turns any
    socket connect / DNS attempt (e.g. `pbpstats`'s period-starter fallback)
    into `IngestNetworkAttempt` -> the game is quarantined; the adapter never
    makes a request.
  - `validate.py` — CourtGraph's independent checks: five per side,
    offense/defense in the game's two teams, possession alternation
    re-derived, score reconciliation against the *independent* box-score total
    (possession points + technical free throws), and per-possession exclusion
    for empty / split-lineup / ambiguous-scoring possessions. Outcomes are
    "accept", "exclude possession", or "quarantine game" — never a fabricated
    value.
  - `stints.py` — maximal runs of consecutive validated possessions with the
    same ten on the floor become one-sided `Stint` rows. Non-contiguous
    re-appearances of the same five are **never merged** (fresh run index ->
    fresh `stint_id`).
  - `manifest.py` / `pipeline.py` — `run_ingest()` writes `stints.jsonl`,
    `quarantine.jsonl`, and a full `manifest.json` audit trail (input hashes,
    `pbpstats` version, `policy_version`, run timestamp, per-game source-event
    counts, reconciliation, every exclusion).
- **CLI**: `courtgraph ingest --snapshot-dir PATH --out-dir DIR
  [--allow-score-mismatch] [--json]`. Exit 0 when >=1 stint is emitted, 1 when
  none, 2 on an invalid snapshot.
- **`.gitignore`**: real snapshots, working copies, `stints.jsonl`,
  `quarantine.jsonl`, `manifest.json` (`DATA_SOURCES.md` §1/§5.1). Fixtures
  under `tests/` are committed.
- **Tests** (`tests/test_nba_ingest.py`, `tests/test_nba_snapshot.py`,
  `tests/_nba_fixtures.py`): a hand-authored builder emits **real
  `playbyplayv2`-shaped JSON** that real `pbpstats` parses (asserted). Covers
  ordinary play, offensive rebounds (same possession), free throws + a
  technical, substitutions (new stint), overtime, non-contiguous same lineup
  (not merged), split-lineup quarantine, score-reconciliation failure (+
  `--allow-score-mismatch`), missing context, truncated/ambiguous pbp,
  malformed snapshot JSON, missing files, snapshot immutability across runs,
  proof that a happy-path ingest opens no socket, that a starters-incomplete
  game is quarantined `network_required`, the audit manifest, and fixture
  ingestion -> `courtgraph fit` -> `courtgraph predict`.

## Honest limitations / not done

- **No real snapshot was ingested** — none was provided and none was
  downloaded (per instructions). Real-data validation is **pending**. To run
  it, supply a `stats_nba_pbpstats/v1` snapshot directory: per game,
  `pbp/stats_<game_id>.json` (`playbyplayv2`),
  `game_details/stats_home_shots_<game_id>.json` and
  `stats_away_shots_<game_id>.json` (`shotchartdetail`), optionally
  `game_details/stats_boxscore_<game_id>.json`, optional `overrides/*.json`,
  and `courtgraph_snapshot.json` with each game's `game_date`, `season`,
  `season_type`, `home_team_id`, `away_team_id`, `days_rest` per team, and
  `reconciliation.final_score` / `.period_scores`.
- This is **not** the contract's independent-parser gate or the multi-game
  reconciliation gate — `pbpstats` is the only reconstruction engine here and
  CourtGraph's checks are within-lineage.
- **Minute reconciliation** (master plan §7.7) is not implemented; only
  final-score reconciliation gates a game. Period-score deltas are recorded
  but informational.
- Split-lineup possessions are **excluded**, not down-weighted (master plan
  §7.4 leaves down-weighting for later).
- The fixtures are shape-correct, not basketball-realistic: predicted values
  from a model fit on them are meaningless (the demo proves plumbing only).

## Verification (final run before commit — all pass)

```
uv lock --locked                                    # Resolved 15 packages (no drift)
uv sync --locked
uv run courtgraph doctor                            # CourtGraph 0.2.0: healthy
uv run python -m unittest discover -s tests -v      # Ran 109 tests ... OK   (38 new)
uv run python -m compileall -q src tests            # OK
uv run ruff check .                                 # All checks passed!
uv run ruff format --check .                        # 48 files already formatted
uv run mypy                                         # Success: no issues in 34 source files
PYTHONPATH=src python3 -m courtgraph doctor         # healthy (dependency-free path)
PYTHONPATH=src python3 -m unittest discover -s tests # Ran 109 tests ... OK (23 skipped: no pbpstats)
```

- `courtgraph demo --bootstrap 0`: synthetic slice unchanged (unseen_pair
  2.42->1.71, unseen_lineup 3.55->2.60, chrono 3.32->3.32); saved artifact has
  0 interaction-ensemble members / 0 references; self-contained HTML report.
- Fixture `courtgraph ingest` -> `fit` -> `predict`: 80 stints from 4
  hand-authored games -> model fit -> decomposition `T + C + K = V`
  (`222.93 - 0.40 + 29.77 = 252.31`). Fixtures are shape-correct, not
  basketball-realistic, so these numbers only prove the plumbing.

## Files changed (branch `task/nba-stint-import`)

- `src/courtgraph/ingest/` — new package (7 modules + `__init__`).
- `src/courtgraph/cli.py` — `ingest` subcommand (lazy imports).
- `pyproject.toml`, `uv.lock` — `ingest` group (`pbpstats==1.3.11`), mypy
  override for the untyped `pbpstats`.
- `.gitignore` — ingest inputs/outputs.
- `tests/_nba_fixtures.py`, `tests/test_nba_ingest.py`,
  `tests/test_nba_snapshot.py` — new.
- `README.md`, `docs/PROJECT_STATUS.md`, `docs/CURRENT_TASK.md` — updated.

No change to `DATA_SOURCES.md`, `RESEARCH_CONTRACT.md`, `docs/MASTER_PLAN.md`,
`.github/`, `pilot/`, or the synthetic chemistry code.

## Next action

Codex approved the importer; `task/nba-stint-import` is committed, pushed, and
has an open PR to `main`. Merge the PR. The next single task (do not start
until activated): obtain a permitted real `stats_nba_pbpstats/v1` snapshot, run
`courtgraph ingest`, and re-run the splits / models / evaluation on real data —
including the reconciliation and independent-parser gates this fixture-only
adapter does not cover.

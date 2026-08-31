# Current Task

Last updated: 2026-08-31

## State

Done — **first real-game ingestion demonstration**. Branch
`task/real-game-ingest` off `origin/main` (`92afc73`, PR #6 merged). Committed
and pushed; PR open. Coding-first workflow: Codex approval required before
merge, not before each commit.

Merged to `main`: dev env + CI (#1), research contract (#2), data-source
registry (#3), data-access & schema pilot (#4), synthetic chemistry slice (#5),
offline snapshot -> stint importer (#6). The parked `task/schema-contract`
worktree is untouched.

## What this adds

The importer now runs on **real NBA play-by-play**, with a readable report.

- **`src/courtgraph/ingest/shufinskiy.py`** (`build_snapshot()`) converts named
  games from a local **SRC-SHUFINSKIY** playoffs archive (`DATA_SOURCES.md` §1
  local-dev-only fallback) into a `stats_nba_pbpstats/v1` snapshot:
  - `playbyplayv2` rows emitted **in the archive's order** (`EVENTNUM` is not
    sorted — it is not always monotonic and pbpstats fixes ordering itself);
  - game date and rest days from the **validated `GAME_DATE`** in `shotdetail`
    (never the UTC event wall-clock, which can roll past midnight);
  - reconciliation target = an operator `official_totals.json` when present,
    otherwise the data.nba.com game feed's own running score, labelled per game
    in `reconciliation.source`;
  - a game whose date / rest / totals can't be determined is written without
    them → importer quarantines it (nothing fabricated);
  - `provenance.json` (consumed-CSV sha256s, pinned `shufinskiy` commit,
    `converter_version`); `display_names.json` (ids → names, report only);
    a `.gitignore` (`*`). Destination is checked for overlap with the archive
    and for symlinked generated files before any write. No network.
- **`src/courtgraph/ingest/report.py`** — one self-contained HTML report: run
  totals, source-provenance table, per-game matchup with real names, the score
  check **against the recorded source stated per game**, busiest stint lineups,
  garbage-time-weighted stints, and every exclusion. `write_report` refuses a
  symlinked path or one inside the snapshot / on an ingest output file.
- **Manifest** now carries `source_provenance` and, per game,
  `reconciliation.official_score_source`.
- **CLI**: `courtgraph snapshot-from-shufinskiy`; `courtgraph ingest --report`.
- `ensure_gitignore_block` **merges and de-duplicates** the patterns already in
  its managed block with the new ones, so writing a second differently named
  report into one directory leaves the first still ignored; caller rules and
  repeated-run stability are preserved.
- **Tests (+24)**: CSV order preserved with non-monotonic `EVENTNUM`; `GAME_DATE`
  used across a UTC-midnight crossing (date + rest); provenance recorded;
  `official_totals.json` preferred; destination overlap / file- and
  directory-symlink rejected (`pbp/`, `game_details/` validated before any
  write); existing snapshot `.gitignore` preserved; report auto-gitignored
  (existing rules kept); two differently named reports in one directory both
  stay `git check-ignore`d; report provenance + recorded score source; banner
  source label derived from provenance (synthetic / unlabelled inputs are not
  called SRC-SHUFINSKIY); unsafe report path rejection.

## The demonstration (local only, not committed)

Game 0042400102 passed strict reconciliation with CLE 121 [25,43,25,28],
MIA 112 [24,27,29,32], and produced 50 stints.

2024-25 Round 1, Cavaliers–Heat, from the local archive, strict checks:

| game | date | recorded final | outcome |
|---|---|---|---|
| `0042400102` (Game 2) | 2025-04-23 | CLE 121 – MIA 112 (data.nba.com feed) | **accepted** — 189 possessions reconstructed, 175 accepted, **50 stints**; reconstructed score matches the feed exactly, all periods; 3 empty + 11 split-lineup possessions excluded |
| `0042400103` (Game 3) | 2025-04-26 | CLE 124 – MIA 87 (feed) | **accepted** — 183 reconstructed, 165 accepted, **47 stints**; score matches; 3 empty + 15 split-lineup excluded |
| `0042400104` (Game 4) | 2025-04-28 | — | **quarantined** — `network_required`: pbpstats cannot infer all five period starters for this game from PBP alone and its only fallback is a stats.nba.com box-score request, refused by the offline guard. (Preserved as-is; recovering it is out of scope.) |

Real lineups appear by name (e.g. *Darius Garland, Donovan Mitchell, Evan Mobley,
Jarrett Allen, Max Strus*). Fixing the `EVENTNUM` sort (finding 1) is what let
Games 2 and 3 through — the previous run's "Game 4 accepted" was an artifact of
that bug. Two accepted games is still **not** proof of predictive accuracy,
calibration, or data quality.

**Official Game 2 totals:** the four-findings note asked to use "the previously
supplied official Game 2 totals". None were present in this task's input, so the
demo uses the data.nba.com feed (source labelled per game) and the
`official_totals.json` mechanism is ready for when they are supplied. Flagged in
the milestone reply.

## Data permission

`SRC-SHUFINSKIY` — acquired 2026-08-30 for `task/schema-contract`, pinned +
checksummed, **no live endpoint contacted**. `DATA_SOURCES.md` classifies it
*fallback — local dev only*: fine for a local, non-redistributed demonstration;
NBA terms still apply. All NBA data, snapshots, ingest outputs, and reports are
gitignored (`data/nba_snapshots/`). No data was downloaded. A truly independent
reconciliation lineage is still unavailable (`DATA_SOURCES.md` §5.2).

## Limitations

- One archive, one series; nothing validated at season scale.
- Unless official totals are supplied, the score check is within-NBA (stats vs
  data.nba.com feed), not independent.
- Box-score minutes / lineup minutes are not reconciled.
- Games needing `pbpstats` overrides are quarantined, not patched.
- No model is fit or evaluated.

## Verification (all pass)

```
uv lock --locked                                    # 15 packages, no drift
uv run courtgraph doctor                             # healthy
uv run python -m unittest discover -s tests -v       # 133 tests OK (24 new this milestone)
uv run python -m compileall -q src tests             # OK
uv run ruff check . ; uv run ruff format --check .   # clean
uv run mypy                                          # 40 source files, clean
PYTHONPATH=src python3 -m courtgraph doctor / unittest  # 133 OK (28 skipped: no pbpstats)
uv run courtgraph demo --bootstrap 0                 # synthetic slice unchanged; 0 ensemble members
courtgraph snapshot-from-shufinskiy ... ; courtgraph ingest ... --report report.html
```

## Next action

Codex review before merge. If the user supplies `official_totals.json`, re-run
the batch so the score check uses the NBA box score. Recovering Game 4 (a
`pbpstats` period-starter override from a box score) is a later task.

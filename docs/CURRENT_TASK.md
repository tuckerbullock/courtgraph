# Current Task

Last updated: 2026-08-31

## State

Done — **five regular NBA seasons ingested** (2020-21 → 2024-25) on
`task/regular-season-ingest`, off `origin/main` (`0a2e30d`). Committed and
pushed, PR open. The importer was generalized to multi-season archives; the
data was acquired locally (gitignored), converted, and run through
`courtgraph ingest` in one pass. No live NBA endpoint was contacted.

This is the first time CourtGraph has held real **regular-season** data — the
window the modeling is designed around. The 2024-25 playoffs archive is
untouched and stays held out for the transport test (`DATA_SOURCES.md` §6).

## Delivered

### Importer generalization (`src/courtgraph/ingest/shufinskiy.py`)

- Was hardcoded to `nbastats_po_2024.csv` / `datanba_po_2024.csv` /
  `shotdetail_po_2024.csv`. Now globs `nbastats*.csv` / `datanba*.csv` /
  `shotdetail*.csv` in the archive dir and concatenates every match, so one
  archive holds several seasons. A provider contributing zero files is a clear
  `ShufinskiyArchiveError`.
- `provenance.json` hashes every consumed CSV; `archive_coverage` keys switch
  from filenames to provider labels (`nbastats` / `datanba` / `shotdetail`).
  `CONVERTER_VERSION` → `cg-shufinskiy/3`.
- **Rest days are counted only against a team's earlier game in the same
  season.** Pooling seasons previously made every season opener look like a game
  with ~150 days' rest carried over from the prior postseason; such a game now
  has `days_rest` omitted and is quarantined (`missing_context`).
- CLI help + `snapshot-from-shufinskiy` human output generalized (no per-game
  line spam past 50 games).
- Tests: `ShufinskiyMultiSeasonTests` (glob discovery + hashing, same-season
  rest bound, cross-season leak guard, missing-provider error). Fixture gains
  `write_multi_season_archive` and a `suffix` arg on `write_raw_archive`.

### Data acquisition (local, gitignored — nothing under `data/` is committed)

- `data/nba_snapshots/_shufinskiy_rs_2020_2024/` — 15 `*.tar.xz`
  (`{nbastats,datanba,shotdetail}_{2020..2024}`, ~102 MB) from
  `shufinskiy/nba_data` @ pinned commit
  `e829d4678be1e075f99e5d41a1c5f97089be446b`, sha256-recorded in `SOURCE.md`
  and `_archives/SHA256SUMS.txt`, extracted to ~1.3 GB of CSV (~6.7 M rows).
- Kept in a **separate directory** from the playoffs archive so
  `--all-games` never pools regular season with playoffs.

## The run (local only, not committed)

```
snapshot-from-shufinskiy --all-games  → 6,000 archive games, 5,998 complete
                                         (2 missing the datanba feed)
ingest                                → 5,158 accepted · 840 quarantined
                                         266,518 stints · 941,897 accepted
                                         possessions · 91,420 excluded
```

- **Coverage: 5,158 / 5,998 games (86%).** 30 teams, 985 players,
  2020-12-25 → 2025-04-13.
- Per season (accepted / total): 2020-21 929/1080 · 2021-22 1039/1230 ·
  2022-23 1069/1230 · 2023-24 1088/1228 · 2024-25 1033/1230.
- Quarantine reasons: **503 `network_required`** (pbpstats wants a
  stats.nba.com box-score call for period starters; the offline guard blocks
  it — the dominant loss and the main lever for a follow-up), 170
  `TeamHasBackToBackPossessionsException` (pbpstats internal), 93
  `score_reconciliation_failed` (reconstructed final ≠ feed final), 68
  `missing_context` (season openers, no same-season prior game), 5
  `possession_alternation_failed`, 1 `AttributeError`.
- Reconciliation on accepted games: reconstructed final and every period match
  the data.nba.com feed exactly (`final_score_matched: true`). This is a
  within-NBA check (stats vs feed), **not** an independent provider.

## Verification

- `uv sync --locked`, `uv run courtgraph doctor` — clean / healthy.
- `uv run python -m unittest discover -s tests -v` — **158 tests** (3 new);
  `compileall`, `ruff check`, `ruff format --check`, `mypy` (45 files) clean;
  `PYTHONPATH=src python3 -m unittest …` — 158 OK (33 skipped, no numpy).
- Real end-to-end run of `snapshot-from-shufinskiy --all-games` → `ingest`
  → counts above; `courtgraph app --ingest-dir …` `/api/state` reports 6,000
  games, 30 teams, 985 players, matching coverage, real names.
- Spot-checked games across all five seasons: `final_score_matched: true`,
  per-period deltas zero. `days_rest_offense` distribution is sane (mode 1,
  back-to-backs 0, All-Star-break gaps 7-8; **no cross-season >100-day
  values**).
- `git status` clean of everything under `data/` (all gitignored).

## Boundaries

- Observational regular-season data only — no model is fit or evaluated here.
- The score check is within-NBA (stats reconstruction vs data.nba.com feed),
  not an independent lineage (`DATA_SOURCES.md` §5.2).
- Box-score minutes / lineup minutes are not reconciled.
- 14% of games are quarantined; the biggest bucket (`network_required`, 503
  games) needs operator-supplied period starters or a permitted box-score
  fetch, both out of scope here.
- 2020-21 is the COVID-shortened 72-game season and is structurally unusual
  (`DATA_SOURCES.md` §6).

## Next candidate tasks (not started)

1. **Run the model on real data** — leakage-safe splits (chronological,
   unseen-pair, unseen-lineup) + the ridge RAPM baseline + the low-rank model
   on these 266k stints; compare to the contract's rung-2/3 references. This is
   the next scientific milestone.
2. Nullable `days_rest_offense` (stint schema v3) to recover the 68
   season-opener quarantines.
3. Recover `network_required` games (503) with operator period-starter files.
4. 2016-17 → 2019-20 seasons — gated by the contract on passing data-quality
   checks first.

The research contract and the parked `task/schema-contract` worktree are
unchanged. `MASTER_PLAN.md` §44 is on `main` (PR #9); `courtgraph app` is on
`main` (PR #10).

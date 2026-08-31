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

- **`src/courtgraph/ingest/shufinskiy.py`** — `build_snapshot()` converts named
  games from a local **SRC-SHUFINSKIY** playoffs archive (CSV re-packaging of
  stats.nba.com / data.nba.com; `DATA_SOURCES.md` §1 — the local-dev-only
  fallback) into a `stats_nba_pbpstats/v1` snapshot: `playbyplayv2` and
  `shotchartdetail` rebuilt from the CSVs; `game_date` / teams derived; the
  **`reconciliation` target taken from the data.nba.com lineage** (a second NBA
  surface, not an independent provider); rest days computed from the archive's
  own game dates. A game whose rest days cannot be derived is written **without**
  `days_rest` (importer quarantines it — nothing fabricated). A
  `display_names.json` sidecar (ids -> names, display only) is written for the
  report. No network.
- **`src/courtgraph/ingest/report.py`** — `render_report()` / `write_report()`:
  one self-contained HTML report from a completed ingest — run totals, per-game
  matchup with real team/player names, the score check vs the data.nba.com
  lineage (per team and per period), busiest stint lineups, garbage-time-weighted
  stints, and every exclusion. Banner states plainly that a few games is not
  evidence of predictive accuracy.
- **CLI**: `courtgraph snapshot-from-shufinskiy --archive-dir DIR --game GID ...
  --out-dir DIR`; `courtgraph ingest ... --report PATH`.
- **Tests** (+9): converter (structure, padded ids, derived-not-fabricated
  metadata, no-prior-game -> no `days_rest`, name sidecar, unknown-game error)
  and the report renderer (self-contained, shows teams/lineups/score check/
  exclusions, honest disclaimer).

## The demonstration (local only, not committed)

Round 1, Cavaliers-Heat, from the local archive:

| game | date | result | outcome |
|---|---|---|---|
| `0042400102` | 2025-04-23 | -- | quarantined: `pbpstats` cannot order two events (needs an override) |
| `0042400103` | 2025-04-26 | -- | quarantined: period starters not inferable from PBP -> a box-score request -> refused by the offline guard |
| `0042400104` | 2025-04-28 | CLE 138 - MIA 83 | **accepted**: 197 possessions reconstructed, 184 accepted, **47 stints**; reconstructed score == data.nba.com lineage exactly, all four periods; 2 empty + 11 split-lineup possessions excluded; 2 stints garbage-time-weighted (0.2) |

Real lineups appear by name (e.g. *Donovan Mitchell, Evan Mobley, Jarrett Allen,
Max Strus, Sam Merrill*). This is one game -- **not** proof of predictive
accuracy, calibration, or data quality at scale.

## Data permission

The archive is `SRC-SHUFINSKIY` -- acquired 2026-08-30 for
`task/schema-contract`, pinned + checksummed, **no live endpoint contacted**.
`DATA_SOURCES.md` classifies it *fallback -- local dev only*: fine for a local,
non-redistributed demonstration; NBA terms still apply. All NBA data, the built
snapshot, the ingest outputs, and the report are gitignored
(`data/nba_snapshots/`). No data was downloaded for this task. A truly
independent reconciliation lineage is still unavailable (`DATA_SOURCES.md` §5.2).

## Limitations

- One archive, one series; nothing validated at season scale.
- Reconciliation is within-NBA (stats vs data.nba.com), not independent.
- Box-score minutes / lineup minutes are not reconciled.
- Games needing `pbpstats` overrides are quarantined, not patched.
- No model is fit or evaluated.

## Verification (all pass)

```
uv lock --locked                                    # 15 packages, no drift
uv run courtgraph doctor                             # healthy
uv run python -m unittest discover -s tests -v       # 118 tests OK (9 new)
uv run python -m compileall -q src tests             # OK
uv run ruff check . ; uv run ruff format --check .   # clean
uv run mypy                                          # 39 source files, clean
PYTHONPATH=src python3 -m courtgraph doctor / unittest  # 118 OK (26 skipped: no pbpstats)
uv run courtgraph demo --bootstrap 0                 # synthetic slice unchanged; 0 ensemble members
courtgraph snapshot-from-shufinskiy ... ; courtgraph ingest ... --report report.html
```

## Next action

Codex review before merge. Next single task (do not start until activated):
recover the two quarantined games with `pbpstats` override files derived from a
box score, and/or extend the run to a full series.

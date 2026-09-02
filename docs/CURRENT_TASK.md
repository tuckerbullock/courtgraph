# Current Task

Last updated: 2026-09-01

## Active

**Task 2 of the autonomous work queue: maximal data acquisition + dual-surface
ingest.** Branch `task/data-acquisition`.

Motivation: (a) the `three_share` hardening showed the interaction models are
power-limited; (b) the ingest silently quarantines 8.4 % of games (840 / 5,998
for RS 2020-24), a rate that would hit every new season. User direction: "just
get as much new data as you can", "proceed, keep it local & research-scoped",
"re-run the tests and models after".

### Done

- **`scripts/fetch_shufinskiy.py`** + `cycle1` plan — pulled from the pinned
  shufinskiy commit (GitHub raw, sha256 TOFU, no NBA endpoint): RS 2016-17…
  2019-20 (`datanba`/`nbastats`/`shotdetail`), playoffs 2016…2023, and the
  `cdnnba`/`nbastatsv3`/`matchups` surfaces for 2020-25. New dirs under
  `data/nba_snapshots/_shufinskiy_{rs_2016_2019,po_2016_2023,2025}/`, each with
  a `SOURCE.md`. All gitignored.
- **Snapshot format `v2`** (`v1` still loads): each game may carry
  `pbp/data_<gid>.json` (the data.nba.com feed). `snapshot-from-shufinskiy`
  emits it from the `datanba` CSVs and now takes several `--archive-dir` dirs.
- **Dual-surface reconstruction** (`possessions.py` / `pipeline.py`): when the
  playbyplayv2 surface needs a network call or raises, the game is retried with
  pbpstats' `data_nba` provider (period starters from the pbp walk, no network
  path). Games that already work are untouched; a `pbp_surface:data_nba`
  manifest flag records the fallback.
- **`courtgraph fetch-live`** + `live_fetch.py` — the optional §5.1 live path
  (single worker, ≥1.5 s, backoff, hard-stop on 403/429, cache-and-freeze).
  stdlib only, no new dep. **stats.nba.com is unreachable from this
  environment** (times out even unsandboxed), so the live path is built and
  tested (fake transport) but must be run by the user from a network where NBA
  endpoints resolve — for full 2025-26 and the `InvalidNumberOfStarters` set.

230 tests pass; ruff / mypy / dep-free clean. 2 commits on the branch.

### In progress

Sequential re-ingest (memory-bounded — the combined 9-season build OOM'd):
`rs_2016_2019/` then `rs_2020_2024_v2/` (v2 rebuild). Then concatenate RS
stints and re-run `baselines` / `roles` / `confirm` at the new scale.

### Next action

1. Finish the re-ingest; record games/stints/quarantine-rate before-vs-after
   and the `stats_nba`-vs-`data_nba` possession-count delta in this file.
2. Regenerate `player_profiles.jsonl`; re-run `courtgraph baselines --rung4`,
   `roles`, `confirm` on the enlarged RS stint set; update
   `docs/INTERACTION_FINDINGS.md` and `docs/PROJECT_STATUS.md`. Preserve every
   result (contract §17).
3. Open the PR.

## The work queue (after this task, in order)

3. **Nullable `days_rest` schema v3** — the season-opener `missing_context`
   quarantines (68 for 2020-24, plus every 2016-17 and new-season opener).
4. **Per-player production ingest** — unlocks §45 player-lift, defensive side.
5. **§45 player-lift** — Phase A (pooled lift scalar) then Phase B.
6. **Defensive-side extension** — roles / redundancy / mechanistic on defense;
   `matchups` surface (now acquired) feeds this.
7. **Turnover / assist mechanistic outcomes** — full `confirm` treatment.
8. **Transaction backtest (T4)** — needs the live roster/transaction fetch
   (`fetch-live`, blocked here) or `prosportstransactions`.
9. **Model-ladder gaps** — rung 1 explicit; rung 5 re-run under bootstrap CI.
10. **Contract deliverables** — strong unseen-pair holdout; seed stability;
    cycle-1 report; decision log.
11. **2025-26** — needs the live fetch or a `cdnnba`→pbp importer path.
12. **Product (§44 / issue #8)**.

Each item lands as its own focused branch + PR. `ChemistryConfig` /
`HierarchicalConfig` / `RoleInteractionConfig` / `RedundancyConfig` defaults
unchanged unless a task explicitly changes them. The 2024-25 playoff archive
stays held out of training.

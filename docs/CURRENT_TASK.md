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

### Re-ingest result (2026-09-02)

Built memory-bounded, one season-range at a time (a combined 9-season + PO
build OOM'd this machine):

| RS window | games in | accepted | quar (rate) | stints | `data_nba` recovery |
|---|---|---|---|---|---|
| 2016-17 … 2019-20 | 4,746 | 4,556 | 190 (4.0 %) | 239,570 | 903 games |
| 2020-21 … 2024-25 | 5,998 | 5,760 | 238 (4.0 %) | 297,404 | 620 games |
| **8-season concat** (`rs_2016_2024/out/stints.jsonl`) | | 10,316 | 428 | **536,974** | 1,523 |

Was 266,518 (2020-24 v1). **All 266,518 prior stints are a strict subset of
the new 297,404** for that window — the `data_nba` fallback is purely additive.
Residual quarantines: `score_reconciliation_failed` 162 (fail-closed on
data.nba.com score ≠ official), `missing_context` 143 (openers → Task 3),
`network_required` 92 (both surfaces need network), `possession_alternation` 29.

### Model re-run at scale (2026-09-02)

`baselines` (rung 2/3) on the 537k 8-season set and on 2020-24 v2 (297k):

| holdout | 8-season r2 | r3 | 2020-24 v2 r2 | r3 |
|---|---|---|---|---|
| chronological | 6.63 | **6.47** | 2.79 | **2.69** |
| unseen_pair | 19.78 | **18.90** | 19.48 | **19.28** |
| unseen_lineup | **4.53** | 4.60 | **4.94** | 4.99 |

rung 3 still wins on the pair/chronological holdouts; unseen_lineup is a wash
(both scales). rung-3 structural-holdout calibration holds (z_sd ≈ 1.0–1.4).
chronological calibration is still broken and slightly worse over the longer
2016 → 2024 span (z_sd 2.48). Variance components essentially unchanged
(τ_off ≈ 2.2, σ ≈ 118) — **the noise floor is structural, not sample-limited**.
rung 4 skipped at this scale (EM over ~8k pairs OOM-risk; it was already a
placebo-matched null in PR #16).

`confirm` (`three_share`, K {3,5,7}, 120-group holdout, 3000-boot) on 2020-24
v2 (297k): `three_share` **holds** — beats rung 3 across all K (CI excludes 0,
~2 % of RMSE), beats the placebo clearly at K = 3, borderline K = 5/7,
mediation with scoring = −0.01. The **points/100 role effect is now null** —
K = 5 delta +0.06 [−0.02, +0.15], CI spans 0 (was barely excluding it on
266k). redundancy still null. **More data sharpened the negative.**

Full write-up in `docs/INTERACTION_FINDINGS.md` → "At 2× the data".

### Status: complete

6 commits + docs on `task/data-acquisition`. PR: opened. Deferred to the user
(needs a network where stats.nba.com resolves): `courtgraph fetch-live` for
2025-26 and the ~92 residual `network_required` games.

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

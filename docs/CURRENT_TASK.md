# Current Task

Last updated: 2026-09-05

## Active — real-data lineup predictor (product side) — DONE

`task/rung3-lineup-predictor`. After the interaction arc closed (below) the
user asked to switch to the product side (issue #8 / master plan §44).
Issue #8's flagship feature ranks lineups by "chemistry surplus" — not
buildable as specified, since chemistry isn't a supported predictive effect.
This task builds the honest version: a user can pick any 5-vs-5 lineup from
real, observed NBA players and get the one validated result (rung 3's
calibrated additive prediction), with no interaction/chemistry field
anywhere in the code path.

- New `rung3_artifact.py` (its own schema, distinct from the synthetic
  `ChemistryModel` artifact) + `fit_rung3_file`/`predict_lineup_rung3` in
  `pipeline.py`, reusing `HierarchicalRidge`'s existing `decompose_row` /
  `group_predictive` — no new model math needed, only packaging.
- New CLI: `courtgraph fit-rung3`, `courtgraph predict-rung3`.
- App: `Observations.player_pool(team)` (observed-in-stints, explicitly not
  an official roster — no dated roster source exists) and
  `Observations.predict()` (rung 3 fit once, lazily, cached in memory); new
  endpoints `GET /api/player-pool`, `POST /api/predict-real`; a new frontend
  panel in the Game explorer view with a permanent no-chemistry-claim banner.
- Manually end-to-end tested against the real 297k-stint dataset (curl +
  browser-driven click-through), which caught and fixed two real bugs: an
  id mismatch between `index.html`'s player-select containers and
  `app.js` (dropdowns silently failed to populate) and a bloated API
  response (the full per-player possessions table, hundreds of entries, was
  being echoed on every prediction — trimmed since `support` already
  summarizes what's needed).
- 260 tests (7 new), ruff/mypy/dependency-free path clean. 2 commits on the
  branch.

Not done (explicitly out of scope, documented in the plan): anything ranked
by chemistry surplus, league-wide lineup finder, real dated rosters,
opponent counter-lineups — all remain backlog until a future estimand
overturns the current null, or a roster data source is acquired.

## Completed — the "could still flip it" arc (user: "yesss all of these")

Different estimands, not "does γ_ij exist". Order: **D → B → C → E.**

- **§45 Phase A — DONE** (`task/player-lift-phase-a`, PR pending). Pooled `λ_i`
  lift scalar on lineup value + player-permutation placebo. **Null**, as §45.2
  predicted: full-fit `τ_λ²` at the grid floor (1e-5), |λ_i| ≤ 0.0003, and the
  placebo recovers the *exact same* `τ_λ` per fold. Held-out RMSE within 0.5 %
  of rung 3, matched by the placebo. Written up in `INTERACTION_FINDINGS.md`
  §45.
- **D — transaction backtest — DONE** (`task/transaction-backtest`, PR pending).
  585 clean cross-season switches + 1,200 phantom non-movers. **Clean null:**
  mean |Δ| real − phantom = −0.30 [−0.63, +0.03] — movers' lineups scatter from
  the additive prediction no more than non-movers', if anything marginally
  less. A player's lineup-value contribution transfers across a team change as
  cleanly as a non-mover's stays put. Best-powered test in the project.
  `INTERACTION_FINDINGS.md` "Transaction backtest".
- **B + C — DONE** (`task/player-production`, PR pending). `courtgraph
  player-production` (per-(player, stint) production, 99.2 % event match on real
  data, validated vs known stars) + `courtgraph phase-b` (§45 Phase B: lift on
  teammate *individual* production). **Null:** the lift model does not beat the
  base-only model (points-only +0.01 [−0.12, +0.13]; +0.5·assists −0.08
  [−0.26, +0.11]). Large in-sample lift coefficients that do not generalise.
  **Closes the last open estimand** — a player's effect on teammates is not a
  transferable quantity beyond additive talent, on any of the three
  measurements (lineup value, roster changes, individual production).
- **E — defensive side — DONE** (`task/defensive-side`, PR pending).
  `courtgraph player-lift --side defense`: the pooled lift keyed on the
  defensive lineup. **Null** — on 297k and 537k the defensive lift terms make
  held-out prediction *worse* than rung 3 and the placebo recovers the
  identical `τ_λ` per fold. A defensive per-pair `γ_ij^def` and a
  `matchups`-surface deep dive remain documented follow-ups but the pooled
  result points the same way as everything else.

## Arc complete

**Every estimand for "a player's effect on teammates" is now tested — every
one null.** Symmetric pairs (rungs 4–5), pooled asymmetric lift on lineup
value (Phase A), across roster changes (transaction backtest, best-powered),
on individual production (Phase B), defensive side. The interaction question
has a defensible, comprehensive answer: **not supported.** Next candidate
work at the time: the cycle-1 research report, or the product side (§44 /
issue #8) — the product side's first slice is done above.

---

## Superseded (Task 2, merged PR #25)

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
12. **Product (§44 / issue #8)** — first slice done (rung-3 real-lineup
    predictor, above). Remaining: real dated rosters, league-wide finder,
    opponent counter-lineups, and everything else in issue #8 that depends
    on data or model support this slice doesn't have.

Each item lands as its own focused branch + PR. `ChemistryConfig` /
`HierarchicalConfig` / `RoleInteractionConfig` / `RedundancyConfig` defaults
unchanged unless a task explicitly changes them. The 2024-25 playoff archive
stays held out of training.

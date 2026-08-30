# CourtGraph Data-Source Registry and Source-Selection Decision

> **Version:** 0.3 (research cycle 1) — revised after a second independent review
> **Status:** engineering assessment — proposed, pending a data pilot and a legal review of release scope
> **Assessment date:** 2026-08-29 (all reachability and documentation checks on this date unless noted)
> **Governing documents:** [`RESEARCH_CONTRACT.md`](RESEARCH_CONTRACT.md) §9, §29; [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) §§6–7

## Scope and disclaimers

This document decides **how CourtGraph obtains the data for its first research
cycle**. It is a research and decision task, not an ingestion implementation:
no data was bulk-downloaded, no pages were scraped, no access controls were
tested or bypassed. A small number of respectful, read-only requests were made
to check reachability and inspect representative schemas.

- **This is an engineering assessment, not legal advice.** A qualified lawyer's
  review is required before any public data release and before scaling ingestion.
- **Endpoint reachability is not permission.** A URL returning JSON does not mean
  its terms allow retrieval, storage, database creation, or redistribution.
- **Terms findings are verified against the official policy pages** cited in
  "Sources" below, as of 2026-08-29. The Disney/ESPN Terms of Use were retrieved
  in full (effective 2024-05-24) and are quoted verbatim. The NBA and Sports
  Reference terms pages block automated retrieval; the findings recorded for them
  reflect direct review of the official pages, not search-result snippets. No
  conclusion in this document rests on a search snippet.
- **Confirmed vs inferred.** First-party facts carry a URL. CourtGraph's own
  conclusions are marked *(inference)*. Two conservative inferences drive
  decisions here and are labelled as such wherever they appear: (i) that
  CourtGraph's private research dataset *may* fall within the NBA
  "comprehensive database" restriction (§1 note), and (ii) that the Sports
  Reference AI-training prohibition *also* reaches using its data to validate or
  benchmark a model (SRC-BREF).
- **Unclear stays unclear.** Where a term or limit is not confirmable, it is
  recorded as `unclear`, not guessed.

### Distinctions this document keeps separate

| Distinction | In this document |
|---|---|
| Data **provider** vs client **library** | `stats.nba.com` / `data.nba.com` are providers; `nba_api`, `pbpstats` are libraries/tools (§5) |
| **Raw events** vs **parsed possessions** | Bronze = raw provider payloads; possessions/stints are a CourtGraph-owned derivation, independently validated (§5) |
| **Technical accessibility** vs **permission** | The registry records reachability *and*, separately, terms / redistribution status |
| **Primary** vs **validation** source | Primary = the raw feed CourtGraph ingests; validation = an independent lineage used only to cross-check |
| **Availability** vs **quality** | The rubric (§3) scores coverage separately from provenance, schema stability, and known anomalies |
| **Confirmed fact** vs **assumption / open question** | Inline *(inference)* tags; open questions in §9 |

---

## 1. Decision summary

| Decision | Recommendation | Status |
|---|---|---|
| **Primary raw-data source** | `stats.nba.com` NBA Stats endpoints (all eras), retrieved via the `nba_api` client under the conservative access policy in §5.1, stored locally as immutable Bronze. `data.nba.com` as a second raw lineage for 2016-17+ (it carries `offense_team_id` on every event). | **Provisional & conditional** — the build may plausibly fall within the NBA "comprehensive database" restriction (§1 note); needs legal review and possibly express consent; interim build stays local, non-commercial, non-redistributed, research-scoped |
| **Possession / parser strategy** | Use `pbpstats` as reconstruction **tooling** (possession boundaries, lineup-on-floor, event-order fixes) against the stored Bronze. CourtGraph builds and owns an **independent possession/stint validator** (master plan §7.7); parser output is derived, never a substitute for raw payloads. Do **not** ingest `pbpstats.com` hosted archive data (an archived play-by-play product — see NBA note). | **Binding** (approach); parser parameters provisional |
| **Independent reconciliation source** | **None confirmed.** No genuinely independent, freely usable provider lineage with sufficiently clear rights exists. Interim validation stack in §5.2 relies on the two NBA surfaces, official NBA totals, CourtGraph's own validator, and manual audits. Licensed **Sportradar** access or written provider permission is the only path to a truly independent lineage. | **Open** — a known gap, not a solved problem |
| **Transaction source** | A **manually curated event cohort**, each transaction supported by an official NBA/team release and at least one dated contemporaneous report, with per-event citations. `prosportstransactions.com` may be a **human research lead only** — not imported or systematically copied. | **Provisional**; compiled table's public release **deferred** |
| **Fallbacks** | (a) `shufinskiy/nba_data` or Kaggle `wyattowalsh/basketball` as a **local-only** frozen snapshot for pilot/dev when live access is blocked — not a reproducibility or release artifact, and subject to the same NBA restrictions; (b) `data.nba.com` where `stats.nba.com` parsing is fragile. | Fallback (local only) |
| **Rejected** | **Basketball-Reference / Sports Reference** — its terms prohibit using its content to train / fine-tune / prompt / instruct AI systems or to support ML methods that predict, classify, label, or score, without permission; CourtGraph **conservatively infers** this reaches model validation and benchmarking too, and removes Basketball-Reference from every CourtGraph pipeline. **ESPN via `hoopR`** — Disney/ESPN terms prohibit automated extraction / data-mining and use in connection with training, testing, benchmarking, or validation of AI/ML tools; `hoopR`'s MIT code licence grants no rights to ESPN data. **`prosportstransactions.com` as an imported dataset** — no reuse terms, all rights reserved, no API. **Sportradar licensed feed** — cost and contract scope disproportionate for cycle 1 (kept as the optional permission path). | Rejected |
| **Public release scope** | Release **only**: CourtGraph code, schemas, synthetic or hand-authored fixtures, methodological reports, and appropriately **aggregated** findings. **Do not release** Bronze payloads or **row-level** Silver/Gold possession/stint/event data. Embeddings, model artifacts, and any derived row-level dataset are **permission / legal-review dependent** and released only if that review clears them. | **Deferred** — pending legal review; seek NBA written consent |

**Binding now:** the parser *approach* (pbpstats-as-tool + an independent
CourtGraph validator); Bronze immutability and content-addressing; the access
conduct policy (§5.1); "no row-level NBA-derived data and no Bronze in the public
release" as the default; the rejections above.

**Provisional (pending the §8 pilot):** primary provider per era; the season
windows (§6); playoff handling; the transaction cohort's size and depth.

**Deferred / open:** whether express NBA consent is sought and obtained; the
legal review of release scope; the absence of any confirmed independent
validation lineage.

### Note — NBA statistics use and release

Reviewed against the NBA Terms of Use (<https://www.nba.com/termsofuse>,
2026-08-29): NBA statistics are made available for **legitimate news reporting or
private, non-commercial use**; **archived play-by-play products are restricted**;
and the terms address use **connected to a website, product, or service that
features a comprehensive, regularly updated database of NBA statistics**, for
which the NBA's **express prior consent** is required. Consequences for
CourtGraph:

- CourtGraph's dataset is a **fixed-cutoff, six-season, private research
  dataset**, not a public service or product. Whether it falls within the
  "comprehensive, regularly updated database" restriction is **not clear** — it
  **may plausibly** be covered. This requires a **legal review** and possibly the
  NBA's **express consent**. Until that is resolved, keep the build strictly
  local, non-commercial, non-redistributed, and research-scoped, and be prepared
  to request written permission.
- **Do not** redistribute Bronze payloads or any row-level NBA-derived data
  (events, possessions, stints, lineup exposures) without clearance.
- The default public research release is limited to code, schemas,
  synthetic/fixture data, methodology, and aggregated results; model artifacts
  and any derived row-level dataset ship only if a legal review clears them.

---

## 2. Data requirements

Translated from `RESEARCH_CONTRACT.md` §§4–13 and `docs/MASTER_PLAN.md` §§6–8.
Every ingested record must ultimately support these fields.

### 2.1 Identifiers (never names as primary keys — master plan §6.4)

- `game_id` — stable canonical game key. `stats.nba.com` uses a 10-digit ID:
  `00` + season-type digit (`1` pre, `2` regular, `3` all-star, `4` playoffs) +
  2-digit season + 5-digit sequence, e.g. `0022300001` = first regular-season
  game of 2023-24. Format is a de-facto standard documented by the community, not
  officially published — <https://nba-stats-tracking.readthedocs.io/en/latest/source/modules.html> (accessed 2026-08-29) *(inference)*.
- `team_id`, `player_id` — `stats.nba.com` numeric IDs. Store relocations /
  abbreviation changes with `valid_from` / `valid_to`.
- `season`, `season_type` (regular / playoffs), `league_id`.
- `event_id` / `event_num`, `period`, and game clock for event ordering.

### 2.2 Event-level (play-by-play)

- period, game clock, wall-clock where available;
- event type and sub-type, and the provider's description strings;
- participant player/team IDs (primary, and secondary for assists / blocks /
  steals / fouls);
- home and away score after the event;
- `offense_team_id` where the provider supplies it (`data.nba.com` on all events;
  `stats.nba.com` does not — §5);
- provider-native event sequence number (to reproduce and audit any re-ordering).

### 2.3 Lineup / substitution inputs

- substitution events (player in / out, period, clock);
- period-start on-court state (inferred from box score + first-period events);
- five-player offensive and defensive sets per possession, canonicalised as an
  unordered set sorted by `player_id` (contract §13);
- `lineup_state_confidence` and partial-/split-possession flags (master plan §7.3–7.4).

### 2.4 Totals and reconciliation targets (master plan §6.5, §7.7)

- official final score; team points by period; number of OT periods;
- player minutes and team minutes; lineup minutes;
- team possession counts within expected methodology differences.

### 2.5 Schedule and phase

- full schedule per season; game date; home / away team; arena;
- season phase label (regular / play-in / playoffs / NBA Cup knockout) kept
  explicit — never silently pooled (contract §9; master plan §6.3).

### 2.6 Rosters and transactions

- roster membership with `valid_from` / `valid_to` per player-team spell;
- transaction records: players and teams involved, destination team, the
  transaction/announcement timestamp **and** the computed debut timestamp (§7);
- coach / team-season metadata **if reasonably available** (Tier-B; not required
  for cycle-1 baselines).

### 2.7 Provenance (attached to every Bronze record — master plan §6.1)

- `retrieved_at` timestamp (UTC);
- `provider` identity and endpoint;
- raw schema / API version string where present;
- `content_hash` (raw payload, content-addressed, append-only);
- `parser_version` and `correction_set_version` on every derived record.

---

## 3. Evaluation rubric

Scored `strong` / `acceptable` / `weak` / `unknown`. No numeric scores are
manufactured. "—" = not applicable to that source's role.

| Dimension | stats.nba.com | data.nba.com | nba_api (lib) | pbpstats (tool) | shufinskiy / Kaggle snapshots | Basketball-Reference | ESPN via hoopR | prosportstransactions | Sportradar (licensed) |
|---|---|---|---|---|---|---|---|---|---|
| Authority / provenance | strong (NBA first-party) | strong (NBA first-party) | — | — | weak (re-hosted NBA data) | acceptable (independent editorial) | acceptable (independent provider) | acceptable (independent editorial) | strong (official NBA data provider) |
| Historical coverage | strong (PBP 1996-97+) | weak (PBP 2016-17+) | — | acceptable | strong (packaged 1996-97+) | strong (PBP 1996-97+) | acceptable (ESPN era) | strong (multi-decade) | strong (contracted) |
| Event detail | strong | acceptable | — | strong (adds possession + lineup fields) | strong (mirrors NBA) | acceptable | acceptable | — | strong |
| Identifier stability | acceptable (stable in practice, undocumented) | acceptable | — | acceptable (uses NBA IDs) | acceptable | weak (BR slugs ≠ NBA IDs) | weak (ESPN IDs ≠ NBA IDs) | weak (name-based) | acceptable |
| Schema stability | weak (undocumented; NBA does not announce changes) | weak (undocumented) | acceptable (community tracks changes) | acceptable | acceptable (frozen archives) | acceptable | acceptable | acceptable | strong (contracted) |
| Accessibility / auth | weak (no auth; automated access often restricted; NBA.com properties timed out here) | weak (no auth; not directly reachable here) | strong (pip) | strong (pip) | acceptable (archive download) | weak (403 to automated client) | weak (Disney terms bar automated extraction) | weak (403 to automated client; search form only) | weak (paid, authenticated) |
| Rate-limit clarity | weak (no official limit; §5.1 self-imposed ceiling) | weak (same) | weak (defers to NBA) | weak | n/a (static files) | n/a (rejected) | n/a (rejected) | unknown | strong (contracted) |
| Terms / redistribution clarity | weak — news-reporting / private non-commercial use; use tied to a comprehensive updated statistics database may need express consent; archived PBP restricted (nba.com/termsofuse) | weak (same NBA terms) | acceptable (MIT code; no data rights) | acceptable (MIT code; no data rights) | weak (no compilation licence; NBA terms still apply) | weak — **prohibits training / fine-tuning / prompting / instructing AI systems and ML methods that predict / classify / label / score, without permission** (sports-reference.com); CourtGraph infers validation/benchmarking is also barred | weak — **prohibits automated extraction / data-mining and use in training, testing, benchmarking, or validation of AI/ML tools** (disneytermsofuse.com) | weak (all rights reserved; no terms page) | acceptable (explicit paid licence) |
| Reproducibility | acceptable (with local Bronze cache) | acceptable | acceptable | strong (ships override files) | acceptable (pin a version; local only) | weak | weak | weak | acceptable |
| Known corrections / anomalies | weak (out-of-order shots/rebounds; pbpstats documents fixes) | acceptable (fewer ordering issues) | — | strong (documents and fixes them) | acceptable | acceptable | acceptable | acceptable | unknown |
| Suitability for immutable raw snapshots | strong (technically) | strong (technically) | — | — | acceptable (local only) | weak | weak | weak | acceptable (per licence) |
| Suitability as a CourtGraph validation input | acceptable (within-lineage only) | acceptable (within-NBA cross-check) | — | — | weak | **rejected** (AI-training term + CourtGraph's conservative reading) | **rejected** (terms bar testing/benchmarking/validation of AI/ML tools) | **rejected as import** (terms) | acceptable (with licence/permission) |
| Suitability for public release | weak (row-level NBA-derived data not releasable) | weak (same) | strong (code) | strong (code) | weak | weak | weak | weak | weak |
| Expected maintenance burden | high (endpoint drift, re-ordering) | medium | low | low–medium | low | n/a | n/a | medium (manual) | low (contracted) |
| Genuinely independent cross-check | no (same NBA lineage) | no (same NBA lineage) | — | — | no (re-hosted NBA) | rejected | rejected | rejected as import | yes (with licence/permission) |

---

## 4. Source registry

### SRC-NBASTATS — NBA Stats endpoints (stats.nba.com)

- **Provider:** National Basketball Association (first-party digital platform).
- **Role in CourtGraph:** recommended **primary raw source** (Bronze), all eras.
- **Canonical URL:** <https://stats.nba.com/> · endpoints under `https://stats.nba.com/stats/`
- **Documentation URL:** none official. Community endpoint map: <https://github.com/swar/nba_api> (accessed 2026-08-29).
- **Terms / policy URL:** <https://www.nba.com/termsofuse> — page **blocks automated retrieval** (timed out from the assessment client, 2026-08-29); findings below are from direct review of the official page. NBA statistics are for **legitimate news reporting or private, non-commercial use**; **archived play-by-play products are restricted**; and use **connected to a website, product, or service featuring a comprehensive, regularly updated NBA-statistics database** requires the NBA's **express prior consent** (whether CourtGraph's private research dataset is covered is unclear — see §1 note).
- **Access method:** unauthenticated HTTPS GET returning JSON; browser-like headers required in practice. See §5.1 for the mandatory access-conduct policy.
- **Authentication required:** no.
- **Data format / key fields:** JSON `resultSets` (row sets + headers). Endpoints incl. `leaguegamelog` / `scoreboardv2` (schedule), `playbyplayv2` (events), `boxscoretraditionalv2` / `boxscoreadvancedv2` (totals, minutes), `commonallplayers` / `commonteamroster` (rosters), `shotchartdetail`.
- **Known / claimed coverage:** play-by-play and shot detail from **1996-97** — packaging that spans this range: <https://github.com/shufinskiy/nba_data> (accessed 2026-08-29).
- **Update cadence:** near-real-time during games; historical seasons static.
- **Rate-limit info:** **no official limit published.** CourtGraph applies the self-imposed ceiling in §5.1. Automated access from non-residential / hosting IPs is frequently restricted; the pilot (§8) must measure actual behaviour.
- **Redistribution implications:** raw payloads and row-level derivations are **not redistributable** under the current terms; building the database itself plausibly requires express NBA consent (§1 note).
- **Known issues:** `stats.nba.com` play-by-play has many out-of-order shots and rebounds that need manual fixing, and events carry **no `offense_team_id`** — <https://github.com/dblackrun/pbpstats/blob/main/docs/quickstart.rst> (accessed 2026-08-29).
- **Reachability result (2026-08-29):** stats endpoint host reachable in principle; NBA.com web properties (`/termsofuse`, `/robots.txt`) timed out from the assessment client. Treat automated access as fragile.
- **Verification date:** 2026-08-29.
- **Decision status:** **provisional & conditional (primary).**
- **Reason:** best coverage + first-party provenance; the blocker is the terms position on database creation and release, not technical fit.
- **Pilot check still required:** yes — access behaviour under §5.1 from the intended environment; endpoint schema snapshot; completeness for the §6 windows; and a decision on seeking NBA consent.

### SRC-DATANBA — NBA live/CDN feeds (data.nba.com / cdn.nba.com)

- **Provider:** National Basketball Association (first-party).
- **Role:** recommended **second raw lineage** for 2016-17+ and within-NBA parser cross-check.
- **Canonical URL:** <https://data.nba.com/> · <https://cdn.nba.com/static/json/>
- **Documentation URL:** none official; usage documented by `pbpstats` — <https://pbpstats.readthedocs.io/en/latest/> (accessed 2026-08-29).
- **Terms / policy URL:** <https://www.nba.com/termsofuse> — same findings and restrictions as SRC-NBASTATS.
- **Access method:** unauthenticated HTTPS GET, JSON. §5.1 policy applies.
- **Authentication required:** no.
- **Data format / key fields:** JSON; **`offense_team_id` on all play-by-play events**, which simplifies possession-change tracking — <https://github.com/dblackrun/pbpstats/blob/main/docs/quickstart.rst> (accessed 2026-08-29).
- **Known / claimed coverage:** play-by-play from **2016-17** — <https://github.com/shufinskiy/nba_data> (accessed 2026-08-29).
- **Update cadence:** real-time.
- **Rate-limit info:** none published; §5.1 applies.
- **Redistribution implications:** same as SRC-NBASTATS.
- **Known issues:** limited history; fewer event-ordering problems than `stats.nba.com` *(inference from pbpstats docs)*.
- **Reachability result (2026-08-29):** not directly tested (NBA.com properties timed out); documented and used by `pbpstats` and `shufinskiy/nba_data`.
- **Verification date:** 2026-08-29.
- **Decision status:** **provisional (secondary raw lineage).**
- **Reason:** a within-NBA cross-check of possession boundaries for the whole recommended window, and simpler possession attribution.
- **Pilot check still required:** yes.

### SRC-NBA-API — `nba_api` Python client (swar/nba_api)

- **Provider:** open-source project (maintainer: swar and contributors). **A client library, not a data provider.**
- **Role:** the retrieval **client** for SRC-NBASTATS / SRC-DATANBA.
- **Canonical / Documentation URL:** <https://github.com/swar/nba_api>
- **Terms / license URL:** MIT — <https://github.com/swar/nba_api/blob/master/LICENSE>. README: "NBA.com has a Terms of Use regarding the use of the NBA's digital platforms." (accessed 2026-08-29).
- **Access method:** `pip install nba_api`; Python 3.10+; `requests`, `numpy` (pandas optional).
- **Authentication required:** no (passes through to NBA endpoints).
- **Data format / key fields:** returns NBA endpoint JSON as dicts / DataFrames; maps hundreds of endpoints and parameters.
- **Known / claimed coverage:** whatever the underlying NBA endpoints expose; "NBA.com does not provide information regarding new, changed, or removed endpoints" (README, accessed 2026-08-29).
- **Update cadence:** actively maintained; frequent releases — <https://github.com/swar/nba_api/releases>.
- **Rate-limit info:** none of its own; CourtGraph imposes §5.1.
- **Redistribution implications:** the **library** is MIT; it confers **no rights over the data** it retrieves.
- **Known issues:** endpoint drift handled reactively via GitHub issues.
- **Reachability result (2026-08-29):** GitHub repo and PyPI package reachable.
- **Verification date:** 2026-08-29.
- **Decision status:** **selected (client).**
- **Reason:** the standard, permissively licensed client; avoids re-deriving endpoint knowledge.
- **Pilot check still required:** yes — confirm a pinned version works from the intended environment under §5.1.

### SRC-PBPSTATS — `pbpstats` parsing / possession library (dblackrun/pbpstats)

- **Provider:** open-source project (maintainer: dblackrun). **Reconstruction tooling.** `pbpstats.com` separately hosts a derived play-by-play archive (2000-01+) — **not used** (see decision).
- **Role:** possession / stint / lineup **reconstruction tool** run against stored Bronze; source of event-order **override files**.
- **Canonical URL:** <https://github.com/dblackrun/pbpstats> · docs <https://pbpstats.readthedocs.io/>
- **Terms / license URL:** MIT — <https://github.com/dblackrun/pbpstats/blob/main/LICENSE> (accessed 2026-08-29).
- **Access method:** `pip install pbpstats`; configurable `dir` (local cache of `pbp` / `schedule` / `game_details` / `overrides`), `source` = `file` | `web`, `data_provider` = `stats_nba` | `data_nba` | `live`.
- **Authentication required:** no.
- **Data format / key fields:** adds per-possession start/end time, score margin, prior-possession end type; lineup-on-floor for all events; shots by zone; fixes some out-of-order events — <https://github.com/dblackrun/pbpstats/blob/main/docs/quickstart.rst> (accessed 2026-08-29).
- **Known / claimed coverage:** works with `stats_nba` (older data), `data_nba` (2016-17+), `live`.
- **Update cadence:** community; per its docs, event-order override files were maintained historically but going forward "you will have to keep up with fixing them manually yourself."
- **Rate-limit info:** none of its own (when it fetches, §5.1 applies).
- **Redistribution implications:** MIT tool; when run against NBA Bronze, its output is NBA-derived row-level data and inherits the NBA restrictions.
- **Known issues:** growing override-maintenance burden; possession logic is opinionated and must be independently validated (contract §13; master plan §7.7).
- **Reachability result (2026-08-29):** GitHub + Read the Docs reachable.
- **Verification date:** 2026-08-29.
- **Decision status:** **selected (tool); `pbpstats.com` hosted archive rejected** (an archived play-by-play product — restricted per the NBA terms).
- **Reason:** mature, documented possession logic; used as a tool (not an oracle) with validation kept in-house.
- **Pilot check still required:** yes — reproduce possessions for a game sample from both providers; diff against CourtGraph's validator and official totals.

### SRC-SHUFINSKIY — `shufinskiy/nba_data` bulk archive

- **Provider:** open-source project (maintainer: shufinskiy); **re-hosts NBA-origin data** (an archived play-by-play product).
- **Role:** **local-only** frozen snapshot for pilot / development when live access is blocked. **Not** a reproducibility anchor and **not** a release artifact.
- **Canonical URL:** <https://github.com/shufinskiy/nba_data> · fields <https://github.com/shufinskiy/nba_data/blob/main/description_fields.md>
- **Terms / license URL:** **no compilation licence stated** (accessed 2026-08-29); the underlying data is `stats.nba.com` / `data.nba.com` / `pbpstats.com` and the NBA terms — including the restriction on archived play-by-play products — apply.
- **Access method:** download `tar.xz` CSV archives from GitHub / Kaggle / Google Drive.
- **Data format / key fields:** `nbastats` (1996-97+), `datanba` (2016-17+), `pbpstats` (2000-01+), `shotdetail` (1996-97+), plus CDN / v3 variants.
- **Update cadence:** periodic archive refreshes.
- **Rate-limit info:** n/a (static files).
- **Redistribution implications:** **not redistributable** — no compilation licence, and the NBA archived-PBP restriction applies.
- **Known issues:** a mirror, not a source of truth; may lag the live endpoints; provenance of a row is the original NBA endpoint.
- **Reachability result (2026-08-29):** GitHub repo + README reachable.
- **Verification date:** 2026-08-29.
- **Decision status:** **fallback (local dev only).**
- **Reason:** lets the pilot proceed without repeatedly hitting live endpoints if the environment is blocked; carries the same NBA restrictions and cannot be released.
- **Pilot check still required:** yes — checksum a pinned archive; confirm it matches a small live sample.

### SRC-NBADB-KAGGLE — `wyattowalsh/basketball` ("NBA Database") + `wyattowalsh/nbadb`

- **Provider:** open-source project (maintainer: wyattowalsh); a publicly hosted, daily-updated relational NBA-statistics database derived from `stats.nba.com` via `nba_api`.
- **Role:** alternative **local-only** relational snapshot for exploratory work.
- **Canonical URL:** <https://www.kaggle.com/datasets/wyattowalsh/basketball> · code <https://github.com/wyattowalsh/nbadb>
- **Terms / license URL:** repo code is **MIT** (<https://github.com/wyattowalsh/nbadb>); the Kaggle dataset licence was **not confirmable** by automated fetch on 2026-08-29 — record as `unclear`. As a *public, regularly updated* NBA-statistics database, this dataset is the kind of product the NBA terms address most directly; its own compliance is not CourtGraph's to assume, and pinning a snapshot of it for local use does not launder the underlying NBA restrictions.
- **Access method:** Kaggle download (SQLite / DuckDB / Parquet / CSV).
- **Data format / key fields:** `dim_*` / `fact_*` / `bridge_*` schema incl. `fact_*` play-by-play, box scores, shot charts.
- **Update cadence:** daily incremental during active seasons.
- **Redistribution implications:** **not redistributable** — dataset licence unconfirmed and NBA terms apply.
- **Known issues:** daily-moving target — pin a version; not a source of truth.
- **Reachability result (2026-08-29):** GitHub reachable; Kaggle page not machine-readable without a session.
- **Verification date:** 2026-08-29.
- **Decision status:** **fallback (local dev only).**
- **Pilot check still required:** yes — confirm the Kaggle licence; pin a snapshot.

### SRC-BREF — Basketball-Reference.com (Sports Reference LLC)

- **Provider:** Sports Reference LLC (independent editorial).
- **Role in CourtGraph:** **none.** Not a feed, not a training input, and — on CourtGraph's conservative reading (below) — not a validation input.
- **Canonical URL:** <https://www.basketball-reference.com/>
- **Terms / policy URLs:** <https://www.sports-reference.com/termsofuse.html>, <https://www.sports-reference.com/data_use.html> — both **block automated retrieval** (HTTP 403, 2026-08-29); findings from direct review of the official pages.
- **Terms finding (official policy, reviewed 2026-08-29):** the current terms prohibit using Sports Reference content to **train, fine-tune, prompt, or instruct any artificial-intelligence system**, and to **support machine-learning methods used to predict, classify, label, or score**, without permission; they also prohibit automated access and building tools from scraped data without permission. The policy does **not** expressly list "testing", "benchmarking", or "validation".
- **CourtGraph inference (conservative, not quoted policy):** training a chemistry model and evaluating it are one workflow, and using Basketball-Reference data as a validation or benchmarking reference for that model would still be "supporting" ML methods that predict and score. CourtGraph therefore treats **any** use of Basketball-Reference data in its modelling *or* validation pipeline as disallowed absent written permission. This is a risk-averse reading; the policy does not state it.
- **Redistribution implications:** Basketball-Reference data is not used anywhere in CourtGraph's pipeline (training or, per the inference above, validation) without written permission from Sports Reference LLC.
- **Reachability result (2026-08-29):** 403 to automated client; normal for human browsing.
- **Verification date:** 2026-08-29.
- **Decision status:** **rejected** from the CourtGraph pipeline absent written permission.
- **Note:** a human may still *read* Basketball-Reference as general background when manually assembling the transaction cohort; its data is never systematically copied, ingested, or used as a validation reference.
- **Pilot check still required:** no (rejected).

### SRC-HOOPR-ESPN — ESPN play-by-play via `hoopR` / sportsdataverse

- **Provider:** ESPN (data, a Disney property); `hoopR` (open-source R client, MIT — <https://github.com/sportsdataverse/hoopR>).
- **Role in CourtGraph:** **none** for cycle 1.
- **Canonical URL:** <https://hoopr.sportsdataverse.org/> · <https://github.com/sportsdataverse/hoopR>
- **Terms / policy URL:** Disney Terms of Use — <https://disneytermsofuse.com/english/> (retrieved in full 2026-08-29; **effective 2024-05-24**).
- **Terms finding (verbatim, 2026-08-29):**
  - §2.B.x prohibits users to "access, monitor, copy or extract the Disney Products using a robot, spider, script, or other automated means, including, for the avoidance of doubt, for the purposes of creating or developing any AI Tool, data mining or web scraping";
  - §2.A (Consumer License) grants "no right to reproduce, distribute … or transform any Disney Product, including in connection with any use, creation, development, modification, prompting, fine-tuning, training, testing, benchmarking or validation of any artificial intelligence or machine learning tool";
  - §3.H prohibits commercial or business-related uses.
- **`hoopR` licence:** MIT — covers the **code only**; it grants **no rights to ESPN's data**.
- **Redistribution / use implications:** obtaining ESPN play-by-play by automated means and using it to test / benchmark / validate CourtGraph models is **prohibited** by the Disney terms.
- **Reachability result (2026-08-29):** `hoopR` GitHub + docs reachable; Disney terms retrieved successfully.
- **Verification date:** 2026-08-29.
- **Decision status:** **rejected / deferred** — usable only with a separately obtained, compatible data licence or written permission from ESPN/Disney.
- **Pilot check still required:** no (rejected for cycle 1).

### SRC-PST — prosportstransactions.com (Pro Sports Transactions Archive)

- **Provider:** Frank Marousek (independent editorial); the site asserts "all rights reserved" and offers no terms-of-use page or API.
- **Role in CourtGraph:** **human research lead only** — a place a person may look to find that a move happened, then verify and cite it independently. **Not imported, not systematically copied, not a base dataset.**
- **Canonical URL:** <https://www.prosportstransactions.com/basketball/>
- **Terms / policy URL:** none located (2026-08-29). A third-party client library states "usage of all information … is subject to all rights reserved by Pro Sports Transactions" — <https://github.com/rsforbes/pro_sports_transactions> (accessed 2026-08-29).
- **Access method:** web search form; no official API or bulk export.
- **Redistribution implications:** all rights reserved; nothing from this site is redistributed or stored as a dataset.
- **Reachability result (2026-08-29):** homepage returned HTTP 403 to the automated client; browsable by humans.
- **Verification date:** 2026-08-29.
- **Decision status:** **rejected as an imported dataset**; permitted only as an informal human lead.
- **Pilot check still required:** no.

### SRC-SPORTRADAR — official NBA licensed data feed

- **Provider:** Sportradar — the NBA's Official Data Provider; Second Spectrum / Genius Sports for official tracking — <https://www.nba.com/news/nba-extends-u-s-betting-data-partnerships-with-sportradar-and-genius-sports-group> (accessed 2026-08-29).
- **Role:** the **only** identified path to a genuinely independent, contractually clear data lineage — via a paid licence or a written research-permission arrangement.
- **Canonical / docs URL:** <https://developer.sportradar.com/basketball/reference/nba-overview>
- **Terms / license URL:** commercial contract + API terms (not reviewed here).
- **Access method:** authenticated REST API; paid tiers.
- **Redistribution implications:** governed by the paid licence; public redistribution of raw feed data is not part of standard tiers *(inference)*.
- **Decision status:** **rejected for cycle 1's default plan; retained as the optional permission/licence path** for independent validation and for a compliant release.
- **Reason:** cost and contract scope are disproportionate for a part-time student project, but it is the clean answer to the validation-lineage gap if resources allow.
- **Pilot check still required:** no.

---

## 5. Provider-versus-tool separation

**What supplies the raw records.** Only the NBA first-party surfaces
(`stats.nba.com`, `data.nba.com`) supply CourtGraph's raw event, box-score,
schedule, and roster records. Everything else in the registry is a **client**
that fetches those records, a **tool** that transforms them, a **re-host** of
them, or a source that has been **rejected** on terms grounds.

**What `nba_api` does and does not guarantee.** `nba_api` is an MIT-licensed
community client that maps NBA.com's undocumented endpoints and returns their
JSON. It **does not** provide the data, warrant its accuracy, grant any right to
use or redistribute it, guarantee endpoint stability ("NBA.com does not provide
information regarding new, changed, or removed endpoints"), or impose a rate
limit. CourtGraph owns throttling (§5.1), retry/backoff, schema-drift detection,
and Bronze persistence.

**What `pbpstats` does, and the provider differences it exposes.** `pbpstats`
parses raw play-by-play into possessions and stints, attaches lineup-on-floor to
every event, and corrects some out-of-order events. It makes the two NBA
providers' differences unavoidable:

- `stats.nba.com` — more history (PBP to 1996-97) but many out-of-order shots and
  rebounds needing manual fixes, and **no `offense_team_id`**, so possession
  attribution depends on correct event order;
- `data.nba.com` — PBP only from 2016-17, but `offense_team_id` on every event
  and fewer ordering problems.

Because possession counts and lineup states are **derived** under a policy, the
same raw game yields different possessions under different providers, parser
versions, and override sets. The research contract already requires that provider
differences in possession definitions be "documented and tested, not hidden"
(contract §24; master plan §7.2).

**Why parser output must not replace raw preservation.** Bronze payloads are
append-only and content-addressed (master plan §6.2). Possessions, stints, and
lineups are **Silver/Gold derivations** carrying `parser_version` and
`correction_set_version`. If `pbpstats` logic, an override file, or a provider
changes, CourtGraph must re-derive every downstream table from the unchanged raw
payloads and diff the result. Storing only parsed possessions would make the
pipeline unauditable and irreproducible.

### 5.1 Access conduct and self-imposed rate ceiling

The NBA publishes no rate limit. CourtGraph therefore adopts a conservative
**self-imposed ceiling** — this is a courtesy policy, **not** an official NBA
allowance:

- a **single worker**, no parallel request streams;
- **at least 1–2 seconds between requests** initially, tuned upward if the pilot
  shows any strain;
- **exponential backoff** on any error or slow response;
- **stop immediately** on any block, `HTTP 429`, or `HTTP 403`, and do not resume
  until a human has reviewed why;
- **never** rotate identities/IPs, spoof to evade blocks, solve anti-bot
  challenges, or otherwise bypass an access control;
- prefer the frozen local snapshot (SRC-SHUFINSKIY, local only) for repeated
  development runs so the live endpoints are hit as little as possible.

### 5.2 Interim validation stack (honest statement of the gap)

There is **no confirmed genuinely independent, freely usable provider lineage
with sufficiently clear rights**. The Disney/ESPN terms expressly bar use of ESPN
content in training, testing, benchmarking, or validation of AI/ML tools; the
Sports Reference terms bar training/fine-tuning/prompting/instructing AI systems
and supporting ML methods that predict or score, which CourtGraph reads
conservatively to also exclude using its data for validation (SRC-BREF). Every
free mirror re-hosts NBA data and inherits the NBA restrictions. Until a licensed
feed (SRC-SPORTRADAR) or written provider permission is obtained, CourtGraph's
validation is **within-lineage plus internal rigor**:

1. **Two NBA raw surfaces compared** — reconstruct possessions from both
   `stats.nba.com` and `data.nba.com` for 2016-17+ and diff possession counts,
   boundaries, and lineup states.
2. **Reconciliation to official NBA aggregates** — final score, period scores,
   team and player minutes, and (within methodology tolerance) team possession
   counts, from the same NBA lineage (master plan §6.5, §7.7).
3. **CourtGraph's independent parser/validator** — a from-scratch possession/stint
   state machine, not a wrapper around `pbpstats`, run in parallel and diffed
   against it.
4. **Documented manual audits of raw events** — event-by-event human review of
   ≥25 sampled games per season, oversampling OT, technical fouls, ejections,
   flagrant sequences, and audit-flagged games; every correction recorded in the
   patch table (master plan §6.2, §7.7).
5. **Acknowledged limitation** — this stack cannot catch a systematic error
   present in *both* NBA surfaces. That residual risk is stated in the data-quality
   report and in any publication. Closing it requires SRC-SPORTRADAR or
   equivalent written permission.

---

## 6. Coverage decision

Constraints (contract §9): NBA regular season **and** playoffs; **3** contiguous
development seasons; **6** contiguous first-cycle seasons ending at a fixed
cutoff; **completed seasons only**; expansion toward 8–10 seasons only after the
data-quality gates pass.

As of 2026-08-29 the last completed NBA season is **2025-26**.

| Window | Recommended seasons | Rationale |
|---|---|---|
| **Development (3)** | **2023-24, 2024-25, 2025-26** | Most recent; fully covered by both NBA surfaces (`data.nba.com` since 2016-17); post-COVID scheduling; smallest re-ordering burden. |
| **First research cycle (6, contiguous)** | **2020-21 → 2025-26** | Most recent contiguous six; entirely dual-surface, so the within-NBA possession-definition check (§5.2) runs for every game; only one structurally anomalous season (2020-21). |
| **Playoffs** | Ingest and **label** for all six seasons; **exclude from cycle-1 training by default**, reserving them for transport / robustness evaluation (contract §29 Q3). | Keeps regular season and playoffs from being silently pooled (master plan §6.3); leaves the playoff-transport test clean. |

**Both windows are provisional** until the pilot confirms, per season: 100% of
expected completed games present or explicitly excluded; final score from events
== official final score within tolerance; every model possession has exactly five
identifiable players per side; player-seconds ≈ box-score totals within tolerance
(master plan §6.5).

### Structurally unusual seasons / breaks to account for

- **2020-21** — 72-game season (not 82), heavily compressed schedule, limited or
  no attendance for much of the year, **play-in tournament introduced**. Highest
  comparability risk in the window. Handle via the Tier-A era/context controls
  (contract §10) and the required robustness runs (contract §19); do **not** drop
  it silently.
- **2021-22** — early-season COVID (Omicron) postponements and hardship
  contracts; minor.
- **2023-24 onward** — **In-Season Tournament / NBA Cup**: group-play games count
  as regular-season games; the Cup Championship game does **not** count toward
  regular-season statistics while the earlier knockout games do. Season-phase
  labelling must capture this.
- General **pace and rule-environment drift** across 2020-21 → 2025-26 — covered
  by the mandatory season/era + pace controls (contract §10).

**Robustness variant (recommended, not binding):** repeat headline results on the
**clean 5-season sub-window 2021-22 → 2025-26** to check that conclusions do not
depend on 2020-21. A COVID-free *contiguous six-season* window does not exist in
the available data (shifting earlier to 2019-20 → 2024-25 adds a second anomalous
season, the 2019-20 bubble). Keep 2020-21 in, handled by the era/context
controls, and rely on the 5-season robustness check; revisit only if the pilot
shows 2020-21 fails its comparability audits outright.

---

## 7. Transaction-data decision

**Approach:** a **manually curated event cohort**, not an imported dataset. For
each transaction admitted to the T4 backtest (contract §12), a person records:

- the players and teams involved and the destination team, resolved to
  `stats.nba.com` IDs (ambiguous matches flagged for review — master plan §6.4);
- the **transaction / announcement timestamp**, cited to an **official NBA or
  team press release** plus **at least one dated contemporaneous report**;
- the **debut timestamp**, computed from CourtGraph's own play-by-play as the
  player's first game (and first shared possession with each core teammate) for
  the destination team;
- a per-event **source citation list**.

**Sources and their roles:**

- **Official NBA / team communications** and **dated contemporaneous reporting** —
  the required evidence for every event.
- **`prosportstransactions.com`** — a **human research lead only**: a person may
  consult it to notice that a move occurred, then find and cite the official
  release and reporting independently. Nothing from the site is stored, imported,
  or systematically copied (all rights reserved).
- **Basketball-Reference** — **not used** (SRC-BREF; its terms bar AI-training use, and CourtGraph conservatively excludes it from validation and cohort construction too).

**Leakage rule (contract §27; master plan §25):**

- the model's **information cutoff must be immediately before the transaction
  event** (the announcement/execution), never at the debut;
- the **outcome window may begin at the debut** and run for the defined
  post-move horizons;
- **debut alone is not a leakage-safe pre-transaction cutoff** — between
  announcement and debut, the destination roster and expectations are already
  known, so a model cut at debut would have seen post-decision information.

**Timestamp precision and hazards:**

- day-level precision is expected; where announcement and official execution
  differ materially (e.g. trades finalised days after agreement), record both and
  use the **earlier** (announcement) as the conservative cutoff;
- name-based identity hazards: common-name collisions, mid-career name changes,
  Jr./Sr. suffixes — all flagged for manual review;
- 10-day / two-way / G-League-only moves are a **separate class**, handled apart
  from the core trade/signing cohort.

**Eligibility gates before a transaction enters the cohort** (master plan §25.2):
sufficient pre-move NBA data; a minimum post-move sample; the destination roster
reconstructable from CourtGraph's own data at the cutoff; the evaluation window
not confounded by an unrelated major roster change (else censor or label).

**Public release of the compiled transaction table:** **deferred** pending the
legal review — it is a derived, curated dataset and its release status follows
the same review as other row-level artifacts (§1).

---

## 8. Required pilot checks before provisional choices become binding

No **provisional** choice in §1, §6, or §7 — the primary provider per era, the
season windows, playoff handling, the transaction cohort's size and depth —
becomes binding until these checks pass. The safeguards marked *binding now* in
§1 (the pbpstats-as-tool + independent-validator approach, Bronze immutability
and content-addressing, the §5.1 access policy, the restricted release scope, and
the source rejections) apply immediately and do **not** wait on the pilot. The
pilot is read-mostly — a few dozen games under the §5.1 policy, not a bulk
ingest.

1. **Access pilot.** From the intended environment, exercise `stats.nba.com` and
   `data.nba.com` via `nba_api` under §5.1; record actual behaviour, including
   any block/`429`/`403`. If blocked, switch to the local frozen snapshot
   (SRC-SHUFINSKIY) and re-plan the ingestion approach.
2. **Schema snapshot.** Capture and hash the response schema of every endpoint in
   §2 for one game and one season; store as the v0 schema contract.
3. **Possession reconciliation.** For ~30 games spanning the window (oversampling
   OT, ejections, flagrant/technical sequences), reconstruct possessions from
   both NBA surfaces with `pbpstats` and with CourtGraph's independent validator;
   diff possessions, stint boundaries, lineup states; reconcile to official
   totals (§5.2).
4. **Completeness check.** For each of the six seasons, confirm game counts
   against the schedule and flag gaps.
5. **Transaction sample.** Manually build the cohort for two seasons per §7;
   record, for each event, the citation set and the announcement→debut gap;
   produce eligibility counts.
6. **Consent & release review.** Decide whether to submit a written
   research-permission request to the NBA; obtain a lawyer's read on (a) a local,
   non-commercial, non-redistributed research build and (b) the §1 release scope
   (code + schemas + fixtures + methodology + aggregates only).
7. **Independent-lineage decision.** Decide whether project resources allow a
   licensed SRC-SPORTRADAR feed or an equivalent written permission to close the
   validation-lineage gap (§5.2); if not, record the residual risk explicitly.

---

## 9. Open questions carried forward

Mapped to `RESEARCH_CONTRACT.md` §29. Resolved items update the contract by
amendment; unresolved items stay here.

| Contract §29 item | Status after this assessment |
|---|---|
| 1. Exact seasons / cutoff | **Provisionally resolved:** dev = 2023-24…2025-26; cycle = 2020-21…2025-26; provisional pending the §8 pilot. |
| 2. Primary PBP provider + secondary for cross-validation | **Provisionally resolved:** primary `stats.nba.com` (all eras) via `nba_api`; secondary `data.nba.com` (2016-17+). **No genuinely independent third lineage** is available for free; SRC-SPORTRADAR or written permission is the only clean option (§5.2). |
| 3. Playoffs in training? | **Provisional recommendation:** ingest + label all six seasons; exclude from cycle-1 training by default. |
| 4. Transaction source + date reliability + eligibility counts | **Provisionally resolved:** a manually curated cohort evidenced by official releases + dated reporting; `prosportstransactions.com` is a human lead only; announcement timestamp is the leakage cutoff, debut only starts the outcome window; eligibility counts from the §8 pilot. |
| 5. Box-score / shot-profile feature source | **Resolved:** `stats.nba.com` box-score + shot-chart endpoints (same lineage as PBP). |
| 6. Legal / licensing constraints on redistribution | **Partly resolved, partly open.** Resolved: NBA statistics are for news-reporting / private non-commercial use, and archived PBP is restricted; the NBA "comprehensive, regularly updated database" restriction addresses websites/products/services and *may plausibly* reach CourtGraph's private dataset (legal review needed). Sports Reference bars training/fine-tuning/prompting/instructing AI systems and supporting ML methods that predict/classify/label/score without permission (CourtGraph *infers* validation/benchmarking use is also barred). Disney/ESPN bars automated extraction and use in training, testing, benchmarking, or validation of AI/ML tools. Open: whether NBA written consent is sought/granted; the lawyer's read on a local research build and on the §1 release scope; the Kaggle `nbadb` dataset licence. |
| 7. Numeric minimum-exposure thresholds | Still deferred to the descriptive stage (needs variance estimates). |
| 8. Leaderboard interval-width threshold, §17 magnitudes | Still deferred to pilot baselines. |
| 9. Garbage-time definition, possession-boundary variants | Partly informed: the two NBA surfaces give at least two boundary lineages to test; the exact deterministic rule is deferred to master plan §7.2 / §7.6 work. |
| 10. Feasible number of rolling-origin folds | Informed by the 6-season window; exact count deferred to the descriptive stage. |

---

## 10. Final recommendation

Proceed to a small **data pilot** (§8), not a build. Its purpose is to (a) confirm
the two NBA surfaces are usable from the intended environment under the §5.1
policy, (b) prove CourtGraph's independent possession validator against official
totals and the second NBA surface, (c) test the manual transaction workflow on
two seasons, and (d) get a legal read on a local research build and on the
restricted release scope, plus a decision on requesting NBA written consent.

Hold as **binding** only: the pbpstats-as-tool + independent-validator approach;
Bronze immutability; the §5.1 access policy; the release scope in §1 (code,
schemas, fixtures, methodology, aggregates only — no Bronze, no row-level
NBA-derived data); and the rejections of Basketball-Reference, ESPN/`hoopR`, and
`prosportstransactions.com`-as-a-dataset.

Everything else — the season windows, the primary provider per era, playoff
handling, the transaction cohort — stays **provisional** until the pilot and the
legal review report back. The absence of a free independent validation lineage is
a **known, documented gap**, not a solved problem.

---

## Sources (all accessed / reviewed 2026-08-29)

- **NBA Terms of Use** — <https://www.nba.com/termsofuse>. Official policy; the
  page blocks automated retrieval (timed out from the assessment client). Findings
  (statistics for news-reporting / private non-commercial use; archived
  play-by-play restricted; express prior consent required for use connected to a
  website/product/service featuring a comprehensive, regularly updated statistics
  database) reflect direct review of the official page. Whether CourtGraph's
  private research dataset is covered is unclear — legal review needed (§1 note).
- **Disney / ESPN Terms of Use** — <https://disneytermsofuse.com/english/>.
  Retrieved in full; **effective 2024-05-24**. §2.B.x (automated means / data
  mining / web scraping / AI Tool), §2.A (Consumer License — no reproduce /
  distribute / transform, incl. training / testing / benchmarking / validation of
  AI/ML tools), §3.H (no commercial use) quoted verbatim in SRC-HOOPR-ESPN.
- **Sports Reference Terms of Use** — <https://www.sports-reference.com/termsofuse.html>;
  **Data Use** — <https://www.sports-reference.com/data_use.html>. Official
  policy; both pages return HTTP 403 to an automated client. Finding, from direct
  review: content may not be used to train / fine-tune / prompt / instruct AI
  systems, or to support ML methods used to predict / classify / label / score,
  without permission; no scraping or tools from scraped data without permission.
  The policy does **not** expressly list "testing", "benchmarking", or
  "validation" — CourtGraph's exclusion of validation/benchmarking use is its own
  conservative inference (SRC-BREF), not quoted policy.
- **`nba_api`** — <https://github.com/swar/nba_api>; licence <https://github.com/swar/nba_api/blob/master/LICENSE>; releases <https://github.com/swar/nba_api/releases>.
- **`pbpstats`** — <https://github.com/dblackrun/pbpstats>; docs <https://pbpstats.readthedocs.io/>; quickstart (provider differences) <https://github.com/dblackrun/pbpstats/blob/main/docs/quickstart.rst>; licence <https://github.com/dblackrun/pbpstats/blob/main/LICENSE>.
- **`shufinskiy/nba_data`** — <https://github.com/shufinskiy/nba_data>; <https://github.com/shufinskiy/nba_data/blob/main/description_fields.md>.
- **`wyattowalsh/basketball` / `nbadb`** — <https://www.kaggle.com/datasets/wyattowalsh/basketball>; <https://github.com/wyattowalsh/nbadb>.
- **`hoopR`** — <https://github.com/sportsdataverse/hoopR>; <https://hoopr.sportsdataverse.org/>.
- **`prosportstransactions.com`** — <https://www.prosportstransactions.com/basketball/>; third-party client <https://github.com/rsforbes/pro_sports_transactions>.
- **NBA game-ID format** (community documentation) — <https://nba-stats-tracking.readthedocs.io/en/latest/source/modules.html>.
- **Sportradar as Official Data Provider** — <https://www.nba.com/news/nba-extends-u-s-betting-data-partnerships-with-sportradar-and-genius-sports-group>; Sportradar NBA API <https://developer.sportradar.com/basketball/reference/nba-overview>.

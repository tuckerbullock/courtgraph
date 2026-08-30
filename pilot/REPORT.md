# NBA data-access & schema pilot — report

> **Date:** 2026-08-29, then a third revision 2026-08-30 (Codex review round 3:
> probe §5.1 immediate-stop, hard-stop manifest audit trail, dynamic-key
> limitation, four new offline tests)
> **Scope:** `DATA_SOURCES.md` §8 steps **1 (access)** and **2 (schema
> fingerprint)** only, for a *few* representative completed games. **Not** a bulk
> download, six-season completeness sweep, 30-game possession reconciliation,
> transaction study, or ingestion build.
> **Branch / worktree:** `task/data-pilot`, isolated from `task/data-sources`
> (PR #3, approved, unmerged — preserved untouched).

## What `pilot/` is

| File | Purpose | Deps |
|---|---|---|
| `nba_access_pilot.py` | §8 steps 1–2 orchestration: fetch a few endpoints under §5.1, split each response into a tracked contract + an ignored manifest | stdlib; network path needs `requests==2.34.2` via `uv run --with` |
| `probe_access.py` | one-shot reachability probe — at most two single `requests.get` calls, no retry, no orchestration; stops before the second provider on a first-provider HTTP 403/429 (§5.1) | `requests==2.34.2` via `uv run --with` |
| `test_pilot.py` | offline checks (fingerprinter units + end-to-end output separation) | stdlib only |
| `schema/` | tracked output location for the `v0_schema_contract.json` (not created yet) | — |
| `_local/` | raw payloads + `run_manifest.json` | gitignored |

Nothing is added to the project runtime.

## Two outputs, cleanly separated

Every run of `nba_access_pilot.py` produces:

- **Schema contract** → `--out PATH` (default gitignored; pass
  `pilot/schema/v0_schema_contract.json` for the tracked file). Contains only:
  `contract_version`, and per **stable logical alias** (`stats_scoreboard`,
  `stats_playbyplay_01`, `stats_boxscore_traditional_01`, `data_scoreboard`,
  `data_playbyplay_01`, … / `snapshot_01`, `snapshot_02`, …) a value-free
  fingerprint — recursive `schema` (dict keys, list-item structure, scalar type
  categories), `schema_sha256`, and an `nba_view` (column names + type
  categories) for `resultSets` payloads. **No dates, game IDs, event text,
  scores, filenames, or local paths.** Array/row counts are not in it.
- **Run manifest** → always `pilot/_local/run_manifest.json` (**gitignored**).
  Contains the provenance: `mode`, `date`, real `game_ids`, per-request
  URLs/params/status, raw filenames, alias→file mapping, and `completeness`
  (array lengths). Never committed.

`test_pilot.py::EndToEndSeparationTests` builds and serializes both objects into
a temp dir (live-shaped and snapshot) and asserts the contract contains none of
`0022300001`, `2023-12-25`, `Jump Ball`, clock/score values, or the temp path,
while the manifest carries the game ID, the date, and the snapshot dir.

## Evidence — access (this host, 2026-08-29)

**Verified method:** two **separate, manual, one-shot** `requests.get(url,
headers, timeout=30)` calls — one per provider, ~2 s apart, **no retry loop and
no orchestration** — issued from an ad-hoc script equivalent to the included
`pilot/probe_access.py`, run as `uv run --with requests==2.34.2 python <script>`.
(The first call was a timeout, not a 403/429, so proceeding to the second
provider was permitted under §5.1; the included probe now enforces that rule —
a first-provider 403/429 stops it before the second call.)

| Provider | Result |
|---|---|
| `stats.nba.com/stats/scoreboardv2?GameDate=2024-01-15…` | **Read timeout at 30 s** — TCP connects, no response body. |
| `data.nba.com/data/10s/prod/v1/20240115/scoreboard.json` | **HTTP 403** in 0.2 s, `application/xml` (263 B, edge "AccessDenied"). |

**`nba_access_pilot.py`'s live orchestration has never been executed** (against
the network or otherwise for a real run). It retries a timeout `MAX_TRIES` times
with backoff and hard-stops before reaching the second provider, so it **could
not** have produced the one-timeout-then-one-403 sequence above — that came from
the manual one-shot probes only. **No NBA JSON response body has been received
by any code in this branch.**

## Evidence — offline

- **Recursive fingerprint** (`test_pilot.py`, 12 stdlib tests): includes nested
  event fields at every depth; excludes cell values; **stable** when scalar
  values *and* array lengths change but structure holds; **changes** when a
  field is added or a scalar's type category changes; deterministic /
  key-order-independent.
- **Output separation** (same file): live-shaped and snapshot runs each produce
  a provenance-free tracked contract and a provenance-carrying ignored manifest,
  asserted token-by-token.
- **§5.1 immediate stop** (same file): a first-provider HTTP 403/429 makes
  `probe_access.main()` return without contacting the second provider; a
  first-provider timeout still proceeds to the second one-shot request.
- **Hard-stop audit trail** (same file): a live `nba_access_pilot` run whose
  first endpoint times out `MAX_TRIES` times exits 2 **and** still writes
  `run_manifest.json` with `mode`, `date`, every attempted request
  (url/status-or-error details), and the `stopped` reason; no schema contract is
  written.
- **Snapshot path** works with no network: `*.json` → recursive fingerprinter,
  `*.csv` → header-only column fingerprint; alias→relative-file mapping stays in
  the manifest.
- **Client library**: `nba_api==1.11.4` installs via
  `uv run --with nba_api==1.11.4` and its endpoints import (`scoreboardv2`,
  `playbyplayv2`, `boxscoretraditionalv2`, `commonteamroster`). Constructing an
  endpoint triggers a live request, so this was **not** done.
- **TLS note**: this machine's bare-stdlib `python3` has no CA bundle
  (`urllib` → `CERTIFICATE_VERIFY_FAILED`); the network path uses `requests`
  (bundles `certifi`), and any future client must too.
- **Static checks**: `py_compile` OK for all three modules every round. `ruff
  check` / `ruff format --check` and `mypy --strict` were clean on the prior
  revision; the 2026-08-30 safety revision (probe stop + hard-stop manifest +
  four new tests) was written to the same config but `ruff`/`mypy` were **not
  re-run** this round (offline, per the instruction to run only the focused
  tests and `git diff --check`). Re-run before merge.

## Findings

1. **Access from this host is blocked** — `stats.nba.com` times out,
   `data.nba.com` returns 403. Consistent with the datacenter-IP blocking noted
   in `DATA_SOURCES.md` (SRC-NBASTATS). An **environment** result.
2. **Live endpoint compatibility and `nba_api` functional behaviour are
   UNVERIFIED.** No NBA JSON response has been received, so the actual response
   shapes, required fields, the `data.nba.com` historical-PBP URL pattern, and
   `nba_api`'s behaviour on live data are all unconfirmed. `nba_api` installs and
   imports — that is the only client fact.
3. **The fingerprinter and the two-object split are proven only offline** —
   against synthetic and snapshot input. They have **not** run against a real
   NBA response. The instrument is *drafted and offline-validated*, **not**
   "ready".
4. **The offline / snapshot path is usable now.**

## Limitations

- Single host, single day, two manual probe requests — no claim about NBA
  availability generally.
- §8 step 2 deliverables (live response shape, required fields, completeness)
  remain **open**; `pilot/schema/v0_schema_contract.json` is **not** created.
- Endpoint parameters and the `data.nba.com`
  `…/mobile_teams/nba/{season}/scores/pbp/{gid}_full_pbp.json` pattern are
  community conventions, unverified live.
- `nba_view` column-type logic is exercised only on synthetic `resultSets`.
- The fingerprinter records **every dict key**. The sampled endpoints key data
  under fixed structural names (`resultSets` / `rowSet` arrays; `data.nba.com`
  `{"g": {"pd": [...]}}`), so no value becomes a key. If a real response instead
  keys an object by a date or game ID, that token would enter the tracked
  `schema`. The first real run must confirm the emitted contract has no such
  dynamic keys before it is trusted.

## Changed files (branch `task/data-pilot`, uncommitted)

- `pilot/nba_access_pilot.py` — new; §8-steps-1–2 orchestration + value-free
  fingerprinter + tracked-contract / ignored-manifest split + partial-manifest
  write on hard stop.
- `pilot/probe_access.py` — new; the one-shot reachability probe (stops before
  the second provider on a first-provider 403/429).
- `pilot/test_pilot.py` — new; 12 offline tests — fingerprinter units,
  end-to-end separation, probe §5.1 stop, hard-stop manifest.
- `pilot/schema/README.md` — new; documents the tracked contract path (file not created).
- `pilot/.gitignore` — new; keeps `pilot/_local/` uncommitted.
- `pilot/REPORT.md` — new; this report.
- `docs/CURRENT_TASK.md` — updated at this milestone.

No change to `DATA_SOURCES.md`, `RESEARCH_CONTRACT.md`, `docs/PROJECT_STATUS.md`,
`src/`, `tests/`, `pyproject.toml`, `uv.lock`, or `.github/`. No dependency added
to the project runtime. No raw NBA data in the repo.

## Recommendation for Codex review

- **`DATA_SOURCES.md` v0.3 stands** — nothing here contradicts it — but provider
  and `nba_api` viability is **unproven**. Confirm from a residential connection
  (or a representative snapshot response) before the full pilot.
- **Next step (do not start yet):** run
  `pilot/nba_access_pilot.py --out pilot/schema/v0_schema_contract.json` from a
  residential network for ~3–5 completed games (normal / OT / playoff), **or**
  the same with `--snapshot-dir` against a pinned SRC-SHUFINSKIY extract. The run
  must exercise the recursive fingerprinter on a **real** NBA response before the
  contract is trusted or the instrument is called ready. Verify the emitted
  `v0_schema_contract.json` is provenance-free (the tests assert the mechanism;
  a human should still eyeball a real one). Then §8 step 3.
- Raw NBA payloads and the run manifest stay out of the repo on every path.

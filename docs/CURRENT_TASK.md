# Current Task

Last updated: 2026-08-30

## State

Done — small NBA **data-access & schema pilot** (`DATA_SOURCES.md` §8 steps
1–2, narrow scope). Revised three times after Codex review (round 3: probe §5.1
immediate-stop, hard-stop manifest audit trail, dynamic-key limitation note,
four new offline tests). **Codex-approved 2026-08-30.** Committed on
`task/data-pilot` and opened as a stacked PR targeting `task/data-sources`.

Isolated on branch `task/data-pilot` in a separate worktree, stacked on
`task/data-sources` (PR #3, Codex-approved, unmerged — preserved untouched).
Earlier tasks (dev environment + CI, research contract) are merged to `main`
(PRs #1, #2).

## Objective

Test respectful access to `stats.nba.com` and `data.nba.com` for a few
representative completed games under the `DATA_SOURCES.md` §5.1 policy, capture
what schema we can as a **value-free** contract, and judge whether the proposed
providers and `nba_api` are technically viable for the later full pilot.
Explicitly **not**: bulk download, six-season completeness, 30-game possession
reconciliation, transaction study, or ingestion architecture.

## Outcome

- **Access from this host is blocked** — `stats.nba.com` read-times-out at 30 s;
  `data.nba.com` returns HTTP 403. Matches the datacenter-IP blocking in
  `DATA_SOURCES.md` (SRC-NBASTATS).
- **Verified probe method:** two **separate manual one-shot** `requests.get`
  calls (one per provider, ~2 s apart, no retry, no orchestration), equivalent
  to the included `pilot/probe_access.py`. The first call timed out (not a
  403/429), so proceeding to the second provider was permitted under §5.1; the
  probe now **enforces** that — a first-provider 403/429 stops it before the
  second call. The `nba_access_pilot.py` live orchestration **has never been
  run** — it retries a timeout and hard-stops before reaching the second
  provider, so it could not have produced the observed sequence. **No NBA JSON
  response has been received by any code here.**
- **Live endpoint compatibility and `nba_api` behaviour are UNVERIFIED.**
  `nba_api==1.11.4` installs and its endpoints import — the only client fact.
- **Two outputs, cleanly separated (proven offline only):**
  - tracked **schema contract** (`--out`): `contract_version` + per stable alias
    (`stats_scoreboard`, `stats_playbyplay_01`, …) a value-free fingerprint
    (recursive schema, `schema_sha256`, `nba_view`). **No** dates, game IDs,
    event text, scores, filenames, paths, or counts.
  - ignored **run manifest** (`pilot/_local/run_manifest.json`): mode, date, real
    game IDs, request details, raw filenames, alias→file map, array counts.
    Now **also written on a hard stop** (partial: mode, date, game IDs resolved
    so far, every attempted request with status/error details, stop reason) so
    a failed run still leaves an audit trail.
- **Fingerprinter + split + safety rules proven offline** by `pilot/test_pilot.py`
  (12 stdlib tests: fingerprinter units, end-to-end serialization of both
  objects with token-level leak assertions, probe §5.1 immediate-stop on
  403/429, hard-stop partial-manifest preservation). **Not** run against a real
  NBA response — instrument is *drafted*, not "ready".
- **Offline / snapshot path** (`--snapshot-dir`) works with no network.

Full evidence, limitations, recommendation: `pilot/REPORT.md`.

## Files changed (branch `task/data-pilot`)

- `pilot/nba_access_pilot.py` — new; §8-steps-1–2 orchestration, value-free
  fingerprinter, tracked-contract / ignored-manifest split, `--out`,
  partial-manifest write on hard stop.
- `pilot/probe_access.py` — new; one-shot reachability probe (≤2 requests, no
  retry; stops before the second provider on a first-provider 403/429).
- `pilot/test_pilot.py` — new; offline fingerprinter + end-to-end separation +
  probe-stop + hard-stop-manifest tests (12 total).
- `pilot/schema/README.md` — new; documents the tracked contract path
  (`v0_schema_contract.json` not created — no NBA response yet).
- `pilot/.gitignore` — new; keeps `pilot/_local/` (raw payloads + `run_manifest.json`) uncommitted.
- `pilot/REPORT.md` — new; pilot report.
- `docs/CURRENT_TASK.md` — this handoff.

No change to `DATA_SOURCES.md`, `RESEARCH_CONTRACT.md`, `docs/PROJECT_STATUS.md`,
`src/`, `tests/`, `pyproject.toml`, `uv.lock`, or `.github/`. No dependency added
to the project runtime. No raw NBA data in the repo.

## Verification (focused, offline)

Round 3 (2026-08-30), per the user's instruction to run only focused offline
tests and `git diff --check`:

- `python3 pilot/test_pilot.py`: 12/12 pass (stdlib only).
- `python3 -m py_compile pilot/*.py`: OK.
- `git diff --check`: clean.

From the prior revision rounds (not re-run this round):

- `ruff check` + `ruff format --check` clean; `mypy --strict` clean. Round-3
  code was written to the same config but `ruff`/`mypy` were **not** re-run
  (no network for `uv run`; instruction limited this round to the focused
  tests + `git diff --check`). Re-run before merge.
- Manual `--snapshot-dir` run: contract carries only aliases + value-free
  fingerprints; manifest carries the game-id-bearing filename + local path;
  `grep` confirms no provenance in the contract.

## Known limitations (accepted at approval)

- Provider + `nba_api` **live viability is unproven** — the only observed fact is
  this host is blocked.
- The fingerprinter and the two-object split are validated only against
  synthetic / snapshot input; a **real** NBA response must exercise them before
  `v0_schema_contract.json` is trusted.
- The fingerprinter records **every dict key**. The sampled endpoints key data
  under fixed structural names, so no value becomes a key — but a response that
  keyed an object by a date or game ID would leak that token into the tracked
  `schema`. The first real run must confirm the emitted contract has no dynamic
  keys (now noted in `pilot/REPORT.md` and `pilot/schema/README.md`).
- Endpoint params and the `data.nba.com` historical-PBP URL pattern are
  community conventions, unverified live.

## Next action

Codex approved (2026-08-30); pilot committed on `task/data-pilot` and opened as
a PR stacked on `task/data-sources` (PR #3). Merge order: PR #3 into `main`
first, then this PR. The next single task — a residential or snapshot run
producing `pilot/schema/v0_schema_contract.json`, then §8 step 3 — does not
begin until the user activates it. `ruff`/`mypy` should be re-run in CI or
locally with `uv` before merge (not run in the offline revision round).

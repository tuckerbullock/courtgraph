# `pilot/schema/` — tracked schema-contract output

This directory is the **explicit tracked output path** for the pilot's
value-free schema contract (`DATA_SOURCES.md` §8 step 2).

`v0_schema_contract.json` is **not created yet** — no NBA JSON response has been
received (see `../REPORT.md`). It is produced by pointing `--out` here, from a
residential network **or** an offline snapshot:

```bash
uv run --with requests==2.34.2 pilot/nba_access_pilot.py \
    --date 2023-12-25 --max-games 3 \
    --out pilot/schema/v0_schema_contract.json

# or, offline, against a pinned SRC-SHUFINSKIY extract:
python3 pilot/nba_access_pilot.py \
    --snapshot-dir /path/to/shufinskiy_extract \
    --out pilot/schema/v0_schema_contract.json
```

## What the contract contains — and does not

The contract holds **only**:

- `contract_version`;
- per **stable logical alias** (`stats_scoreboard`, `stats_playbyplay_01`,
  `stats_boxscore_traditional_01`, `data_scoreboard`, `data_playbyplay_01`, … or
  `snapshot_01`, `snapshot_02`, …): a recursive value-free `schema` (dict keys,
  list-item structure, scalar type categories), a `schema_sha256`, and — for NBA
  `resultSets` payloads — an `nba_view` of column names + type categories.

It contains **no** dates, game IDs, event descriptions, scores, clock values,
source filenames, or local paths. Array/row counts are **not** in it either.

One caveat for the first real run: the fingerprinter keeps every dict key. The
sampled endpoints use fixed structural keys, so no value becomes a key — but a
response that keyed an object by a date or game ID would put that token in the
`schema`. Eyeball the first emitted contract for dynamic keys before trusting it.

## Where the provenance goes

Everything else — `mode`, `date`, real `game_ids`, request URLs/params, raw
filenames, the alias→file mapping, and array-length `completeness` — is written
to `../_local/run_manifest.json`, which is **gitignored and never committed**.
Raw payloads live in `../_local/raw/` (also gitignored).

If a live run hits a §5.1 hard stop (HTTP 403/429 or repeated timeouts), no
contract is written, but `../_local/run_manifest.json` still gets a partial
record — `mode`, `date`, game IDs resolved so far, every attempted request with
its status/error details, and the `stopped` reason — as the failure's audit
trail.

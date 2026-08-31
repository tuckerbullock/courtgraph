# Current Task

Last updated: 2026-08-31

## State

Done — **local browser app** on `task/local-lineup-app`, off `origin/main`
(`ca63f38`). Implemented by Codex (user-authorized), then brought into the
repository, re-verified, and finished by Claude Code; committed and pushed, PR
open. No monitoring, data downloads, or public deployment occurred.

Claude finished one gap Codex left: the synthetic sandbox pool was too small
for the cross-fitted interaction fit to clear its out-of-fold selection gate,
so every "interaction surplus" was identically zero and the A/B comparison was
meaningless. The sandbox now generates ~12.7k deterministic stints (fits in
~7 s at startup) and recovers a non-zero interaction component, with a
regression test.

## Delivered

- `courtgraph app`: local browser app, bound only to `127.0.0.1` (default 8765).
- Game explorer reads an explicitly supplied ingest output and optional names
  sidecar. It verifies the stint checksum and accepted-game exposure/date
  consistency. Game, offensive-team, player, and minimum-possession filters;
  observed offensive lineup rates; source, score-check, and exclusion panels.
- Whole-archive conversion via `snapshot-from-shufinskiy --all-games`, with
  coverage recorded in provenance. The local 2025 playoff archive has 84 games:
  83 complete inputs and attempted, 62 accepted, 21 quarantined, and one missing
  a required input. The UI exposes every game and exact failure category rather
  than presenting the old three-game batch as the available dataset.
- Synthetic sandbox fits a deterministic model (~12.7k stints) in memory at
  startup (~7 s). Two independently chosen fives share an opposing five and
  context; compare talent, interaction, context, offensive value, individual
  training support, and approximate interaction intervals. No real observations
  enter this model.
- Fixed asset routes, origin/host checks, bounded JSON requests, and no file
  browsing, upload, telemetry, external assets, or new dependencies.

## Run

```bash
uv run courtgraph app
```

Then open `http://127.0.0.1:8765`. Stop with Ctrl+C. To use the complete local
playoff archive (the `_shufinskiy_src` archive is gitignored local-dev data,
not in the repo):

```bash
uv run courtgraph snapshot-from-shufinskiy \
  --archive-dir data/nba_snapshots/_shufinskiy_src --all-games \
  --out-dir data/nba_snapshots/all_2025_playoffs/snap
uv run courtgraph ingest \
  --snapshot-dir data/nba_snapshots/all_2025_playoffs/snap \
  --out-dir data/nba_snapshots/all_2025_playoffs/out
uv run courtgraph app \
  --ingest-dir data/nba_snapshots/all_2025_playoffs/out \
  --names data/nba_snapshots/all_2025_playoffs/snap/display_names.json
```

The generated local dataset has 3,325 stints, 10,852 accepted possessions, 1,236
recorded exclusions, 16 teams, and 210 players. It remains gitignored and the
three source CSV hashes are unchanged. Provenance records converter
`cg-shufinskiy/2` and the pinned source commit.

## Verification

Re-run by Claude Code on the committed branch (`uv` environment, Python 3.13):

- `uv sync --locked` — 15 packages, no drift.
- `uv run courtgraph doctor` — healthy.
- `uv run python -m unittest discover -s tests -v` — **155 tests passed**
  (1 new regression test for the sandbox interaction signal).
- `uv run python -m compileall -q src tests` — OK.
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy` — clean, 45 source files.
- `node --check src/courtgraph/app/static/app.js` — passed.
- `PYTHONPATH=src python3 -m unittest discover -s tests` — 155 OK (33 skipped:
  no `numpy` on that interpreter).
- **Full-archive pipeline, end to end on the local archive:**
  `snapshot-from-shufinskiy --all-games` → coverage `archive_games=84`,
  `complete_games=83`, one game missing `datanba_po_2024.csv`; `ingest` →
  62 accepted, 21 quarantined, 3,325 stints, 10,852 accepted possessions,
  1,236 excluded; `courtgraph app --ingest-dir …` → `/api/state` reports 84
  games, 16 teams, 210 players, matching coverage counts, real player names
  resolved.
- `courtgraph app` (sandbox only): `/api/state` and `/api/compare` return the
  synthetic catalog and decomposition; `Host: evil.com` → 403.
- Codex's Browser tool was blocked by its admin policy; the coverage screen's
  underlying data is verified above, a manual look at the rendered UI is still
  worthwhile but no longer a blocker.

## Boundaries and next action

This is one postseason observational archive plus a fictional sandbox. It is
not a validated NBA model, complete dated-roster optimizer, injury forecast,
or betting product. Raw observed rates exclude quarantined/split possessions
and do not apply garbage-time weights. Training support in the sandbox is
individual exposure, not evidence that the exact selected five has a large
sample. Bootstrap intervals concern interaction only, not total performance.

Next candidate task (not started): improve the importer for the 21 quarantined
playoff games. The next scientific milestone remains permitted multi-season NBA
data validation and chronological baseline evaluation. The research contract and
the parked `task/schema-contract` worktree are unchanged. `MASTER_PLAN.md`
section 44 (appended future capabilities) is already on `main` (PR #9).

# Current Task

Last updated: 2026-08-29

## State

`DATA_SOURCES.md` **v0.3 approved by Codex**. Committed on `task/data-sources`
and opened as a PR into `main`; not merged. Prior tasks (dev environment + CI,
research contract) are merged to `main` (PRs #1, #2).

## Objective

Create a root-level `DATA_SOURCES.md` deciding how CourtGraph lawfully,
reproducibly, and reliably obtains cycle-1 data — a research and decision
document, not an ingestion implementation. All source records, the rubric, and
the binding / provisional / deferred split live in that file (see §1 and §10).

## Files changed

- `DATA_SOURCES.md` — the source registry and selection decision (revised to v0.3).
- `docs/CURRENT_TASK.md` — this handoff.
- `docs/PROJECT_STATUS.md` — record the merged research contract and the
  in-progress data-source decision; correct the stale "next task" line.

No code, `pyproject.toml`, `uv.lock`, `.github/`, `README.md`, or
`RESEARCH_CONTRACT.md` changed. No data acquired, no ingestion code, no ADR.

## What the v0.3 revision changed (applying Codex review #2)

- **Sports Reference:** its terms prohibit training / fine-tuning / prompting /
  instructing AI systems and supporting ML methods that predict / classify /
  label / score — they do **not** expressly list "testing, benchmarking,
  validation". The document now quotes that scope and labels the extension to
  model validation/benchmarking as **CourtGraph's conservative inference**, not
  quoted policy. Basketball-Reference stays rejected from the pipeline.
- **NBA:** the terms address use connected to a website, product, or service
  featuring a comprehensive, regularly updated statistics database. The document
  no longer says CourtGraph's fixed-cutoff six-season private research dataset is
  "exactly" covered — only that it **may plausibly** fall within the restriction,
  needing legal review and possibly express consent. The conservative
  public-release policy is retained: no Bronze or row-level NBA-derived data
  without clearance.
- **§8 contradiction resolved:** it now says no *provisional* choice (primary
  provider per era, season windows, playoff handling, transaction cohort depth)
  becomes binding until the pilot passes, while the safeguards explicitly marked
  *binding now* in §1 (parser approach, Bronze immutability, §5.1 access policy,
  restricted release scope, source rejections) apply immediately.
- Two conservative inferences are now flagged as such wherever they appear
  (scope/disclaimers bullet; §1 note; SRC-BREF; §9 item 6; Sources).
- Unchanged from v0.2: ESPN/`hoopR` rejected (Disney terms do expressly bar
  testing/benchmarking/validation of AI/ML tools); the interim within-NBA
  validation stack and its acknowledged gap; the manually curated transaction
  cohort; the §5.1 access ceiling; coverage (dev 2023-24…2025-26; cycle
  2020-21…2025-26).

## Verification

Documentation-only task; proportional checks on `task/data-sources`:

- `git diff --check`: clean.
- `uv run courtgraph doctor`: `healthy`; 7 unit tests OK (code untouched).
- Structure: 10 numbered sections + scope/disclaimers + registry (10 `SRC-*`) +
  sources; internal links resolve; no tabs / trailing whitespace.
- `git status`: only `DATA_SOURCES.md`, `docs/CURRENT_TASK.md`, and
  `docs/PROJECT_STATUS.md` changed.

## Review outcome

Codex approved `DATA_SOURCES.md` v0.3 (2026-08-29). Points from the review that
remain open are recorded below and in `DATA_SOURCES.md` §§1, 8, 9 — they are
downstream work (legal review, the data pilot), not blockers for merging the
document.

## Open review concerns

- NBA and Sports Reference terms pages block automated retrieval; findings are
  from direct review and labelled as such. A human should re-confirm the exact
  clause wording before any release decision.
- The two conservative inferences (NBA restriction may reach the private dataset;
  Sports Reference AI-training ban read to also cover validation) are the
  document's most consequential judgment calls — confirm they are acceptably
  cautious rather than over- or under-reaching.
- The primary-source recommendation is provisional & conditional pending legal
  review; the independent-validation-lineage gap is real and unresolved without a
  licensed feed or written permission.

## Next action

Merge the `task/data-sources` PR into `main` when the user approves. The single
task after that is the §8 data pilot or possession-rule work (master plan §7);
neither begins until the user activates it.

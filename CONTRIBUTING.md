# Contributing

CourtGraph is currently in the research-design stage. Contributions should preserve the project’s core standard: every result must be traceable from source data through a leakage-safe evaluation protocol.

## Before implementation

- Read [the master plan](docs/MASTER_PLAN.md).
- Record material design choices as architecture decision records.
- Register hypotheses before final test evaluation.
- Do not introduce an advanced model before its required baselines exist.

## Research integrity

- Never use future data in historical features, embeddings, priors, or roster snapshots.
- Do not select case studies or transactions based on favorable outcomes.
- Preserve failed and null experiments.
- Report uncertainty, support, and model/data versions with predictions.
- Use causal language only when the identification strategy supports it.
- Keep raw source data immutable and version derived snapshots.

## Engineering expectations

- Add regression fixtures for every possession-parser bug.
- Test lineup permutation invariance where applicable.
- Keep reusable logic outside notebooks.
- Make commands reproducible from committed configuration.
- Compare models on the same split manifests and metrics.

Implementation-specific development instructions will be added when Stage 0 begins.

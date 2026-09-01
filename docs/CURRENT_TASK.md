# Current Task

Last updated: 2026-09-01

## State

Done — the chemistry model was reworked to sparse linear algebra so it runs at
NBA scale, and the **full low-rank interaction model was fit and evaluated
against the additive baseline on the 266,518 real regular-season stints**.
Branch `task/chemistry-sparse-scale` off `origin/main` (`a4c04f0`). Committed
and pushed, PR open.

**Result: the low-rank chemistry model does not beat the additive baseline on
this real regular-season data.** A genuine null / unfavourable result, kept per
`RESEARCH_CONTRACT.md` ("Preserve failed, null, and unfavorable experiments").

## The sparse rework (`ed3aa2c`, `71ae257`)

`ChemistryModel.fit` used dense `(n_stints × n_players)` one-hots and
`O(n · n_players²)` Gram matmuls, plus a ~6 GB `(n · n_players · rank)` buffer
per ALS half-sweep — one fit did not finish in over an hour at 985 players.

Each stint touches 5 of ~985 players, so every Gram / rhs is now accumulated by
scattering the 25 (player, player) index pairs per row with `np.bincount`
(`O(n · 25 · rank²)`), then a direct `np.linalg.solve` on the cheaply-built
dense Gram. **Same math, pure numpy, no new dependency, no artifact-format
change.**

- `features.DesignMatrices` drops `offense_onehot` / `defense_onehot` (no
  consumer outside `baseline.py`).
- `baseline._normal_equations` (`_pair_gram` / `_ctx_gram` / `_idx_rhs`) replaces
  `_assemble`; `_select_l2_player` builds each fold's Gram once and re-solves per
  grid value; `predict` / `decompose_row` use a padded index gather.
- `chemistry_model.LowRankInteraction.fit`'s `half_step` accumulates the
  `(n_players·rank)²` Gram by `bincount`; ALS loop / seed / standardization /
  convergence check unchanged.

Equivalence, checked three ways:
- 164 tests pass, including `test_model_fit_is_deterministic` and the recovery /
  no-spurious-chemistry guards (unchanged).
- New `SparseGramEquivalenceTests`: sparse Gram / rhs / predict == an
  independent dense one-hot reference to `atol ≤ 1e-9`.
- Pre- vs post-rework full fit on `recovery_synthetic`: every coefficient agrees
  to **max |Δ| ≈ 1e-13**; identical discrete L2 picks.
- New `test_chemistry_scale.py`: ~19k stints / 330 players fits in ~19 s,
  `tracemalloc` peak < 1 GB.

Timing on the real data: `courtgraph fit --evaluate --bootstrap 0` on 266k
stints / 985 players ran in **~16 min** (was: never finished).

## Full model vs additive baseline — 266k real regular-season stints

Held-out **macro** RMSE (possession-weighted group means — the contract
headline), points per 100 possessions. `courtgraph fit
data/nba_snapshots/rs_2020_2024/out/stints.jsonl --evaluate --bootstrap 0`;
result in `data/nba_snapshots/rs_2020_2024/chem_full_eval.json` (gitignored).

| holdout | additive | full (low-rank) | improvement |
|---|---|---|---|
| chronological | 3.523 | 3.523 | **+0.0%** (nil) |
| unseen_pair | 3.738 | 3.741 | **−0.08%** |
| unseen_lineup | 4.349 | 4.712 | **−8.4%** |

- The full-run interaction-L2 selection picked `l2 = 200` — the **top** of the
  grid `(8, 25, 70, 200)`, i.e. "no interaction helps, shrink it toward zero"
  (`_select_interaction`'s documented default). Even so the residual interaction
  term is slightly to clearly harmful on the pair / lineup holdouts.
- `evaluate_suite` refits per holdout; on the unseen-lineup training split the
  internal CV apparently let a smaller L2 through, and the resulting interaction
  overfit — held-out unseen lineups are 8.4% worse than additive-only. This is
  the model getting fooled by a weak / absent signal, which is exactly what the
  unseen-lineup gate exists to catch.
- Micro (stint) RMSE moves < 0.05 ppp100 in all three — no meaningful change.

**Interpretation:** on 266k real regular-season stints, a rank-3 provision/need
low-rank interaction adds **no** held-out predictive signal beyond additive
offense/defense talent, and mildly hurts on unseen lineups. This does not prove
"NBA lineup chemistry does not exist" — it means *this model, at this rank, on
this data, with this evaluation* finds none. Consistent with the literature that
lineup non-additivity is a small residual. Next moves are on the contract's
ladder, not a bigger neural model.

## Next candidate tasks (not started)

1. **Rungs 1 & 3** — descriptive/shrinkage diagnostics and a hierarchical
   (partial-pooling) impact model with a proper prior, and a wider `l2_player`
   grid; the current ridge selection maxes out shrinkage (`l2_player = 100`)
   on real data, suggesting the flat prior is too weak.
2. **Explicit pair interactions (rung 4)** before more low-rank — measure
   whether *any* non-additive parameterization beats additive on unseen pairs.
3. Recover the 840 quarantined games (503 `network_required`); nullable
   `days_rest` (schema v3) for the 68 season-opener quarantines.
4. Playoffs transport test — the 2024-25 playoffs archive is still held out.

`ChemistryConfig` defaults are unchanged. If real fits need to be faster, a
later opt-in `ChemistryConfig.large_data()` preset — not done here.

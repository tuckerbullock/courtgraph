# CourtGraph: A Research-Grade Blueprint for NBA Lineup Chemistry Embeddings

> **Living master plan — version 0.1**  
> **Status:** research and implementation blueprint  
> **Last updated:** 2026-08-31  
> **Working product names:** CourtGraph, LineupLab, NBA Chemistry Engine  
> **North-star question:** *Can we estimate how NBA players will fit together before we have observed that exact combination on the court?*

---

## Document purpose

This is the operating blueprint for an ambitious, research-grade NBA lineup chemistry project. It is intended to remain useful from the first raw play-by-play download through a paper-quality analysis, a reproducible modeling system, and an interactive roster-construction product.

The project is not “NBA2Vec with prettier charts,” a lineup net-rating leaderboard, or a black-box fifth-player recommender. Its central claim must be stronger and testable:

> **Learn context-aware representations of NBA players; separate individual talent from pairwise and higher-order interaction value; quantify uncertainty; and test whether the learned chemistry signal generalizes to unseen lineups, unseen pairs, future seasons, and post-transaction environments.**

The project succeeds only if it can answer all four of these questions convincingly:

1. What does “chemistry” mean statistically?
2. How do we know the metric is not just individual talent, lineup luck, coaching, or opponent quality?
3. Can it predict fit before the relevant players share the floor?
4. Does it remain useful when evaluated chronologically and under realistic roster decisions?

This plan deliberately starts with data validity and conservative baselines. Advanced neural, graph, and hypergraph models are later stages, not proof of rigor by themselves.

### How to maintain this document

- Treat section 36 as the operational roadmap and section 37 as the milestone scoreboard.
- Record material design choices with the section 41 decision template; do not silently rewrite history.
- Replace indicative durations with observed estimates after each milestone.
- Promote hypotheses to “supported,” “not supported,” or “inconclusive” only through registered experiments.
- Keep formulas and metric cards versioned when definitions change.
- Update the risk register and selected bibliography at each research release.
- Move implementation detail into linked design documents when it would obscure the research logic here.

---

## Table of contents

1. [Vision and research thesis](#1-vision-and-research-thesis)
2. [Definitions and estimands](#2-definitions-and-estimands)
3. [Research questions and hypotheses](#3-research-questions-and-hypotheses)
4. [Literature-grounded framing](#4-literature-grounded-framing)
5. [Scope, non-goals, and claims discipline](#5-scope-non-goals-and-claims-discipline)
6. [Data strategy and governance](#6-data-strategy-and-governance)
7. [Possession and stint construction](#7-possession-and-stint-construction)
8. [Database and feature-store design](#8-database-and-feature-store-design)
9. [Descriptive analysis and data audits](#9-descriptive-analysis-and-data-audits)
10. [Prediction targets and statistical units](#10-prediction-targets-and-statistical-units)
11. [The model ladder](#11-the-model-ladder)
12. [Baseline models](#12-baseline-models)
13. [RAPM and adjusted impact](#13-rapm-and-adjusted-impact)
14. [Hierarchical and Bayesian models](#14-hierarchical-and-bayesian-models)
15. [Pair and higher-order interactions](#15-pair-and-higher-order-interactions)
16. [Matrix factorization and latent complementarity](#16-matrix-factorization-and-latent-complementarity)
17. [Neural player embeddings](#17-neural-player-embeddings)
18. [Permutation-invariant lineup encoders](#18-permutation-invariant-lineup-encoders)
19. [Attention and multi-set interaction models](#19-attention-and-multi-set-interaction-models)
20. [Graph and hypergraph models](#20-graph-and-hypergraph-models)
21. [Context, roles, and temporal dynamics](#21-context-roles-and-temporal-dynamics)
22. [Chemistry metric system](#22-chemistry-metric-system)
23. [Uncertainty, shrinkage, and calibration](#23-uncertainty-shrinkage-and-calibration)
24. [Evaluation and generalization](#24-evaluation-and-generalization)
25. [Historical trade and roster backtests](#25-historical-trade-and-roster-backtests)
26. [Ablation and sensitivity program](#26-ablation-and-sensitivity-program)
27. [Leakage prevention](#27-leakage-prevention)
28. [Causal caveats and robustness](#28-causal-caveats-and-robustness)
29. [Interpretability and embedding probes](#29-interpretability-and-embedding-probes)
30. [Dashboard and product experience](#30-dashboard-and-product-experience)
31. [APIs and serving architecture](#31-apis-and-serving-architecture)
32. [Repository architecture](#32-repository-architecture)
33. [Testing and quality gates](#33-testing-and-quality-gates)
34. [Reproducibility and experiment tracking](#34-reproducibility-and-experiment-tracking)
35. [Compute strategy](#35-compute-strategy)
36. [Staged execution roadmap](#36-staged-execution-roadmap)
37. [Milestones and exit criteria](#37-milestones-and-exit-criteria)
38. [Publication-quality outputs](#38-publication-quality-outputs)
39. [Resume, portfolio, and interview framing](#39-resume-portfolio-and-interview-framing)
40. [Risk register](#40-risk-register)
41. [Decision log and experiment registry templates](#41-decision-log-and-experiment-registry-templates)
42. [Definition of done](#42-definition-of-done)
43. [Selected references](#43-selected-references)
44. [Appended product capabilities and example scenarios](#44-appended-product-capabilities-and-example-scenarios)

---

# 1. Vision and research thesis

## 1.1 The problem

Traditional lineup analysis usually asks which five-player groups produced the best historical offensive rating, defensive rating, or net rating. Those tables are useful descriptions, but they are poor estimates of transferable chemistry because:

- most five-player lineups have tiny samples;
- lineup membership is not randomly assigned;
- starters and bench players face different opponents;
- coaches deploy lineups in different game states and tactical contexts;
- great lineups may simply contain great players;
- teammates often share minutes so consistently that their effects are hard to separate;
- one season’s observed lineup may never recur;
- results after a trade are affected by coaching, role, health, schedule, and adaptation.

The hard problem is therefore not retrospective ranking. It is **structured generalization under sparse, confounded observations**.

## 1.2 The conceptual decomposition

The project starts from the decomposition

\[
V(L_o,L_d,c,t)
=
T(L_o,L_d,t)
+ C(L_o,L_d,c,t)
+ K(c,t),
\]

where:

- \(L_o\) is the offensive five-player set;
- \(L_d\) is the defensive five-player set;
- \(c\) is possession context;
- \(t\) is time or season;
- \(T\) is additive individual talent;
- \(C\) is non-additive interaction value, the candidate chemistry signal;
- \(K\) is non-player context such as home court, score state, rest, era, and possession start.

A practical lineup chemistry surplus is:

\[
\widehat{C}(L)
=
\widehat{V}_{full}(L)
-
\widehat{V}_{additive}(L).
\]

This definition prevents a five-star lineup from automatically receiving a high chemistry score. If its performance is exactly what the players’ additive talent predicts, its chemistry surplus is near zero even if the lineup itself is elite.

## 1.3 The primary contribution

The intended contribution is a **validation framework and metric system**, not merely a new architecture:

1. Construct trustworthy possession and stint data with explicit auditability.
2. Estimate additive offensive and defensive player impact.
3. Model observed and latent interactions at pair, trio, and lineup levels.
4. Quantify posterior or predictive uncertainty.
5. Define distinct fit concepts rather than one overloaded “chemistry” score.
6. Test transfer to unseen combinations and future roster environments.
7. expose the results through reproducible research artifacts and a transparent product.

## 1.4 The revolutionary standard

“Revolutionary” cannot mean “uses a GNN.” It must mean the project changes the interpretation of lineup data from:

> “This lineup was +14.2 per 100 possessions.”

to:

> “This lineup is estimated to be +6.4 per 100 in the specified context; +5.1 comes from additive player value, +1.3 is interaction surplus, the 80% interval for that surplus is [−0.4, +2.9], and a pre-lineup model trained without this exact combination predicted +0.9 of surplus.”

The user should always see:

- expected performance;
- talent contribution;
- chemistry surplus;
- offensive and defensive decomposition;
- uncertainty;
- observation status: seen, partially seen, or entirely unseen;
- data cutoff and model version;
- a short explanation of the main interaction drivers.

---

# 2. Definitions and estimands

Chemistry is not a directly observed quantity. It is a model-dependent residual concept, so definitions must be written before modeling.

## 2.1 Core vocabulary

| Term | Operational definition | What it is not |
|---|---|---|
| **Observed lineup performance** | Points or margin per possession for a particular lineup/opponent sample | A stable estimate of true ability |
| **Individual talent** | Player-specific offensive and defensive contribution after adjustment and shrinkage | Box-score production alone |
| **Pair chemistry** | Incremental predicted value from two teammates sharing a unit beyond additive player effects | Raw two-man net rating |
| **Lineup chemistry** | Full model prediction minus an additive counterfactual for the same players and context | Full lineup net rating |
| **Complementarity** | Expected interaction gain between two player profiles, including pairs never observed together | Similarity |
| **Compatibility** | Directional fit of player A in player B’s or a lineup’s environment | Necessarily symmetric |
| **Portability** | Stability of a player’s marginal value across diverse teammate and system contexts | Overall player quality |
| **Dependency** | Degree to which value varies with specific teammate profiles or contexts | A moral judgment or “system player” label |
| **Replacement value preservation** | Expected retention of lineup function when one player is swapped for another | Nearest neighbor in a visualization |
| **Chemistry centrality** | Network-based breadth and strength of positive, uncertainty-adjusted interaction links | Popularity or total minutes |
| **Confidence** | Model-estimated uncertainty and data support | Probability that the model is philosophically correct |

## 2.2 Primary estimands

Maintain separate estimands instead of collapsing them:

1. **Possession scoring distribution**  
   \(P(Y \in \{0,1,2,3,4+\}\mid L_o,L_d,c,t)\).

2. **Expected offensive points**  
   \(E[Y\mid L_o,L_d,c,t]\).

3. **Expected lineup net value**  
   predicted offensive output minus predicted defensive concession against a standardized opponent and context distribution.

4. **Pair interaction surplus**  
   \(\gamma_{ij}^{off}\) and \(\gamma_{ij}^{def}\), with distinct same-team and cross-team terms.

5. **Five-player surplus**  
   \(C(L)=V_{full}(L)-V_{additive}(L)\), averaged over a declared context and opponent reference distribution.

6. **Marginal roster move value**  
   \(\Delta V=V(L\setminus\{a\}\cup\{b\})-V(L)\), decomposed into talent and fit.

7. **Transported fit**  
   pre-event prediction for a player entering a team or lineup, evaluated only on subsequent data.

## 2.3 Reference distributions

Every context-averaged number needs an explicit reference population. Supported choices should include:

- league-average opponent and neutral context;
- expected opponent distribution for a selected team;
- actual historical contexts for an observed sample;
- playoff-caliber opponent distribution;
- user-defined roster and minutes scenario.

Do not compare figures averaged over different distributions without labeling them.

## 2.4 Symmetry choices

Some chemistry concepts should be symmetric; others should not.

- Pair surplus \(\gamma_{ij}=\gamma_{ji}\) may be symmetric for a simple teammate model.
- “Player A helps Player B” can be directional when modeling role provision and need vectors.
- Offensive and defensive embeddings should be separate or partially shared.
- Teammate and opponent interactions must be distinguished.
- Home and away team order must not affect results after corresponding context adjustment.

---

# 3. Research questions and hypotheses

## 3.1 Primary research question

> Do latent interaction models predict the performance of unobserved NBA player combinations more accurately and more reliably than additive impact models, raw lineup statistics, and observed pair effects?

## 3.2 Secondary research questions

1. How much lineup variance is attributable to additive talent, pair effects, higher-order effects, context, and noise?
2. Do low-rank interaction models outperform independently estimated pair coefficients in sparse settings?
3. Are separate offensive and defensive embeddings necessary?
4. Does explicit permutation invariance improve stability and unseen-lineup prediction?
5. Does attention yield useful pair attributions beyond simpler pair interactions?
6. Do graph or hypergraph models add value after controlling for parameter count and tuning budget?
7. Can embeddings trained only on possession outcomes recover recognizable basketball skills when probed after training?
8. Which players have the most portable impact across teams, roles, coaches, and teammate configurations?
9. Which players are highly complementary but not stylistically similar?
10. Can a model trained before a trade rank destination fit in a way associated with post-trade outcomes?

## 3.3 Pre-registered hypotheses

Write and timestamp hypotheses before the final test runs. Initial hypotheses:

- **H1:** Shrunk additive RAPM will outperform raw lineup and pair ratings on chronological prediction.
- **H2:** Pair interaction ridge will improve in-sample explanation but not necessarily unseen-pair performance because pair identifiers cannot extrapolate.
- **H3:** Low-rank interaction factorization will beat explicit pair coefficients on unseen-pair and sparse-pair tests.
- **H4:** A talent-plus-interaction neural model will outperform a single entangled neural pathway on transfer and interpretability.
- **H5:** Permutation-invariant set models will be more stable under player order perturbations and at least as accurate as concatenation models.
- **H6:** Context improves possession prediction but may reduce the apparent magnitude of chemistry estimates.
- **H7:** Most raw high-chemistry pairs will shrink substantially after opponent, teammate, and context adjustment.
- **H8:** Portability will correlate positively with impact but remain a distinct construct.
- **H9:** Trade-fit predictions will show modest signal with wide uncertainty; any strong causal language will be unjustified.

## 3.4 Falsification conditions

The central thesis should be weakened or rejected if:

- no interaction model beats additive RAPM on unseen-lineup or chronological tests;
- gains disappear under team-season or coach-aware controls;
- predictions are badly calibrated for sparse groups;
- embedding neighbors and complementarity rankings are unstable across seeds;
- trade backtest performance is indistinguishable from talent-only and team-strength baselines;
- “chemistry” leaderboards are driven mostly by one team, era, or extreme samples.

A null result is a valid research outcome: it would show how little transferable chemistry can be identified from public possession data.

---

# 4. Literature-grounded framing

## 4.1 Adjusted plus-minus lineage

Adjusted plus-minus treats scoring margin over stints as a regression problem with indicators for the players on the floor. Regularized versions address severe multicollinearity and unstable estimates. This project must reproduce a conventional RAPM-style baseline before claiming interaction value.

Key implication: lineup outcomes cannot be interpreted without accounting for teammates and opponents, but adjusted coefficients are still associational and sensitive to modeling choices.

## 4.2 Sparse lineup estimation

Recent Lineup RAPM work highlights the structural sparsity: an NBA team can use more than 600 lineups in a season, leaving the average lineup only about 25–30 possessions. Its proposed use of informed priors is directly relevant to predicting low-sample lineups. The project should reproduce or closely approximate this prior-informed idea as a dedicated baseline rather than using raw lineup ratings.

## 4.3 Representation learning

[NBA2Vec](https://arxiv.org/abs/2302.13386) demonstrates that dense player representations can be learned by predicting possession outcomes from offensive and defensive players. CourtGraph’s contribution must go beyond this foundation by:

- explicitly separating additive talent and interaction pathways;
- defining multiple fit metrics;
- quantifying uncertainty;
- enforcing leakage-safe unseen-combination tests;
- evaluating pre-transaction transport;
- comparing low-rank, set, graph, and hypergraph formulations under one protocol.

## 4.4 Set learning

A five-player lineup is a set, not a sequence. [Deep Sets](https://arxiv.org/abs/1703.06114) motivates symmetric pooling, while the [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) uses attention to model interactions while retaining permutation invariance. A possession contains two related sets—offense and defense—making multi-set cross-attention a natural later-stage model.

## 4.5 Graph and hypergraph formulations

Pairwise graphs can represent teammate and opponent relations, while a five-player lineup is also a higher-order object that cannot always be reduced to pair edges. [Hypergraph adjusted plus-minus](https://arxiv.org/abs/2403.20214) provides a direct precedent for jointly representing individuals, subgroups, and lineups. CourtGraph should compare graph and hypergraph models with simpler alternatives rather than assuming higher-order structure is automatically superior.

## 4.6 Bayesian workflow

Hierarchical priors can partially pool players, pairs, seasons, and lineups. Prior predictive checks, posterior predictive checks, convergence diagnostics, and simulation-based calibration should be treated as required workflow, following the [Stan User’s Guide](https://mc-stan.org/docs/stan-users-guide/index.html), not optional decoration.

## 4.7 Data tooling precedent

[pbpstats](https://pbpstats.readthedocs.io/en/latest/) can add lineups to events and construct detailed possessions, including start/end time and score margin. It is a strong practical foundation, but its own documentation notes provider differences and raw event-order issues. The project must preserve raw inputs, version the parser, record corrections, and independently validate the derived data.

## 4.8 Research gap

The project’s target gap is the intersection of:

- adjusted individual impact;
- explicit interaction decomposition;
- latent out-of-sample compatibility;
- temporal and transaction-based validation;
- decision-facing uncertainty;
- transparent, reproducible public tooling.

The novel claim is not that embeddings exist. It is that **chemistry estimates can be operationalized, stress-tested, and evaluated as transportable predictions rather than post-hoc stories**.

---

# 5. Scope, non-goals, and claims discipline

## 5.1 Initial scope

- NBA regular season and playoffs.
- Target first window: 8–10 seasons, expanding only after validation.
- Possession-level and stint-level parallel datasets.
- Publicly accessible play-by-play, rosters, schedules, and box scores.
- Player embeddings learned primarily from on-court outcomes.
- Optional post-training probes using box-score and shot-profile features.
- Historical roster moves with defensible transaction timestamps.

## 5.2 Possible later extensions

- WNBA or G League transfer learning.
- Historical NBA eras where lineup reconstruction is reliable.
- Injury and availability modeling.
- Coach/system embeddings.
- Play-type or spatial shot context.
- Salary-cap-aware roster optimization.
- Minute allocation optimization.
- Tracking-data integration if legally and practically available.

## 5.3 Explicit non-goals for version 1

- Claiming causal proof that two players “make” each other better.
- Predicting championships from a single lineup score.
- Automating betting decisions.
- Using scraped data without provenance or respect for terms and rate limits.
- Treating UMAP plots as evidence of learned basketball truth.
- Producing a single universal chemistry ranking with no uncertainty.
- Starting with transformers/GNNs before validating possessions and RAPM.

## 5.4 Claims ladder

Use the strongest claim supported by the experiment—never stronger:

1. **Descriptive:** the pair outscored opponents in observed minutes.
2. **Adjusted association:** the pair coefficient remains positive after stated controls.
3. **Predictive:** the model predicts held-out outcomes better than baselines.
4. **Transport predictive:** the model predicts new teammate or team contexts.
5. **Causal:** the pairing itself changes performance. This level generally requires stronger identification than public observational lineup data provides.

---

# 6. Data strategy and governance

## 6.1 Source hierarchy

Preferred hierarchy:

1. raw provider responses from NBA data endpoints where permitted;
2. pbpstats parsing and possession reconstruction;
3. independent schedule, roster, and box-score reconciliation;
4. transaction data with event dates from a stable, citable source;
5. derived public stats used only for validation, not silently merged as truth.

Each source gets a registry entry with:

- source name and URL;
- provider and access method;
- retrieval timestamp;
- season/date coverage;
- terms or rate-limit notes;
- raw schema/version hash;
- known issues;
- transformation owner;
- last validation date.

## 6.2 Raw-data immutability

Raw responses are append-only and content-addressed. Never “fix” a raw file in place. Corrections live in a structured patch table:

| correction_id | game_id | event_id | rule | old_value | new_value | reason | evidence | parser_version |
|---|---|---|---|---|---|---|---|---|

Derived datasets record the exact raw hashes and correction-set version used.

## 6.3 Coverage plan

Begin with two consecutive seasons for pipeline development. After all audits pass, expand backward and forward to the intended 8–10-season research window. Keep regular season and playoffs labeled; never silently pool them.

Recommended release tiers:

- **Bronze:** raw endpoint payloads.
- **Silver:** normalized events, rosters, game metadata.
- **Gold:** possessions, stints, lineup exposure, model-ready matrices.
- **Platinum:** versioned predictions, embeddings, uncertainty, and product aggregates.

## 6.4 Entity resolution

Build canonical mappings for:

- player IDs and name variants;
- team IDs, relocations, and abbreviations;
- season and league identifiers;
- games and rescheduled dates;
- transaction players and teams;
- coaches if used later.

Never use player names as primary keys. Store valid-from and valid-to dates for team membership. Flag ambiguous or unmatched identities for manual review.

## 6.5 Data-quality service levels

Before a season enters training:

- 100% of expected completed games are present or explicitly excluded;
- team and opponent IDs reconcile;
- final score from events equals official final score, except documented anomalies;
- every model possession has exactly five identifiable players per side;
- total player seconds approximate team box-score totals within declared tolerance;
- substitution transitions preserve valid five-player states;
- duplicate events and games are zero after canonicalization;
- possession counts fall within plausible league and team distributions;
- exclusions and corrections are summarized in a season-level report.

---

# 7. Possession and stint construction

This is the foundation. A sophisticated model on corrupted lineups is worse than a simple model on trusted data.

## 7.1 Canonical event ordering

For each game:

1. ingest all raw events with provider sequence identifiers;
2. normalize period and clock values;
3. apply documented same-clock ordering rules;
4. resolve substitutions, technical free throws, replay changes, and jump balls;
5. infer starting lineups from box score and period-start state;
6. maintain a lineup state machine for each team;
7. emit an audit trail for every inferred or corrected transition.

Same-clock events require special care because a substitution may occur before or after a free throw or timeout at the same displayed time.

## 7.2 Possession boundary policy

Write a formal rules specification covering:

- made field goals;
- defensive rebounds;
- turnovers;
- shooting fouls and free-throw sequences;
- technical and flagrant free throws;
- offensive rebounds and continuation;
- jump balls and alternating control;
- end-of-period heaves;
- team rebounds;
- lane violations and nullified attempts;
- replay reversals;
- simultaneous fouls;
- possessions spanning substitutions or timeouts.

Store both:

- `raw_possession_id`: parser’s natural possession;
- `analysis_possession_id`: research policy after filters and corrections.

## 7.3 Possession record

Minimum canonical fields:

```text
possession_id
game_id
season
game_date
period
possession_index
offense_team_id
defense_team_id
offense_player_ids[5]
defense_player_ids[5]
start_clock_seconds
end_clock_seconds
duration_seconds
start_score_margin_offense
end_score_margin_offense
points_scored
possession_end_type
possession_start_type
second_chance_flag
transition_proxy
timeout_before_flag
home_offense_flag
playoff_flag
garbage_time_weight
lineup_state_confidence
raw_source_hash
parser_version
```

Additional optional fields:

- shot count and shot zones;
- turnover type;
- free-throw counts;
- offensive rebound sequence count;
- foul/bonus state;
- seconds since substitution;
- starters on floor;
- rest days and back-to-back flags;
- travel proxy;
- team/coach/system identifiers.

## 7.4 Split-lineup possessions

A possession can contain a substitution. Maintain two analysis options:

1. assign to the lineup responsible for the substantive live-ball segment under a declared policy;
2. split into micro-stints/events for sensitivity analysis.

Primary training should exclude or downweight ambiguous split-lineup possessions until their frequency and effect are understood.

## 7.5 Stint construction

A stint is a maximal interval with unchanged ten-player membership. Store:

- start/end event and clock;
- offense and defense possessions;
- points for and against;
- elapsed seconds;
- starting and ending score margin;
- exact lineups;
- partial-possession flags;
- garbage-time weight;
- context aggregates.

RAPM should be reproduced with multiple targets:

- point differential per stint;
- point differential per 100 possessions;
- separate offensive and defensive points;
- possession-weighted or duration-weighted variants.

## 7.6 Garbage time

Do not simply delete all “garbage time.” Implement:

- an explicit deterministic rule as a baseline;
- a leverage or win-probability-derived continuous weight;
- full-data and filtered sensitivity runs.

Report how chemistry rankings change. Garbage time can contain real bench-player information, and deletion can introduce its own selection effects.

## 7.7 Validation against external totals

For every game and season, reconcile:

- final score;
- team possessions within expected methodology differences;
- player minutes;
- lineup minutes;
- team points by period;
- substitutions and ejections;
- number of overtime periods.

Randomly sample at least 25 games per new season for event-by-event manual review, oversampling overtime games, technical fouls, ejections, and games flagged by automated audits.

---

# 8. Database and feature-store design

## 8.1 Technology path

Development:

- Parquet files for immutable analytical snapshots;
- DuckDB for local exploration and pipeline tests;
- optional PostgreSQL for product APIs and multi-user access;
- object storage abstraction for raw and model artifacts.

Do not require distributed infrastructure until local profiling proves it necessary.

## 8.2 Core relational schema

### Dimensions

```sql
dim_player(
  player_id bigint primary key,
  display_name text,
  first_seen_date date,
  last_seen_date date,
  metadata_version text
)

dim_team(
  team_id bigint primary key,
  franchise_id bigint,
  abbreviation text,
  valid_from date,
  valid_to date
)

dim_game(
  game_id bigint primary key,
  season_id text,
  game_date date,
  home_team_id bigint,
  away_team_id bigint,
  game_type text,
  status text,
  source_hash text
)

bridge_roster_membership(
  player_id bigint,
  team_id bigint,
  valid_from timestamp,
  valid_to timestamp,
  source_id text
)
```

### Facts

```sql
fact_event(... raw and normalized event fields ...)
fact_possession(... canonical possession fields ...)
fact_stint(... canonical stint fields ...)
fact_player_game(... box-score reconciliation fields ...)
fact_transaction(... timestamped roster moves ...)
```

### Lineup structures

```sql
dim_lineup(
  lineup_id text primary key,
  p1 bigint, p2 bigint, p3 bigint, p4 bigint, p5 bigint,
  canonical_hash text
)

bridge_lineup_player(
  lineup_id text,
  player_id bigint,
  slot_for_storage_only smallint,
  primary key(lineup_id, player_id)
)

fact_lineup_exposure(
  season_id text,
  lineup_id text,
  opponent_lineup_id text,
  possessions bigint,
  points_for bigint,
  points_against bigint,
  context_slice_id text
)
```

Player ordering in `dim_lineup` is canonical and has no basketball meaning.

## 8.3 Model artifact schema

```sql
model_registry(
  model_id text primary key,
  git_commit text,
  data_snapshot_id text,
  config_hash text,
  trained_at timestamp,
  cutoff_date date,
  status text
)

player_embedding(
  model_id text,
  player_id bigint,
  season_id text,
  pathway text,       -- offense, defense, provision, need
  vector float[],
  posterior_sd float[] null
)

pair_prediction(
  model_id text,
  player_a bigint,
  player_b bigint,
  context_id text,
  chemistry_mean float,
  chemistry_sd float,
  observed_together boolean,
  shared_possessions bigint
)

lineup_prediction(
  model_id text,
  lineup_id text,
  reference_context_id text,
  expected_net float,
  talent_component float,
  chemistry_component float,
  lower_interval float,
  upper_interval float,
  novelty_class text
)
```

## 8.4 Snapshot manifests

Every dataset snapshot has a machine-readable manifest:

```yaml
snapshot_id: possessions_v1_2026_08_29
created_at: 2026-08-29T00:00:00Z
raw_hashes: [...]
parser_version: 0.3.1
correction_set: corrections_2026_08_28
seasons: [2016-17, ..., 2025-26]
row_count: 0
excluded_games: []
schema_hash: ...
quality_report: reports/data_quality/...
```

---

# 9. Descriptive analysis and data audits

Before ML, publish a “why naïve chemistry fails” report.

## 9.1 Required descriptive tables

- games, possessions, and stints by season;
- unique players, pairs, trios, and five-player lineups;
- lineup possession distribution: median, percentiles, maximum;
- pair and lineup repeat rates across seasons;
- fraction of test lineups and pairs unseen in training;
- player/team missingness;
- number and types of parser corrections;
- ambiguous lineup-state rate;
- garbage-time share;
- possessions by end type;
- home/away and playoff splits.

## 9.2 Required demonstrations

1. Raw net rating versus possession count.
2. First-half versus second-half stability of pair and lineup ratings.
3. Season-to-season stability.
4. Shrinkage trajectories for low-, medium-, and high-sample pairs.
5. Examples where raw pair value changes sign after adjustment.
6. Correlation between pair net rating and the quality of the other three teammates.
7. The frequency of exact lineup recurrence.
8. Sensitivity to garbage-time policy.

## 9.3 Negative controls

- Randomly permute player identities within team-season while preserving exposure structure; interaction signal should collapse.
- Randomize outcomes within context strata; advanced models should not show stable chemistry.
- Reorder player slots in set models; predictions must remain invariant.
- Use future-only features accidentally in a test fixture and confirm leakage tests fail.

---

# 10. Prediction targets and statistical units

No single target is sufficient. Maintain a small target suite.

## 10.1 Possession-level categorical target

Predict \(Y\in\{0,1,2,3,4+\}\) points. Use categorical cross-entropy or an ordinal/count alternative. Advantages:

- preserves full scoring distribution;
- supports probability calibration;
- directly yields expected points;
- keeps each possession as a natural observation.

Limitations: high irreducible variance and within-game dependence.

## 10.2 Possession expected-points regression

Predict points scored with squared, Huber, Poisson, or negative-binomial-style losses where appropriate. Compare distributional assumptions rather than assuming Gaussian residuals.

## 10.3 Stint margin target

Predict point differential with possession count as exposure/weight. This aligns with RAPM and reduces row count but aggregates context and may obscure possession mechanisms.

## 10.4 Two-stage offensive/defensive targets

Train separate pathways:

- offensive lineup versus defensive lineup → expected points;
- player offensive and defensive effects;
- teammate and opponent interaction components.

This is preferable to a single plus/minus coefficient when the objective includes basketball interpretation.

## 10.5 Cluster-aware inference

Possessions are not independent. Use:

- game-clustered bootstrap intervals;
- team-season block bootstrap;
- grouped cross-validation;
- hierarchical random effects;
- cluster-robust standard errors where applicable.

---

# 11. The model ladder

Every rung must have a reason, a baseline comparison, and an exit criterion.

| Rung | Model | What it tests | Must beat |
|---:|---|---|---|
| 0 | Global/context mean | Target sanity and context value | Constant mean |
| 1 | Raw/shrunk lineup ratings | Value of direct lineup history | Global/context mean |
| 2 | Additive ridge RAPM | Adjusted individual talent | Raw lineup ratings |
| 3 | Hierarchical player model | Partial pooling and uncertainty | Ridge RAPM on calibration/stability |
| 4 | Explicit pair interaction RAPM | Observed pair surplus | Additive RAPM on seen pairs |
| 5 | Low-rank pair factorization | Transfer across sparse/unseen pairs | Explicit pair model |
| 6 | Talent + neural embeddings | Nonlinear latent interactions | Low-rank factorization |
| 7 | Deep Sets | Permutation-invariant nonlinear lineup value | Sum/mean embedding model |
| 8 | Set attention / multi-set transformer | Adaptive within/across-lineup interactions | Deep Sets |
| 9 | Graph model | Typed teammate/opponent relations | Set model under matched budget |
| 10 | Hypergraph model | Higher-order subgroup/lineup effects | Graph and low-rank models |
| 11 | Dynamic hierarchical model | Time-varying player/fit state | Static models chronologically |

Promotion rule: an advanced model proceeds to product use only if it improves a pre-declared primary metric, does not materially worsen calibration, is stable across seeds, and offers an acceptable compute/interpretability tradeoff.

---

# 12. Baseline models

## 12.1 Constant and context-only baselines

- league mean expected points;
- season mean;
- home/away and season effects;
- team offense plus opponent defense;
- context-only generalized linear model.

These reveal how much signal comes from player identities versus broad environment.

## 12.2 Raw lineup baseline

For lineup \(L\):

\[
\hat V_L = 100\cdot\frac{PF_L-PA_L}{Poss_L}.
\]

Implement minimum-possession filters only for display, never as a substitute for uncertainty modeling.

## 12.3 Empirical Bayes lineup baseline

Shrink raw lineup value toward a team, season, or composition-informed prior:

\[
\hat V_L^{shrunk}
=w_L\hat V_L+(1-w_L)\mu_L,
\qquad
w_L=\frac{n_L}{n_L+k}.
\]

Tune \(k\) only inside training data. Candidate priors:

- league mean;
- team-season mean;
- sum of player ratings;
- L-RAPM-inspired informed lineup prior.

## 12.4 Additive team/player baselines

- team offense and defense fixed effects;
- box-score-informed player priors;
- single-season and multi-season ridge RAPM;
- exponentially decayed RAPM.

## 12.5 Nearest-history baselines

For an unseen lineup, estimate from:

- average of its ten constituent teammate pairs;
- average of four-player subsets;
- nearest observed lineup by player overlap;
- additive sum of current player ratings.

These are important because a complex model must beat simple basketball-informed heuristics.

---

# 13. RAPM and adjusted impact

## 13.1 Classical design

For stint \(s\), construct a design matrix with separate offense and defense columns:

\[
y_s = \alpha
+ \sum_i x^{off}_{si}\beta_i^{off}
+ \sum_i x^{def}_{si}\beta_i^{def}
+ z_s^T\delta + \epsilon_s.
\]

Possible encodings:

- offense players `+1`, defense players `−1` for a combined net coefficient;
- two columns per player for offensive and defensive effects;
- team-season intercepts;
- home-court and era effects;
- score-state/context covariates.

Fit ridge regression:

\[
\hat\beta
=\arg\min_\beta
\left\|W^{1/2}(y-X\beta)\right\|_2^2
+\lambda\|\beta\|_2^2.
\]

Tune \(\lambda\) using grouped chronological validation, never random stints from the same games across folds.

## 13.2 Weighting variants

Compare:

- possession-count weights;
- duration weights;
- equal-stint weights;
- leverage/garbage-time weights;
- recency decay;
- inverse-frequency or capped weights to prevent a few long stints dominating.

Declare one primary specification and use others as sensitivity checks.

## 13.3 Multi-season player state

Candidate specifications:

1. one coefficient per player across all seasons;
2. player-season coefficients with independent shrinkage;
3. player-season coefficients shrunk toward a career mean;
4. random-walk or state-space evolution;
5. exponentially weighted historical contribution.

For roster decisions, model state must reflect the player at the decision date, not his full-career average or post-decision performance.

## 13.4 Priors and box-score information

Use box-score or public impact metrics only as explicit priors/baselines. Maintain a possession-only model to test what the on-court interaction data learns independently. Never use a metric calculated with future season data as a pre-trade prior.

## 13.5 RAPM deliverables

- offense, defense, and total estimates;
- regularization path plots;
- coefficient stability across folds and seasons;
- intervals from clustered bootstrap or Bayesian posterior;
- comparison against published/public reference estimates where licensing permits;
- synthetic recovery test with known player effects;
- documentation of intercept, coding, weights, and scale.

## 13.6 RAPM exit criterion

Do not proceed to chemistry modeling until:

- synthetic coefficients are recovered within tolerance;
- signs and magnitudes are stable under reasonable coding variants;
- future-season prediction beats team/context-only baselines;
- duplicate or shuffled player columns trigger tests;
- a reproducible report explains differences from reference RAPM values.

---

# 14. Hierarchical and Bayesian models

## 14.1 Why hierarchy is essential

Players, pairs, and lineups have radically different sample sizes. Independent estimates overfit low-minute groups, while complete pooling erases genuine variation. Partial pooling provides a principled middle ground and propagates uncertainty into derived chemistry metrics.

## 14.2 Player hierarchy

A starting model:

\[
\begin{aligned}
y_s &\sim \mathcal{N}(\mu_s, \sigma^2/w_s),\\
\mu_s &= \alpha + X_s\beta + Z_s\delta,\\
\beta_{i,t}^{off} &\sim \mathcal{N}(\theta_i^{off}+a_{i,t}^{off},\tau_{off}^2),\\
a_{i,t}^{off} &\sim \mathcal{N}(\rho a_{i,t-1}^{off},\sigma_a^2),
\end{aligned}
\]

with analogous defensive effects. Alternatives include Student-t priors for robustness and position/role-informed group priors.

## 14.3 Pair hierarchy

For teammate pair \(i,j\):

\[
\gamma_{ij,t}
\sim \mathcal{N}(u_i^Tv_j + b_{team,t},\tau_{pair}^2).
\]

This combines explicit observed-pair residuals with a low-rank prior capable of unseen-pair prediction.

## 14.4 Lineup hierarchy

Lineup effect:

\[
\eta_L
\sim \mathcal{N}\left(
\sum_{i<j\in L}\gamma_{ij},
\tau_{lineup}^2
\right).
\]

This treats higher-order lineup chemistry as deviation from accumulated pair chemistry. It directly estimates whether there is meaningful residual five-player interaction variance after pair terms.

## 14.5 Prior design

Prior families must be justified through prior predictive simulation. Check that implied per-100-possession values are plausible. Candidate priors:

- regularizing zero-centered Normal;
- regularized horseshoe for sparse explicit interactions;
- Student-t for occasional large effects;
- hierarchical scale priors by experience or exposure;
- dynamic priors centered on previous season.

Avoid priors that silently force chemistry to exist or vanish.

## 14.6 Bayesian workflow checklist

- prior predictive checks;
- synthetic parameter recovery;
- multiple chains and dispersed initial values;
- \(\hat R\), effective sample size, divergence, tree-depth, and energy diagnostics;
- posterior predictive checks by season, team, lineup exposure, and score state;
- simulation-based calibration for reduced synthetic versions;
- leave-future-period-out predictive evaluation;
- sensitivity to prior scale and likelihood family;
- storage of posterior draws or compact approximations for derived metrics.

## 14.7 Approximate inference strategy

Full Bayesian inference over all pairs may be infeasible. Stage options:

1. fit smaller seasons/subsets in Stan or PyMC for gold-standard diagnostics;
2. use sparse matrices and marginalization where possible;
3. compare variational inference or Laplace approximation against MCMC on tractable subsets;
4. use empirical Bayes for production while retaining full Bayesian benchmark studies;
5. calibrate approximate intervals with bootstrap or conformal methods.

Approximation must be labeled; do not call variational standard deviations exact posterior uncertainty.

---

# 15. Pair and higher-order interactions

## 15.1 Explicit teammate pair model

Extend additive RAPM:

\[
\mu_s
=X_s\beta
+\sum_{i<j\in L_o(s)}\gamma_{ij}^{off}
+\sum_{i<j\in L_d(s)}\gamma_{ij}^{def}
+Z_s\delta.
\]

There are ten teammate pairs per five-player lineup on each side. Use sparse feature construction.

## 15.2 Cross-team matchups

Potentially model 25 offense-defense matchups:

\[
\sum_{i\in L_o}\sum_{j\in L_d}\omega_{ij}.
\]

This can capture defender/scorer or scheme matchup interactions, but parameter growth is enormous. Add only after teammate effects and evaluation infrastructure work.

## 15.3 Pair identifiability

Many teammates share nearly all minutes. Diagnose:

- pair exposure matrix condition structure;
- variance inflation or posterior correlation;
- share of each player’s minutes with the partner;
- connected components in the co-play graph;
- coefficient dependence on regularization;
- recovery in synthetic schedules mimicking real rotations.

Do not display pair scores when the model cannot distinguish them from shared teammates.

## 15.4 Sparse interaction regularization

Compare:

- ridge;
- lasso or elastic net;
- group lasso separating offense and defense;
- hierarchical shrinkage;
- regularized horseshoe;
- exposure-dependent penalties;
- low-rank priors.

Tune with group- and time-safe validation. The scientific question is not which method creates the most extreme leaderboard, but which yields stable transported predictions.

## 15.5 Trio and higher-order effects

Explicit trio indicators scale combinatorially. Restrict experiments to:

- high-exposure trios with strong shrinkage;
- residual trio terms after player and pair effects;
- tensor factorization;
- hierarchical decomposition;
- graph/hypergraph models.

Quantify how much predictive variance remains after pair terms. If higher-order variance is negligible or unstable, say so.

## 15.6 Interaction decomposition

For each lineup prediction, compute:

\[
V(L)
=\alpha
+\sum_i \beta_i
+\sum_{i<j}\gamma_{ij}
+\eta_L
+K(c).
\]

Product explanations should show:

- additive talent;
- top positive pair contributions;
- top negative pair contributions;
- higher-order residual;
- opponent/context adjustment;
- interval/uncertainty contribution.

Do not imply the decomposition is unique; it depends on constraints and model parameterization.

---

# 16. Matrix factorization and latent complementarity

## 16.1 Low-rank pair structure

Instead of a free coefficient for every pair:

\[
\gamma_{ij}=u_i^Tv_j.
\]

For symmetric chemistry, constrain or parameterize:

\[
\gamma_{ij}=e_i^TMe_j,
\]

where \(M\) may be symmetric. For directional provision/need:

\[
\gamma_{i\rightarrow j}=p_i^Tn_j,
\quad
\gamma_{ij}=p_i^Tn_j+p_j^Tn_i.
\]

This says a player can provide traits that satisfy another player’s needs. It separates **similarity** from **complementarity**.

## 16.2 Offensive and defensive factorization

Maintain separate spaces:

- offensive teammate complementarity;
- defensive teammate complementarity;
- offense-versus-defense matchup factors;
- optional shared base representation with pathway-specific projections.

## 16.3 Cold-start hierarchy

Pure ID embeddings cannot estimate rookies. Cold-start tiers:

1. known NBA player with no shared minutes with target partner: use learned latent vectors;
2. low-minute NBA player: shrink toward role/biographical prior;
3. rookie: initialize from pre-NBA or draft features if included in an explicitly separate model;
4. unknown player: use broad prior and show high uncertainty.

Primary research claims should focus on known NBA players and unseen combinations, not pretend the public PBP model solves rookie projection.

## 16.4 Rank selection

Evaluate latent dimensions \(d\in\{2,4,8,16,32,64\}\). Select using validation performance and stability, not visualization aesthetics. Report:

- predictive metric by dimension;
- seed-to-seed variance;
- effective rank/singular spectrum;
- probe performance;
- nearest-neighbor stability;
- compute cost.

## 16.5 Identifiability and alignment

Latent vectors are invariant to rotations and sometimes scale/sign transformations. For comparisons:

- align runs with orthogonal Procrustes;
- compare pair predictions rather than raw coordinates;
- bootstrap subspace stability;
- never interpret an individual raw dimension without probe evidence.

---

# 17. Neural player embeddings

## 17.1 Base architecture

Each player ID maps to trainable vectors:

```text
player_id
 ├── offensive talent scalar
 ├── defensive talent scalar
 ├── offensive interaction embedding
 ├── defensive interaction embedding
 ├── provision embedding
 └── need embedding
```

Possession model:

```text
offensive player set ─┐
                      ├─ lineup interaction encoder ─┐
defensive player set ─┘                              ├─ outcome head
context features ───────── context encoder ──────────┘
additive talent path ────────────────────────────────┘
```

## 17.2 Disentangled talent and chemistry

Use an explicitly additive skip path:

\[
\hat y
=\alpha
+\sum_{i\in L_o}\beta_i^{off}
+\sum_{j\in L_d}\beta_j^{def}
+f_{int}(E_o,E_d)
+g(c).
\]

Regularize the interaction pathway toward zero and compare with:

- talent-only;
- interaction-only;
- entangled single network;
- residual training where interaction learns only errors from a frozen/cross-fitted additive model.

To prevent the flexible pathway from re-encoding all talent, consider:

- residualization;
- orthogonality penalties;
- zero-sum centering across reference lineups;
- capacity limits;
- cross-fitting the additive baseline;
- variance decomposition on held-out data.

## 17.3 Outcome heads

Compare:

- categorical points distribution;
- expected-points regression;
- hurdle model for score/no-score then point value;
- ordinal logits;
- mixture density or distributional head.

Use proper scoring rules and calibration, not accuracy alone.

## 17.4 Training protocol

- group mini-batches by games only if necessary for efficiency, but shuffle games within the training window;
- validation split by time/game/team according to experiment;
- early stopping on a pre-declared metric;
- weight decay, embedding norm control, and gradient clipping;
- at least five seeds for final comparisons;
- saved configs, environment, checkpoints, and data hashes;
- no test-set hyperparameter decisions.

## 17.5 Multi-task objectives

Optional auxiliary tasks:

- possession points;
- team win-probability change;
- shot-zone distribution;
- turnover/rebound/foul outcome;
- next-season player impact;
- masked-player reconstruction.

Auxiliary tasks must be ablated. Do not add them merely to make the architecture appear complex.

---

# 18. Permutation-invariant lineup encoders

## 18.1 Required invariance

For any permutation \(\pi\) of a five-player lineup:

\[
f(\{e_1,\dots,e_5\})
=f(\{e_{\pi(1)},\dots,e_{\pi(5)}\}).
\]

Offense and defense are different sets; swapping the two should not be invariant because their roles differ.

## 18.2 Deep Sets baseline

Encode each player then pool:

\[
h_o=\rho_o\left(\sum_{i\in L_o}\phi_o(e_i)\right),
\qquad
h_d=\rho_d\left(\sum_{j\in L_d}\phi_d(e_j)\right).
\]

Combine with context and cross-lineup features. Compare sum, mean, max, and learned pooling.

## 18.3 Pairwise invariant encoder

Explicitly aggregate unordered pair functions:

\[
h_L
=\rho\left(
\sum_i\phi(e_i)
+\sum_{i<j}\psi(e_i,e_j)
\right).
\]

This can be easier to interpret than attention while allowing nonlinear pair effects.

## 18.4 Multi-set architecture

Treat offense and defense as separate permutation-invariant sets with:

- within-offense aggregation;
- within-defense aggregation;
- cross-set interactions;
- context-conditioned pooling.

Tests must permute offense and defense inputs independently and assert identical predictions to numerical tolerance.

## 18.5 Variable-size utility

Although NBA lineups normally contain five players, variable-size support helps with:

- missing-player uncertainty tests;
- three- and four-player subgroup queries;
- transfer to other sports or unusual data;
- masked-player training.

Never treat an unknown lineup player as a zero vector without an explicit mask.

---

# 19. Attention and multi-set interaction models

## 19.1 Attention objectives

Attention should allow context-specific weighting of relationships. For example, the relevance of a creator–rim-runner pair may change against a switching defense.

Architecture candidates:

1. self-attention inside the offensive set;
2. self-attention inside the defensive set;
3. offensive queries attending to defenders;
4. defensive queries attending to offensive players;
5. pooled matchup representation plus additive talent and context.

## 19.2 Role-aware attention without hard positions

Use learned role tokens or soft role assignment rather than fixed PG/SG/SF/PF/C labels. A player may occupy different roles in different lineups. Contextual role weights can be inferred from the lineup composition and possibly recent usage features.

## 19.3 Interpretability caution

Attention weights are not automatically faithful explanations. Validate them through:

- edge/pair deletion tests;
- counterfactual player swaps;
- integrated gradients or SHAP-like approximations where tractable;
- agreement with explicit interaction decomposition;
- stability across seeds;
- controlled synthetic tasks with known relevant pairs.

## 19.4 Capacity-matched evaluation

Compare attention with Deep Sets and pairwise encoders under matched:

- parameter count;
- tuning budget;
- training data;
- compute budget;
- number of seeds.

Report both best-run and mean/standard-deviation results.

---

# 20. Graph and hypergraph models

## 20.1 Possession graph

Represent ten players as nodes. Typed edges:

- offensive teammate;
- defensive teammate;
- offensive-player versus defender;
- optional historical chemistry edge;
- optional same-team/previous-team relation.

Node features can include player embeddings, current latent state, and non-outcome metadata available at prediction time.

## 20.2 Message passing

Use relation-aware message passing:

\[
h_i^{(l+1)}
=\sigma\left(
W_0h_i^{(l)}
+\sum_r\sum_{j\in N_r(i)}
\alpha_{ijr}W_rh_j^{(l)}
\right).
\]

Pool offense and defense nodes separately before the outcome head.

## 20.3 Hypergraph representation

Hyperedges may represent:

- the offensive five;
- the defensive five;
- teammate pairs;
- trios;
- historical roster groups;
- the full ten-player possession context.

The model should allow individual, subgroup, and lineup effects without pretending all group value decomposes into pairs.

## 20.4 Graph-specific experiments

- remove opponent edges;
- remove teammate edges;
- collapse edge types;
- compare one, two, and three message-passing layers;
- test over-smoothing;
- hold out entire co-play edges;
- test inductive node generalization only where node features support it;
- compare against a set model with identical base embeddings.

## 20.5 Graph model go/no-go

A graph/hypergraph model earns inclusion only if it offers at least one of:

- significant unseen-lineup improvement;
- better calibrated higher-order uncertainty;
- stable and validated subgroup attributions;
- a demonstrated advantage on synthetic higher-order interactions.

If it only adds complexity with negligible gain, it remains an appendix experiment.

---

# 21. Context, roles, and temporal dynamics

## 21.1 Context hierarchy

### Tier A: mandatory

- season/era;
- home court;
- regular season versus playoffs;
- score margin bucket or spline;
- period and time remaining;
- possession start type;
- garbage-time/leverage weight;
- team offense and defense environment.

### Tier B: recommended

- rest days;
- back-to-back status;
- travel/time-zone proxy;
- starters on floor;
- seconds/minutes since substitution;
- bonus state;
- transition/second-chance indicators;
- coach/team-season effect.

### Tier C: advanced

- shot profile and spacing proxies;
- role/usage state estimated only from prior data;
- injury return status;
- playoff series/game number;
- altitude;
- tactical or play-type features;
- tracking-derived spatial context.

## 21.2 Context leakage rule

A feature is eligible only if it would be known at the prediction timestamp. For a possession prediction, do not include end-of-possession features. For a trade forecast, use only pre-trade player and team data.

## 21.3 Team and coach effects

Chemistry may actually reflect a system. Compare:

- no team effect;
- team-season intercepts;
- coach-season effects;
- player-by-team adaptation effects;
- interactions with team style embeddings.

For portability, explicitly evaluate whether player fit survives removal of team/system effects.

## 21.4 Temporal embeddings

Player ability and style change. Options:

- player-season embeddings;
- base career embedding plus season delta;
- random-walk latent state;
- time-decayed training;
- quarterly/rolling updates;
- meta-learned update from recent possessions.

Align temporal embeddings before comparing trajectories. Penalize implausibly abrupt changes unless supported by data.

## 21.5 Role modeling

Avoid relying solely on listed positions. Learn soft roles from prior-only features or latent behavior:

- creator;
- spacer;
- rim pressure;
- screener/roller;
- connective passer;
- offensive rebounder;
- point-of-attack defender;
- rim protector;
- switchable defender.

These are interpretive probes or auxiliary labels, not assumed ground truth. A player can occupy a mixture of roles depending on lineup context.

---

# 22. Chemistry metric system

The product must expose a family of related metrics, each with a precise definition and uncertainty.

## 22.1 Pair Chemistry Surplus (PCS)

For teammates \(i,j\), averaged over reference contexts:

\[
PCS_{ij}
=E_c[\hat V_{full}(i,j,c)-\hat V_{no\ interaction}(i,j,c)].
\]

Report offense, defense, total, shared possessions, whether the pair was observed, and an interval.

## 22.2 Lineup Chemistry Surplus (LCS)

\[
LCS_L
=E_c[\hat V_{full}(L,c)-\hat V_{additive}(L,c)].
\]

Decompose into observed pair, latent pair, and higher-order components when the model supports it.

## 22.3 Lineup Synergy Added (LSA)

An attribution of lineup chemistry to players:

\[
LSA_i
=E_{L\ni i}[w_L\cdot\phi_i(C(L))],
\]

where \(\phi_i\) may be a Shapley-style contribution and \(w_L\) defines a declared exposure or equal-lineup weighting. Because exact Shapley computation is expensive and model-dependent, use approximation and report the attribution policy.

## 22.4 Portability Score

Estimate a player’s marginal impact across varied teammate/context draws:

\[
M_i(c,L_{-i})
=V(L_{-i}\cup\{i\},c)-V(L_{-i},c).
\]

Define portability using both mean and dispersion:

\[
Portability_i
=E[M_i]-\lambda\sqrt{Var(M_i)}.
\]

Also publish the raw mean, dispersion, 10th percentile, and coverage of evaluated contexts. A universally mediocre player should not rank highly merely because his value is stable.

## 22.5 Dependency Index

Quantify conditional variability and concentration:

- variance of marginal value across teammate archetypes;
- performance drop outside top-fit contexts;
- mutual information between teammate profile and predicted player contribution;
- share of value associated with a small number of partner clusters.

High dependency is not inherently bad; it may indicate a powerful specialist who requires a compatible system.

## 22.6 Complementarity Score

Directional provision/need model:

\[
Comp(i,j)=p_i^Tn_j+p_j^Tn_i.
\]

Display complementarity separately from cosine similarity. Two players can be dissimilar but highly complementary.

## 22.7 Replacement Preservation Score

For replacing player \(a\) with candidate \(b\) in lineup \(L\):

\[
RPS(b;a,L)
=-\left|V(L-a+b)-V(L)\right|
-\lambda U(b,L),
\]

or rank by expected value subject to preservation of offensive/defensive function. Product variants:

- closest functional replacement;
- best value replacement;
- best chemistry replacement;
- lowest-uncertainty replacement.

## 22.8 Chemistry Centrality

Construct an uncertainty-adjusted network where edge weight might be:

\[
w_{ij}
=E[PCS_{ij}]\cdot P(PCS_{ij}>0)\cdot reliability_{ij}.
\]

Measure:

- weighted degree/strength;
- eigenvector centrality;
- betweenness with caution;
- cross-community bridges;
- positive-edge breadth;
- temporal centrality stability.

Avoid minutes-driven centrality by using predicted edges across a standardized candidate set and publish sensitivity to edge thresholds.

## 22.9 Fit Frontier

For every roster decision, show a frontier across:

- expected net value;
- chemistry surplus;
- uncertainty;
- portability;
- optional salary/cost later.

This avoids pretending one scalar captures every decision objective.

---

# 23. Uncertainty, shrinkage, and calibration

## 23.1 Sources of uncertainty

- possession randomness;
- limited shared minutes;
- confounding and model specification;
- parameter estimation;
- embedding seed instability;
- temporal drift;
- context transport;
- lineup-state data errors;
- transaction adaptation.

## 23.2 Uncertainty outputs

Every prediction should return:

- point estimate;
- 50%, 80%, and/or 95% interval as appropriate;
- probability chemistry is positive;
- sample/exposure support;
- novelty class;
- model disagreement;
- data cutoff and model version.

Use intervals sparingly in the UI but make full details accessible.

## 23.3 Shrinkage display

Show raw and shrunk values together for educational examples. A pair with +7.2 over 53 possessions should not outrank a +3.8 estimate over 4,000 possessions without showing the uncertainty difference.

## 23.4 Predictive calibration

For categorical possession outcomes:

- reliability diagrams;
- expected calibration error with caveats;
- classwise Brier score;
- log loss;
- calibration by season, team, lineup novelty, and exposure.

For expected points or lineup values:

- interval coverage;
- calibration slope/intercept;
- residual plots by predicted value;
- conformal interval coverage where exchangeability assumptions are discussed;
- coverage under chronological and unseen-lineup shifts.

## 23.5 Uncertainty calibration for chemistry

Use simulation and resampling because chemistry itself is latent:

- synthetic data with known pair effects;
- posterior coverage tests;
- bootstrap reproducibility;
- sign stability across seasons;
- relationship between predicted interval width and realized error;
- held-out pair/lineup coverage.

## 23.6 Leaderboard eligibility

Default leaderboards require:

- adequate model support;
- interval width below a declared threshold;
- stability across seeds/specifications;
- not being flagged as an extrapolation outside training support.

Users may inspect all estimates, but uncertain entries must be visibly labeled.

---

# 24. Evaluation and generalization

## 24.1 Evaluation philosophy

Random possession splits are secondary diagnostics only. They allow nearly identical games, lineups, and pairs to appear on both sides and overstate roster-construction performance.

## 24.2 Primary split suite

### A. Chronological season holdout

Train through season \(t\), validate on early \(t+1\), test on the remainder or full \(t+1\). Repeat rolling-origin folds.

### B. Future-date rolling window

At several cutoff dates, train only on earlier data and predict the next 30/60/90 days.

### C. Exact unseen-lineup holdout

Select lineups with sufficient total test exposure. Remove every training possession containing that exact five-player offensive unit. The players and subgroups may remain observed.

### D. Unseen-pair holdout

Select teammate pairs and remove all training possessions where they share a team lineup. Retain each player individually in other contexts. Evaluate pair and lineup outcomes when they later play together.

### E. Strong unseen-pair holdout

Remove pair co-play from all earlier data and require the pair’s first observed partnership to occur in the test period, often via trade/signing.

### F. Leave-team-out transport

Hold out a team-season or franchise window. This tests reliance on team identifiers and system-specific effects.

### G. Player-team transfer

Train on a player’s prior teams, then evaluate his first window with a new team.

### H. Playoff transport

Train on regular seasons and test playoff possessions, with cautious interpretation due to distribution shift and selection.

## 24.3 Holdout construction details

For exact lineup holdouts:

1. define lineup as an unordered five-player set;
2. exclude it on both offense and defense if the task requires complete absence;
3. verify no duplicate/canonicalization mismatch remains;
4. compute training exposure for all constituent pairs and trios;
5. stratify results by subgroup familiarity;
6. select eligible lineups without looking at their outcomes;
7. keep tuning and test holdouts disjoint.

For unseen-pair holdouts:

1. choose pairs based on dates/exposure, not performance;
2. remove all co-play rows in training;
3. retain separate-player histories;
4. verify zero shared training possessions;
5. evaluate first-N and cumulative post-pair possessions;
6. cluster uncertainty by pair and team.

## 24.4 Metrics

### Possession prediction

- negative log likelihood;
- Brier score;
- expected-points MAE/RMSE;
- calibration error and slope;
- rank correlation for aggregated groups.

### Lineup/pair prediction

- possession-weighted and equal-group MAE/RMSE;
- Spearman/Kendall ranking correlation;
- top-k precision/recall for genuinely positive groups, with uncertainty;
- sign accuracy only for sufficiently precise outcomes;
- interval coverage and width;
- regret in fifth-player/replacement choices.

### Decision evaluation

- uplift over talent-only ranking;
- realized value of top-ranked candidate versus alternatives;
- pairwise ranking accuracy among actual candidate sets;
- calibration of “positive fit” probabilities;
- utility under uncertainty penalties.

## 24.5 Aggregation discipline

Report both:

- micro metrics weighted by possessions;
- macro metrics giving each pair/lineup equal weight;
- stratified metrics by exposure, novelty, team, season, and star/bench status.

A model can look good by predicting only high-minute lineups. Macro and sparse-group results expose this.

## 24.6 Statistical comparison

Use:

- paired game/team-season block bootstrap;
- confidence intervals for metric differences;
- Diebold-Mariano-style time-series comparisons only if assumptions fit;
- multiple-comparison correction for broad leaderboard claims;
- seed-level paired comparisons for neural models.

Practical significance matters: report the magnitude and decision impact, not only p-values.

## 24.7 Evaluation matrix

Final paper table:

| Model | Random diagnostic | Future season | Unseen lineup | Unseen pair | New team | Trade backtest | Calibration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Context mean |  |  |  |  |  |  |  |
| Shrunk lineup |  |  |  | N/A |  |  |  |
| RAPM |  |  |  |  |  |  |  |
| Hierarchical RAPM |  |  |  |  |  |  |  |
| Pair RAPM |  |  |  |  |  |  |  |
| Low-rank factorization |  |  |  |  |  |  |  |
| Neural embedding |  |  |  |  |  |  |  |
| Deep Sets |  |  |  |  |  |  |  |
| Set attention |  |  |  |  |  |  |  |
| Graph/hypergraph |  |  |  |  |  |  |  |

---

# 25. Historical trade and roster backtests

## 25.1 The Trade Test

For transaction \(k\) at timestamp \(T_k\):

1. freeze all data at \(T_k^-\);
2. train or load a model whose cutoff precedes the move;
3. construct destination-team lineups from players available immediately after the move;
4. predict incoming player fit with core teammates and projected units;
5. store predictions before examining post-move outcomes;
6. evaluate defined windows after adaptation.

This directly asks whether representations predict fit before observation in the new environment.

## 25.2 Event eligibility

Predefine inclusion rules:

- player had sufficient pre-move NBA data;
- player logged a minimum post-move sample;
- transaction and debut dates are known;
- no immediate season-ending injury;
- destination roster can be reconstructed;
- evaluation window does not cross unrelated major roster upheaval, or such events are censored/labeled;
- loans/10-day contracts/two-way moves handled separately.

Do not choose trades based on famous successes or failures.

## 25.3 Outcome windows

Evaluate:

- first 100 possessions with each core pair;
- first 500 team possessions;
- first 20 games;
- rest of season;
- next season where roster continuity permits.

Separate immediate adaptation from longer-term performance.

## 25.4 Trade baselines

- incoming player RAPM only;
- destination team strength only;
- average of current lineup ratings;
- similarity to outgoing player;
- public box-score projection available at cutoff;
- random eligible destination;
- talent-only roster simulation.

## 25.5 Counterfactual limits

Only the chosen destination is observed. We cannot directly observe how the player would have performed on every alternative team. Therefore:

- primary evaluation predicts observed post-move outcomes;
- destination rankings are exploratory unless candidate sets can be reconstructed;
- use matched comparisons cautiously;
- avoid stating the model proved a different trade would have been better.

## 25.6 Adaptation curve

Model post-move fit as a function of shared possessions:

\[
Fit_{k}(n)=\theta_{k,\infty}+(\theta_{k,0}-\theta_{k,\infty})e^{-n/\tau}.
\]

This tests whether predicted complementarity appears immediately or develops with familiarity. Treat this as descriptive unless identification improves.

## 25.7 Backtest report

For each move:

- pre-move model/version/cutoff;
- predicted talent and chemistry components;
- uncertainty;
- core lineups and assumed minutes;
- realized post-move adjusted value;
- contextual disruptions;
- error versus each baseline;
- no hindsight-edited narrative.

Aggregate results must include all eligible moves, not only illustrative case studies.

---

# 26. Ablation and sensitivity program

## 26.1 Architecture ablations

- remove additive talent path;
- remove interaction path;
- remove opponent embeddings;
- remove within-lineup interactions;
- remove cross-lineup interactions;
- replace attention with mean pooling;
- remove higher-order hyperedges;
- share versus separate offense/defense embeddings;
- freeze versus jointly train RAPM initialization.

## 26.2 Context ablations

- no context;
- no team/coach effects;
- no score state;
- no rest/travel;
- no garbage-time filter/weight;
- regular season only;
- current season only versus multi-season.

## 26.3 Data ablations

- one, three, five, and ten seasons;
- possessions versus stints;
- exclude ambiguous lineup states;
- provider A versus provider B where possible;
- high-exposure only versus full sample;
- playoffs excluded;
- remove box-score priors.

## 26.4 Representation ablations

- embedding dimensions;
- provision/need versus single symmetric vector;
- role features versus ID only;
- static versus dynamic embeddings;
- pairwise factorization rank;
- embedding norm and orthogonality penalties.

## 26.5 Robustness specifications

- ridge penalty ranges;
- different likelihoods;
- different possession-boundary policies;
- multiple garbage-time definitions;
- opponent-reference distributions;
- player minimum-exposure policies;
- seed variation;
- bootstrap resamples;
- alternate lineup holdout selection.

## 26.6 Ablation reporting

Publish one table with:

- delta on primary predictive metric;
- delta on unseen-lineup metric;
- delta on calibration;
- runtime/parameters;
- conclusion.

Do not present only favorable ablations.

---

# 27. Leakage prevention

## 27.1 Leakage taxonomy

1. **Temporal leakage:** future games inform current embeddings or features.
2. **Group leakage:** same lineup/pair appears in training and purported unseen test.
3. **Target leakage:** end-of-possession information enters features.
4. **Preprocessing leakage:** scalers, PCA, roles, or priors fit on full data.
5. **Roster leakage:** future team membership used in historical prediction.
6. **Selection leakage:** test lineups selected by observed performance.
7. **Hyperparameter leakage:** repeated test evaluation guides choices.
8. **Publication leakage:** case studies chosen after seeing outcomes.

## 27.2 Split manifests

Every experiment creates immutable files listing:

- train/validation/test game IDs;
- cutoff dates;
- held-out lineup hashes;
- held-out pair IDs;
- feature-fit scope;
- eligible entities;
- exclusion reasons;
- split-generation code version.

## 27.3 Feature timestamp contract

Every feature has:

- `event_time`;
- `available_time`;
- `computed_from_start`;
- `computed_from_end`;
- lookback window;
- source snapshot.

Pipeline assertion: `available_time <= prediction_time`.

## 27.4 Cross-fitting chemistry residuals

If chemistry is defined as residual from an additive model, avoid using an additive fit trained on the same outcomes without accounting for overfit. Options:

- fit additive baseline on fold A, generate residuals on fold B;
- joint hierarchical model with identified components;
- nested cross-fitting;
- evaluate full-versus-additive predictions only on held-out data.

## 27.5 Automated leakage tests

- fail if any test game ID exists in train;
- fail if an exact held-out lineup occurs in train;
- fail if unseen pair has shared training possessions;
- fail if scaler/role model cutoff exceeds training cutoff;
- fail if transaction model artifact date follows transaction date;
- deliberately insert a leaked feature and ensure the guard catches it.

---

# 28. Causal caveats and robustness

## 28.1 Why observational chemistry is confounded

Coaches choose lineups based on information not fully observed. Players share roles, opponents, injuries, play calls, and game states. A positive interaction residual may reflect:

- coaching scheme;
- stagger pattern;
- weak opponent benches;
- unmeasured health;
- stable third-player effects;
- substitution timing;
- role changes;
- selective deployment.

## 28.2 Primary language

Use:

- “estimated interaction surplus”;
- “predictive fit”;
- “adjusted association”;
- “model-implied complementarity.”

Avoid:

- “Player A causes Player B to improve”;
- “the trade added exactly X wins”;
- “true chemistry” without qualification.

## 28.3 Robustness designs

- player/team/season fixed or random effects;
- coach/system controls;
- opponent and score-state adjustment;
- matched-context comparisons;
- within-team-season analysis;
- first-difference/on-off contrasts around substitutions, cautiously;
- negative-control pairings;
- sensitivity to unobserved confounding;
- natural-experiment case studies only where assumptions are credible.

## 28.4 Causal extension candidates

Possible future research:

- instrumental variables based on plausibly exogenous injuries, with exclusion restrictions scrutinized;
- regression discontinuity around rotation decisions, rarely clean;
- difference-in-differences around trades with parallel-trend checks;
- causal forests for heterogeneous lineup effects;
- target-trial emulation for roster moves.

These require a separate causal protocol. They should not be mixed casually into the predictive paper.

## 28.5 Support and extrapolation

For any hypothetical lineup, measure distance from training support:

- player exposure;
- pair/trio exposure;
- embedding-space distance;
- team/system novelty;
- context density;
- model ensemble disagreement.

Flag or refuse confident predictions outside support.

---

# 29. Interpretability and embedding probes

## 29.1 Interpretation hierarchy

1. additive player effects;
2. explicit pair factors;
3. model decomposition;
4. counterfactual swap effects;
5. post-hoc embedding probes;
6. 2D visualization.

The lower items are more exploratory and should not override validated predictive evidence.

## 29.2 Intrinsic embedding analysis

- cosine/euclidean neighborhoods after normalization choices;
- provision versus need neighbors;
- offensive versus defensive spaces;
- cluster stability across seeds and seasons;
- Procrustes-aligned temporal trajectories;
- singular spectrum/effective dimensionality;
- neighborhood overlap metrics.

## 29.3 External probes

After training, regress known basketball traits on frozen embeddings:

- usage rate;
- assist rate;
- turnover rate;
- three-point attempt rate;
- rim frequency;
- free-throw rate;
- offensive rebound rate;
- block/steal rates;
- shot-distance distribution;
- listed size/position;
- public role labels where available.

Use nested chronological splits. Probe success means information is encoded, not that a dimension has a unique causal meaning.

## 29.4 Linear concept directions

Fit simple linear probes and inspect directions such as spacing, creation, rim protection, or rebounding. Validate with held-out players/seasons and report uncertainty. Avoid naming axes from cherry-picked examples.

## 29.5 Counterfactual lineup explanations

When replacing A with B, report:

```text
Predicted net change                         +2.1 / 100
  Individual talent change                  +0.7
  Pair complementarity with Player 1        +0.8
  Pair complementarity with Player 2        +0.3
  Other interaction/higher-order change     +0.5
  Context/role adjustment                   -0.2
80% predictive interval                     [-0.9, +5.0]
```

Counterfactuals should hold opponent and context fixed unless the user changes them.

## 29.6 Explanation faithfulness tests

- remove the supposedly important partner and measure prediction change;
- swap with a matched player lacking the identified trait;
- compare explanation rank with exhaustive pair deletion;
- test explanation stability across seeds;
- synthetic ground-truth interaction tests;
- report when explanations disagree across models.

## 29.7 UMAP/PCA display rules

- label the model, season, pathway, and projection method;
- include projection seed and parameters;
- allow switching PCA/UMAP;
- warn that 2D distance distorts high-dimensional geometry;
- use high-dimensional metrics for actual nearest-neighbor results;
- never treat visually separated clusters as formal evidence without quantitative validation.

---

# 30. Dashboard and product experience

## 30.1 Product thesis

The dashboard is not a decorative layer on top of the research. It is a structured interface for asking counterfactual questions while preserving the model’s epistemic limits.

Every page must answer:

- What is the question being estimated?
- What is the comparison or counterfactual?
- What data were available?
- How uncertain is the answer?
- Is this combination observed, partially observed, or unobserved?
- Which parts of the result come from talent, chemistry, and context?

## 30.2 Global interaction design

Persistent controls:

- model version;
- data cutoff;
- season or date range;
- offensive/defensive/total view;
- reference opponent distribution;
- context preset;
- regular season/playoffs;
- uncertainty level;
- observed-only versus predictions allowed.

Persistent status banner:

```text
Model: courtgraph-set-v3
Data through: 2025-26 regular season
Scenario: neutral court, league-average opponent
Prediction support: partially observed lineup
```

All exported charts should include this metadata.

## 30.3 Home / Research Overview

The landing page should communicate the research contribution in under 60 seconds:

1. one-sentence thesis;
2. animated or static decomposition of lineup value;
3. model comparison on unseen lineups;
4. calibration summary;
5. one trade-backtest result with uncertainty;
6. prominent link to methodology, limitations, and reproducibility.

Avoid opening with a cherry-picked star pairing. Lead with the problem and the validated result.

## 30.4 Player Explorer

### Header

- player name, team at selected date, season;
- offensive, defensive, and total adjusted talent;
- uncertainty and exposure;
- model’s support status.

### Latent Profile

- nearest functional neighbors;
- nearest offensive-profile neighbors;
- nearest defensive-profile neighbors;
- most complementary partners;
- the distinction between similarity and complementarity.

### Fit Profile

- portability score and distribution;
- dependency index;
- teammate archetypes that raise/lower predicted marginal value;
- team/system sensitivity;
- best and worst supported pair fits;
- observed versus inferred partnership labels.

### Temporal Profile

- adjusted talent over seasons;
- embedding movement after alignment;
- uncertainty bands;
- team changes, major injuries if sourced, and coach changes as annotations;
- clear warning that co-occurrence is not attribution.

### Evidence drawer

For every metric:

- formal definition;
- reference distribution;
- possessions or contexts supporting it;
- interval;
- stability across models/seeds;
- link to model card.

## 30.5 Pair Explorer

Input any two players, including those who have never been teammates.

Output:

- predicted pair chemistry, offense/defense/total;
- observed pair rating if applicable;
- adjusted observed pair effect;
- latent predicted effect;
- uncertainty and shared possessions;
- top provision/need matches;
- likely redundancies;
- lineup contexts in which fit improves or degrades;
- similar historical pairs by interaction profile;
- “not enough support” warning where appropriate.

The pair page should never hide disagreement between raw and adjusted results. That disagreement is often the most educational finding.

## 30.6 Lineup Builder

Core interaction:

1. select a team/date context or start from a blank lineup;
2. choose five players;
3. optionally choose an opposing lineup or standardized opponent;
4. choose neutral/current-team/custom context;
5. receive a decomposition and uncertainty.

Primary display:

```text
Expected net rating                    +8.3 / 100
  Additive player talent               +6.1
  Pairwise chemistry                   +1.6
  Higher-order lineup chemistry        +0.6

Offensive value                        91st percentile
Defensive value                        77th percentile
Prediction interval                    [+2.0, +14.1]
Model support                          partially observed
```

Secondary panels:

- pair interaction matrix;
- offense and defense radar with calibrated benchmark dimensions;
- most influential relationships;
- closest observed analog lineups;
- performance under alternative opponents;
- uncertainty sources;
- “what would change this estimate?” sensitivity panel.

## 30.7 Fifth-Player Finder

Given four players:

- rank candidate fifth players by expected net value;
- separately rank by chemistry surplus, fit floor, portability, and confidence;
- allow roster/team, position/role, minutes, availability, and optional salary constraints;
- compare candidates against an additive talent-only ranking;
- show whether recommendation comes from talent or fit;
- surface Pareto-optimal candidates rather than only one winner.

Required evaluation link: every ranked page must display the corresponding historical held-out-task performance for the model, such as average regret on unseen fifth-player tests.

## 30.8 Replacement Finder

Given a lineup and outgoing player:

- **Functional replacement:** preserves predicted role and structure.
- **Best net replacement:** maximizes predicted performance.
- **Best chemistry replacement:** maximizes interaction surplus.
- **Safest replacement:** maximizes lower confidence bound.
- **Budget replacement:** future extension using salary/cap data.

Display what is lost and gained, not just a list of names.

## 30.9 Team Fit Lab

For a selected team/date:

- roster chemistry network;
- core-lineup decomposition;
- players with high/low portability within the current system;
- lineup coverage and uncertainty;
- candidate additions;
- fragile dependencies;
- rotation-level scenario builder;
- fit gaps by learned provision/need traits.

Use current rosters only when current data are actively maintained. Historical snapshots must never be overwritten by today’s membership.

## 30.10 Trade Simulator

### Inputs

- one or more outgoing players;
- one or more incoming players;
- team and date;
- candidate rotation/minute scenarios;
- reference opponent distribution;
- optional “information available as of” cutoff for historical replay.

### Outputs

- estimated team/rotation net change;
- talent versus chemistry change;
- lineup-by-lineup impact;
- role/provision gaps created or resolved;
- uncertainty and support;
- nearest historical analog transactions;
- explicit non-causal disclaimer.

The simulator should not estimate wins until a separate minute-allocation and season simulation model is validated.

## 30.11 Chemistry Network

Interactive graph controls:

- positive, negative, or all edges;
- observed versus predicted edges;
- minimum certainty/exposure;
- season and team;
- offense/defense;
- centrality measure;
- community detection method.

Selecting an edge opens the pair-evidence panel. Edge width must not conflate magnitude with certainty; use separate visual encodings.

## 30.12 Model Comparison Lab

Let technical users compare:

- raw lineup rating;
- RAPM;
- pair RAPM;
- matrix factorization;
- neural/set/graph model.

Show where models agree, where they disagree, and how each performed on the relevant holdout category. This makes the system scientifically inspectable rather than presenting the final model as an oracle.

## 30.13 Data Quality Observatory

Public or internal page with:

- games processed by season;
- missing/excluded games;
- parser corrections;
- lineup-state confidence;
- score/minute reconciliation;
- latest pipeline run;
- source/provider status;
- current known issues.

This page establishes that data engineering is part of the research result.

## 30.14 Accessibility and communication

- never rely on color alone;
- keyboard navigation;
- readable intervals and units;
- plain-language tooltips beside formal definitions;
- mobile-safe core pages;
- exportable CSV and SVG/PNG;
- accessible tables behind visualizations;
- no unlabeled percentile or rating scales.

## 30.15 Product telemetry and ethics

If telemetry is used:

- collect only necessary anonymous interactions;
- document it;
- never use private user simulations as training data without consent;
- provide opt-out;
- do not imply endorsements from players, teams, or the NBA.

---

# 31. APIs and serving architecture

## 31.1 Architecture overview

```text
Raw/normalized data
       ↓
Versioned feature snapshots
       ↓
Training + evaluation pipelines
       ↓
Model/artifact registry
       ↓
Offline prediction materialization
       ↓
PostgreSQL / analytical store
       ↓
Read-only prediction API
       ↓
Dashboard / notebooks / paper figures
```

Most expensive pair and lineup estimates should be materialized offline. On-demand neural inference is reserved for genuinely novel scenarios.

## 31.2 API principles

- version every endpoint;
- return model and data version with every response;
- canonicalize player ordering server-side;
- validate lineup uniqueness and date eligibility;
- include uncertainty and support metadata;
- use stable player IDs;
- separate observed facts from model predictions;
- rate limit hypothetical bulk queries;
- never silently fall back to a different model.

## 31.3 Candidate REST endpoints

```text
GET  /v1/models
GET  /v1/players/{player_id}
GET  /v1/players/{player_id}/embedding-neighbors
GET  /v1/players/{player_id}/fit-profile
GET  /v1/pairs/{player_a}/{player_b}
POST /v1/lineups/predict
POST /v1/lineups/replacements
POST /v1/lineups/fifth-player
POST /v1/teams/{team_id}/trade-simulate
GET  /v1/teams/{team_id}/chemistry-network
GET  /v1/evaluations/{model_id}
GET  /v1/data-quality/seasons/{season_id}
```

## 31.4 Lineup request example

```json
{
  "offense_player_ids": [1, 2, 3, 4, 5],
  "defense_player_ids": [6, 7, 8, 9, 10],
  "as_of_date": "2026-01-15",
  "reference_context": "neutral_standard",
  "model_id": "courtgraph-set-v3",
  "uncertainty_level": 0.80
}
```

## 31.5 Lineup response example

```json
{
  "prediction": {
    "expected_points": 1.148,
    "expected_points_per_100": 114.8,
    "additive_component": 7.1,
    "pair_chemistry_component": 1.4,
    "higher_order_component": 0.3,
    "context_component": 0.2,
    "interval": [105.9, 123.6]
  },
  "support": {
    "novelty_class": "unseen_exact_lineup",
    "constituent_pairs_seen": 8,
    "constituent_pairs_total": 10,
    "minimum_player_possessions": 842,
    "out_of_support": false
  },
  "provenance": {
    "model_id": "courtgraph-set-v3",
    "data_snapshot_id": "possessions_2026_01_14",
    "data_cutoff": "2026-01-14"
  }
}
```

## 31.6 Error behavior

Use actionable errors:

- duplicate player in lineup;
- inactive/unknown player at historical cutoff;
- insufficient support;
- model incompatible with requested date;
- invalid opponent/reference context;
- scenario outside supported league/season.

For out-of-support scenarios, return a refusal or explicitly low-confidence result—not an ordinary-looking number.

## 31.7 Serving paths

### MVP

- FastAPI or similar typed Python API;
- DuckDB/Parquet for read-heavy local prototype;
- cached materialized results;
- single-process deployment.

### Production-style portfolio release

- containerized API;
- PostgreSQL for entity/prediction tables;
- object storage for model artifacts;
- background job for large scenario batches;
- CDN/static delivery for public aggregates;
- monitoring and structured logs.

## 31.8 Performance targets

- cached player/pair query p95 < 200 ms;
- cached lineup query p95 < 300 ms;
- uncached single-lineup inference p95 < 1 s on CPU where feasible;
- fifth-player search < 3 s for a bounded candidate pool;
- every response reproducible from model/data IDs.

Do not optimize prematurely; measure first.

## 31.9 Security and reliability

- read-only public API database role;
- request-size limits;
- input validation;
- no arbitrary model artifact paths;
- model checksum validation;
- safe serialization format where possible;
- dependency scanning;
- health, readiness, and model-loaded endpoints;
- structured error rates and latency monitoring.

---

# 32. Repository architecture

## 32.1 Proposed monorepo

```text
courtgraph/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock                         # or another committed lockfile
├── Makefile                        # thin convenience commands
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── configs/
│   ├── data/
│   ├── models/
│   ├── experiments/
│   ├── evaluation/
│   └── product/
│
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── corrections/
│   └── samples/                    # tiny legal fixtures only
│
├── src/courtgraph/
│   ├── __init__.py
│   ├── cli.py
│   ├── settings.py
│   ├── logging.py
│   │
│   ├── ingest/
│   │   ├── schedules.py
│   │   ├── play_by_play.py
│   │   ├── box_scores.py
│   │   ├── rosters.py
│   │   └── transactions.py
│   │
│   ├── normalize/
│   │   ├── events.py
│   │   ├── identities.py
│   │   ├── corrections.py
│   │   └── schemas.py
│   │
│   ├── possessions/
│   │   ├── state_machine.py
│   │   ├── boundaries.py
│   │   ├── lineups.py
│   │   ├── stints.py
│   │   └── audits.py
│   │
│   ├── features/
│   │   ├── context.py
│   │   ├── exposure.py
│   │   ├── roles.py
│   │   ├── matrices.py
│   │   └── timestamps.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── baselines.py
│   │   ├── rapm.py
│   │   ├── hierarchical.py
│   │   ├── pair_rapm.py
│   │   ├── factorization.py
│   │   ├── embeddings.py
│   │   ├── deep_sets.py
│   │   ├── attention.py
│   │   ├── graph.py
│   │   ├── hypergraph.py
│   │   └── dynamic.py
│   │
│   ├── metrics/
│   │   ├── chemistry.py
│   │   ├── portability.py
│   │   ├── dependency.py
│   │   ├── complementarity.py
│   │   ├── replacement.py
│   │   └── centrality.py
│   │
│   ├── evaluation/
│   │   ├── splits.py
│   │   ├── leakage.py
│   │   ├── predictive.py
│   │   ├── calibration.py
│   │   ├── uncertainty.py
│   │   ├── ablations.py
│   │   ├── trades.py
│   │   └── reports.py
│   │
│   ├── interpretability/
│   │   ├── probes.py
│   │   ├── attributions.py
│   │   ├── alignment.py
│   │   └── counterfactuals.py
│   │
│   ├── registry/
│   │   ├── datasets.py
│   │   ├── models.py
│   │   └── artifacts.py
│   │
│   └── api/
│       ├── app.py
│       ├── schemas.py
│       ├── dependencies.py
│       └── routes/
│
├── app/
│   ├── README.md
│   ├── package.json
│   ├── src/
│   └── tests/
│
├── pipelines/
│   ├── ingest.yaml
│   ├── build_possessions.yaml
│   ├── train.yaml
│   ├── evaluate.yaml
│   └── publish.yaml
│
├── experiments/
│   ├── registry.yaml
│   ├── hypotheses/
│   └── archived_configs/
│
├── notebooks/
│   ├── README.md
│   ├── exploratory/
│   └── published/                  # executed and frozen
│
├── reports/
│   ├── data_quality/
│   ├── evaluation/
│   ├── figures/
│   ├── tables/
│   └── paper/
│
├── artifacts/
│   ├── README.md
│   └── manifests/                  # artifacts themselves external/ignored
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── property/
│   ├── regression/
│   ├── synthetic/
│   ├── leakage/
│   └── fixtures/
│
├── docs/
│   ├── architecture/
│   ├── data_dictionary.md
│   ├── possession_rules.md
│   ├── metric_cards/
│   ├── model_cards/
│   ├── decisions/
│   └── runbooks/
│
└── scripts/
    ├── bootstrap_environment.sh
    ├── verify_snapshot.sh
    └── reproduce_release.sh
```

## 32.2 Package boundaries

- `ingest` retrieves but does not reinterpret source data.
- `normalize` standardizes raw schemas and identity mappings.
- `possessions` owns basketball state and boundary logic.
- `features` produces timestamp-safe model inputs.
- `models` consumes typed datasets and returns standardized predictions.
- `metrics` derives basketball concepts from predictions.
- `evaluation` owns splits and test protocols; models cannot choose their test data.
- `registry` provides provenance and immutable identifiers.
- `api` is a thin serving layer, not a second implementation of metrics.

## 32.3 Configuration

Use composable YAML/TOML configs validated into typed objects. A resolved config is saved with each run. No hidden notebook constants.

Example:

```yaml
experiment_id: E042_unseen_pair_factorization
data:
  snapshot_id: possessions_v1
  train_end: 2024-06-30
split:
  type: unseen_pair_temporal
  manifest_id: split_unseen_pair_v2
model:
  family: pair_factorization
  rank: 16
  ridge_player: 25.0
  ridge_pair: 100.0
evaluation:
  primary_metric: macro_pair_mae
  secondary: [nll, calibration_slope, interval_coverage]
seed: 24017
```

## 32.4 Notebook policy

- notebooks are for exploration and narrative reports;
- reusable logic must move into `src/`;
- published notebooks run top-to-bottom from a snapshot;
- outputs are cleared or frozen consistently;
- no production model exists only in a notebook;
- each notebook states data snapshot, model ID, and purpose.

## 32.5 Command interface

Target commands:

```text
courtgraph ingest --season 2024-25
courtgraph build-possessions --snapshot raw_2025_06
courtgraph audit-data --season 2024-25
courtgraph make-split --config configs/evaluation/unseen_lineup.yaml
courtgraph train --config configs/experiments/E042.yaml
courtgraph evaluate --run-id ...
courtgraph backtest-trades --cutoff 2025-02-01
courtgraph materialize-product --model-id ...
courtgraph serve
```

## 32.6 Documentation artifacts

Required early documents:

- data dictionary;
- possession-boundary rules;
- data-source registry;
- model interface contract;
- metric cards;
- evaluation split specification;
- reproducibility runbook;
- public limitations statement.

---

# 33. Testing and quality gates

## 33.1 Testing pyramid

### Unit tests

- clock parsing;
- event ordering;
- possession boundaries;
- substitution state transitions;
- lineup canonicalization;
- pair/trio enumeration;
- context features;
- metric formulas;
- uncertainty transformations;
- API validation.

### Property-based tests

- any lineup permutation yields the same canonical ID;
- any offense player permutation leaves set-model output unchanged;
- adding and removing the same player restores state;
- each valid lineup has five unique players;
- ten teammate pairs are generated from a five-player lineup;
- talent-plus-chemistry decomposition sums to total within tolerance;
- intervals remain ordered;
- no future timestamp passes feature validation.

### Integration tests

- raw game → normalized events → possessions → stints;
- snapshot → design matrix → RAPM fit → predictions;
- model checkpoint → registry → API response;
- historical cutoff → features contain no future rows;
- dashboard request → API schema and provenance.

### Regression tests

- known complex games preserve expected possession/lineup output;
- season aggregates remain within tolerance after parser changes;
- model metrics do not change unexpectedly on frozen miniature data;
- published figure/table hashes or semantic values remain stable.

## 33.2 Possession fixture library

Hand-curate small event sequences for:

- ordinary made basket;
- miss + offensive rebound + make;
- shooting foul with substitutions between free throws;
- technical free throw;
- flagrant and retained possession;
- jump ball changing possession;
- replay overturn;
- lane violation;
- team rebound;
- period-ending heave;
- overtime start;
- ejection and replacement;
- same-clock timeout/substitution/free throw.

Each fixture documents expected lineups, possessions, points, and why.

## 33.3 Synthetic model-recovery suite

Generate schedules resembling real co-play structure with known:

- additive player talent;
- sparse pair interactions;
- low-rank complementarity;
- higher-order lineup effects;
- context effects;
- temporal drift;
- data missingness.

Required tests:

- RAPM recovers additive effects;
- pair model recovers observed pair effects under sufficient support;
- factorization predicts withheld low-rank pairs;
- set model learns a nonlinear symmetric function;
- graph/hypergraph model recovers a true higher-order signal;
- uncertainty intervals achieve approximate coverage;
- no-signal simulations do not produce stable chemistry leaderboards.

## 33.4 Data quality gates

Pipeline fails when:

- final scores do not reconcile beyond tolerance;
- five-player lineup invariant is violated;
- duplicate game/event/possession keys appear;
- missingness exceeds season threshold;
- unexpected schema changes occur;
- player minutes materially disagree with box scores;
- source coverage drops;
- correction count spikes without review.

## 33.5 Model quality gates

A model cannot become `candidate` unless:

- training is reproducible from config;
- leakage suite passes;
- all primary folds complete;
- calibration report exists;
- seed variability is measured;
- performance is compared with required baselines;
- model card is drafted;
- artifact checksum and environment are stored.

It cannot become `production` unless:

- pre-declared promotion criteria are met;
- dashboard/API contract tests pass;
- support/refusal behavior is tested;
- known limitations are approved in the release report;
- a frozen reproduction run succeeds.

## 33.6 Continuous integration tiers

### Pull request, fast

- formatting/linting/type checks;
- unit/property tests;
- tiny integration fixture;
- leakage unit tests;
- API schema tests.

### Nightly or scheduled

- complete fixture corpus;
- miniature end-to-end training;
- parser regression suite;
- synthetic recovery;
- dependency/security scans.

### Release

- full data audit;
- required training/evaluation runs or verified immutable artifacts;
- paper table/figure regeneration;
- dashboard smoke test;
- model/data card validation.

## 33.7 Test ownership

Every discovered data or modeling bug must add a regression test. Parser rules cannot change without a fixture, rationale, and before/after season audit.

---

# 34. Reproducibility and experiment tracking

## 34.1 Reproducibility levels

1. **Code reproducibility:** same code and input produce same outputs within tolerance.
2. **Computational reproducibility:** clean environment can rebuild the release.
3. **Statistical reproducibility:** conclusions survive seeds/folds/specifications.
4. **Data reproducibility:** source snapshots or documented acquisition scripts recreate inputs where legally possible.
5. **Decision reproducibility:** a historical prediction can be reconstructed exactly as of its cutoff.

## 34.2 Run provenance

Each run records:

- run/experiment ID;
- Git commit and dirty-state indicator;
- resolved config;
- data and split manifest IDs;
- environment lock hash;
- host/device information;
- random seeds;
- start/end time;
- metrics by fold;
- artifact hashes;
- parent/base model IDs;
- notes and status.

## 34.3 Experiment tracking stack

Start with a lightweight local system such as MLflow or Weights & Biases if desired, plus repository-owned YAML/CSV registry for durable summaries. The external service is convenient; the committed registry is the long-term index.

Artifact types:

- resolved config;
- logs;
- checkpoints;
- coefficient tables;
- embeddings;
- posterior summaries/draw references;
- split manifests;
- evaluation tables;
- calibration plots;
- model card draft.

## 34.4 Experiment lifecycle

```text
proposed → preregistered → running → completed
         ↘ cancelled     ↘ failed
completed → accepted / rejected / superseded
```

“Rejected” experiments remain in the registry with their result. Failed ideas are valuable evidence against repeated dead ends.

## 34.5 Determinism

- set Python/NumPy/framework seeds;
- record data-loader and CUDA determinism settings;
- accept that some GPU operations are nondeterministic and document them;
- compare multiple seeds rather than trusting a nominally deterministic run;
- use tolerance-based artifact tests.

## 34.6 Environment management

- pin Python and dependency versions;
- commit lockfile;
- containerize release path;
- record GPU/CUDA versions when relevant;
- isolate developer conveniences from required dependencies;
- offer CPU-only baseline reproduction.

## 34.7 Data/version tools

Options include DVC, lakeFS, object-store manifests, or content-addressed Parquet. Minimum acceptable design:

- raw and derived hashes;
- schema version;
- row counts;
- source lineage;
- immutable snapshot ID;
- no silent overwrite.

## 34.8 Reproduction tiers

Provide:

- **Smoke reproduction:** tiny fixture in minutes.
- **Baseline reproduction:** selected seasons/RAPM on CPU in hours.
- **Paper reproduction:** all prepared snapshots and models, potentially GPU/multiday.
- **From-source reconstruction:** documented, subject to provider availability and terms.

## 34.9 Paper artifact contract

Every figure and table is generated from code with:

- source result IDs;
- plotting script/function;
- data/model version annotation;
- deterministic export settings;
- publication and web formats.

No manually edited numerical figure values.

---

# 35. Compute strategy

## 35.1 Principle

Spend compute only after the data, split, and baseline are trusted. Most early stages fit on a modern laptop.

## 35.2 Workload estimates to measure

- raw data size by season;
- normalized event and possession row counts;
- sparse design matrix dimensions and nonzeros;
- pair/trio feature counts;
- batch throughput;
- peak memory;
- training time per epoch/fold/seed;
- inference latency;
- posterior sampling diagnostics and effective samples per second.

Do not lock architecture based on guessed scale.

## 35.3 Compute phases

### Phase A: local CPU

- ingestion and parsing;
- DuckDB/Parquet analytics;
- descriptive reports;
- ridge RAPM;
- synthetic tests;
- API/dashboard prototype.

### Phase B: single GPU

- matrix factorization at larger scale;
- neural embeddings;
- Deep Sets and attention;
- moderate graph experiments;
- seed/fold sweeps.

### Phase C: targeted cloud or university compute

- broad hyperparameter searches;
- dynamic models over many seasons;
- graph/hypergraph models;
- approximate Bayesian large-scale models;
- release reproduction.

Use UT research computing resources if legitimately available and permitted; keep the project reproducible without assuming private infrastructure.

## 35.4 Sparse linear algebra

RAPM and explicit pair models require sparse matrices. Use sparse CSR/CSC representations and solvers that do not densify. Profile:

- coordinate construction;
- matrix storage;
- iterative ridge solvers;
- cross-validation reuse;
- pair feature pruning or hashing only if evaluated for collisions.

## 35.5 Neural training efficiency

- pre-encode player indices and context tensors;
- store compact columnar shards;
- mixed precision after correctness validation;
- gradient accumulation if needed;
- early stopping;
- batch-size scaling study;
- cache repeated lineup encodings only when it does not break training gradients;
- profile input pipeline before adding hardware.

## 35.6 Search strategy

Use staged tuning:

1. broad low-fidelity search on a development window;
2. narrow search on full training folds;
3. five-seed confirmation of finalists;
4. one untouched test evaluation.

Prefer random/Bayesian search over exhaustive grids for neural models. Match tuning budgets across competing families.

## 35.7 Bayesian compute strategy

- validate model structure on synthetic and one-season subsets;
- use non-centered parameterizations;
- exploit sufficient statistics for aggregated likelihoods;
- benchmark MCMC against approximations;
- store compact posterior draws strategically;
- do not run full-scale MCMC before sampler health is proven.

## 35.8 Cost controls

- per-experiment time/cost estimate in registry;
- automatic stop for divergence or no learning;
- cache immutable features;
- reuse split manifests;
- checkpoint recoverable long runs;
- artifact retention policy;
- monthly compute budget;
- no unbounded sweep on the final test set.

## 35.9 Carbon and efficiency reporting

For large final experiments, record hardware hours and approximate energy/carbon if practical. Report when a simpler model provides nearly equal performance at a fraction of the cost.

---

# 36. Staged execution roadmap

The roadmap is deliberately sequential in scientific dependency, but tasks inside a stage can run in parallel. Time ranges assume part-time student development and should be updated after the first two milestones.

## Stage 0 — Research contract and project skeleton

**Indicative duration:** 1 week  
**Goal:** freeze the question, claims, conventions, and reproducible shell before touching large data.

### Tasks

- choose working name and create repository;
- write concise README vision;
- copy this blueprint into `docs/MASTER_PLAN.md`;
- select Python/runtime/package manager;
- create source/test/config/report directories;
- add lint, formatting, type checking, and CI;
- write data-source registry template;
- write first architecture decision record;
- define canonical units: points per possession and per 100;
- define offense/defense signs;
- define lineup canonicalization;
- pre-register primary research question and initial hypotheses;
- identify legal/terms constraints for each data source;
- establish experiment and dataset ID formats.

### Deliverables

- repository boots from a clean environment;
- smoke test passes;
- `RESEARCH_CONTRACT.md`;
- `DATA_SOURCES.md`;
- `possession_rules.md` skeleton;
- initial project board/issues.

### Exit criteria

- a new contributor can install and run tests from the README;
- primary outcome, split philosophy, and claims ladder are written;
- no secret or unversioned configuration is required for fixture tests.

## Stage 1 — Raw ingestion and immutable bronze layer

**Indicative duration:** 2–3 weeks  
**Goal:** obtain two development seasons with complete provenance.

### Tasks

- ingest schedules/game IDs;
- download raw play-by-play and box scores with respectful retry/rate limiting;
- cache raw payloads using content hashes;
- collect roster/player metadata;
- record source timestamps and response schemas;
- implement resumable ingestion;
- detect missing games and duplicate payloads;
- build command-line entry points;
- create source availability dashboard/report;
- add a tiny committed fixture set.

### Deliverables

- immutable raw snapshot manifest;
- completeness report for two seasons;
- retry/failure log;
- source schema documentation;
- reproducible ingestion runbook.

### Exit criteria

- all expected completed games are present or explicitly listed as exclusions;
- repeated ingestion is idempotent;
- raw payloads are never silently overwritten;
- every file is traceable to source and retrieval time.

## Stage 2 — Event normalization and lineup state machine

**Indicative duration:** 3–5 weeks  
**Goal:** reconstruct who is on the floor at every relevant event.

### Tasks

- normalize event types, clock, teams, players, and scores;
- resolve starting lineups;
- implement substitution state machine;
- canonicalize same-clock ordering;
- add structured correction table;
- create high-risk event fixtures;
- compare parser output with pbpstats on sampled games;
- reconcile period scores and minutes;
- calculate lineup-state confidence;
- generate per-game audit traces.

### Deliverables

- normalized event schema;
- tested lineup state machine;
- 25+ manually audited development games;
- correction registry;
- season data-quality report v1.

### Exit criteria

- all modeled events have valid five-player lineups or are explicitly excluded;
- final scores reconcile;
- minute discrepancies stay within declared tolerance;
- edge-case fixtures cover every documented correction rule;
- parser differences from pbpstats are explained for samples.

## Stage 3 — Possessions, stints, and analytical gold layer

**Indicative duration:** 3–4 weeks  
**Goal:** produce trustworthy model-ready observations.

### Tasks

- formalize possession-boundary rules;
- implement possession construction;
- implement stint construction;
- label partial and ambiguous possessions;
- create garbage-time/leverage variants;
- derive lineup/pair/trio exposure tables;
- reconcile totals at game/team/player levels;
- output Parquet and DuckDB snapshots;
- create schema and lineage manifests;
- benchmark row count, storage, and build time.

### Deliverables

- gold possession and stint snapshots;
- data dictionary;
- reconciliation report;
- descriptive coverage report;
- possession fixture suite.

### Exit criteria

- final-score and player-minute checks pass;
- possession counts are plausible and explained relative to provider totals;
- snapshot is immutable and fully traceable;
- ambiguous cases are quantified rather than hidden.

## Stage 4 — Descriptive paper: why raw lineup chemistry fails

**Indicative duration:** 2 weeks  
**Goal:** establish the sparsity and reliability problem before modeling.

### Tasks

- calculate raw lineup/pair ratings;
- chart exposure distributions;
- measure within-season and cross-season reliability;
- show opponent/teammate confounding examples;
- build basic shrinkage baseline;
- measure exact-lineup recurrence;
- quantify unseen-pair/lineup rates in chronological splits;
- run negative controls;
- create a polished exploratory report.

### Deliverables

- “Why naïve chemistry fails” report;
- publication-ready sparsity figures;
- baseline metric table;
- candidate evaluation cohort definitions.

### Exit criteria

- project can explain with evidence why raw net rating is insufficient;
- no unresolved coverage anomaly undermines model training;
- initial holdout feasibility is known.

## Stage 5 — RAPM reproduction

**Indicative duration:** 3–4 weeks  
**Goal:** establish a defensible additive player-impact baseline.

### Tasks

- build sparse stint design matrix;
- fit combined and separate offense/defense ridge models;
- compare target/weighting variants;
- tune regularization chronologically;
- run synthetic recovery;
- bootstrap by game/team-season;
- compare coefficient stability across seasons;
- reconcile with public reference estimates where appropriate;
- document coding and units;
- publish model card.

### Deliverables

- reproducible RAPM pipeline;
- player estimate table with uncertainty;
- regularization/stability report;
- future-season baseline metrics;
- RAPM model card.

### Exit criteria

- synthetic recovery passes;
- RAPM beats context/team baselines out of sample;
- estimates are directionally and numerically explainable;
- leakage tests pass;
- every output includes cutoff/snapshot/model identifiers.

## Stage 6 — Hierarchical and lineup-prior baselines

**Indicative duration:** 3–5 weeks  
**Goal:** quantify partial pooling and provide strong sparse-lineup comparators.

### Tasks

- implement empirical Bayes lineup shrinkage;
- reproduce an L-RAPM-style informed-prior baseline;
- fit tractable hierarchical player model;
- run prior/posterior predictive checks;
- compare bootstrap and posterior intervals;
- test multi-season player hierarchy;
- calibrate intervals by exposure;
- define standard uncertainty outputs.

### Deliverables

- shrunk lineup baseline;
- hierarchical player benchmark;
- uncertainty calibration report;
- posterior/sampler diagnostic appendix.

### Exit criteria

- low-sample predictions improve or a null result is established;
- interval coverage is acceptable on simulations and holdouts;
- prior sensitivity is documented;
- approximation versus MCMC differences are understood.

## Stage 7 — Explicit pair chemistry

**Indicative duration:** 4–6 weeks  
**Goal:** build the first direct chemistry estimator.

### Tasks

- construct sparse teammate pair design;
- fit ridge/elastic-net/hierarchical variants;
- separate offensive and defensive pair effects;
- analyze pair identifiability and co-play graph;
- create observed-pair evaluation cohort;
- run chronological/seen-pair tests;
- quantify shrinkage and sign stability;
- implement pair explorer prototype;
- test whether pair effects add value beyond RAPM.

### Deliverables

- pair interaction model and card;
- pair effect table with uncertainty/exposure;
- adjusted-versus-raw pair case studies;
- identifiability report;
- initial Pair Chemistry Surplus definition.

### Exit criteria

- pair effects improve a declared task or are retained only as descriptive estimates;
- low-exposure extremes are controlled;
- estimates are not displayed without support metadata;
- pair model behavior is validated on synthetic schedules.

## Stage 8 — Low-rank complementarity and unseen-pair prediction

**Indicative duration:** 4–6 weeks  
**Goal:** move from memorizing pairs to learning transferable interaction structure.

### Tasks

- implement symmetric low-rank pair factorization;
- implement directional provision/need factorization;
- compare ranks and priors;
- align latent spaces across seeds;
- construct unseen-pair and first-partnership splits;
- evaluate sparse/novel pairs;
- build complementarity and similarity metrics;
- probe embeddings after training;
- compare with explicit pair RAPM and pair-average baselines.

### Deliverables

- factorization models;
- unseen-pair benchmark table;
- Complementarity Score;
- embedding stability report;
- nearest-neighbor/probe report.

### Exit criteria

- factorization beats explicit pair identifiers on unseen/sparse pairs or the hypothesis is rejected;
- gains persist across folds and seeds;
- similarity and complementarity are separated in code/UI;
- latent claims survive alignment/stability tests.

## Stage 9 — Neural embeddings with talent separation

**Indicative duration:** 4–6 weeks  
**Goal:** test whether nonlinear representations improve possession and lineup transfer.

### Tasks

- implement standardized PyTorch/JAX model interface;
- initialize or compare with RAPM/factorization;
- build additive talent skip pathway;
- build residual interaction pathway;
- compare outcome heads;
- enforce timestamp-safe loaders;
- tune on development splits;
- run five-seed finalist comparison;
- calibrate probabilities/intervals;
- implement counterfactual decomposition.

### Deliverables

- neural embedding model;
- talent-versus-chemistry ablation;
- seed/calibration report;
- standardized embedding artifact;
- initial Lineup Chemistry Surplus implementation.

### Exit criteria

- model beats factorization/RAPM on a primary transfer metric or remains experimental;
- flexible pathway is shown not merely to duplicate talent;
- decomposition sums exactly;
- predictions are stable enough for interpretation.

## Stage 10 — Deep Sets and unseen-lineup benchmark

**Indicative duration:** 4–6 weeks  
**Goal:** make lineup composition structurally correct and answer the north-star question.

### Tasks

- implement Deep Sets and pairwise invariant encoders;
- write permutation property tests;
- create exact unseen-lineup splits;
- stratify by constituent-pair/trio familiarity;
- compare pooling operators;
- evaluate macro and micro metrics;
- measure fifth-player decision regret;
- calibrate by novelty class;
- build Lineup Builder prototype.

### Deliverables

- unseen-lineup benchmark;
- permutation-invariant model card;
- lineup decomposition API;
- fifth-player held-out evaluation;
- north-star result memo.

### Exit criteria

- exact held-out lineup manifests pass zero-overlap audit;
- invariant tests pass;
- result clearly answers whether interactions improve prediction beyond additive RAPM;
- uncertainty expands appropriately on novel lineups.

## Stage 11 — Attention, graph, and hypergraph research track

**Indicative duration:** 6–10 weeks  
**Goal:** test adaptive and higher-order interactions without losing the baseline discipline.

### Tasks

- implement set self- and cross-attention;
- implement typed possession graph;
- implement a tractable hypergraph formulation;
- match capacity/tuning budgets;
- run synthetic higher-order recovery;
- perform edge/hyperedge ablations;
- evaluate explanation faithfulness;
- compare compute and stability;
- decide paper-main-text versus appendix status.

### Deliverables

- attention/graph/hypergraph benchmark;
- higher-order variance analysis;
- explanation faithfulness report;
- complexity-versus-benefit table.

### Exit criteria

- each advanced architecture has a documented scientific contribution;
- no model is promoted merely for novelty;
- synthetic and real-data findings are consistent enough to interpret;
- matched-budget comparison is complete.

## Stage 12 — Temporal models and portability/dependency

**Indicative duration:** 4–7 weeks  
**Goal:** model changing players and distinguish versatile fit from context dependency.

### Tasks

- implement player-season or dynamic latent states;
- align states across time;
- define reference lineup/context sampler;
- compute marginal value distributions;
- implement Portability Score and Dependency Index;
- validate against team-change holdouts;
- test sensitivity to team/coach effects;
- create temporal profile visualizations;
- audit fairness of labels such as “dependent.”

### Deliverables

- dynamic model benchmark;
- portability/dependency metric cards;
- temporal player reports;
- cross-team validation analysis.

### Exit criteria

- metrics are distinct from total impact by construction and empirically;
- rankings are stable across reasonable reference distributions;
- team/coach confounding is disclosed;
- uncertainty and support accompany every score.

## Stage 13 — Historical transaction backtest

**Indicative duration:** 5–8 weeks  
**Goal:** test practical pre-observation fit prediction.

### Tasks

- source and validate transaction timestamps;
- define eligibility without outcomes;
- build as-of feature/model snapshots;
- identify destination core lineups at cutoff;
- generate frozen pre-move predictions;
- evaluate multiple post-move windows;
- compare with talent/team/similarity baselines;
- bootstrap by transaction;
- document injuries/roster disruptions;
- write selected cases only after aggregate analysis.

### Deliverables

- transaction dataset and protocol;
- all-eligible-trade results;
- calibration and rank/regret analysis;
- trade case-study cards;
- strict causal limitations section.

### Exit criteria

- no transaction prediction uses later data;
- all eligibility rules are reproducible;
- aggregate results include failures;
- conclusions are framed as predictive associations.

## Stage 14 — Product/API implementation

**Indicative duration:** 6–10 weeks, overlapping late research  
**Goal:** turn validated results into a transparent, usable system.

### Tasks

- materialize player/pair/lineup predictions;
- implement typed versioned API;
- build Player Explorer, Pair Explorer, and Lineup Builder;
- add Fifth-Player/Replacement Finder;
- add evidence drawers and model comparison;
- implement support/refusal logic;
- add exportable tables/charts;
- test accessibility and responsive layout;
- performance profile and cache;
- deploy a stable public or recorded demo.

### Deliverables

- dashboard MVP then release candidate;
- API documentation;
- product analytics/model limitation copy;
- demo video/GIF;
- deployment runbook.

### Exit criteria

- no number lacks provenance and uncertainty access;
- historical roster/date behavior is correct;
- out-of-support cases are not presented confidently;
- critical user flows pass automated and manual tests;
- p95 performance targets are measured.

## Stage 15 — Paper, release, and defense

**Indicative duration:** 4–8 weeks  
**Goal:** produce a publication-quality, independently reproducible final result.

### Tasks

- freeze data/model/split release candidates;
- run untouched final test set once;
- finalize tables, figures, and appendices;
- write paper and executive summary;
- complete data and model cards;
- run clean-room reproduction;
- create poster/slides/demo;
- conduct adversarial internal review;
- write resume bullets from measured outcomes;
- archive artifacts and assign release DOI if possible.

### Deliverables

- preprint/technical report;
- final repository release;
- data-quality report;
- model/evaluation cards;
- interactive demo;
- slide deck/poster;
- reproducibility package.

### Exit criteria

- claims match evidence;
- all required baselines and splits are reported;
- final test was not used for tuning;
- another person can reproduce at least the baseline and paper tables;
- limitations and null results are visible, not buried.

---

# 37. Milestones and exit criteria

## 37.1 Milestone scoreboard

| Milestone | Evidence produced | Scientific gate | Product gate |
|---|---|---|---|
| M0 Foundation | research contract + repo | hypotheses fixed | clean install |
| M1 Trusted events | audit report | lineups valid | quality observatory draft |
| M2 Trusted possessions | gold snapshot | outcome units valid | data explorer |
| M3 RAPM | baseline report | adjusted talent credible | player table |
| M4 Explicit chemistry | pair model | pair signal stable | pair explorer |
| M5 Transferable chemistry | unseen-pair result | factorization beats baselines or null | complementarity view |
| M6 Unseen lineups | exact holdout result | north-star answered | lineup builder |
| M7 Higher-order study | matched model comparison | added value justified | explanation panel |
| M8 Trade test | frozen historical backtest | practical transport assessed | trade simulator |
| M9 Research release | final test + paper | claims defended | public demo |

## 37.2 North-star success criteria

Before experiments, choose quantitative thresholds based on baseline variance. A reasonable structure:

- statistically and practically meaningful improvement over hierarchical RAPM on macro unseen-lineup error;
- improvement on unseen-pair error for low-rank or learned embeddings;
- no material degradation in calibration;
- stable ranking improvement across at least three rolling-origin folds;
- seed variance smaller than the improvement;
- trade backtest better than talent-only baseline with uncertainty reported;
- prediction intervals attain near-nominal coverage by novelty class.

Do not hard-code arbitrary percentage gains without pilot evidence. Record thresholds before final test evaluation.

## 37.3 Minimum viable research result

The project is already portfolio-worthy if it reaches:

- trusted multi-season possession data;
- RAPM reproduction;
- explicit pair model;
- low-rank unseen-pair experiment;
- exact unseen-lineup benchmark;
- rigorous negative or positive result;
- polished technical report.

Graph models and a full product are stretch goals, not prerequisites for intellectual validity.

## 37.4 Stop conditions

Pause or redesign when:

- data inconsistencies exceed correction capacity;
- a provider’s use restrictions make the intended release inappropriate;
- split sizes are too small for reliable comparisons;
- uncertainty makes leaderboard claims meaningless;
- repeated models show no transferable interaction signal;
- compute cost is disproportionate to expected evidence.

Stopping an unproductive architecture is progress. The research question can still yield a valuable null result.

## 37.5 Progress review cadence

Weekly:

- task outcomes;
- data/test failures;
- experiment status;
- decisions needed;
- next week’s smallest verifiable deliverable.

At every milestone:

- re-run data/leakage audits;
- compare work to exit criteria;
- update risk register;
- archive configs/results;
- revise roadmap estimates;
- record a decision memo.

---

# 38. Publication-quality outputs

## 38.1 Main research paper outline

### Abstract

- problem: sparse, confounded lineup observations;
- method: talent/interaction/context decomposition;
- key evaluation: unseen pairs, unseen lineups, chronological transfer;
- principal quantitative result or honest null;
- practical implication and limitation.

### 1. Introduction

- why lineup fit matters;
- why raw lineup ratings do not generalize;
- north-star question;
- contributions list.

### 2. Related Work

- adjusted plus-minus and regularization;
- lineup/pair/group interaction models;
- representation learning/NBA2Vec;
- set, graph, and hypergraph learning;
- uncertainty and temporal sports models.

### 3. Data

- sources and coverage;
- possession/stint construction;
- reconciliation and exclusions;
- sparsity and recurrence;
- ethics/terms and reproducibility.

### 4. Problem Formulation

- estimands;
- talent/chemistry/context decomposition;
- prediction tasks;
- reference distributions;
- non-causal scope.

### 5. Methods

- baselines and RAPM;
- hierarchical models;
- explicit interactions;
- low-rank factorization;
- set/attention/graph models;
- uncertainty.

### 6. Evaluation Protocol

- chronological folds;
- unseen-lineup/pair construction;
- trade backtest;
- metrics and statistical comparison;
- leakage safeguards.

### 7. Results

- overall model table;
- calibration;
- novelty/exposure stratification;
- ablations;
- trade backtest;
- compute/complexity.

### 8. Interpretation

- latent probes;
- complementarity versus similarity;
- portability/dependency;
- carefully selected cases after aggregate results.

### 9. Limitations

- observational confounding;
- public-data limitations;
- model dependence of chemistry;
- changing roles/systems;
- trade counterfactual problem;
- extrapolation and uncertainty.

### 10. Conclusion

- exactly what was demonstrated;
- what was not demonstrated;
- next research step.

## 38.2 Required main figures

1. System overview: data → talent/interaction/context → evaluation/product.
2. Distribution of lineup and pair exposure.
3. Raw versus shrunk lineup estimates.
4. Model performance across random/future/unseen-lineup/unseen-pair tests.
5. Calibration by novelty class.
6. Talent versus chemistry decomposition examples.
7. Low-rank or embedding complementarity map with caveats.
8. Trade-backtest predicted versus realized outcome.
9. Portability versus impact scatter with uncertainty.
10. Model complexity/compute versus performance.

## 38.3 Required main tables

- dataset coverage and exclusions;
- model definitions/parameter counts;
- primary evaluation matrix;
- unseen-lineup results by subgroup familiarity;
- unseen-pair results by exposure;
- ablation results;
- calibration/coverage;
- trade backtest versus baselines;
- stability across seeds/specifications.

## 38.4 Appendix

- full possession rules;
- data corrections and reconciliation;
- hyperparameters/search spaces;
- prior choices and posterior diagnostics;
- every ablation;
- subgroup results;
- alternate target/garbage-time policies;
- synthetic recovery;
- additional case studies;
- reproducibility checklist;
- metric cards.

## 38.5 Model card template

Each released model card states:

- purpose and non-purpose;
- training data and cutoff;
- inputs and outputs;
- architecture and parameter count;
- evaluation results by split;
- calibration;
- uncertainty method;
- known failure modes;
- support boundaries;
- ethical/interpretive risks;
- version/change log.

## 38.6 Data card template

- source/provider;
- seasons/leagues;
- retrieval dates;
- raw/derived schemas;
- possession policy;
- corrections;
- exclusions/missingness;
- identity resolution;
- terms/distribution limits;
- intended uses;
- known biases;
- validation report.

## 38.7 Metric cards

One page each for PCS, LCS, LSA, Portability, Dependency, Complementarity, Replacement Preservation, and Centrality:

- plain-language question;
- formula;
- model dependence;
- reference distribution;
- uncertainty;
- minimum support;
- valid comparisons;
- invalid interpretations;
- worked example.

## 38.8 Public release package

- polished README with one validated headline result;
- architecture and research diagrams;
- installation and smoke reproduction;
- frozen report/preprint;
- model/data/metric cards;
- demo application or recorded walkthrough;
- sample data only if raw distribution is constrained;
- artifact download instructions;
- citation file;
- explicit license boundaries.

## 38.9 Presentation assets

### 60-second recruiter view

- problem;
- scale;
- one model diagram;
- unseen-lineup result;
- interactive lineup demo.

### 10-minute interview presentation

- why raw ratings fail;
- data construction challenge;
- RAPM baseline;
- interaction representation;
- leakage-safe validation;
- result and limitation;
- product demo.

### 30-minute technical talk

Add:

- hierarchy and uncertainty;
- matrix factorization;
- set/graph comparisons;
- ablations;
- trade protocol;
- future causal/decision work.

## 38.10 Venue targets

Potential targets depend on final quality and eligibility:

- MIT Sloan Sports Analytics Conference research-paper or poster tracks;
- Journal of Quantitative Analysis in Sports;
- sports analytics workshops;
- university undergraduate research symposium;
- arXiv preprint;
- reproducible open-source release.

Verify current submission rules and dates before targeting any venue.

---

# 39. Resume, portfolio, and interview framing

## 39.1 Resume framing rule

Use measured facts only after results exist. Never pre-fill improvement percentages or claim millions of possessions until the pipeline confirms them.

## 39.2 Pre-result project description

> Building a research-grade NBA lineup modeling system that reconstructs possession-level lineups, estimates adjusted player and interaction effects, and evaluates learned player embeddings on chronologically held-out and unseen lineup combinations.

## 39.3 Post-result bullet templates

Fill brackets with verified numbers:

- Processed **[N] NBA possessions across [S] seasons** into an audited lineup database, reconciling scores, substitutions, player minutes, and possession boundaries with **[X]% valid-lineup coverage**.
- Implemented ridge RAPM, hierarchical shrinkage, low-rank pair factorization, and permutation-invariant neural lineup encoders; improved **unseen-lineup [metric] by [X]%** over the strongest additive baseline.
- Designed leakage-safe evaluation for exact unseen lineups, unseen teammate pairs, future seasons, and **[N] historical transactions**, with calibrated uncertainty and model-ablation analysis.
- Built a versioned API and interactive roster-construction dashboard that decomposes lineup predictions into **individual talent, pair fit, higher-order chemistry, context, and uncertainty**.
- Open-sourced a reproducible experiment pipeline with immutable data/split manifests, synthetic parameter-recovery tests, model cards, and publication-quality reports.

## 39.4 Strong interview narrative

### Situation

Raw five-player lineup ratings appear useful but most groups play tiny, non-random samples.

### Task

Estimate whether player combinations create value beyond individual talent and determine if the signal transfers before the exact group is observed.

### Action

- reconstructed and audited possessions;
- reproduced RAPM;
- added hierarchical/interaction models;
- learned low-rank and permutation-invariant representations;
- built group/time-safe evaluation;
- quantified uncertainty and leakage;
- translated results into an evidence-aware product.

### Result

State the actual result, including a null if necessary. The strongest intellectual point is the validation design, not the fanciest model name.

## 39.5 Interview questions this project should answer

- Why not use raw net rating?
- Why ridge for RAPM?
- How did you choose the regularization parameter?
- What makes chemistry identifiable?
- How did you prevent the interaction network from re-learning talent?
- Why must the lineup encoder be permutation invariant?
- How do explicit pair terms differ from factorization?
- How do you predict an unseen pair?
- Why is a random split misleading?
- How did you construct the unseen-lineup holdout?
- How was uncertainty calibrated?
- What does an embedding dimension mean?
- Why isn’t attention a guaranteed explanation?
- What causal claims can and cannot be made?
- What would you do with tracking data?
- Which model won, and was the gain worth the compute?
- What was the hardest data bug?
- How could the model fail after a trade?

## 39.6 Portfolio homepage structure

1. one-sentence outcome;
2. interactive or recorded lineup decomposition;
3. unseen-combination model comparison;
4. possession pipeline diagram;
5. key finding and uncertainty;
6. trade test;
7. limitations;
8. links to code, report, demo, and reproducibility.

## 39.7 Avoid weak framing

Do not say:

- “used AI to find the best NBA lineups”;
- “built a GNN that predicts chemistry” without evidence;
- “proved Player X makes Player Y better”;
- “achieved high accuracy” on possession points without baseline/context;
- “created embeddings like word2vec” as the main contribution.

Prefer:

> “I turned a sparse, confounded roster-construction question into a set of falsifiable out-of-sample tasks and built the data, statistical baselines, representation models, uncertainty system, and product needed to answer them.”

---

# 40. Risk register

| Risk | Probability | Impact | Early warning | Mitigation | Fallback |
|---|---|---|---|---|---|
| Provider/API instability | High | High | missing games/schema changes | immutable cache, adapters, backoff | use existing snapshots; limit coverage |
| Data-use restrictions | Medium | High | unclear redistribution terms | source review, distribute code/manifests | release samples and derived aggregates only |
| Incorrect lineup state | Medium | Critical | minute/score mismatches | fixtures, audits, correction registry | exclude low-confidence possessions |
| Possession definition disputes | High | Medium | provider count differences | formal policy + sensitivity runs | stint-based primary model |
| RAPM multicollinearity | High | High | unstable coefficients | ridge/hierarchy, diagnostics | emphasize predictive not individual rank claims |
| Pair parameter explosion | High | High | memory/overfit | sparse ops, shrinkage, factorization | high-exposure descriptive pairs only |
| No transferable chemistry signal | Medium | High | no unseen gains | strong preregistered null study | publish limits of public lineup inference |
| Interaction path absorbs talent | Medium | High | chemistry correlated nearly 1 with impact | residualization, orthogonality, ablation | explicit low-rank model |
| Bad calibration on novel groups | High | High | undercoverage | novelty-specific calibration, ensembles | refuse/flag extrapolations |
| Temporal player drift | High | Medium | old models fail recent seasons | dynamic state/recency | shorter rolling windows |
| Team/coach confounding | High | High | effects vanish with controls | system effects, transfer tests | narrower “predictive fit” claim |
| Trade selection/counterfactual bias | High | High | famous-case dependence | all-eligible protocol | treat as observational case study |
| Small transaction sample | High | Medium | wide intervals | pool seasons, hierarchical eval | focus on unseen-pair task |
| Compute cost | Medium | Medium | slow sweeps/OOM | profiling, staged search | simpler factorization/set model |
| Embedding instability | Medium | Medium | neighbors change by seed | alignment, ensembles, stability metrics | report pair predictions only |
| Dashboard overstates certainty | Medium | High | users quote single scores | evidence drawers, warnings, lower-bound rank | limit public features |
| Scope explosion | High | High | no completed milestones | gated roadmap | stop after minimum viable research result |
| Research novelty weaker than assumed | Medium | High | related work overlap | living literature review | emphasize benchmark/validation contribution |
| Reproducibility failure | Medium | High | clean run differs | locks, manifests, release rehearsal | publish prepared artifacts + baseline path |
| Individual player reputational harm | Low | Medium | reductive labels | neutral language, uncertainty | avoid sensational leaderboards |

## 40.1 Scope-control rules

- no tracking data until public PBP result is complete;
- no salary-cap optimizer until prediction is validated;
- no GNN before low-rank and Deep Sets baselines;
- no live current-season product before historical snapshots work;
- no single-score chemistry leaderboard before uncertainty and metric cards;
- no public causal language without a dedicated identification study.

## 40.2 Risk review trigger

Revisit this register after each milestone and whenever:

- a source changes;
- a model is promoted;
- a new public feature is added;
- a major claim changes;
- final test evaluation begins.

---

# 41. Decision log and experiment registry templates

## 41.1 Architecture decision record

```markdown
# ADR-00X: [Decision title]

Date: YYYY-MM-DD
Status: proposed | accepted | superseded
Owners: ...

## Context
What problem requires a decision?

## Options considered
1. ...
2. ...

## Decision
What was chosen?

## Evidence
Benchmarks, constraints, references, or experiments.

## Consequences
Positive, negative, and follow-up work.

## Revisit trigger
What new evidence would change this decision?
```

## 41.2 Experiment proposal

```markdown
# E0XX: [Experiment name]

Status: proposed
Hypothesis: ...
Scientific question: ...
Primary metric: ...
Secondary metrics: ...
Minimum practical effect: ...

Data snapshot: ...
Split manifest: ...
Train cutoff: ...
Validation protocol: ...
Untouched test: ...

Models/baselines: ...
Search budget: ...
Seeds: ...
Expected compute: ...

Leakage checks: ...
Failure/falsification condition: ...
Decision rule: ...
```

## 41.3 Experiment result

```markdown
# E0XX result

Run IDs: ...
Completed: YYYY-MM-DD
Outcome: accepted | rejected | inconclusive

## Primary result
Estimate, interval, and baseline difference.

## Calibration/stability
...

## Diagnostics
...

## Surprises and failures
...

## Interpretation
What is supported and not supported?

## Decision
Promote, revise, stop, or run follow-up.

## Artifacts
Links/hashes for tables, plots, configs, checkpoints.
```

## 41.4 Metric definition record

```markdown
# Metric: [Name]

Version: ...
Question answered: ...
Formula: ...
Unit: ...
Reference population: ...
Required model outputs: ...
Uncertainty method: ...
Minimum support: ...
Valid interpretations: ...
Invalid interpretations: ...
Validation evidence: ...
```

## 41.5 Data correction record

```yaml
correction_id: C000123
game_id: ...
event_id: ...
detected_by: reconciliation_check
issue: ...
source_value: ...
corrected_value: ...
evidence: ...
rule_scope: one_event | game | provider_pattern
implemented_in: ...
regression_fixture: ...
reviewed_by: ...
```

## 41.6 Weekly operating note

```markdown
# Week of YYYY-MM-DD

## Shipped
- ...

## Evidence
- tests/reports/metrics produced

## Blockers or risks
- ...

## Decisions
- ...

## Next smallest verifiable outcomes
1. ...
2. ...
3. ...
```

---

# 42. Definition of done

The project is genuinely complete when all of the following are true.

## 42.1 Data

- [ ] Multi-season raw inputs are versioned and traceable.
- [ ] Entity resolution is documented.
- [ ] Possession and stint rules are explicit.
- [ ] Final scores, minutes, and lineup validity reconcile within tolerance.
- [ ] Corrections and exclusions are public in the quality report.
- [ ] Development and paper snapshots are immutable.

## 42.2 Baselines

- [ ] Constant, context, team, raw lineup, and shrunk lineup baselines exist.
- [ ] RAPM is reproduced and validated synthetically.
- [ ] Hierarchical/informed-prior sparse-lineup baseline exists.
- [ ] Baselines use the same split and metric protocol as advanced models.

## 42.3 Chemistry modeling

- [ ] Additive talent and interaction components are explicit.
- [ ] Pair chemistry is estimated with shrinkage and uncertainty.
- [ ] Low-rank complementarity supports unseen-pair prediction.
- [ ] At least one permutation-invariant lineup encoder is implemented.
- [ ] Higher-order models are compared or explicitly deferred.
- [ ] Metric cards define chemistry, portability, dependency, complementarity, replacement, and centrality.

## 42.4 Evaluation

- [ ] Rolling chronological validation is complete.
- [ ] Exact unseen-lineup test is zero-overlap audited.
- [ ] Unseen-pair test is zero-co-play audited.
- [ ] Player-team transfer or trade backtest is complete.
- [ ] Macro/micro performance and calibration are reported.
- [ ] Ablations and sensitivity analyses are reported.
- [ ] Negative controls and synthetic recovery pass.
- [ ] Final test remained untouched until model freeze.

## 42.5 Uncertainty and interpretation

- [ ] Every product prediction includes uncertainty and support.
- [ ] Intervals are empirically calibrated by novelty/exposure.
- [ ] Embedding stability across seeds/time is measured.
- [ ] Probe and explanation claims pass faithfulness checks.
- [ ] Causal limitations are explicit.
- [ ] Out-of-support behavior is visible and tested.

## 42.6 Engineering

- [ ] Clean install and smoke reproduction work.
- [ ] Unit, property, integration, regression, synthetic, and leakage tests run in appropriate CI tiers.
- [ ] Runs preserve config, code, data, split, environment, and artifact provenance.
- [ ] API is versioned and returns provenance.
- [ ] Dashboard has accessible evidence-aware core flows.
- [ ] Release artifacts are checksummed and documented.

## 42.7 Research communication

- [ ] Technical report/preprint is complete.
- [ ] Dataset, model, and metric cards are complete.
- [ ] Required tables/figures regenerate from code.
- [ ] README leads with a verified result.
- [ ] Null and failed results are retained.
- [ ] Resume bullets contain only measured claims.
- [ ] Demo and interview presentation are ready.

## 42.8 Final standard

The finished project must permit a skeptical reader to reproduce this reasoning chain:

```text
Raw events are trustworthy enough
        ↓
Possessions and lineups are reconstructed correctly
        ↓
Strong additive and shrinkage baselines are established
        ↓
Chemistry is defined as a model-dependent non-additive quantity
        ↓
Interaction models improve—or fail to improve—held-out prediction
        ↓
Unseen pair and lineup tests establish the degree of transport
        ↓
Uncertainty and robustness bound the claim
        ↓
Historical transactions test practical relevance
        ↓
The product communicates the result without hiding limitations
```

If any arrow is unsupported, the downstream claim must be narrowed.

---

# 43. Selected references

This is a starting bibliography, not a completed systematic review. Before publication, expand it through forward/backward citation search and verify every bibliographic field.

## Basketball impact, lineups, and interactions

1. Sill, J. (2010). *Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing.* Proceedings of the MIT Sloan Sports Analytics Conference. [Conference paper search entry](https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing)
2. Fearnhead, P., & Taylor, B. M. (2011). *On Estimating the Ability of NBA Players.* Journal of Quantitative Analysis in Sports, 7(3). [DOI](https://doi.org/10.2202/1559-0410.1298)
3. Josephs, N., & Upton, E. (2024; revised 2025). *Hypergraph adjusted plus-minus.* [arXiv:2403.20214](https://arxiv.org/abs/2403.20214)
4. Petridis, C., & Pelechrinis, K. (2026). *Lineup Regularized Adjusted Plus-Minus (L-RAPM): Basketball Lineup Ratings with Informed Priors.* [arXiv:2601.15000](https://arxiv.org/abs/2601.15000)
5. Guan, W., Javed, N., & Lu, P. (2023). *NBA2Vec: Dense feature representations of NBA players.* [arXiv:2302.13386](https://arxiv.org/abs/2302.13386)

## Set, attention, graph, and representation learning

6. Zaheer, M., et al. (2017). *Deep Sets.* [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
7. Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* ICML/PMLR 97. [PMLR](https://proceedings.mlr.press/v97/lee19d.html)
8. Selby, K. A., Rashid, A., Kobyzev, I., Rezagholizadeh, M., & Poupart, P. (2022). *Learning Functions on Multiple Sets using Multi-Set Transformers.* UAI/PMLR 180. [PMLR](https://proceedings.mlr.press/v180/selby22a.html)
9. Hamilton, W. L. (2020). *Graph Representation Learning.* Morgan & Claypool. [Author resource](https://www.cs.mcgill.ca/~wlh/grl_book/)
10. Battaglia, P. W., et al. (2018). *Relational inductive biases, deep learning, and graph networks.* [arXiv:1806.01261](https://arxiv.org/abs/1806.01261)

## Bayesian workflow, uncertainty, and evaluation

11. Stan Development Team. *Stan User’s Guide: Prior and Posterior Predictive Checks, Cross-Validation, and Simulation-Based Calibration.* [Documentation](https://mc-stan.org/docs/stan-users-guide/index.html)
12. Gelman, A., et al. (2020). *Bayesian Workflow.* [arXiv:2011.01808](https://arxiv.org/abs/2011.01808)
13. Vehtari, A., Gelman, A., & Gabry, J. (2017). *Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.* Statistics and Computing, 27, 1413–1432. [DOI](https://doi.org/10.1007/s11222-016-9696-4)
14. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
15. Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* Journal of the American Statistical Association, 102(477), 359–378. [DOI](https://doi.org/10.1198/016214506000001437)

## Data and tooling

16. `pbpstats` documentation. Features include lineup-on-floor event enrichment and detailed possession construction. [Documentation](https://pbpstats.readthedocs.io/en/latest/)
17. `pbpstats` quickstart and source-provider notes. [GitHub documentation](https://github.com/dblackrun/pbpstats/blob/main/docs/quickstart.rst)

## Reproducibility and reporting

18. Pineau, J., et al. (2021). *Improving Reproducibility in Machine Learning Research.* Journal of Machine Learning Research, 22(164), 1–20. [JMLR](https://www.jmlr.org/papers/v22/20-303.html)
19. Mitchell, M., et al. (2019). *Model Cards for Model Reporting.* FAT*. [ACM](https://doi.org/10.1145/3287560.3287596)
20. Gebru, T., et al. (2021). *Datasheets for Datasets.* Communications of the ACM, 64(12), 86–92. [ACM](https://doi.org/10.1145/3458723)

---

## Immediate next action

Do **not** begin with a transformer, GNN, or public dashboard.

The first implementation milestone is intentionally narrow:

> **Build and audit a trustworthy two-season possession/stint dataset, then reproduce a leakage-safe ridge RAPM baseline.**

That foundation creates the only credible path to the larger ambition. Once it passes, climb the model ladder one rung at a time and require every additional layer to demonstrate what it learned that the simpler model could not.

---

# 44. Appended product capabilities and example scenarios

**Added:** 2026-08-31. **Status:** future backlog, not active implementation.

These seven additions and refinements extend CourtGraph after the existing backlog. They do not replace features, reorder the roadmap, or change the current milestone. Where an idea already appears in the blueprint, the entry below makes its product behavior concrete rather than creating a duplicate project. The earlier lineup, individual-impact, injury/rotation, and market-research ideas remain recorded in [product backlog issue #8](https://github.com/tuckerbullock/courtgraph/issues/8).

## 44.1 Player roles beyond listed positions

**Example:** Two players share a listed position; show which provides creation, spacing, rim pressure, passing, rebounding, or particular defensive functions.

- **Inputs:** permitted, dated measurements of skills, opportunities, and tendencies, with relevant lineup context.
- **Output:** a profile that allows multiple roles per player, plus supported explanations of role overlap and complementarity within a lineup.
- **Insufficient evidence:** omit unsupported traits or label them unknown. An embedding or a listed position alone does not establish a basketball role.

Refines §21.5 (role modeling), §22.6 (complementarity), and §30.4 (Player Explorer).

## 44.2 Player portability

**Example:** Compare how a candidate's predicted contribution varies across different teammate combinations and systems.

- **Inputs:** dated player estimates and supported alternative five-player lineups, with explicit replacement references and comparable opponent/context assumptions.
- **Output:** the level and variation of predicted contribution, contexts where it is stronger or weaker, and coverage/uncertainty. Consistently low predicted value must not be mistaken for excellent portability.
- **Insufficient evidence:** flag unfamiliar teammates, systems, or roles; do not claim an observed association will transfer to every team.

Refines §22.4–§22.5 (portability and dependency) and §30.4 (Player Explorer).

## 44.3 Full-game rotation planning

**Example:** Compare playing two stars together with staggering their minutes, including the resulting bench units.

- **Inputs:** a dated eligible roster, player availability assumptions, minute limits, rest constraints, opponent scenarios, and supported lineup estimates.
- **Output:** feasible rotation alternatives with time on court, offense/defense estimates, and uncertainty. In regulation, account for 48 minutes of five-player units and 240 player-minutes; handle overtime separately.
- **Insufficient evidence:** identify infeasible constraints and weakly supported combinations. Keep a suggested rotation separate from a forecast of what the coach will actually do.

Extends §30.9 (Team Fit Lab). Minute-allocation optimization remains future scope under the research contract.

## 44.4 Opponent counter-lineups

**Example:** Generate alternatives against an opponent's likely small or large units and show how the preferred five changes.

- **Inputs:** eligible player pools, opponent lineups or a declared distribution of likely units, and information available at the prediction cutoff.
- **Output:** several lineup alternatives, with talent/interaction/context decomposition and matchup-specific trade-offs. Compare candidates under the same opponent assumptions before changing the scenario.
- **Insufficient evidence:** show sparse or unseen matchups and sensitivity to opponent uncertainty. Do not use the opponent's eventual rotation as though it were known beforehand or promise how its coach will respond.

Extends §15.2 (cross-team matchups) and §30.6 (Lineup Builder).

## 44.5 What would change this answer?

**Example:** Show whether a preferred lineup remains attractive if a player receives six fewer minutes, or a future estimated market advantage disappears at a slower pace.

- **Inputs:** a saved scenario, explicit assumptions, and defensible ranges for minutes, availability, pace, opponent, and player estimates.
- **Output:** the assumptions that move the result most, ranges where the ranking changes, and alternatives that remain useful across scenarios.
- **Insufficient evidence:** distinguish user-selected sensitivity ranges from calibrated probability intervals. Report when the apparent advantage is smaller than model or input uncertainty.

Extends §23 (uncertainty), §26 (sensitivity), and §29.5 (lineup explanations). Any market application remains outside cycle 1.

## 44.6 Prediction history and error breakdown

**Example:** Reopen a forecast exactly as it existed before a game, then inspect where predicted minutes, pace, scoring, and individual production differed from observed outcomes.

- **Inputs:** the original prediction timestamp, model version, information cutoff, assumptions, and later permitted observations with their provenance.
- **Output:** immutable forecast history and separate component-level error summaries, preserving unsuccessful predictions as well as successful ones.
- **Insufficient evidence:** mark missing or revised outcomes. Component errors can overlap; do not claim a unique causal explanation for a miss without evidence. Realized minutes may support a labeled after-the-fact diagnostic, never silently replace the original forecast inputs.

Extends §24 (evaluation), §27.3 (feature timestamps), and §34 (reproducibility). Component reporting becomes available only as the relevant forecasting models exist.

## 44.7 Evidence-based similar-player search

**Example:** Replace an absent player with someone who preserves a needed lineup function, rather than merely matching scoring averages.

- **Inputs:** a selected player or role need, permitted dated skill measurements, an eligible candidate pool, and the remaining lineup/opponent context.
- **Output:** separate views for closest functional replacement, best predicted overall replacement, and better-supported choices; explain what each candidate preserves or changes.
- **Insufficient evidence:** label missing traits and unseen contexts. Similarity does not imply complementarity, equal talent, or equal predicted team value.

Refines §22.7 (replacement preservation), §30.4 (functional neighbors), and §30.8 (Replacement Finder).

## 44.8 Example scenarios to define before coding

Create a small, fixed development challenge set. Each case records its input cutoff, required inputs, expected output type, and insufficient-evidence behavior; it does not prescribe a winning player or a favorable numeric result.

| Scenario | Behavior to specify |
|---|---|
| Missing lead guard | Show replacement minutes and creation responsibilities, with supporting evidence or explicit unknowns. |
| Missing defensive center | Separate offensive and defensive implications and compare supported functional replacements. |
| Returning player with a minutes restriction | Respect the cap across the full rotation and show sensitivity to alternative available minutes. |
| Unfamiliar rookie | Expose limited individual/interaction support; use the permitted fallback or abstain instead of inventing chemistry. |
| Two high-usage stars | Compare shared and staggered minutes, including bench consequences and uncertainty about roles. |
| Completely unseen lineup | Show novelty/support flags, compare alternatives under a common opponent/context, then inspect changes across opponent scenarios. |

Use synthetic or clearly labeled hypothetical examples until suitable data permissions and model support exist. Select any historical cases before inspecting model results. These development examples support product design and later checks; they are not an untouched evaluation set and cannot establish real-world predictive value.

**Boundaries:** The existing research contract and data-source restrictions remain unchanged. This appendix authorizes no implementation, downloads, paid services, redistribution, deployment, wagers, agent dispatch, or monitoring. Research cycle 1 is not expanded; future capabilities require their applicable evidence and scope gates. Monitoring remains disabled.

---

# 45. Player-lift: a player's effect on teammates' individual production

**Added:** 2026-09-01. **Status:** Phase A **done and null** (2026-09-02,
`courtgraph player-lift` — see `docs/INTERACTION_FINDINGS.md` §45); Phase B and
the transaction backtest active. Does not reorder the roadmap or expand
research cycle 1.

## 45.1 Why this is a distinct question

Rungs 3–5 tested **symmetric** interaction — a joint `γ_ij` (or `u_i·v_j`)
attached to a *lineup's* value — and it is **not supported** on real data
(`docs/INTERACTION_FINDINGS.md`): four leakage-safe evaluation tasks, no gain
over additive talent, per-pair terms indistinguishable from a placebo.

"Does a good player make teammates better?" is a different estimand:
**asymmetric** ("giver" vs. "receiver"), **pooled** across all of a player's
teammates rather than estimated per pair, and — in its strong form — measured
on a teammate's **individual** production, not the lineup's net rating. A
lineup-value model cannot separate "no interaction" from "a lift effect that is
collinear with the giver's own additive talent"; that ambiguity is exactly
what the null leaves open, and what this item targets.

Note the relationship to rung 5: `lift_i · receptivity_k` is a rank-1
provision/need term with the receiver side **pinned to observed talent**. Rung
5's general low-rank form already failed, so Phase A is a lower-variance
variant of something with a negative prior, and must clear the same evidence
bar rather than a softer one.

## 45.2 Phase A — pooled lift on lineup value (cheap; likely confirms the null)

Add one EM-shrunk scalar per player, `λ_i ~ N(0, τ_λ²)`, to the rung-3 frame:

```
μ_s = context + Σ_{i∈off} α_i − Σ_{j∈def} β_j + Σ_{i∈off} λ_i · (A_off,s − α_i)
```

where `A_off,s = Σ_{i∈off} α_i` is total offensive talent on the floor, so the
lift term rewards lineups where high-`λ` players share the court with strong
teammates. Fit by two-stage (freeze `α` from rung 3, regress the residual on
the `λ_i·(A_off − α_i)` design) or by alternating; `τ_λ` learned as a fourth
variance component.

- **Evaluation:** the four existing leakage-safe tasks (`compare_rungs`,
  `transport`), macro RMSE + calibration vs. rung 3, plus a **placebo** —
  permute the `λ_i → player` assignment (same count, same exposure). Supported
  only if it beats rung 3 out of sample **and** beats its placebo, with
  maintained calibration and seed stability (contract §17, §25).
- **Cost:** ~1 day. Reuses the rung-4 EM and placebo machinery.
- **Expected:** another null, differently shaped. Recorded either way
  (§26); a clean null here further constrains where chemistry could hide.

## 45.3 Phase B — direct per-player on-court production model (the real deliverable)

Needs a **data extension**: per-player offensive production per stint (or per
game, on/off), attributed from the play-by-play already in the snapshots
(`pbpstats` gives player points; assists and a usage proxy are available). No
new download — the raw inputs are the same `stats_nba_pbpstats/v1` snapshots.

Model each player-stint's offensive production:

```
prod_{k,s} = base_k + Σ_{i∈off, i≠k} lift_{i} + context + noise
```

with `lift_i ~ N(0, τ_lift²)` a **pooled giver effect** — "the average bump a
teammate's per-possession offense gets when player i is also on the floor,"
holding the receiver's own level and context fixed. This is a RAPM-style
design on the player-production outcome rather than the lineup outcome.

- **Output:** a per-player `lift_i` in points/100 with a calibrated interval
  and an exposure/support flag — a genuine "makes teammates better" number,
  the first CourtGraph estimate that is not lineup-value.
- **Leakage-safe holdouts:** unseen giver-receiver pairs; chronological;
  transaction cohort (a player who changed teams — does `lift_i` predict the
  new teammates' production shift?). Placebo: shuffle the giver identity.
- **Guards:** attribution choices (what counts as "production": points only vs.
  points + assist credit) are a registered research choice stored in config,
  reported both ways; garbage-time weighting as in the stint pipeline; a
  player's own production is never a regressor on itself.
- **Cost:** ~1–2 weeks (the ingest extension dominates).

## 45.4 What "supported" requires

Contract §17's bar, adapted: a positive Phase-B result needs `lift_i` estimates
that (1) improve out-of-sample teammate-production prediction over a
receiver-only baseline, (2) beat the giver-shuffle placebo, (3) stay calibrated
and seed-stable, (4) show a non-trivial transaction-cohort signal (the
strongest test — the lift moves with the player, not the roster). Anything
less is "not supported" or "inconclusive," reported as a finding.

**Boundaries:** Same as §44. No new data acquisition beyond attributing
player production from snapshots already ingested; no schema change without a
contract amendment; research cycle 1 is not expanded. Product backlog issue #8
remains the index; this section makes the estimand and evidence bar concrete.

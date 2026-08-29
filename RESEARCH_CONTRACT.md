# CourtGraph Research Contract

> **Version:** 0.1 (first research cycle)
> **Status:** binding scientific specification — proposed, pending independent review
> **Last updated:** 2026-08-29
> **Governing document:** [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) (sections 1–5, 10–18, 21–29, 33–34, 37)

This contract converts CourtGraph's north-star ambition into precise, falsifiable
commitments for **research cycle 1**. Where the master plan lists options, this
document chooses; the master plan remains the broader blueprint.

**How this contract binds.** No model may be described as producing a "chemistry"
estimate, and no result may be published or shown in the product, unless it
satisfies the units, baselines, evaluation tasks, leakage rules, calibration
requirements, and evidence thresholds defined here. Material changes require a
dated decision-log entry (master plan §41) and a new contract version. Hypotheses
and numeric thresholds are frozen, with a timestamp, before the held-out test set
is evaluated.

---

## 1. Mission and north-star research question

CourtGraph is a research-grade system for learning and evaluating NBA lineup
chemistry, where

```text
lineup value = individual talent + player interactions + context
```

Chemistry is treated as a **model-dependent predictive interaction quantity**: the
part of a lineup's or pair's performance that is not explained by the additive
talent of its members or by measured context. It is not a claim that one player
psychologically or causally improves another.

**North-star question.**

> Can we estimate how NBA players will fit together before we have observed that
> exact combination on the court?

The hard problem is **structured generalization under sparse, confounded
observation**, not retrospective ranking of lineups that have already been seen.

---

## 2. Research questions

### 2.1 Primary research question

> Do latent interaction models predict the performance of **unobserved** NBA
> player combinations more accurately and more reliably than additive
> impact models, raw lineup statistics, and explicitly estimated pair effects,
> under leakage-safe evaluation?

### 2.2 Secondary research questions

1. How is lineup outcome variance partitioned among additive talent, observed
   pair effects, higher-order effects, context, and irreducible noise?
2. Do low-rank interaction models outperform independently estimated pair
   coefficients in sparse and unseen-pair settings?
3. Are separate offensive and defensive representations necessary, or does a
   shared representation suffice?
4. Does explicit permutation invariance improve stability and unseen-lineup
   accuracy relative to order-dependent encoders?
5. Does adding measured context improve possession prediction while shrinking the
   apparent magnitude of chemistry estimates?
6. How much do raw high-chemistry pairs shrink after opponent, teammate, and
   context adjustment?
7. Is a player's marginal (replacement) value \(M_i\) (§8) portable across
   teammates, roles, and systems, and is portability distinct from overall
   quality?
8. Can a model frozen before a transaction rank destination fit in a way
   associated with realized post-transaction outcomes?

---

## 3. Falsifiable hypotheses

Registered before final test evaluation. Each is promoted only to **supported**,
**not supported**, or **inconclusive**, and only through a registered experiment.

- **H1.** Shrunk additive RAPM outperforms raw lineup and raw pair ratings on
  chronological (future-season) prediction.
- **H2.** Explicit pair-interaction ridge improves in-sample explanation but does
  **not** reliably improve unseen-pair prediction, because pair identifiers
  cannot extrapolate.
- **H3.** Low-rank interaction factorization beats explicit pair coefficients on
  unseen-pair and sparse-pair tests.
- **H4.** A model with separate additive-talent and interaction pathways
  outperforms a single entangled pathway on transfer and decomposability.
- **H5.** Permutation-invariant set encoders are more stable under player-order
  perturbation and at least as accurate as concatenation encoders.
- **H6.** Tier-A context improves possession prediction but reduces estimated
  chemistry magnitude.
- **H7.** Most raw high-chemistry pairs shrink substantially after adjustment.
- **H8.** Portability correlates positively with impact but is a distinct
  construct.
- **H9.** Transaction-fit predictions show modest signal with wide uncertainty;
  strong causal language is unjustified.

**Thesis falsification.** The central thesis is weakened or rejected if: no
interaction model beats additive RAPM on unseen-lineup or chronological tests;
gains vanish under team-season or coach-aware controls; sparse-group predictions
are badly calibrated; latent neighborhoods are seed-unstable; the transaction
backtest is indistinguishable from talent-only and team-strength baselines; or
"chemistry" leaderboards are driven by one team, era, or a few extreme samples. A
well-characterized **null result is a valid outcome**.

---

## 4. Formal definition of lineup value

For an offensive five-player set \(L_o\), a defensive five-player set \(L_d\),
possession context \(c\), and season/time index \(t\):

- the **actual possession points** are \(Y_p \in \{0, 1, 2, \ldots\}\), the points
  the offense scores on that possession (3+ occurs via and-ones and multi-shot
  sequences);
- the **modeled categorical label** is \(Z_p \in \{0, 1, 2, 3, 4+\}\), a
  collapsed version of \(Y_p\) used by the classification target (§7);
- **expected points per possession** is
  \(\mu(L_o, L_d, c, t) = \mathbb{E}[Y_p \mid L_o, L_d, c, t]\), obtained from the
  separate expected-points regression target or from an explicitly calibrated
  conditional expectation for the `4+` tail — **never** by treating the `4+`
  class as exactly four points;
- **expected lineup value** is \(V = 100\,\mu\), in points per 100 possessions —
  the unit for every reported figure in this contract.

Decompose the outcome and the expected value as

\[
Y_p \;=\; \mu + \varepsilon_p,
\qquad
V \;=\; T(L_o, L_d, t) \;+\; C(L_o, L_d, c, t) \;+\; K(c, t) .
\]

In plain language:

- \(T\) is **additive individual talent**, in points per 100 possessions:
  \[
  T \;=\; \alpha \;+\; \sum_{i \in L_o} \beta_i^{\text{off}} \;-\; \sum_{j \in L_d} \beta_j^{\text{def}} .
  \]
  Both \(\beta_i^{\text{off}}\) and \(\beta_j^{\text{def}}\) are signed so that
  **larger is better**: a strong defender has a large positive
  \(\beta_j^{\text{def}}\) (points prevented), which is *subtracted* from the
  offense's expected output. \(T\) is what a pure "sum of the parts" model
  predicts.
- \(C\) is the **interaction surplus** (the candidate chemistry signal): the part
  of \(V\) not captured by \(T\) or \(K\). Operationally, for a lineup,
  \(\widehat{C}(L) = \widehat{V}_{\text{full}}(L) - \widehat{V}_{\text{additive}}(L)\),
  averaged over the declared reference distribution in §8.
- \(K\) is **non-player context**: home court, score state, period and time
  remaining, rest, era/pace, possession start type, and team/coach environment.
- \(\varepsilon_p\) is **residual unexplained variation**: the gap between a
  single possession's points and \(\mu\), plus model misspecification not
  attributed to \(T\), \(C\), or \(K\). NBA possessions are high-variance; most
  of what happens on any single possession is not predictable from the ten
  identities on the floor.

A **net** lineup value subtracts the symmetric expectation for the same players
on defense against a declared reference opponent. A five-star lineup that
performs exactly as its additive talent predicts has \(\widehat{C}(L) \approx 0\)
even though \(V\) is large: chemistry is the **surplus**, not the level.

---

## 5. Formal distinctions among the core concepts

| Concept | Formal definition | Plain language, and what it is not |
|---|---|---|
| **Individual talent** \(\beta_i^{\text{off}}, \beta_i^{\text{def}}\) | Per-player adjusted, shrunk contribution to expected points per 100, marginal to teammates and opponents | How much a player moves an average possession regardless of who surrounds him. Not box-score production or raw on-off. |
| **Player dependency** \(\mathrm{Dep}_i\) | Dispersion of the marginal-value function \(M_i\) (§8) across the declared replacement, teammate, opponent, and context distribution — e.g. \(\sqrt{\operatorname{Var}(M_i)}\) plus concentration of value on a few partner clusters | How much a player's value swings with who he plays with. Not a "system player" verdict; high dependency can mean a strong specialist. |
| **Complementarity** \(\mathrm{Comp}(i,j)\) | Directional provision/need inner product \(p_i^{\top} n_j + p_j^{\top} n_i\); defined even for pairs never observed together | Whether one player supplies what another needs. Not stylistic similarity — dissimilar players can be complementary. |
| **Pair and higher-order chemistry** \(\gamma_{ij}\), \(C(L)\) | Pair surplus \(\gamma_{ij}\) as defined in §8 (expected value with the \(i\)–\(j\) interaction present minus absent, averaged over complete lineups, opponents, and contexts), split offense/defense and same-team/cross-team; \(C(L)\) as in §4, decomposed into observed-pair, latent-pair, and higher-order parts | Predicted value of a pair or group beyond the sum of its members and context. Not raw two-man or five-man net rating. |
| **Context** \(K(c,t)\) | Fixed/random effects and covariates for home court, score margin, period/time, rest, era, possession start, team/coach environment | Everything about the situation that is not the ten identities. Removed before attributing surplus — not part of chemistry. |
| **Residual unexplained variation** \(\varepsilon_p\) | Possession-level variance and specification error not assigned to \(T\), \(C\), or \(K\) | The noise floor; single possessions are mostly unpredictable. Apparent surplus inside the noise band is not a finding. |

Any product surface and any paper claim must keep these six quantities
**separate**, each with its own uncertainty. Collapsing them into one "chemistry
score" is prohibited (§22).

---

## 6. Primary unit of analysis

**Binding choice.**

- **Possession** is the primary statistical unit for outcome modeling. Each
  possession is one row: an offensive five, a defensive five, a context vector,
  and an outcome. Possessions maximize row count, preserve the full scoring
  distribution, and attach context naturally.
- **Stint** (a maximal interval of unchanged ten-player membership, with
  possession count as exposure weight) is the parallel unit for the RAPM-family
  design matrix. Stints align with the adjusted plus-minus literature and reduce
  within-possession dependence.
- **Lineups, player pairs, and players** are **derived** units. Their estimands
  (§8) are *defined* at the pair and lineup level but *estimated* from
  possession/stint data and a model, never read directly off raw pair/lineup
  splits.

**Justification.** Estimating a residual concept requires the finest trustworthy
unit: possessions give the resolution needed to adjust for teammates, opponents,
and context and to calibrate probabilities; stints give a lower-variance,
RAPM-compatible view. Both are built and audited before any chemistry model is
fit. Possession non-independence is handled by clustered/block resampling and
hierarchical effects (§14–15), not by pretending rows are i.i.d.

---

## 7. Prediction targets and outcome variables

A small target suite, not a single target:

1. **Possession scoring distribution (primary modeling target).**
   \(P(Z_p \mid L_o, L_d, c, t)\) over the categorical label
   \(Z_p \in \{0,1,2,3,4+\}\), fit with categorical cross-entropy (ordinal/count
   alternatives compared). Supports calibration.
2. **Possession expected points** \(\mu = \mathbb{E}[Y_p \mid L_o, L_d, c, t]\) on
   actual points \(Y_p\), fit as a separate regression target (squared, Huber,
   and Poisson/NB-style losses compared) or recovered from target 1 with an
   explicitly calibrated `4+`-tail expectation — never by scoring `4+` as four.
3. **Stint point differential per 100 possessions**, with possession count as
   exposure weight — the RAPM target.
4. **Two-stage offensive/defensive targets.** Separate offense-vs-defense
   expected-points pathways, preferred over a single net coefficient wherever
   basketball interpretation is required.

All reported values are in **points per 100 possessions** (\(V = 100\,\mu\)).

---

## 8. Core estimands and their interpretation

Every estimand is a full-lineup quantity averaged over an explicitly declared
**reference distribution** \(\mathcal{R}\) — jointly over the players completing
the offensive lineup, the defensive lineup \(L_d\), and context \(c\). No
estimand compares lineups of different sizes or evaluates an incomplete
\(V(i,j,c)\). Supported choices for \(\mathcal{R}\) (never mixed without a label):
league-average opponent and neutral context; a selected team's expected opponent
distribution; the actual historical contexts of an observed sample; a
playoff-caliber opponent distribution; or a user-defined roster/minutes scenario.

1. **Possession scoring distribution** — the calibrated outcome model itself.
2. **Expected offensive points** \(\mathbb{E}[Y_p\mid\cdot]\) — offense-only
   expectation for a matchup and context.
3. **Expected lineup net value** — predicted offense minus predicted defense for
   the same players, under \(\mathcal{R}\).
4. **Pair interaction surplus** \(\gamma_{ij}^{\text{off}}, \gamma_{ij}^{\text{def}}\).
   For teammates \(i,j\),
   \[
   \gamma_{ij} = \mathbb{E}_{\mathcal{R}}\big[\, V_{\text{full}}(L, L_d, c) - V_{\text{no-}ij}(L, L_d, c) \,\big],
   \]
   where \(L\) ranges over complete offensive lineups containing \(i\) and \(j\)
   (the other three teammates drawn from \(\mathcal{R}\)), and
   \(V_{\text{no-}ij}\) is the same model with the \(i\)–\(j\) interaction term
   set to zero. Distinct same-team and cross-team (opponent) versions.
5. **Five-player chemistry surplus** \(C(L) = V_{\text{full}}(L) - V_{\text{additive}}(L)\),
   averaged over \(L_d\) and \(c\) under \(\mathcal{R}\) — the headline chemistry
   estimand, decomposed into observed-pair, latent-pair, and higher-order parts
   when the model supports it.
6. **Marginal (replacement) value.** For player \(i\) versus a replacement \(b\)
   in the four-teammate context \(L_{-i}\),
   \[
   M_i(b, L_{-i}, c) = V\big(L_{-i}\cup\{i\}, L_d, c\big) - V\big(L_{-i}\cup\{b\}, L_d, c\big),
   \]
   averaged over the declared distribution of \(b\), \(L_{-i}\), \(L_d\), and
   \(c\). Both lineups have five players. Reported decomposed into a talent
   change and a fit change; \(b\) defaults to a replacement-level slot unless a
   candidate set is specified.
7. **Transported fit** — a pre-event prediction for a player entering a new team
   or lineup, evaluated **only** on subsequent data.

---

## 9. Initial population and analysis scope

- **League and phase:** NBA regular season and playoffs.
- **Development window:** 3 contiguous recent seasons for pipeline construction,
  audits, and baseline development.
- **First research-cycle window:** 6 contiguous seasons ending at a fixed cutoff,
  used for the RAPM, pair, factorization, embedding, and unseen-combination
  experiments. Expansion toward the master plan's 8–10 season target happens only
  after the pipeline passes its data-quality gates and the compute budget is
  measured.
- **Inputs:** publicly accessible play-by-play, schedules, rosters, and box
  scores; historical roster moves with defensible transaction timestamps.
- **Representations:** learned primarily from on-court possession outcomes;
  box-score and shot-profile features used only as post-training probes or
  explicit priors, never as unaudited leakage paths.

Exact season identifiers, cutoff dates, and provider selection are **deferred to
the data-source task** (§29); the 3- and 6-season counts are targets subject to
confirmed data availability and terms.

---

## 10. Required contextual controls

Any model that reports a chemistry estimate must, at minimum, adjust for the
**Tier-A** context set:

- season / era (and a pace or possessions-per-48 proxy);
- home court;
- regular season vs playoffs;
- score-margin bucket or spline;
- period and time remaining;
- possession start type (made basket, live rebound, dead ball, turnover, etc.);
- garbage-time / leverage weight;
- team offensive and defensive environment.

**Tier-B** controls (rest days, back-to-back, starters-on-floor, seconds since
substitution, coach / team-season effects) are required for the portability and
transaction analyses and recommended elsewhere. **Context leakage rule:** a
feature is eligible only if its value is known at the prediction timestamp;
end-of-possession information must not enter possession features, and only
pre-transaction data may enter transaction forecasts.

---

## 11. Model ladder for cycle 1

Cycle 1 covers **rungs 0–7**. Each rung must exist, be evaluated on the same
split manifests, and be reported together before the next is attempted. No
chemistry claim is admissible until at least rungs 0–5 are complete.

| Rung | Model | Tests | Must beat (out of sample) |
|---:|---|---|---|
| 0 | Global / context-only mean | context value, target sanity | constant mean |
| 1 | Raw + empirical-Bayes shrunk lineup ratings | value of direct lineup history | rung 0 |
| 2 | Additive ridge RAPM (combined, and separate O/D) | adjusted individual talent | rung 1 |
| 3 | Hierarchical / partially pooled player model | partial pooling, uncertainty | rung 2 on calibration and stability |
| 4 | Explicit teammate-pair interaction RAPM | observed pair surplus | rung 2 on seen pairs |
| 5 | Low-rank pair factorization (symmetric and provision/need) | transfer to sparse / unseen pairs | rung 4 |
| 6 | Neural player embeddings with **separate additive-talent and interaction pathways** | nonlinear latent interactions; talent/chemistry separation | rung 5 on a primary transfer metric |
| 7 | Permutation-invariant lineup encoder (Deep Sets) | symmetric nonlinear lineup value; order stability | rung 6, and a sum/mean-embedding control |

Rungs 2 and 3 are the **reference baselines** against which chemistry usefulness
is judged (§17). Rungs 6–7 keep learned embeddings as a central CourtGraph
contribution and make H4 and H5 testable, but are **strictly gated**: they are
attempted only after rungs 0–5 pass their exit criteria, must use timestamp-safe
loaders and ≥5-seed finalist comparison, and are promoted only under §26. Master
plan **rungs 8+** (attention / set-transformer, graph, hypergraph, dynamic
hierarchical) are **out of scope for cycle 1** (§27).

---

## 12. Primary evaluation tasks

All four are required. In-sample lineup fit and conventional lineup net rating are
**not** acceptable evidence.

- **T1 — Randomly unseen lineups.** Exact five-player units with adequate test
  exposure are removed entirely from training; their players and subgroups may
  remain observed.
- **T2 — Structurally unseen teammate pairs.** Selected pairs have **all**
  shared-lineup possessions removed from training. A **strong** variant requires
  the pair's first partnership to fall in the test period (typically via a
  trade or signing).
- **T3 — Future-season temporal prediction.** Rolling-origin chronological
  holdouts (train through season \(t\), test \(t+1\)) plus fixed-cutoff windows
  predicting the next 30/60/90 days.
- **T4 — Historical transaction / roster-change evaluation.** For each eligible
  move, freeze the model at \(T_k^-\), predict incoming-player fit with
  reconstructed destination lineups, store the prediction before observing
  outcomes, and evaluate defined post-move windows.

Random possession splits are retained only as a **secondary diagnostic**; they may
never be cited as evidence of roster-construction generalization.

---

## 13. Split construction and leakage-prevention rules

- **Lineup identity** is an unordered five-player set (canonical sort by player
  ID); offensive and defensive sets are distinguished; home/away ordering must
  not affect results after context adjustment.
- **Holdout removal is exhaustive and outcome-blind:** held-out lineups and pairs
  are chosen by date and exposure, never by performance; every training
  possession containing a held-out exact lineup or any co-play of a held-out
  pair is removed and verified absent.
- **Feature timestamp contract:** every feature carries `event_time` and
  `available_time`; the pipeline asserts `available_time <= prediction_time`.
  Scalers, role models, cluster definitions, and priors are fit only on data
  within the training cutoff.
- **Cross-fitting for residual chemistry:** when chemistry is a residual from an
  additive fit, the additive model is trained on one fold and residuals are
  formed on another (or a jointly identified hierarchical model is used);
  full-vs-additive comparisons are made only on held-out data.
- **Immutable split manifests:** each experiment writes train/validation/test
  game IDs, cutoff dates, held-out lineup hashes and pair IDs, feature-fit scope,
  eligibility and exclusion reasons, and the split-code version.
- **Automated leakage tests are CI gates:** fail on any test game ID in training,
  any exact held-out lineup in training, any unseen pair with shared training
  possessions, any preprocessing cutoff past the training cutoff, or any
  transaction-model artifact dated after its transaction. A deliberately leaked
  feature must be caught by the guard.
- **The held-out test set is evaluated once,** after hypotheses and thresholds
  are frozen.

---

## 14. Primary predictive metrics

- **Possession prediction:** negative log likelihood, multiclass Brier score,
  expected-points MAE and RMSE, calibration error and slope.
- **Lineup / pair prediction:** possession-weighted (**micro**) and
  equal-group (**macro**) MAE and RMSE; Spearman and Kendall rank correlation;
  top-\(k\) precision/recall for genuinely positive groups with uncertainty;
  sign accuracy only for sufficiently precise estimates; interval coverage and
  width.
- **Decision evaluation:** uplift over a talent-only ranking; realized value of
  the top-ranked candidate vs alternatives; pairwise ranking accuracy within
  actual candidate sets; regret in fifth-player and replacement choices.

**Headline metric:** macro (equal-weight) error on **unseen / sparse groups** —
strong performance only on high-minute lineups does not answer the north-star
question. All metrics are reported micro, macro, and stratified by exposure,
novelty, team, season, and starter/bench status. Model comparisons use paired
game / team-season block bootstrap with intervals on the metric difference,
seed-level pairing for stochastic models, and multiple-comparison correction for
leaderboard claims.

---

## 15. Calibration and probabilistic evaluation

- **Categorical outcomes:** reliability diagrams, expected calibration error
  (with its known bias caveats), classwise Brier score, and log loss, stratified
  by season, team, lineup novelty, and exposure.
- **Expected points and lineup values:** interval coverage at 50/80/95%,
  calibration slope and intercept, residual-vs-prediction plots, and conformal
  coverage (with exchangeability discussed) — including coverage **under
  chronological and unseen-lineup shift**.
- **Chemistry-specific** (chemistry is latent): synthetic data with known pair
  effects, posterior/interval coverage tests, bootstrap reproducibility,
  cross-season sign stability, and the relationship between predicted interval
  width and realized error.

**Gate:** a model whose calibration is materially worse than the rung-3 reference
cannot be promoted, regardless of point-estimate accuracy.

---

## 16. Uncertainty requirements and reporting standards

Every published estimate — pair, lineup, player-derived, or transaction — carries:

- a point estimate (points per 100);
- at least an 80% interval (50% and 95% where useful);
- the probability the surplus is positive;
- sample / exposure support;
- a **novelty class**: seen, partially seen, or entirely unseen;
- model version and data cutoff;
- for hypothetical lineups, a distance-from-support flag and model-disagreement
  indicator.

**Leaderboard eligibility** requires adequate model support, interval width below
a declared threshold, stability across seeds and specifications, and not being
flagged as an extrapolation outside training support. Users may inspect all
estimates, but uncertain entries are visibly labeled. Raw and shrunk values are
shown together in explanatory contexts.

---

## 17. Minimum evidence thresholds for claiming a useful chemistry signal

Numeric thresholds are calibrated to baseline metric variance measured in the
descriptive stage and **recorded, with a timestamp, before final test
evaluation**. The structure below is binding now; all six conditions must hold to
claim "useful transferable chemistry signal":

1. The best interaction model (rung 4–7) shows a statistically and practically
   significant improvement over the hierarchical RAPM baseline (rung 3) on
   **macro unseen-lineup error**, sustained across at least three rolling-origin
   folds.
2. A rung-5–7 model improves on explicit pair identifiers (rung 4) on
   **unseen-pair error**.
3. No material degradation in calibration relative to rung 3.
4. Seed-to-seed variance in the improvement smaller than the improvement itself.
5. Sign stability of estimated pair surplus across seasons for supported pairs.
6. Transaction backtest (T4) better than both talent-only and team-strength
   baselines, with uncertainty reported.

Failing any condition yields **"not supported"** or **"inconclusive"**, which are
reported as findings, not hidden.

---

## 18. Required ablations

Reported in one table with the delta on the primary predictive metric, the delta
on the unseen-lineup metric, the delta on calibration, and runtime/parameters —
including unfavorable results:

- remove the additive-talent pathway;
- remove the interaction pathway;
- (rung 6) single entangled pathway vs separate talent + interaction pathways;
- (rung 7) concatenation / mean pooling vs permutation-invariant pooling;
- no context vs Tier-A context;
- no team / coach effect;
- possessions vs stints as the unit;
- 1 vs 3 vs 6 development seasons;
- shared vs separate offensive/defensive representations;
- factorization rank and embedding dimension;
- (rung 5) symmetric vs provision/need factorization.

---

## 19. Robustness and sensitivity requirements

Every headline result is checked against: a range of ridge penalties; alternative
likelihoods (squared / Huber / Poisson / NB-style); at least two
possession-boundary policies; at least two garbage-time definitions; multiple
opponent-reference distributions; multiple player minimum-exposure policies; at
least five random seeds for stochastic components; game / team-season block
bootstrap resampling; and an alternative held-out lineup/pair selection. A
conclusion that does not survive these is reported as fragile.

---

## 20. Interpretability requirements

Interpretation follows a strict hierarchy, from most to least trustworthy:
(1) additive player effects, (2) explicit pair factors, (3) exact model
decomposition, (4) counterfactual swap effects, (5) post-hoc embedding probes,
(6) 2D visualization. Lower items never override validated predictive evidence.

- Any additive-plus-interaction decomposition **must sum exactly** to the model
  prediction.
- Counterfactual swap explanations hold opponent and context fixed unless the
  user changes them, and report the talent/pair/higher-order/context breakdown
  with an interval.
- **Faithfulness tests** are required: remove the supposedly important partner
  and measure the prediction change; swap in a matched player lacking the
  identified trait; compare explanation rank with exhaustive pair deletion on a
  sample.
- Embedding probes use nested chronological splits, report uncertainty, and are
  labeled exploratory; probe success means information is encoded, not that a
  dimension has a unique causal meaning. Latent axes are not named from
  cherry-picked examples.
- UMAP/PCA plots are labeled with model, season, pathway, projection method, and
  seed, are never presented as evidence, and never override high-dimensional
  nearest-neighbor results.

---

## 21. Permitted claims

Use the **strongest claim the experiment supports and no stronger** (master plan
§5.4):

1. **Descriptive** — "the pair outscored opponents in observed minutes."
2. **Adjusted association** — "the pair coefficient stays positive after the
   stated controls."
3. **Predictive** — "the model predicts held-out outcomes better than the
   baselines."
4. **Transport-predictive** — "the model predicts new teammate or team contexts
   better than the baselines."

Research cycle 1 targets levels 3 and 4. Every published number is explicitly
tagged as an observed fact, an adjusted association, or a prediction.

---

## 22. Prohibited claims

- Causal claims that a pairing itself changes performance — e.g. "Player A makes
  Player B better," "the trade added X wins."
- Psychological or interpersonal claims — "locker-room chemistry," "they enjoy
  playing together," "leadership intangibles."
- "True chemistry" or "real chemistry" without the model-dependent qualifier.
- Predicting championships or season win totals from a single lineup or pair
  score.
- Any betting or wagering recommendation.
- Treating UMAP/PCA clusters, cosine neighbors, or a single visualization as
  evidence of learned basketball truth.
- Publishing one universal chemistry ranking with no uncertainty or support
  metadata.
- Presenting selected trades or lineups chosen after seeing favorable outcomes.

---

## 23. Known identification problems and confounders

Observational lineup data is confounded. A positive interaction residual may
reflect, rather than chemistry:

- non-random lineup assignment and selective deployment by coaches;
- coaching scheme, stagger patterns, and substitution timing;
- weak opponent bench units faced by specific lineups;
- unmeasured player health and load management;
- collinearity among teammates who almost always share minutes;
- role endogeneity — a player's role is chosen partly for his teammates;
- survivorship in which players get traded and to where;
- small-sample noise mistaken for signal;
- era, pace, and rule-environment drift across the study window.

The contract's response is adjustment (Tier-A/B context, team/coach effects),
within-team-season contrasts, negative-control pairings, sensitivity analysis to
unobserved confounding, and conservative language — not a claim that these
threats are eliminated.

---

## 24. Research integrity and reproducibility commitments

- Hypotheses and numeric thresholds are registered with a timestamp before the
  test set is touched (§13).
- Raw source inputs are immutable; corrections and exclusions require an audit
  trail; provider differences in possession definitions are documented and
  tested, not hidden.
- Historical predictions use only information available at their stated cutoff.
- Observed facts, adjusted associations, predictions, and (any future) causal
  statements are reported separately; no case study, trade, or lineup is
  selected after seeing its outcome.
- Failed, null, and unfavorable experiments are preserved in the experiment
  registry with their results.
- Every result records run provenance: git commit and dirty-state flag, resolved
  config, data and split-manifest IDs, environment lock hash, seeds, host/device,
  per-fold metrics, and artifact hashes.
- Material design choices are recorded as decision-log entries / ADRs; this
  contract is versioned, not overwritten.

---

## 25. Initial success criteria (research cycle 1)

Cycle 1 succeeds — and is portfolio- and paper-worthy — when it delivers:

- a trusted multi-season possession **and** stint dataset that passes its
  data-quality gates, with provenance;
- a reproducible RAPM baseline that beats context/team baselines out of sample,
  with uncertainty and a model card;
- an explicit pair-interaction model with characterized shrinkage and sign
  stability;
- a low-rank unseen-pair experiment with a clear positive or null result;
- neural embeddings (rung 6, separate talent/interaction pathways) and a
  permutation-invariant encoder (rung 7), each with a ≥5-seed comparison and an
  exact additive-plus-interaction decomposition;
- an exact unseen-lineup benchmark table spanning rungs 0–7 on all four
  evaluation tasks;
- a defensible answer to the north-star question — positive or null — with
  calibrated uncertainty;
- a polished technical report generated entirely from code.

A positive chemistry result additionally requires all six thresholds in §17.

---

## 26. Failure criteria and model-escalation gates

**Escalation gate.** A rung is attempted only after its predecessor meets its
exit criteria (out-of-sample improvement over its required predecessor,
maintained calibration, seed stability, acceptable compute/interpretability cost)
**or** is shown to be insufficient for the target task. In particular the neural
rungs 6–7 begin only after rungs 0–5 are complete. An advanced model that merely
predicts outcomes without improving out-of-sample error, calibration, or decision
usefulness under leakage-safe evaluation is **not** promoted to the product.

**Stop conditions** (pause and redesign, and consider reporting a null result):

- data inconsistencies exceed correction capacity;
- a provider's use restrictions make the intended release inappropriate;
- split sizes are too small for reliable comparison;
- uncertainty makes any leaderboard claim meaningless;
- successive models show no transferable interaction signal;
- compute cost is disproportionate to the expected evidence.

Abandoning an unproductive model is progress, not failure.

---

## 27. Explicit non-goals for research cycle 1

- Causal proof that two players make each other better.
- Predicting championships, playoff series, or win totals from lineup scores.
- Any betting product.
- Attention / set-transformer, graph, hypergraph, or dynamic hierarchical models
  (master-plan rungs 8+) — later cycles only.
- Tracking / spatial data, WNBA or G League transfer, coach/system embeddings as
  a primary model, injury-availability modeling.
- Salary-cap-aware or minute-allocation optimization.
- Real-time or production serving infrastructure.
- Acquiring data without provenance or respect for provider terms and rate
  limits.
- A single scalar "chemistry rating."
- Expanding the season window, target list, or model ladder mid-cycle without a
  contract amendment.

---

## 28. Decision table — binding choices for cycle 1

| Dimension | Binding choice |
|---|---|
| North-star framing | Predict fit for **unobserved** combinations; chemistry = model-dependent interaction surplus |
| Value decomposition | \(Y_p = \mu + \varepsilon_p\) (\(Y_p\) = actual points); \(V = 100\mu = T + C + K\), with \(T = \alpha + \sum \beta^{\text{off}} - \sum \beta^{\text{def}}\); chemistry \(= V_{\text{full}} - V_{\text{additive}}\) |
| Primary unit | Possession (outcome model) + stint (RAPM design); lineup/pair/player outputs derived |
| Primary modeling target | Categorical label \(Z_p\in\{0,1,2,3,4+\}\) via cross-entropy; \(\mu = \mathbb{E}[Y_p\mid\cdot]\) from a separate regression or a calibrated `4+` tail, never `4+` scored as four |
| Units | Points per 100 possessions; offensive and defensive \(\beta\) both signed larger-is-better (defense = points prevented, subtracted); net vs a labeled reference |
| Estimands | Full-lineup quantities averaged over a declared reference \(\mathcal{R}\) (teammates, opponents, context); no size-mismatched or incomplete comparisons |
| Lineup identity | Unordered 5-set, canonical ID sort; offense/defense distinguished; order-invariant results required |
| Population | NBA RS + playoffs; 3 dev seasons, 6-season first cycle (subject to data availability) |
| Mandatory controls | Tier-A context set (§10); Tier-B for portability and transactions |
| Model ladder (cycle 1) | Rungs 0–7: context mean → shrunk lineup → RAPM → hierarchical → pair RAPM → low-rank factorization → neural embeddings (separate talent/interaction pathways) → permutation-invariant Deep Sets |
| Reference baselines for "useful" | Rung 2 (RAPM) and rung 3 (hierarchical) |
| Evaluation tasks | T1 unseen lineups, T2 unseen/strong-unseen pairs, T3 future-season, T4 transaction backtest; random split = diagnostic only |
| Headline metric | Macro (equal-weight) error on unseen / sparse groups |
| Calibration gate | No promotion if calibration materially worse than rung 3 |
| Uncertainty | Every estimate: point + ≥80% interval + P(>0) + support + novelty class + version/cutoff |
| Evidence bar for chemistry claim | All six §17 conditions, thresholds frozen pre-test |
| Claims ceiling | Transport-predictive (level 4); causal and psychological claims prohibited |
| Test-set usage | Evaluated once, after hypotheses and thresholds are registered |
| Out of scope | Attention/graph/hypergraph/dynamic models (rungs 8+), tracking data, betting, optimization, production serving |

---

## 29. Unresolved questions deferred to the data-source task

These cannot be settled until data access, provider terms, and pilot variance are
known. None may be resolved by assumption:

1. **Exact seasons and cutoff dates** for both windows, and whether 6 or 8
   seasons is feasible within the compute budget.
2. **Primary play-by-play provider** and its canonical possession definition, and
   which secondary provider is used for cross-validation.
3. **Whether playoffs enter cycle-1 training** or are reserved for
   transport/robustness evaluation only.
4. **Transaction list source** and the reliability of transaction/debut
   timestamps for T4, plus event-eligibility counts.
5. **Box-score / shot-profile feature source** for priors and post-training
   probes.
6. **Legal and licensing constraints** on redistributing derived data and on the
   release package.
7. **Numeric minimum-exposure thresholds** for players, pairs, and lineups
   (needs descriptive-stage variance).
8. **Numeric leaderboard interval-width threshold** and the §17 improvement
   magnitudes (frozen only after pilot baselines exist).
9. **Exact garbage-time definition** and the possession-boundary policy variants
   to test.
10. **Feasible number of rolling-origin folds** given the confirmed window
    length.

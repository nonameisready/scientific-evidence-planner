# Algorithms

Every algorithm in the framework, in pipeline order.

## 1. Question loading (`question_loader.py`)

Questions arrive as JSON objects with required keys `question_id` and
`question`. Unknown top-level keys are folded into `Question.metadata`, so
datasets exported by earlier projects load without modification. Supported
container formats: single object, JSON list, JSONL.

## 2. Hypothesis generation (`hypothesis_generator.py`)

Two interchangeable generators behind the `HypothesisGenerator` protocol:

- **StaticHypothesisGenerator** — loads curated hypothesis sets from a JSON
  file keyed by `question_id`. This is the normal path for imported datasets.
- **TemplateHypothesisGenerator** — fallback producing four generic
  methodological alternatives (real effect / analysis artifact / confounder /
  measurement systematic) that apply to any empirical claim in any field.

`normalize_priors` rescales priors to sum to 1, assigning uniform priors when
none are given.

## 3. Evidence graph (`evidence_graph.py`)

A directed multigraph. Nodes: `paper`, `observation`, `dataset`, `claim`,
`hypothesis`. Edges: `supports`, `contradicts`, `derived_from`, `observes`,
each with a confidence in [0, 1] (validated on insert).

Key queries:

- `evidence_for` / `evidence_against(h)` — incoming `supports` /
  `contradicts` edges of a hypothesis node.
- `provenance_roots(n)` — transitively follows `derived_from` edges to the
  root sources of a node. Two claims sharing a root are **not independent**
  evidence; this powers both gap analysis and the novelty estimator.
- `filter_before(cutoff)` — sub-graph of nodes with `timestamp < cutoff`
  (lexicographic, correct for ISO dates and years). Untimestamped nodes
  (e.g. hypothesis nodes) are kept; edges touching hidden nodes are dropped.
  This implements historical simulation.

## 4. Gap analysis (`gap_analysis.py`)

For each hypothesis, in order, the analyzer emits the first gap that applies:

1. **No evidence at all** → gap with confidence 1.0.
2. **Too few evidence edges** (`< min_evidence_count`) → confidence
   `0.5 + 0.5 * (1 - count / min_evidence_count)`.
3. **Weak support** (summed support confidence `< min_support_confidence`)
   → confidence `0.4 + 0.4 * deficit`.
4. **No independent corroboration** (all evidence descends from a single
   provenance root, when `require_independent_sources` is on) → confidence
   0.78.

Gaps are returned sorted by confidence, most certain first.

## 5. Action generation (`action_generator.py`)

A catalog of seven generic action templates (read paper, search archive,
re-run analysis, collect observation, run simulation, compare datasets,
search review) carries default cost/time/resources; it is overridable via
`configs/action_catalog.yaml`.

For every gap, the generator instantiates the subset of templates that
plausibly closes that kind of gap (keyword-matched on the gap's missing
evidence: independence gaps get independent-source actions, confidence gaps
get re-analysis actions, evidence-free hypotheses get broad search actions;
anything else gets the full catalog). Each instantiated action targets the
gap's hypothesis and records the gap reason in its metadata.

## 6. Utility estimation (`utility.py`)

```
utility = w_ig * information_gain + w_nov * novelty
        + w_tract * tractability - w_cost * cost
```

All four weights come from `UtilityConfig` (YAML-loadable). Components:

### Information gain — `ExpectedEntropyReductionEstimator`

Two-outcome preposterior analysis. The action is modeled as producing a
binary outcome ("favors target hypothesis" / "disfavors it") with likelihood
ratio `r` given by the action type's *discriminative power* (configurable
table; e.g. collecting an independent observation discriminates more sharply
than reading a paper):

- `P(favorable | target true) = r / (r + 1)`
- `P(favorable | target false) = 1 / (r + 1)`

The estimate is `H(prior) - E_outcome[H(posterior)]` in bits — the expected
entropy reduction over the whole hypothesis set, always ≥ 0 (information
never hurts in expectation).

### Novelty — `ProvenanceNoveltyEstimator`

Each action type has a base *source novelty* (new observation = 1.0,
re-reading literature ≈ 0.3). The base is boosted by the provenance
concentration of the target hypothesis's existing evidence:
`novelty = min(base * (1 + 1/max(roots, 1)), 1)`. If everything known about
the hypothesis descends from one dataset, independent evidence is maximally
valuable.

### Tractability — `ResourceTractabilityEstimator`

`(1 - estimated_time)`, multiplied by `missing_resource_penalty` (default
0.5) for each required resource not in the declared available set. With no
declared set, all resources are assumed available.

### Cost — `SimpleCostEstimator`

Passes through the action's configured relative cost.

## 7. Planning (`planner.py`)

The planner scores every candidate with the injected utility estimator,
sorts by total utility, truncates to top-K, assigns ranks and generates a
reasoning string naming the target hypothesis, the four component scores and
the primary driver. It contains no scoring logic of its own.

## 8. Belief update (`belief_update.py`)

`BayesianBeliefUpdater` implements exact discrete Bayes over the hypothesis
set (assumed mutually exclusive and exhaustive):

```
posterior(h) ∝ prior(h) * P(evidence | h)
```

Likelihoods come with the evidence (`NewEvidence.likelihoods`); absent
entries default to a neutral 1.0. If all posterior mass is zero the priors
are kept (the evidence is uninformative for this set). The update is
non-destructive and records the evidence id on each hypothesis's supporting
or contradicting list according to the direction of the probability change.
Richer updaters (soft evidence, correlated hypotheses) implement the same
`BeliefUpdater` protocol.

## 9. Historical benchmark (`evaluation.py`)

1. Filter the full graph to nodes before the cutoff.
2. Run the supplied `PlannerPipeline` on the visible sub-graph.
3. Determine which hidden ground-truth evidence the plan would have reached
   (matching action type + target hypothesis).
4. Replay that evidence through the belief updater.
5. Compute the seven metrics (see `docs/evaluation.md`).

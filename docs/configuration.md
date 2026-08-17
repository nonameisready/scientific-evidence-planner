# Configuration

All tunable behavior lives in YAML under `configs/` or in dataclass configs
with documented defaults. Nothing domain-specific is hardcoded.

## `configs/default.yaml`

### `utility` — weights of the utility combination

| Parameter | Default | Meaning |
| --- | --- | --- |
| `information_gain_weight` | 10.0 | Weight of expected entropy reduction. Information gain is in bits (usually well below 1 per action) while the other components are [0, 1] scores; the higher default puts them on a comparable scale. |
| `novelty_weight` | 1.0 | Weight of evidence-independence novelty |
| `tractability_weight` | 1.0 | Weight of feasibility |
| `cost_weight` | 1.0 | Weight of (subtracted) cost |

Loaded with `planner.load_utility_config(path)`.

### `gap_analysis` — thresholds (`GapAnalyzerConfig`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_evidence_count` | 2 | Minimum evidence edges (either polarity) per hypothesis |
| `min_support_confidence` | 0.5 | Minimum summed confidence of supporting edges |
| `require_independent_sources` | true | Flag hypotheses whose evidence shares a single provenance root |

### `planner`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `top_k` | 5 | Number of recommendations returned |

### `benchmark`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `cutoff` | "2020" | Historical-simulation cutoff; the planner sees only nodes strictly before it (ISO date or year, compared lexicographically) |

## `configs/paper3_informed.yaml`

Alternative utility weights calibrated against Paper 3's historical-outcome
findings (`information_gain_weight: 10.0`, `novelty_weight: 0.5`,
`tractability_weight: 1.5`, `cost_weight: 1.0`). The empirical rationale for
each change is documented inline in the file; use `configs/default.yaml` for
the neutral, theory-derived weighting.

## `configs/action_catalog.yaml`

One entry per generic action type; controls the candidate actions the
generator can instantiate:

| Field | Meaning |
| --- | --- |
| `action_type` | One of the seven `ActionType` values |
| `description` | Template description |
| `estimated_cost` | Relative cost in [0, 1] |
| `estimated_time` | Relative time in [0, 1] |
| `required_resources` | Resource tags checked by the tractability estimator |

Load with `planner.load_catalog(path)` and pass to `ActionGenerator`.

## Code-level configuration (constructor injection)

| Object | Parameter | Default | Meaning |
| --- | --- | --- | --- |
| `ExpectedEntropyReductionEstimator` | `discriminative_power` | per-type table | Likelihood ratio of each action type's simulated outcome (how sharply it separates hypotheses) |
| `ProvenanceNoveltyEstimator` | `source_novelty` | per-type table | Base novelty of each action type's evidence source |
| `ResourceTractabilityEstimator` | `available_resources` | `None` (= all available) | Set of resource tags actually available |
| `ResourceTractabilityEstimator` | `missing_resource_penalty` | 0.5 | Multiplier applied per missing resource |
| `BayesianBeliefUpdater` | `neutral_likelihood` | 1.0 | Likelihood for hypotheses absent from the evidence's likelihood map |

Both per-type tables accept any `dict[ActionType, float]`, so they can be
built from YAML by the caller if desired.

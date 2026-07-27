# Evaluation

## Historical simulation

The benchmark (`HistoricalBenchmark`) evaluates the planner the only way an
offline system can be evaluated honestly: against history.

1. **Cutoff.** Choose a time cutoff (e.g. `"2020"`). The planner sees only
   the sub-graph of nodes with `timestamp < cutoff`
   (`EvidenceGraph.filter_before`); untimestamped structural nodes are kept.
2. **Plan.** A `PlannerPipeline` (protocol — any wiring of the modules) runs
   on the visible sub-graph and returns ranked recommendations plus the
   hypothesis set used.
3. **Hidden ground truth.** Evidence that appeared after the cutoff is
   expressed as `GroundTruthEvidence` records: which hypothesis it bore on,
   which action type historically produced it, and its per-hypothesis
   likelihoods.
4. **Replay.** Ground-truth items *reached* by the plan (a recommendation
   matches the item's action type and target hypothesis) are replayed
   through the belief updater to measure how much uncertainty the plan
   could have removed.

The central question: **did the planner recommend actions that would have
resolved the scientific question earlier?**

## Metrics

All metrics are pure functions in `src/planner/evaluation.py`.

### `top_k_utility(recommendations, k)`
Mean total utility of the top-K recommendations. Measures the quality of
what the planner actually puts in front of a researcher.

### `entropy_reduction(before, after)`
Shannon entropy (bits) of the hypothesis distribution before the plan minus
after replaying the reached ground-truth evidence. Positive values mean the
plan would have genuinely reduced scientific uncertainty.

### `average_information_gain(recommendations)`
Mean *expected* information gain across recommendations — how informative
the planner believes its plan is (compare with `entropy_reduction`, which is
what history says).

### `action_diversity(recommendations)`
Distinct action types divided by the number of recommendations, in (0, 1].
Penalizes degenerate plans that recommend the same kind of step repeatedly.

### `hypothesis_coverage(recommendations, hypotheses)`
Fraction of competing hypotheses targeted by at least one recommendation.
A good plan probes the space of explanations, not just the front-runner.

### `resolution_efficiency(recommendations, ground_truth)`
Fraction of hidden ground-truth evidence matched by some recommendation
(same action type, same target hypothesis). This is the "would we have found
it" proxy: 1.0 means every piece of evidence that historically resolved the
question was on the recommended path.

### `expected_utility(recommendations)`
Rank-discounted utility over the whole ranking with weights `1/rank`
(normalized), modeling that earlier recommendations are more likely to be
executed.

## Extending

- New metrics: add a pure function taking recommendations / hypotheses /
  ground truth; wire it into `HistoricalBenchmark.run`.
- New pipelines: implement the two-method `PlannerPipeline` protocol.
- Better ground-truth matching (e.g. semantic rather than
  type-and-target): subclass `HistoricalBenchmark` and override
  `_reachable_ground_truth`.

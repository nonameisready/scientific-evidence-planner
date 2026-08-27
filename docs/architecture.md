# Architecture

The framework is a pipeline of independent modules that communicate only
through shared dataclasses (`schemas.py`) and the `EvidenceGraph` interface.
No module imports another module's internals; composition happens at the
edge (e.g. `experiments/run_benchmark.py`) via dependency injection.

## Module diagram

```mermaid
flowchart TD
    Q[Question JSON] --> QL[question_loader]
    QL --> HG[hypothesis_generator]
    G[Evidence graph JSON] --> EG[evidence_graph.EvidenceGraph]
    HG --> GA[gap_analysis.GapAnalyzer]
    EG --> GA
    GA -->|GapReport| AG[action_generator.ActionGenerator]
    AG -->|CandidateAction list| PL[planner.Planner]
    HG --> PL
    EG --> PL
    UT[utility.UtilityEstimator] --> PL
    PL -->|Top-K Recommendations| OUT[Best next actions]
    OUT -.->|executed offline, new evidence observed| BU[belief_update]
    BU -->|updated hypothesis probabilities| GA
```

The dashed edge closes the loop: after evidence is (externally) acquired, the
belief updater revises hypothesis probabilities and the cycle repeats. The
framework itself never executes actions.

## Utility estimator composition

```mermaid
flowchart LR
    A[CandidateAction] --> IG[ExpectedEntropyReductionEstimator]
    A --> NV[ProvenanceNoveltyEstimator]
    A --> TR[ResourceTractabilityEstimator]
    A --> CO[SimpleCostEstimator]
    IG --> U[UtilityEstimator<br/>weighted sum]
    NV --> U
    TR --> U
    CO --> U
    W[UtilityConfig weights<br/>from YAML] --> U
    U --> B[UtilityBreakdown]
```

Each component implements the same `ComponentEstimator` protocol
(`estimate(action, hypotheses, graph) -> float`) and is injected into
`UtilityEstimator`, so any component can be replaced independently.

## Evaluation harness

```mermaid
flowchart TD
    FG[Full timestamped graph] -->|filter_before cutoff| VG[Visible sub-graph]
    VG --> PP[PlannerPipeline.plan]
    PP -->|recommendations| M[Metric functions]
    GT[Hidden post-cutoff ground truth] --> M
    PP -->|hypotheses| BU[BayesianBeliefUpdater]
    GT -->|reachable evidence replayed| BU
    BU -->|posterior| M
    M --> R[BenchmarkResult]
```

`PlannerPipeline` is a protocol: the benchmark drives any concrete wiring of
the modules without depending on one.

## Coupling rules

- `schemas.py` has no imports from other modules; everything imports it.
- `evidence_graph.py` depends only on schemas.
- `gap_analysis`, `action_generator`, `utility`, `planner`, `belief_update`
  depend on schemas (+ the graph interface where needed) — never on each other.
- `evaluation.py` depends on the belief-update and utility *interfaces* plus
  schemas; concrete pipelines are supplied by the caller.

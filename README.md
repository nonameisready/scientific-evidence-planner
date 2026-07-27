# Scientific Evidence Planner

A research framework for **active scientific discovery**: given an existing
scientific question, competing hypotheses and the current evidence graph, plan
the **next evidence-acquisition step** that maximizes expected scientific
progress under uncertainty.

The system is an **offline planner**. It does not generate questions, does not
run agents or browsers, does not call instrument APIs and does not execute
experiments — everything works from static datasets.

> Given the current scientific evidence, what should we investigate next to
> maximize expected scientific progress?

## Workflow

```
Scientific Question
        ↓
Competing Hypotheses
        ↓
Current Evidence Graph
        ↓
Evidence Gap Analysis
        ↓
Candidate Evidence Actions
        ↓
Expected Utility Estimation
        ↓
Best Next Action
        ↓
Evidence Update
        ↓
      Repeat
```

## Quick start

```bash
pip install pyyaml pytest
python -m pytest tests/          # run the test suite
python experiments/run_benchmark.py   # end-to-end demo + historical benchmark
```

The demo loads an example question ("Is the reported low water abundance real
or a retrieval artifact?"), plans on the evidence visible before the 2020
cutoff, and evaluates the plan against post-cutoff ground truth. The domain
appears only in the *data* — every module is domain-independent.

## Repository layout

```
configs/       YAML configuration (utility weights, thresholds, action catalog)
docs/          Architecture, algorithms, schemas, metrics, configuration
src/planner/   The framework (importable as `planner` with src/ on the path)
tests/         Unit tests for every module
experiments/   Example datasets and the benchmark runner
results/       Benchmark outputs (generated)
paper/         Paper materials
```

## Modules

| Module | Responsibility |
| --- | --- |
| `question_loader` | Load question JSON from this or upstream projects |
| `hypothesis_generator` | Produce structured competing hypotheses (static sets or generic templates) |
| `evidence_graph` | Generic graph of papers/observations/datasets/claims/hypotheses with confidence-weighted `supports` / `contradicts` / `derived_from` / `observes` edges |
| `gap_analysis` | Identify hypotheses with insufficient or non-independent evidence |
| `action_generator` | Instantiate generic candidate actions per evidence gap |
| `utility` | Utility = information gain + novelty + tractability − cost, each an independent, swappable estimator with configurable weights |
| `planner` | Rank actions, return top-K recommendations with reasoning |
| `belief_update` | Modular Bayesian update of hypothesis probabilities |
| `evaluation` | Historical-simulation benchmark and seven evaluation metrics |
| `schemas` | Shared dataclasses; the only coupling between modules |

See `docs/architecture.md` for the module diagram, `docs/algorithms.md` for
each algorithm, `docs/schemas.md` for every data schema,
`docs/evaluation.md` for metric definitions and `docs/configuration.md` for
all configurable parameters.

## Design principles

- Modules communicate only through the schemas and the graph interface —
  no module imports another module's internals.
- Estimators and generators are injected (dependency injection), so any
  component can be replaced without touching the rest.
- All tunable behavior lives in YAML under `configs/`.
- No domain knowledge in code: astronomy, biology or economics enter only
  through datasets and configuration.

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
cutoff, and evaluates the plan against post-cutoff ground truth — which
follows the documented historical outcome (the subsolar-abundance premise was
refuted by independent re-analyses). The domain appears only in the *data* —
every module is domain-independent.

## Integration with Paper 3

[Paper 3](https://github.com/nonameisready/scientific-question-outcome-prediction-indicators)
released 980 questions frozen at five historical cutoffs with future-outcome
labels. Two integration points ship here:

- `python experiments/import_paper3.py [questions.jsonl] [labels.jsonl]` —
  loads Paper 3 question/label records directly into planner objects
  (a three-record sample is vendored for offline use).
- `configs/paper3_informed.yaml` — utility weights calibrated against
  Paper 3's controlled effects (novelty down-weighted, tractability up;
  rationale documented in the file).
- `python experiments/paper3_benchmark.py` — the **scaled benchmark**:
  batch-converts all 374 of Paper 3's tension questions (evidence clusters
  vendored verbatim under `experiments/data/paper3_tension/`) into evidence
  graphs via transparent rules, runs the planner at each question's own
  historical cutoff, and reports metric distributions plus an
  outcome-stratified analysis (results in `results/paper3_benchmark.json`).
  Headline: planner utility is systematically higher on questions the
  community subsequently neglected (permutation p ≈ 0.004) — the planner
  measures neglect, the community follows preparation.

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

"""End-to-end demo: run the full planning loop and the historical benchmark.

Usage (from the repository root)::

    python experiments/run_benchmark.py

Loads the example question/hypotheses/graph, plans the next actions on
the pre-cutoff sub-graph, evaluates against post-cutoff ground truth,
and writes results to ``results/``.

This script is also the reference for wiring the modules together: each
module is constructed independently and composed here (dependency
injection), never inside another module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml

from planner import (
    ActionGenerator,
    EvidenceGraph,
    GapAnalyzer,
    GapAnalyzerConfig,
    GroundTruthEvidence,
    HistoricalBenchmark,
    Planner,
    StaticHypothesisGenerator,
    UtilityEstimator,
    load_question,
    load_utility_config,
)

DATA = REPO_ROOT / "experiments" / "data"
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"


class DefaultPipeline:
    """Concrete PlannerPipeline wiring all modules together."""

    def __init__(self, question, hypothesis_generator, config: dict) -> None:
        self.question = question
        self.hypothesis_generator = hypothesis_generator
        gap_cfg = config.get("gap_analysis", {})
        self.gap_analyzer = GapAnalyzer(
            GapAnalyzerConfig(
                min_evidence_count=gap_cfg.get("min_evidence_count", 2),
                min_support_confidence=gap_cfg.get("min_support_confidence", 0.5),
                require_independent_sources=gap_cfg.get(
                    "require_independent_sources", True
                ),
            )
        )
        self.action_generator = ActionGenerator()
        self.planner = Planner(
            UtilityEstimator(config=load_utility_config(CONFIGS / "default.yaml"))
        )

    def plan(self, graph: EvidenceGraph, top_k: int):
        hypotheses = self.hypothesis_generator.generate(self.question)
        report = self.gap_analyzer.analyze(self.question, hypotheses, graph)
        actions = self.action_generator.generate(report)
        recommendations = self.planner.rank(actions, hypotheses, graph, top_k=top_k)
        return recommendations, hypotheses


def main() -> None:
    config = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    top_k = config.get("planner", {}).get("top_k", 5)
    cutoff = str(config.get("benchmark", {}).get("cutoff", "2020"))

    question = load_question(DATA / "example_question.json")
    generator = StaticHypothesisGenerator(DATA / "example_hypotheses.json")
    full_graph = EvidenceGraph.load(DATA / "example_graph.json")
    ground_truth = [
        GroundTruthEvidence(
            evidence_id=row["evidence_id"],
            hypothesis_id=row["hypothesis_id"],
            action_type=row["action_type"],
            likelihoods=row.get("likelihoods", {}),
            metadata=row.get("metadata", {}),
        )
        for row in json.loads((DATA / "example_ground_truth.json").read_text())
    ]

    pipeline = DefaultPipeline(question, generator, config)

    # Planning step on the pre-cutoff graph (what a researcher in `cutoff`
    # would have seen).
    visible = full_graph.filter_before(cutoff)
    recommendations, _ = pipeline.plan(visible, top_k)
    print(f"Question: {question.question}")
    print(f"Cutoff: {cutoff} — planner sees {len(visible)} of {len(full_graph)} nodes\n")
    print(f"Top-{top_k} recommended next actions:")
    for rec in recommendations:
        print(
            f"  {rec.rank}. [{rec.action.action_type.value}] "
            f"utility={rec.utility.total:.3f} -> {rec.action.target_hypothesis}"
        )
        print(f"     {rec.reasoning}")

    # Historical benchmark against the hidden post-cutoff evidence.
    benchmark = HistoricalBenchmark(full_graph, ground_truth, cutoff)
    result = benchmark.run(pipeline, question.question_id, top_k=top_k)
    print("\nBenchmark metrics:")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.4f}")

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"benchmark_{question.question_id}.json"
    out_path.write_text(
        json.dumps(
            {
                "benchmark": result.to_dict(),
                "recommendations": [r.to_dict() for r in recommendations],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

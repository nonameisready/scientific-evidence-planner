"""Offline benchmark: historical simulation and evaluation metrics.

Historical simulation
---------------------
The benchmark takes a *full* evidence graph with timestamped nodes and a
cutoff (e.g. ``"2020"``). The planner only sees the sub-graph strictly
before the cutoff; nodes at or after the cutoff become hidden ground
truth. Evaluation asks whether the planner recommended actions that
would have produced that later evidence — i.e. whether it would have
resolved the question earlier.

Metrics (all pure functions, individually testable)
---------------------------------------------------
top_k_utility
    Mean total utility of the top-K recommendations.
entropy_reduction
    Hypothesis-distribution entropy before minus after folding in the
    ground-truth evidence reachable through recommended actions.
average_information_gain
    Mean expected information gain across recommendations.
action_diversity
    Distinct action types among recommendations / number of
    recommendations.
hypothesis_coverage
    Fraction of competing hypotheses targeted by at least one
    recommendation.
resolution_efficiency
    Fraction of hidden ground-truth evidence whose action type and
    target hypothesis the planner recommended — a proxy for "would this
    plan have found the resolving evidence".
expected_utility
    Probability-weighted utility over the whole ranking (rank-discounted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .belief_update import BayesianBeliefUpdater, BeliefUpdater
from .evidence_graph import EvidenceGraph
from .schemas import Hypothesis, NewEvidence, Recommendation
from .utility import entropy


# ----------------------------------------------------------------------
# Metric functions
# ----------------------------------------------------------------------


def top_k_utility(recommendations: list[Recommendation], k: int) -> float:
    """Mean total utility of the top-K recommendations."""
    top = recommendations[:k]
    if not top:
        return 0.0
    return sum(r.utility.total for r in top) / len(top)


def hypothesis_entropy(hypotheses: list[Hypothesis]) -> float:
    """Shannon entropy (bits) of the hypothesis probability distribution."""
    return entropy([h.prior for h in hypotheses])


def entropy_reduction(
    before: list[Hypothesis], after: list[Hypothesis]
) -> float:
    """Entropy(before) - Entropy(after); positive means uncertainty fell."""
    return hypothesis_entropy(before) - hypothesis_entropy(after)


def average_information_gain(recommendations: list[Recommendation]) -> float:
    """Mean expected information gain across all recommendations."""
    if not recommendations:
        return 0.0
    return sum(r.utility.information_gain for r in recommendations) / len(
        recommendations
    )


def action_diversity(recommendations: list[Recommendation]) -> float:
    """Distinct action types / number of recommendations, in (0, 1]."""
    if not recommendations:
        return 0.0
    types = {r.action.action_type for r in recommendations}
    return len(types) / len(recommendations)


def hypothesis_coverage(
    recommendations: list[Recommendation], hypotheses: list[Hypothesis]
) -> float:
    """Fraction of hypotheses targeted by at least one recommendation."""
    if not hypotheses:
        return 0.0
    targeted = {
        r.action.target_hypothesis
        for r in recommendations
        if r.action.target_hypothesis is not None
    }
    covered = sum(1 for h in hypotheses if h.hypothesis_id in targeted)
    return covered / len(hypotheses)


def resolution_efficiency(
    recommendations: list[Recommendation],
    ground_truth: list["GroundTruthEvidence"],
) -> float:
    """Fraction of hidden ground-truth evidence matched by a recommendation.

    A ground-truth item is matched when some recommendation shares its
    action type and targets the hypothesis the evidence bears on.
    """
    if not ground_truth:
        return 0.0
    matched = 0
    for item in ground_truth:
        for rec in recommendations:
            if (
                rec.action.action_type.value == item.action_type
                and rec.action.target_hypothesis == item.hypothesis_id
            ):
                matched += 1
                break
    return matched / len(ground_truth)


def expected_utility(recommendations: list[Recommendation]) -> float:
    """Rank-discounted expected utility of the whole ranking.

    Weight of rank i is 1/i (normalized), modeling that earlier
    recommendations are more likely to actually be executed.
    """
    if not recommendations:
        return 0.0
    weights = [1.0 / (i + 1) for i in range(len(recommendations))]
    total_weight = sum(weights)
    return (
        sum(w * r.utility.total for w, r in zip(weights, recommendations))
        / total_weight
    )


# ----------------------------------------------------------------------
# Historical benchmark
# ----------------------------------------------------------------------


@dataclass
class GroundTruthEvidence:
    """One piece of hidden post-cutoff evidence.

    ``action_type`` is the action that historically produced this
    evidence; ``likelihoods`` allows replaying it through the belief
    updater (see :class:`~planner.schemas.NewEvidence`).
    """

    evidence_id: str
    hypothesis_id: str
    action_type: str
    likelihoods: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_new_evidence(self) -> NewEvidence:
        return NewEvidence(
            evidence_id=self.evidence_id,
            description=self.metadata.get("description", ""),
            likelihoods=dict(self.likelihoods),
            metadata=self.metadata,
        )


@dataclass
class BenchmarkResult:
    """All metrics for one benchmark run."""

    question_id: str
    cutoff: str
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "cutoff": self.cutoff,
            "metrics": dict(self.metrics),
        }


class PlannerPipeline(Protocol):
    """Interface the benchmark drives.

    Implementations run the full plan step on a (pre-cutoff) graph and
    return ranked recommendations plus the hypothesis set used. Keeping
    this a protocol decouples evaluation from any concrete pipeline
    wiring.
    """

    def plan(
        self, graph: EvidenceGraph, top_k: int
    ) -> tuple[list[Recommendation], list[Hypothesis]]:
        ...


class HistoricalBenchmark:
    """Run the planner on a time-truncated graph and score it against
    hidden post-cutoff ground truth."""

    def __init__(
        self,
        full_graph: EvidenceGraph,
        ground_truth: list[GroundTruthEvidence],
        cutoff: str,
        belief_updater: BeliefUpdater | None = None,
    ) -> None:
        self.full_graph = full_graph
        self.ground_truth = ground_truth
        self.cutoff = cutoff
        self.belief_updater = belief_updater or BayesianBeliefUpdater()

    def run(
        self, pipeline: PlannerPipeline, question_id: str, top_k: int = 5
    ) -> BenchmarkResult:
        visible_graph = self.full_graph.filter_before(self.cutoff)
        recommendations, hypotheses = pipeline.plan(visible_graph, top_k)

        # Replay the hidden evidence that the plan would have reached to
        # measure how much uncertainty it could have removed.
        reachable = self._reachable_ground_truth(recommendations)
        posterior = hypotheses
        for item in reachable:
            posterior = self.belief_updater.update(
                posterior, item.to_new_evidence()
            )

        metrics = {
            "top_k_utility": top_k_utility(recommendations, top_k),
            "entropy_reduction": entropy_reduction(hypotheses, posterior),
            "average_information_gain": average_information_gain(recommendations),
            "action_diversity": action_diversity(recommendations),
            "hypothesis_coverage": hypothesis_coverage(recommendations, hypotheses),
            "resolution_efficiency": resolution_efficiency(
                recommendations, self.ground_truth
            ),
            "expected_utility": expected_utility(recommendations),
        }
        return BenchmarkResult(
            question_id=question_id, cutoff=self.cutoff, metrics=metrics
        )

    def _reachable_ground_truth(
        self, recommendations: list[Recommendation]
    ) -> list[GroundTruthEvidence]:
        reachable = []
        for item in self.ground_truth:
            for rec in recommendations:
                if (
                    rec.action.action_type.value == item.action_type
                    and rec.action.target_hypothesis == item.hypothesis_id
                ):
                    reachable.append(item)
                    break
        return reachable

"""Utility estimation for candidate actions.

Default utility::

    utility = w_ig * information_gain
            + w_nov * novelty
            + w_tract * tractability
            - w_cost * cost

Each component is an independent estimator behind a small protocol, so
any of them can be replaced without touching the others, and all
weights are configurable (YAML-loadable via :func:`load_utility_config`).

Components
----------
Information gain
    Expected reduction of hypothesis-distribution entropy (in bits),
    computed by a two-outcome preposterior analysis: the action either
    yields evidence favoring its target hypothesis or evidence against
    it, with a likelihood ratio given by the action type's
    discriminative power.

Novelty
    Whether the evidence the action would produce is independent of the
    existing evidence, measured through provenance roots in the graph.

Tractability
    Feasibility: inverse of estimated time, penalized for required
    resources that are not declared available.

Cost
    The action's configured relative cost, passed through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from .evidence_graph import EvidenceGraph
from .schemas import ActionType, CandidateAction, Hypothesis, UtilityBreakdown


def entropy(probabilities: list[float]) -> float:
    """Shannon entropy in bits; zero-probability entries contribute 0."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


class ComponentEstimator(Protocol):
    """Interface every utility component implements."""

    def estimate(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> float:
        ...


# ----------------------------------------------------------------------
# Information gain
# ----------------------------------------------------------------------

#: How sharply each action type discriminates between hypotheses,
#: expressed as the likelihood ratio P(outcome | target) / P(outcome | other).
DEFAULT_DISCRIMINATIVE_POWER: dict[ActionType, float] = {
    ActionType.READ_PAPER: 1.5,
    ActionType.SEARCH_ARCHIVE: 1.5,
    ActionType.SEARCH_REVIEW: 1.8,
    ActionType.COMPARE_DATASETS: 2.5,
    ActionType.RERUN_ANALYSIS: 3.0,
    ActionType.RUN_SIMULATION: 2.5,
    ActionType.COLLECT_OBSERVATION: 4.0,
}


class ExpectedEntropyReductionEstimator:
    """Preposterior expected information gain over the hypothesis set.

    Model: the action produces a binary outcome ("favors target" /
    "disfavors target"). The likelihood of the favorable outcome is
    ``r / (r + 1)`` under the target hypothesis and ``1 / (r + 1)``
    under every other hypothesis, where ``r`` is the action type's
    discriminative power. The estimate is the expected entropy of the
    posterior subtracted from the current entropy — always >= 0.
    """

    def __init__(self, discriminative_power: dict[ActionType, float] | None = None) -> None:
        self.discriminative_power = discriminative_power or DEFAULT_DISCRIMINATIVE_POWER

    def estimate(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> float:
        if not hypotheses or action.target_hypothesis is None:
            return 0.0
        ids = [h.hypothesis_id for h in hypotheses]
        if action.target_hypothesis not in ids:
            return 0.0

        priors = [max(h.prior, 0.0) for h in hypotheses]
        total = sum(priors)
        if total <= 0:
            return 0.0
        priors = [p / total for p in priors]

        r = self.discriminative_power.get(action.action_type, 1.0)
        p_fav_target = r / (r + 1.0)  # P(favorable outcome | target true)
        p_fav_other = 1.0 / (r + 1.0)  # P(favorable outcome | target false)

        target_index = ids.index(action.target_hypothesis)
        prior_entropy = entropy(priors)

        expected_posterior_entropy = 0.0
        for favorable in (True, False):
            joint = []
            for i, p in enumerate(priors):
                if favorable:
                    lik = p_fav_target if i == target_index else p_fav_other
                else:
                    lik = (1 - p_fav_target) if i == target_index else (1 - p_fav_other)
                joint.append(p * lik)
            p_outcome = sum(joint)
            if p_outcome <= 0:
                continue
            posterior = [j / p_outcome for j in joint]
            expected_posterior_entropy += p_outcome * entropy(posterior)

        return max(prior_entropy - expected_posterior_entropy, 0.0)


# ----------------------------------------------------------------------
# Novelty
# ----------------------------------------------------------------------

#: Whether an action type produces evidence from a genuinely new source
#: (1.0) or reprocesses existing material (closer to 0).
DEFAULT_SOURCE_NOVELTY: dict[ActionType, float] = {
    ActionType.READ_PAPER: 0.3,
    ActionType.SEARCH_ARCHIVE: 0.5,
    ActionType.SEARCH_REVIEW: 0.3,
    ActionType.COMPARE_DATASETS: 0.7,
    ActionType.RERUN_ANALYSIS: 0.4,
    ActionType.RUN_SIMULATION: 0.6,
    ActionType.COLLECT_OBSERVATION: 1.0,
}


class ProvenanceNoveltyEstimator:
    """Novelty = source novelty of the action, boosted when the target
    hypothesis's existing evidence is provenance-concentrated.

    If all current evidence for the target descends from one provenance
    root, independent evidence is maximally valuable, so the boost is
    largest; with many independent roots the boost vanishes.
    """

    def __init__(self, source_novelty: dict[ActionType, float] | None = None) -> None:
        self.source_novelty = source_novelty or DEFAULT_SOURCE_NOVELTY

    def estimate(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> float:
        base = self.source_novelty.get(action.action_type, 0.5)
        target = action.target_hypothesis
        if target is None or not graph.has_node(target):
            return base

        roots: set[str] = set()
        for source in graph.evidence_sources(target):
            roots |= graph.provenance_roots(source)
        # 0 or 1 roots -> full boost; more roots -> diminishing boost.
        concentration = 1.0 / max(len(roots), 1)
        return min(base * (1.0 + concentration), 1.0)


# ----------------------------------------------------------------------
# Tractability
# ----------------------------------------------------------------------


class ResourceTractabilityEstimator:
    """Tractability = (1 - estimated_time), scaled down for each
    required resource not declared available.

    ``available_resources = None`` means "assume everything is
    available" (no penalty).
    """

    def __init__(
        self,
        available_resources: set[str] | None = None,
        missing_resource_penalty: float = 0.5,
    ) -> None:
        self.available_resources = available_resources
        self.missing_resource_penalty = missing_resource_penalty

    def estimate(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> float:
        score = max(0.0, 1.0 - action.estimated_time)
        if self.available_resources is not None:
            for resource in action.required_resources:
                if resource not in self.available_resources:
                    score *= self.missing_resource_penalty
        return score


# ----------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------


class SimpleCostEstimator:
    """Pass through the action's configured relative cost."""

    def estimate(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> float:
        return action.estimated_cost


# ----------------------------------------------------------------------
# Combined utility
# ----------------------------------------------------------------------


@dataclass
class UtilityConfig:
    """Weights for the utility combination. All configurable."""

    information_gain_weight: float = 1.0
    novelty_weight: float = 1.0
    tractability_weight: float = 1.0
    cost_weight: float = 1.0


def load_utility_config(path: str | Path) -> UtilityConfig:
    """Load utility weights from a YAML file with a ``utility:`` section."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    section = data.get("utility", data)
    return UtilityConfig(
        information_gain_weight=float(section.get("information_gain_weight", 1.0)),
        novelty_weight=float(section.get("novelty_weight", 1.0)),
        tractability_weight=float(section.get("tractability_weight", 1.0)),
        cost_weight=float(section.get("cost_weight", 1.0)),
    )


@dataclass
class UtilityEstimator:
    """Combine the four component estimators under configurable weights.

    Component estimators are injected, so any of them can be swapped
    (dependency injection instead of hardwired composition).
    """

    config: UtilityConfig = field(default_factory=UtilityConfig)
    information_gain: ComponentEstimator = field(
        default_factory=ExpectedEntropyReductionEstimator
    )
    novelty: ComponentEstimator = field(default_factory=ProvenanceNoveltyEstimator)
    tractability: ComponentEstimator = field(
        default_factory=ResourceTractabilityEstimator
    )
    cost: ComponentEstimator = field(default_factory=SimpleCostEstimator)

    def estimate(
        self,
        action: CandidateAction,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> UtilityBreakdown:
        ig = self.information_gain.estimate(action, hypotheses, graph)
        nov = self.novelty.estimate(action, hypotheses, graph)
        tract = self.tractability.estimate(action, hypotheses, graph)
        cost = self.cost.estimate(action, hypotheses, graph)
        total = (
            self.config.information_gain_weight * ig
            + self.config.novelty_weight * nov
            + self.config.tractability_weight * tract
            - self.config.cost_weight * cost
        )
        return UtilityBreakdown(
            information_gain=ig,
            novelty=nov,
            tractability=tract,
            cost=cost,
            total=total,
        )

"""Modular Bayesian belief updating.

The interface (:class:`BeliefUpdater`) takes the current hypothesis set
and a piece of :class:`~planner.schemas.NewEvidence` carrying
per-hypothesis likelihoods, and returns hypotheses with updated
probabilities. The default implementation is exact discrete Bayes over
the (assumed exhaustive and mutually exclusive) hypothesis set; richer
models — soft evidence, correlated hypotheses, hierarchical priors —
can implement the same protocol without touching callers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from .schemas import Hypothesis, NewEvidence


class BeliefUpdater(Protocol):
    """Interface: fold new evidence into hypothesis probabilities."""

    def update(
        self, hypotheses: list[Hypothesis], evidence: NewEvidence
    ) -> list[Hypothesis]:
        ...


class BayesianBeliefUpdater:
    """Exact discrete Bayes update.

    For each hypothesis ``h``::

        posterior(h) ∝ prior(h) * P(evidence | h)

    Likelihoods come from ``evidence.likelihoods``; hypotheses missing
    from the map get the neutral likelihood 1.0. If every posterior mass
    is zero (evidence impossible under all hypotheses), priors are kept
    unchanged rather than dividing by zero — such evidence is
    uninformative for this hypothesis set.

    The update is non-destructive: input hypotheses are not mutated. The
    evidence id is appended to each hypothesis's supporting or
    contradicting list depending on whether its probability rose or fell.
    """

    def __init__(self, neutral_likelihood: float = 1.0) -> None:
        self.neutral_likelihood = neutral_likelihood

    def update(
        self, hypotheses: list[Hypothesis], evidence: NewEvidence
    ) -> list[Hypothesis]:
        updated = deepcopy(hypotheses)
        weights = []
        for h in updated:
            likelihood = evidence.likelihoods.get(
                h.hypothesis_id, self.neutral_likelihood
            )
            if likelihood < 0:
                raise ValueError(
                    f"negative likelihood for {h.hypothesis_id}: {likelihood}"
                )
            weights.append(h.prior * likelihood)

        total = sum(weights)
        if total <= 0:
            return updated  # evidence uninformative; keep priors

        for h, w in zip(updated, weights):
            posterior = w / total
            if posterior > h.prior:
                h.supporting_evidence.append(evidence.evidence_id)
            elif posterior < h.prior:
                h.contradicting_evidence.append(evidence.evidence_id)
            h.prior = posterior
        return updated


def apply_evidence_sequence(
    hypotheses: list[Hypothesis],
    evidence_items: list[NewEvidence],
    updater: BeliefUpdater | None = None,
) -> list[Hypothesis]:
    """Fold a sequence of evidence into the hypotheses, in order."""
    updater = updater or BayesianBeliefUpdater()
    current = hypotheses
    for item in evidence_items:
        current = updater.update(current, item)
    return current

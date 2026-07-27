"""Rank candidate actions and recommend the top-K next steps.

The planner is a thin orchestrator: it scores every candidate action
with the injected :class:`~planner.utility.UtilityEstimator`, sorts by
total utility, and attaches human-readable reasoning to each
recommendation. It contains no scoring logic of its own, keeping the
ranking policy fully swappable.
"""

from __future__ import annotations

from .evidence_graph import EvidenceGraph
from .schemas import CandidateAction, Hypothesis, Recommendation, UtilityBreakdown
from .utility import UtilityEstimator


class Planner:
    """Score, rank and explain candidate evidence actions."""

    def __init__(self, utility_estimator: UtilityEstimator | None = None) -> None:
        self.utility_estimator = utility_estimator or UtilityEstimator()

    def rank(
        self,
        actions: list[CandidateAction],
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
        top_k: int | None = None,
    ) -> list[Recommendation]:
        """Return recommendations sorted by total utility (best first).

        ``top_k=None`` returns the full ranking.
        """
        scored: list[Recommendation] = []
        for action in actions:
            utility = self.utility_estimator.estimate(action, hypotheses, graph)
            scored.append(
                Recommendation(
                    action=action,
                    utility=utility,
                    reasoning=self._explain(action, utility, hypotheses),
                )
            )
        scored.sort(key=lambda r: r.utility.total, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        for i, rec in enumerate(scored):
            rec.rank = i + 1
        return scored

    @staticmethod
    def _explain(
        action: CandidateAction,
        utility: UtilityBreakdown,
        hypotheses: list[Hypothesis],
    ) -> str:
        target = next(
            (h for h in hypotheses if h.hypothesis_id == action.target_hypothesis),
            None,
        )
        target_text = (
            f"hypothesis {target.hypothesis_id} ({target.description})"
            if target
            else "no specific hypothesis"
        )
        drivers = sorted(
            [
                ("information gain", utility.information_gain),
                ("novelty", utility.novelty),
                ("tractability", utility.tractability),
            ],
            key=lambda kv: kv[1],
            reverse=True,
        )
        main_driver = drivers[0][0]
        return (
            f"Targets {target_text}. Expected information gain "
            f"{utility.information_gain:.3f} bits, novelty {utility.novelty:.2f}, "
            f"tractability {utility.tractability:.2f}, cost {utility.cost:.2f}. "
            f"Primary driver: {main_driver}."
        )

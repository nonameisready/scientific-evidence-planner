"""Evidence gap analysis.

Given a question, its competing hypotheses and the current evidence
graph, identify which hypotheses have insufficient evidence and why.

A hypothesis is considered under-evidenced when any of these hold:

* it has fewer than ``min_evidence_count`` evidence edges;
* its total support confidence is below ``min_support_confidence``;
* all of its evidence descends from a single provenance root, i.e. no
  independent corroboration exists.

The gap confidence expresses how certain the analyzer is that the gap
is real: it grows as the evidence count and confidence shrink.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence_graph import EvidenceGraph
from .schemas import EvidenceGap, GapReport, Hypothesis, Question


@dataclass
class GapAnalyzerConfig:
    """Thresholds for calling a hypothesis under-evidenced."""

    min_evidence_count: int = 2
    min_support_confidence: float = 0.5
    require_independent_sources: bool = True


class GapAnalyzer:
    """Identify hypotheses that lack sufficient evidence."""

    def __init__(self, config: GapAnalyzerConfig | None = None) -> None:
        self.config = config or GapAnalyzerConfig()

    def analyze(
        self,
        question: Question,
        hypotheses: list[Hypothesis],
        graph: EvidenceGraph,
    ) -> GapReport:
        gaps: list[EvidenceGap] = []
        for hypothesis in hypotheses:
            gap = self._analyze_hypothesis(hypothesis, graph)
            if gap is not None:
                gaps.append(gap)
        # Most confident gaps first.
        gaps.sort(key=lambda g: g.confidence, reverse=True)
        return GapReport(question_id=question.question_id, gaps=gaps)

    def _analyze_hypothesis(
        self, hypothesis: Hypothesis, graph: EvidenceGraph
    ) -> EvidenceGap | None:
        cfg = self.config
        hyp_id = hypothesis.hypothesis_id

        if graph.has_node(hyp_id):
            supporting = graph.evidence_for(hyp_id)
            contradicting = graph.evidence_against(hyp_id)
            sources = graph.evidence_sources(hyp_id)
        else:
            supporting, contradicting, sources = [], [], set()

        evidence_count = len(supporting) + len(contradicting)
        support_confidence = sum(e.confidence for e in supporting)

        if evidence_count == 0:
            return EvidenceGap(
                hypothesis_id=hyp_id,
                missing_evidence="any evidence (supporting or contradicting)",
                confidence=1.0,
                reason="No evidence in the graph touches this hypothesis.",
            )

        if evidence_count < cfg.min_evidence_count:
            confidence = 1.0 - evidence_count / cfg.min_evidence_count
            return EvidenceGap(
                hypothesis_id=hyp_id,
                missing_evidence="additional supporting or contradicting evidence",
                confidence=round(0.5 + 0.5 * confidence, 3),
                reason=(
                    f"Only {evidence_count} evidence edge(s); at least "
                    f"{cfg.min_evidence_count} required."
                ),
            )

        if support_confidence < cfg.min_support_confidence:
            deficit = 1.0 - support_confidence / cfg.min_support_confidence
            return EvidenceGap(
                hypothesis_id=hyp_id,
                missing_evidence="higher-confidence supporting evidence",
                confidence=round(0.4 + 0.4 * deficit, 3),
                reason=(
                    f"Total support confidence {support_confidence:.2f} is below "
                    f"threshold {cfg.min_support_confidence:.2f}."
                ),
            )

        if cfg.require_independent_sources:
            roots: set[str] = set()
            for source in sources:
                roots |= graph.provenance_roots(source)
            if len(roots) <= 1:
                return EvidenceGap(
                    hypothesis_id=hyp_id,
                    missing_evidence="evidence from an independent source",
                    confidence=0.78,
                    reason="No independent observation exists: all evidence "
                    "descends from a single provenance root.",
                )

        return None

"""Core data schemas for the scientific evidence planner.

All schemas are plain dataclasses so that modules stay decoupled: every
module communicates exclusively through these types (plus the
:class:`~planner.evidence_graph.EvidenceGraph`), never through each
other's internals.

Serialization convention: every schema provides ``to_dict`` /
``from_dict`` so objects round-trip through JSON without extra
dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class NodeType(str, enum.Enum):
    """Types of nodes in the evidence graph (domain-independent)."""

    PAPER = "paper"
    OBSERVATION = "observation"
    DATASET = "dataset"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"


class EdgeType(str, enum.Enum):
    """Types of edges in the evidence graph."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    OBSERVES = "observes"


@dataclass
class Question:
    """A scientific question under investigation.

    The planner never generates questions; it consumes them.
    """

    question_id: str
    question: str
    domain: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            domain=data.get("domain", "generic"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Hypothesis:
    """A structured competing hypothesis for a question.

    ``prior`` is the current probability mass assigned to the hypothesis.
    ``supporting_evidence`` / ``contradicting_evidence`` reference node
    ids in the evidence graph.
    """

    hypothesis_id: str
    description: str
    prior: float = 0.0
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "prior": self.prior,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id=data["hypothesis_id"],
            description=data["description"],
            prior=data.get("prior", 0.0),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            contradicting_evidence=list(data.get("contradicting_evidence", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvidenceNode:
    """A node in the evidence graph.

    ``timestamp`` (ISO date string or year) supports historical
    simulation: the benchmark hides nodes newer than the cutoff.
    """

    node_id: str
    node_type: NodeType
    label: str = ""
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceNode":
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            label=data.get("label", ""),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvidenceEdge:
    """A directed, confidence-weighted edge in the evidence graph."""

    source: str
    target: str
    edge_type: EdgeType
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=EdgeType(data["edge_type"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvidenceGap:
    """One entry of a gap report: a hypothesis with insufficient evidence."""

    hypothesis_id: str
    missing_evidence: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "missing_evidence": self.missing_evidence,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceGap":
        return cls(
            hypothesis_id=data["hypothesis_id"],
            missing_evidence=data["missing_evidence"],
            confidence=data["confidence"],
            reason=data["reason"],
        )


@dataclass
class GapReport:
    """Full output of the gap analyzer for one question."""

    question_id: str
    gaps: list[EvidenceGap] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "gaps": [g.to_dict() for g in self.gaps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GapReport":
        return cls(
            question_id=data["question_id"],
            gaps=[EvidenceGap.from_dict(g) for g in data.get("gaps", [])],
        )


class ActionType(str, enum.Enum):
    """Generic evidence-acquisition action types (not domain-specific)."""

    READ_PAPER = "read_paper"
    SEARCH_ARCHIVE = "search_archive"
    RERUN_ANALYSIS = "rerun_analysis"
    COLLECT_OBSERVATION = "collect_observation"
    RUN_SIMULATION = "run_simulation"
    COMPARE_DATASETS = "compare_datasets"
    SEARCH_REVIEW = "search_review"


@dataclass
class CandidateAction:
    """A possible next evidence-acquisition step.

    ``estimated_cost`` and ``estimated_time`` are unit-free relative
    scores in [0, 1]; ``target_hypothesis`` is the hypothesis the action
    is expected to inform most.
    """

    action_id: str
    action_type: ActionType
    description: str
    estimated_cost: float
    estimated_time: float
    required_resources: list[str] = field(default_factory=list)
    target_hypothesis: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "estimated_cost": self.estimated_cost,
            "estimated_time": self.estimated_time,
            "required_resources": list(self.required_resources),
            "target_hypothesis": self.target_hypothesis,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateAction":
        return cls(
            action_id=data["action_id"],
            action_type=ActionType(data["action_type"]),
            description=data["description"],
            estimated_cost=data["estimated_cost"],
            estimated_time=data["estimated_time"],
            required_resources=list(data.get("required_resources", [])),
            target_hypothesis=data.get("target_hypothesis"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UtilityBreakdown:
    """Per-component utility scores for one candidate action."""

    information_gain: float
    novelty: float
    tractability: float
    cost: float
    total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "information_gain": self.information_gain,
            "novelty": self.novelty,
            "tractability": self.tractability,
            "cost": self.cost,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UtilityBreakdown":
        return cls(
            information_gain=data["information_gain"],
            novelty=data["novelty"],
            tractability=data["tractability"],
            cost=data["cost"],
            total=data["total"],
        )


@dataclass
class Recommendation:
    """One ranked planner recommendation."""

    action: CandidateAction
    utility: UtilityBreakdown
    reasoning: str
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "utility": self.utility.to_dict(),
            "reasoning": self.reasoning,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recommendation":
        return cls(
            action=CandidateAction.from_dict(data["action"]),
            utility=UtilityBreakdown.from_dict(data["utility"]),
            reasoning=data["reasoning"],
            rank=data.get("rank", 0),
        )


@dataclass
class NewEvidence:
    """Observed evidence handed to the belief updater.

    ``likelihoods`` maps hypothesis_id -> P(evidence | hypothesis).
    Hypotheses absent from the map get a neutral likelihood of 1.0.
    """

    evidence_id: str
    description: str
    likelihoods: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "description": self.description,
            "likelihoods": dict(self.likelihoods),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NewEvidence":
        return cls(
            evidence_id=data["evidence_id"],
            description=data["description"],
            likelihoods=dict(data.get("likelihoods", {})),
            metadata=data.get("metadata", {}),
        )

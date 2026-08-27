import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from planner import (
    EdgeType,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    Hypothesis,
    NodeType,
    Question,
)


@pytest.fixture
def question() -> Question:
    return Question(
        question_id="q1",
        question="Is the reported effect real or an artifact?",
    )


@pytest.fixture
def hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis("H1", "The effect is real.", prior=0.4),
        Hypothesis("H2", "The effect is an analysis artifact.", prior=0.3),
        Hypothesis("H3", "The effect is a measurement systematic.", prior=0.3),
    ]


@pytest.fixture
def graph() -> EvidenceGraph:
    """Small graph: H1 supported by two claims sharing one dataset root,
    H2 supported by one claim, H3 untouched."""
    g = EvidenceGraph()
    g.add_node(EvidenceNode("H1", NodeType.HYPOTHESIS))
    g.add_node(EvidenceNode("H2", NodeType.HYPOTHESIS))
    g.add_node(EvidenceNode("H3", NodeType.HYPOTHESIS))
    g.add_node(EvidenceNode("d1", NodeType.DATASET, timestamp="2018"))
    g.add_node(EvidenceNode("p1", NodeType.PAPER, timestamp="2019"))
    g.add_node(EvidenceNode("c1", NodeType.CLAIM, timestamp="2019"))
    g.add_node(EvidenceNode("c2", NodeType.CLAIM, timestamp="2019"))
    g.add_node(EvidenceNode("c3", NodeType.CLAIM, timestamp="2021"))
    g.add_edge(EvidenceEdge("p1", "d1", EdgeType.DERIVED_FROM))
    g.add_edge(EvidenceEdge("c1", "p1", EdgeType.DERIVED_FROM))
    g.add_edge(EvidenceEdge("c2", "p1", EdgeType.DERIVED_FROM))
    g.add_edge(EvidenceEdge("c1", "H1", EdgeType.SUPPORTS, confidence=0.6))
    g.add_edge(EvidenceEdge("c2", "H1", EdgeType.SUPPORTS, confidence=0.5))
    g.add_edge(EvidenceEdge("c3", "H2", EdgeType.SUPPORTS, confidence=0.4))
    return g

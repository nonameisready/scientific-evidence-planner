import pytest

from planner import EdgeType, EvidenceEdge, EvidenceGraph, EvidenceNode, NodeType


def test_add_and_query_nodes(graph):
    assert len(graph) == 8
    assert graph.get_node("H1").node_type == NodeType.HYPOTHESIS
    assert {n.node_id for n in graph.nodes(NodeType.CLAIM)} == {"c1", "c2", "c3"}


def test_edge_requires_existing_nodes():
    g = EvidenceGraph()
    g.add_node(EvidenceNode("a", NodeType.CLAIM))
    with pytest.raises(KeyError):
        g.add_edge(EvidenceEdge("a", "missing", EdgeType.SUPPORTS))


def test_edge_confidence_validated():
    g = EvidenceGraph()
    g.add_node(EvidenceNode("a", NodeType.CLAIM))
    g.add_node(EvidenceNode("b", NodeType.HYPOTHESIS))
    with pytest.raises(ValueError):
        g.add_edge(EvidenceEdge("a", "b", EdgeType.SUPPORTS, confidence=1.5))


def test_evidence_for_and_against(graph):
    assert {e.source for e in graph.evidence_for("H1")} == {"c1", "c2"}
    assert graph.evidence_against("H1") == []
    assert graph.evidence_sources("H1") == {"c1", "c2"}


def test_provenance_roots(graph):
    # c1 -> p1 -> d1, so the root of c1 is the dataset d1.
    assert graph.provenance_roots("c1") == {"d1"}
    # c3 has no derived_from parents; it is its own root.
    assert graph.provenance_roots("c3") == {"c3"}


def test_filter_before_hides_future_nodes(graph):
    visible = graph.filter_before("2020")
    assert visible.has_node("c1")
    assert not visible.has_node("c3")  # timestamp 2021
    # Untimestamped hypothesis nodes are kept.
    assert visible.has_node("H1")
    # Edges touching hidden nodes are dropped.
    assert visible.evidence_for("H2") == []


def test_serialization_roundtrip(graph, tmp_path):
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = EvidenceGraph.load(path)
    assert len(loaded) == len(graph)
    assert {e.source for e in loaded.evidence_for("H1")} == {"c1", "c2"}
    assert loaded.get_node("d1").timestamp == "2018"

"""Generic, domain-independent evidence graph.

Nodes are papers, observations, datasets, claims and hypotheses; edges
are ``supports`` / ``contradicts`` / ``derived_from`` / ``observes``
relations, each carrying a confidence in [0, 1]. The implementation is
a plain adjacency structure with no domain assumptions, so any field of
science can populate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import EdgeType, EvidenceEdge, EvidenceNode, NodeType


class EvidenceGraph:
    """Directed multigraph of evidence with confidence-weighted edges."""

    def __init__(self) -> None:
        self._nodes: dict[str, EvidenceNode] = {}
        self._edges: list[EvidenceEdge] = []
        self._out: dict[str, list[EvidenceEdge]] = {}
        self._in: dict[str, list[EvidenceEdge]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node: EvidenceNode) -> None:
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])

    def add_edge(self, edge: EvidenceEdge) -> None:
        for endpoint in (edge.source, edge.target):
            if endpoint not in self._nodes:
                raise KeyError(f"unknown node in edge: {endpoint!r}")
        if not 0.0 <= edge.confidence <= 1.0:
            raise ValueError(f"edge confidence must be in [0, 1], got {edge.confidence}")
        self._edges.append(edge)
        self._out[edge.source].append(edge)
        self._in[edge.target].append(edge)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> EvidenceNode:
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self, node_type: NodeType | None = None) -> list[EvidenceNode]:
        if node_type is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def edges(self, edge_type: EdgeType | None = None) -> list[EvidenceEdge]:
        if edge_type is None:
            return list(self._edges)
        return [e for e in self._edges if e.edge_type == edge_type]

    def out_edges(self, node_id: str, edge_type: EdgeType | None = None) -> list[EvidenceEdge]:
        edges = self._out.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def in_edges(self, node_id: str, edge_type: EdgeType | None = None) -> list[EvidenceEdge]:
        edges = self._in.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def evidence_for(self, hypothesis_id: str) -> list[EvidenceEdge]:
        """Edges that support the given hypothesis node."""
        return self.in_edges(hypothesis_id, EdgeType.SUPPORTS)

    def evidence_against(self, hypothesis_id: str) -> list[EvidenceEdge]:
        """Edges that contradict the given hypothesis node."""
        return self.in_edges(hypothesis_id, EdgeType.CONTRADICTS)

    def evidence_sources(self, hypothesis_id: str) -> set[str]:
        """Node ids of all evidence touching a hypothesis (either polarity)."""
        return {
            e.source
            for e in self.in_edges(hypothesis_id)
            if e.edge_type in (EdgeType.SUPPORTS, EdgeType.CONTRADICTS)
        }

    def provenance_roots(self, node_id: str) -> set[str]:
        """Follow ``derived_from`` edges to find the root sources of a node.

        Used by the novelty estimator: two pieces of evidence derived
        from the same dataset are not independent.
        """
        roots: set[str] = set()
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            parents = [e.target for e in self.out_edges(current, EdgeType.DERIVED_FROM)]
            if not parents:
                roots.add(current)
            else:
                stack.extend(parents)
        return roots

    # ------------------------------------------------------------------
    # Historical simulation support
    # ------------------------------------------------------------------
    def filter_before(self, cutoff: str) -> "EvidenceGraph":
        """Return a new graph containing only nodes with timestamp < cutoff.

        Nodes without a timestamp are kept (they are assumed to be
        structural, e.g. hypothesis nodes). Timestamps compare
        lexicographically, which is correct for ISO dates and years.
        """
        filtered = EvidenceGraph()
        for node in self._nodes.values():
            if node.timestamp is None or str(node.timestamp) < str(cutoff):
                filtered.add_node(node)
        for edge in self._edges:
            if filtered.has_node(edge.source) and filtered.has_node(edge.target):
                filtered.add_edge(edge)
        return filtered

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceGraph":
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(EvidenceNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(EvidenceEdge.from_dict(edge_data))
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_components(
        cls, nodes: Iterable[EvidenceNode], edges: Iterable[EvidenceEdge]
    ) -> "EvidenceGraph":
        graph = cls()
        for node in nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)
        return graph

    def __len__(self) -> int:
        return len(self._nodes)

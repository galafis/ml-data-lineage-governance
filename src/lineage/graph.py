"""
Directed acyclic graph (DAG) for data lineage representation.

Provides a lightweight, dependency-free graph structure that models
data assets as nodes and transformations as edges, enabling upstream
and downstream traversal, impact analysis, and DOT export.
"""

from __future__ import annotations

import copy
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class NodeType(Enum):
    """Classification of nodes within the lineage graph."""

    DATASET = "dataset"
    TRANSFORMATION = "transformation"
    MODEL = "model"
    FEATURE = "feature"
    PREDICTION = "prediction"
    EXTERNAL = "external"


class LineageNode:
    """
    Represents a single asset in the lineage graph.

    Attributes:
        node_id: Unique identifier for the node.
        name: Human-readable name.
        node_type: Classification from NodeType.
        metadata: Arbitrary key-value metadata.
        created_at: Timestamp of node creation.
    """

    __slots__ = ("node_id", "name", "node_type", "metadata", "created_at")

    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: NodeType,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return f"LineageNode(id={self.node_id!r}, name={self.name!r}, type={self.node_type.value})"


class LineageEdge:
    """
    Represents a directed relationship between two lineage nodes.

    Attributes:
        source_id: Origin node identifier.
        target_id: Destination node identifier.
        transformation: Description of the transformation applied.
        metadata: Additional edge metadata (e.g. SQL query, function name).
        created_at: Timestamp of edge creation.
    """

    __slots__ = ("source_id", "target_id", "transformation", "metadata", "created_at")

    def __init__(
        self,
        source_id: str,
        target_id: str,
        transformation: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.transformation = transformation
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "transformation": self.transformation,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return f"LineageEdge({self.source_id!r} -> {self.target_id!r})"


class LineageGraph:
    """
    A lightweight directed acyclic graph for data lineage.

    Nodes represent data assets (datasets, models, features, etc.)
    and edges represent transformations or data flows between them.
    The graph supports upstream/downstream traversal, impact analysis,
    topological sorting, and DOT-format export for visualization.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._reverse_adjacency: dict[str, list[str]] = {}
        self._edges: dict[tuple[str, str], LineageEdge] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: NodeType,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LineageNode:
        """Add a node to the graph or update it if it already exists."""
        node = LineageNode(node_id, name, node_type, metadata)
        self._nodes[node_id] = node
        self._adjacency.setdefault(node_id, [])
        self._reverse_adjacency.setdefault(node_id, [])
        return node

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Return the node with the given identifier, or None."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its incident edges."""
        if node_id not in self._nodes:
            return False
        for target in list(self._adjacency.get(node_id, [])):
            self._edges.pop((node_id, target), None)
            adj = self._reverse_adjacency.get(target, [])
            if node_id in adj:
                adj.remove(node_id)
        for source in list(self._reverse_adjacency.get(node_id, [])):
            self._edges.pop((source, node_id), None)
            adj = self._adjacency.get(source, [])
            if node_id in adj:
                adj.remove(node_id)
        self._adjacency.pop(node_id, None)
        self._reverse_adjacency.pop(node_id, None)
        del self._nodes[node_id]
        return True

    @property
    def nodes(self) -> list[LineageNode]:
        return list(self._nodes.values())

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        transformation: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> LineageEdge:
        """
        Add a directed edge from *source_id* to *target_id*.

        Both nodes must already exist in the graph.  Adding the edge is
        rejected if it would create a cycle.

        Raises:
            ValueError: If either node is missing or the edge creates a cycle.
        """
        if source_id not in self._nodes:
            raise ValueError(f"Source node '{source_id}' not found in graph")
        if target_id not in self._nodes:
            raise ValueError(f"Target node '{target_id}' not found in graph")
        if self._would_create_cycle(source_id, target_id):
            raise ValueError(
                f"Edge {source_id} -> {target_id} would create a cycle"
            )

        edge = LineageEdge(source_id, target_id, transformation, metadata)
        self._edges[(source_id, target_id)] = edge
        if target_id not in self._adjacency[source_id]:
            self._adjacency[source_id].append(target_id)
        if source_id not in self._reverse_adjacency[target_id]:
            self._reverse_adjacency[target_id].append(source_id)
        return edge

    def get_edge(self, source_id: str, target_id: str) -> Optional[LineageEdge]:
        return self._edges.get((source_id, target_id))

    @property
    def edges(self) -> list[LineageEdge]:
        return list(self._edges.values())

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def get_downstream(self, node_id: str, max_depth: int = -1) -> list[str]:
        """
        Return all downstream (successor) node IDs reachable from *node_id*.

        Args:
            node_id: Starting node.
            max_depth: Maximum traversal depth (-1 for unlimited).
        """
        return self._bfs(node_id, self._adjacency, max_depth)

    def get_upstream(self, node_id: str, max_depth: int = -1) -> list[str]:
        """
        Return all upstream (predecessor) node IDs reachable from *node_id*.
        """
        return self._bfs(node_id, self._reverse_adjacency, max_depth)

    def get_direct_successors(self, node_id: str) -> list[str]:
        return list(self._adjacency.get(node_id, []))

    def get_direct_predecessors(self, node_id: str) -> list[str]:
        return list(self._reverse_adjacency.get(node_id, []))

    def get_roots(self) -> list[str]:
        """Return nodes with no incoming edges (source datasets)."""
        return [
            nid for nid in self._nodes
            if not self._reverse_adjacency.get(nid)
        ]

    def get_leaves(self) -> list[str]:
        """Return nodes with no outgoing edges (final outputs)."""
        return [
            nid for nid in self._nodes
            if not self._adjacency.get(nid)
        ]

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def impact_analysis(self, node_id: str) -> dict[str, Any]:
        """
        Analyse the downstream impact of a change in *node_id*.

        Returns a dictionary containing:
            - affected_nodes: list of downstream node IDs
            - affected_by_type: count of affected nodes grouped by NodeType
            - paths: all paths from node_id to every affected leaf
            - total_affected: total number of affected nodes
        """
        affected = self.get_downstream(node_id)
        by_type: dict[str, int] = {}
        for nid in affected:
            ntype = self._nodes[nid].node_type.value
            by_type[ntype] = by_type.get(ntype, 0) + 1

        paths = self._all_paths_to_leaves(node_id)

        return {
            "source_node": node_id,
            "affected_nodes": affected,
            "affected_by_type": by_type,
            "paths": paths,
            "total_affected": len(affected),
        }

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for target in self._adjacency.get(nid, []):
                in_degree[target] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for target in self._adjacency.get(nid, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        if len(result) != len(self._nodes):
            raise RuntimeError("Graph contains a cycle; topological sort is impossible")
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dot(self, title: str = "DataLineage") -> str:
        """Export the graph in Graphviz DOT format."""
        type_shapes = {
            NodeType.DATASET: "cylinder",
            NodeType.TRANSFORMATION: "box",
            NodeType.MODEL: "hexagon",
            NodeType.FEATURE: "parallelogram",
            NodeType.PREDICTION: "diamond",
            NodeType.EXTERNAL: "ellipse",
        }
        type_colors = {
            NodeType.DATASET: "#4A90D9",
            NodeType.TRANSFORMATION: "#F5A623",
            NodeType.MODEL: "#7B68EE",
            NodeType.FEATURE: "#50C878",
            NodeType.PREDICTION: "#FF6347",
            NodeType.EXTERNAL: "#A9A9A9",
        }
        lines = [
            f'digraph {title} {{',
            '  rankdir=LR;',
            '  node [style=filled, fontname="Helvetica"];',
        ]
        for node in self._nodes.values():
            shape = type_shapes.get(node.node_type, "ellipse")
            color = type_colors.get(node.node_type, "#CCCCCC")
            label = f"{node.name}\\n[{node.node_type.value}]"
            lines.append(
                f'  "{node.node_id}" [label="{label}", shape={shape}, '
                f'fillcolor="{color}", fontcolor="white"];'
            )
        for edge in self._edges.values():
            label = edge.transformation
            if label:
                lines.append(
                    f'  "{edge.source_id}" -> "{edge.target_id}" [label="{label}"];'
                )
            else:
                lines.append(f'  "{edge.source_id}" -> "{edge.target_id}";')
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire graph to a plain dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of the graph."""
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "roots": self.get_roots(),
            "leaves": self.get_leaves(),
            "node_types": {
                nt.value: sum(
                    1 for n in self._nodes.values() if n.node_type == nt
                )
                for nt in NodeType
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bfs(
        self,
        start: str,
        adj: dict[str, list[str]],
        max_depth: int = -1,
    ) -> list[str]:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    if max_depth != -1 and depth + 1 > max_depth:
                        continue
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Return True if adding source -> target would create a cycle."""
        if source == target:
            return True
        visited: set[str] = set()
        queue = deque([target])
        while queue:
            current = queue.popleft()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self._adjacency.get(current, []):
                queue.append(neighbor)
        return False

    def _all_paths_to_leaves(self, start: str) -> list[list[str]]:
        """Find all paths from *start* to leaf nodes."""
        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            current, path = stack.pop()
            successors = self._adjacency.get(current, [])
            if not successors:
                if len(path) > 1:
                    paths.append(path)
            else:
                for neighbor in successors:
                    stack.append((neighbor, path + [neighbor]))
        return paths

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __repr__(self) -> str:
        return (
            f"LineageGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
        )

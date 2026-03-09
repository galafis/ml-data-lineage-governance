"""
Lineage tracker for recording data transformations in ML pipelines.

Provides a high-level API to register inputs, outputs, and
transformations, automatically building the underlying lineage DAG
and maintaining an audit log of every recorded operation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from src.lineage.graph import LineageGraph, NodeType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformationRecord:
    """Immutable record of a single data transformation event."""

    __slots__ = (
        "record_id",
        "source_ids",
        "target_id",
        "transformation_name",
        "parameters",
        "timestamp",
        "user",
        "description",
    )

    def __init__(
        self,
        source_ids: list[str],
        target_id: str,
        transformation_name: str,
        parameters: Optional[dict[str, Any]] = None,
        user: str = "system",
        description: str = "",
    ) -> None:
        self.source_ids = source_ids
        self.target_id = target_id
        self.transformation_name = transformation_name
        self.parameters = parameters or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.user = user
        self.description = description
        self.record_id = self._generate_id()

    def _generate_id(self) -> str:
        raw = f"{self.source_ids}-{self.target_id}-{self.timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_ids": self.source_ids,
            "target_id": self.target_id,
            "transformation_name": self.transformation_name,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "user": self.user,
            "description": self.description,
        }


class LineageTracker:
    """
    High-level lineage tracker that wraps a LineageGraph.

    Provides methods to register data inputs, outputs, and
    transformations, maintaining both the DAG and a chronological
    audit log.  Supports querying upstream/downstream dependencies
    and exporting the full lineage.
    """

    def __init__(self) -> None:
        self.graph = LineageGraph()
        self._audit_log: list[TransformationRecord] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def track_input(
        self,
        dataset_id: str,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        node_type: NodeType = NodeType.DATASET,
    ) -> None:
        """Register a source dataset (root node) in the lineage graph."""
        self.graph.add_node(dataset_id, name, node_type, metadata)
        logger.info("Tracked input: %s (%s)", name, dataset_id)

    def track_output(
        self,
        output_id: str,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        node_type: NodeType = NodeType.PREDICTION,
    ) -> None:
        """Register an output artifact (typically a leaf node)."""
        self.graph.add_node(output_id, name, node_type, metadata)
        logger.info("Tracked output: %s (%s)", name, output_id)

    def track_transformation(
        self,
        source_ids: list[str],
        target_id: str,
        target_name: str,
        transformation_name: str,
        target_type: NodeType = NodeType.TRANSFORMATION,
        parameters: Optional[dict[str, Any]] = None,
        user: str = "system",
        description: str = "",
        target_metadata: Optional[dict[str, Any]] = None,
    ) -> TransformationRecord:
        """
        Record a data transformation that produces *target_id* from
        one or more *source_ids*.

        Creates the target node if it does not already exist and adds
        edges from every source to the target.
        """
        if target_id not in self.graph:
            self.graph.add_node(target_id, target_name, target_type, target_metadata)

        for src in source_ids:
            if src not in self.graph:
                raise ValueError(
                    f"Source node '{src}' must be tracked before "
                    f"recording a transformation"
                )
            if not self.graph.get_edge(src, target_id):
                self.graph.add_edge(
                    src,
                    target_id,
                    transformation=transformation_name,
                    metadata=parameters,
                )

        record = TransformationRecord(
            source_ids=source_ids,
            target_id=target_id,
            transformation_name=transformation_name,
            parameters=parameters,
            user=user,
            description=description,
        )
        self._audit_log.append(record)
        logger.info(
            "Tracked transformation '%s': %s -> %s",
            transformation_name,
            source_ids,
            target_id,
        )
        return record

    def track_model(
        self,
        model_id: str,
        name: str,
        feature_ids: list[str],
        metadata: Optional[dict[str, Any]] = None,
        user: str = "system",
    ) -> TransformationRecord:
        """Convenience method for tracking a model trained on features."""
        return self.track_transformation(
            source_ids=feature_ids,
            target_id=model_id,
            target_name=name,
            transformation_name="model_training",
            target_type=NodeType.MODEL,
            parameters=metadata,
            user=user,
            description=f"Model '{name}' trained on {len(feature_ids)} features",
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_upstream(self, node_id: str) -> list[str]:
        """Return all upstream dependencies for the given node."""
        return self.graph.get_upstream(node_id)

    def get_downstream(self, node_id: str) -> list[str]:
        """Return all downstream dependents for the given node."""
        return self.graph.get_downstream(node_id)

    def impact_analysis(self, node_id: str) -> dict[str, Any]:
        """Perform downstream impact analysis."""
        return self.graph.impact_analysis(node_id)

    def get_transformation_history(
        self, target_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Return the audit log as a list of dicts, optionally filtered
        by target node.
        """
        records = self._audit_log
        if target_id:
            records = [r for r in records if r.target_id == target_id]
        return [r.to_dict() for r in records]

    def get_full_lineage(self, node_id: str) -> dict[str, Any]:
        """
        Compile full lineage information for a single node, including
        upstream inputs, downstream consumers, and transformation history.
        """
        node = self.graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found")

        upstream = self.get_upstream(node_id)
        downstream = self.get_downstream(node_id)

        upstream_details = []
        for uid in upstream:
            u_node = self.graph.get_node(uid)
            if u_node:
                upstream_details.append(u_node.to_dict())

        downstream_details = []
        for did in downstream:
            d_node = self.graph.get_node(did)
            if d_node:
                downstream_details.append(d_node.to_dict())

        return {
            "node": node.to_dict(),
            "upstream": upstream_details,
            "downstream": downstream_details,
            "transformation_history": self.get_transformation_history(node_id),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_dot(self, title: str = "DataLineage") -> str:
        """Export the lineage graph in Graphviz DOT format."""
        return self.graph.to_dot(title)

    def export_json(self) -> str:
        """Export the full lineage (graph + audit log) as JSON."""
        return json.dumps(
            {
                "graph": self.graph.to_dict(),
                "audit_log": [r.to_dict() for r in self._audit_log],
            },
            indent=2,
            default=str,
        )

    def summary(self) -> dict[str, Any]:
        """High-level summary of the lineage tracker state."""
        graph_summary = self.graph.summary()
        graph_summary["total_transformations_recorded"] = len(self._audit_log)
        return graph_summary

    def __repr__(self) -> str:
        return (
            f"LineageTracker(nodes={len(self.graph)}, "
            f"transformations={len(self._audit_log)})"
        )

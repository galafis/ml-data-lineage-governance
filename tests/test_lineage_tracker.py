"""Tests for src.lineage.tracker — LineageTracker high-level API."""

import pytest

from src.lineage.graph import NodeType
from src.lineage.tracker import LineageTracker


class TestLineageTracker:

    def test_track_input_creates_node(self):
        t = LineageTracker()
        t.track_input("ds1", "Dataset 1")
        assert "ds1" in t.graph

    def test_track_transformation_builds_edge(self):
        t = LineageTracker()
        t.track_input("ds1", "Raw")
        t.track_transformation(
            source_ids=["ds1"],
            target_id="ds2",
            target_name="Clean",
            transformation_name="clean",
        )
        assert "ds2" in t.graph
        assert t.graph.get_edge("ds1", "ds2") is not None

    def test_transformation_requires_tracked_source(self):
        t = LineageTracker()
        with pytest.raises(ValueError, match="must be tracked"):
            t.track_transformation(
                source_ids=["nonexistent"],
                target_id="ds2",
                target_name="Clean",
                transformation_name="clean",
            )

    def test_audit_log_records_transformations(self):
        t = LineageTracker()
        t.track_input("a", "A")
        t.track_transformation(["a"], "b", "B", "step1")
        history = t.get_transformation_history()
        assert len(history) == 1
        assert history[0]["transformation_name"] == "step1"

    def test_full_lineage_includes_upstream_and_downstream(self):
        t = LineageTracker()
        t.track_input("a", "A")
        t.track_transformation(["a"], "b", "B", "step1")
        t.track_transformation(["b"], "c", "C", "step2")

        lineage = t.get_full_lineage("b")
        upstream_ids = [n["node_id"] for n in lineage["upstream"]]
        downstream_ids = [n["node_id"] for n in lineage["downstream"]]
        assert "a" in upstream_ids
        assert "c" in downstream_ids

    def test_export_json_is_valid(self):
        import json
        t = LineageTracker()
        t.track_input("x", "X")
        data = json.loads(t.export_json())
        assert "graph" in data
        assert "audit_log" in data

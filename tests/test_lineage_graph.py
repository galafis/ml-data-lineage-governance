"""Tests for src.lineage.graph — LineageGraph DAG operations."""

import pytest

from src.lineage.graph import LineageGraph, NodeType


class TestLineageGraphNodes:
    """Node CRUD operations."""

    def test_add_and_retrieve_node(self):
        g = LineageGraph()
        node = g.add_node("n1", "Dataset A", NodeType.DATASET)
        assert node.node_id == "n1"
        assert g.get_node("n1") is not None
        assert "n1" in g

    def test_remove_node_cleans_edges(self):
        g = LineageGraph()
        g.add_node("a", "A", NodeType.DATASET)
        g.add_node("b", "B", NodeType.TRANSFORMATION)
        g.add_edge("a", "b", "transform")
        assert g.remove_node("a")
        assert "a" not in g
        assert g.get_edge("a", "b") is None

    def test_node_not_found_returns_none(self):
        g = LineageGraph()
        assert g.get_node("missing") is None


class TestLineageGraphEdges:
    """Edge operations and cycle detection."""

    def test_add_edge(self):
        g = LineageGraph()
        g.add_node("a", "A", NodeType.DATASET)
        g.add_node("b", "B", NodeType.TRANSFORMATION)
        edge = g.add_edge("a", "b", "etl")
        assert edge.source_id == "a"
        assert edge.target_id == "b"

    def test_edge_rejects_missing_node(self):
        g = LineageGraph()
        g.add_node("a", "A", NodeType.DATASET)
        with pytest.raises(ValueError, match="not found"):
            g.add_edge("a", "ghost")

    def test_edge_rejects_cycle(self):
        g = LineageGraph()
        g.add_node("a", "A", NodeType.DATASET)
        g.add_node("b", "B", NodeType.TRANSFORMATION)
        g.add_edge("a", "b")
        with pytest.raises(ValueError, match="cycle"):
            g.add_edge("b", "a")

    def test_self_loop_rejected(self):
        g = LineageGraph()
        g.add_node("x", "X", NodeType.DATASET)
        with pytest.raises(ValueError, match="cycle"):
            g.add_edge("x", "x")


class TestLineageGraphTraversal:
    """Upstream/downstream traversal and impact analysis."""

    @pytest.fixture()
    def pipeline_graph(self) -> LineageGraph:
        g = LineageGraph()
        g.add_node("raw", "Raw", NodeType.DATASET)
        g.add_node("clean", "Clean", NodeType.DATASET)
        g.add_node("feat", "Features", NodeType.FEATURE)
        g.add_node("model", "Model", NodeType.MODEL)
        g.add_node("pred", "Predictions", NodeType.PREDICTION)
        g.add_edge("raw", "clean", "clean")
        g.add_edge("clean", "feat", "engineer")
        g.add_edge("feat", "model", "train")
        g.add_edge("model", "pred", "infer")
        return g

    def test_get_downstream(self, pipeline_graph: LineageGraph):
        downstream = pipeline_graph.get_downstream("raw")
        assert set(downstream) == {"clean", "feat", "model", "pred"}

    def test_get_upstream(self, pipeline_graph: LineageGraph):
        upstream = pipeline_graph.get_upstream("pred")
        assert set(upstream) == {"model", "feat", "clean", "raw"}

    def test_get_roots_and_leaves(self, pipeline_graph: LineageGraph):
        assert pipeline_graph.get_roots() == ["raw"]
        assert pipeline_graph.get_leaves() == ["pred"]

    def test_impact_analysis(self, pipeline_graph: LineageGraph):
        impact = pipeline_graph.impact_analysis("clean")
        assert impact["total_affected"] == 3
        assert "feat" in impact["affected_nodes"]

    def test_topological_sort(self, pipeline_graph: LineageGraph):
        order = pipeline_graph.topological_sort()
        assert order.index("raw") < order.index("clean")
        assert order.index("clean") < order.index("feat")
        assert order.index("feat") < order.index("model")

    def test_dot_export_contains_nodes(self, pipeline_graph: LineageGraph):
        dot = pipeline_graph.to_dot("TestGraph")
        assert "digraph TestGraph" in dot
        assert '"raw"' in dot
        assert '"pred"' in dot

"""Tests for src.quality.profiler — DataProfiler statistics."""

import numpy as np
import pandas as pd
import pytest

from src.quality.profiler import DataProfiler


@pytest.fixture()
def profiler() -> DataProfiler:
    return DataProfiler(sample_size=3)


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, 30.0, None, 50.0],
            "category": ["a", "b", "a", "c", "b"],
        }
    )


class TestProfile:

    def test_profile_returns_all_columns(self, profiler: DataProfiler, df: pd.DataFrame):
        result = profiler.profile(df, "test_dataset")
        assert result["column_count"] == 3
        assert len(result["columns"]) == 3

    def test_completeness_calculation(self, profiler: DataProfiler, df: pd.DataFrame):
        result = profiler.profile(df)
        value_profile = next(c for c in result["columns"] if c["name"] == "value")
        assert value_profile["completeness"] == 0.8  # 4 out of 5


class TestSchemaDetection:

    def test_detect_schema(self, profiler: DataProfiler, df: pd.DataFrame):
        schema = profiler.detect_schema(df)
        assert "id" in schema
        assert schema["category"] == "object"

    def test_compare_schemas_detects_changes(self, profiler: DataProfiler):
        old = {"a": "int64", "b": "object", "c": "float64"}
        new = {"a": "int64", "b": "object", "d": "bool"}
        diff = profiler.compare_schemas(old, new)
        assert "c" in diff["removed"]
        assert "d" in diff["added"]
        assert diff["is_compatible"] is False

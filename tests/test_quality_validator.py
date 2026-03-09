"""Tests for src.quality.validator — DataQualityValidator rules."""

import numpy as np
import pandas as pd
import pytest

from src.quality.validator import DataQualityValidator, RuleSeverity


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "Diana", "Eve"],
            "age": [25, 30, -5, 45, 200],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "bad-email",
                "diana@example.com",
                "eve@example.com",
            ],
            "status": ["active", "active", "inactive", "active", "unknown"],
        }
    )


class TestNotNullRule:

    def test_detects_nulls(self, sample_df: pd.DataFrame):
        v = DataQualityValidator("test")
        v.add_not_null_rule("name")
        report = v.validate(sample_df)
        result = report.results[0]
        assert result["passed"] is False
        assert result["failing_rows"] == 1


class TestUniqueRule:

    def test_passes_unique_column(self, sample_df: pd.DataFrame):
        v = DataQualityValidator("test")
        v.add_unique_rule("id")
        report = v.validate(sample_df)
        assert report.results[0]["passed"] is True

    def test_detects_duplicates(self):
        df = pd.DataFrame({"col": [1, 2, 2, 3]})
        v = DataQualityValidator("test")
        v.add_unique_rule("col")
        report = v.validate(df)
        assert report.results[0]["passed"] is False


class TestRangeCheck:

    def test_detects_out_of_range(self, sample_df: pd.DataFrame):
        v = DataQualityValidator("test")
        v.add_range_check("age", min_value=0, max_value=120)
        report = v.validate(sample_df)
        result = report.results[0]
        assert result["passed"] is False
        assert result["failing_rows"] >= 2  # -5 and 200


class TestRegexRule:

    def test_detects_invalid_emails(self, sample_df: pd.DataFrame):
        v = DataQualityValidator("test")
        v.add_regex_rule(
            "email",
            r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
        )
        report = v.validate(sample_df)
        assert report.results[0]["passed"] is False


class TestReferentialIntegrity:

    def test_detects_orphans(self, sample_df: pd.DataFrame):
        v = DataQualityValidator("test")
        v.add_referential_integrity_rule(
            "status",
            reference_values=["active", "inactive"],
        )
        report = v.validate(sample_df)
        result = report.results[0]
        assert result["passed"] is False
        assert "unknown" in result["sample_failures"]


class TestQualityReport:

    def test_overall_pass_when_only_warnings_fail(self):
        df = pd.DataFrame({"col": [None, "a", "b"]})
        v = DataQualityValidator("test")
        v.add_not_null_rule("col", severity=RuleSeverity.WARNING)
        report = v.validate(df)
        assert report.overall_pass is True  # only warnings, no errors

    def test_quality_score_calculation(self, sample_df: pd.DataFrame):
        v = DataQualityValidator("test")
        v.add_not_null_rule("id")  # passes
        v.add_unique_rule("id")   # passes
        report = v.validate(sample_df)
        assert report.quality_score == 100.0

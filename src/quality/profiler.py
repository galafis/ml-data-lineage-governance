"""
Data profiling engine for statistical analysis and schema detection.

Produces per-column statistics (completeness, uniqueness, distributions,
anomaly indicators) and supports schema comparison between two DataFrames
to detect structural drift.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ColumnProfile:
    """Statistical profile of a single DataFrame column."""

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype
        self.total_count: int = 0
        self.null_count: int = 0
        self.completeness: float = 0.0
        self.unique_count: int = 0
        self.uniqueness: float = 0.0
        self.most_common: list[tuple[Any, int]] = []
        self.min_value: Any = None
        self.max_value: Any = None
        self.mean: Optional[float] = None
        self.median: Optional[float] = None
        self.std_dev: Optional[float] = None
        self.percentiles: dict[str, float] = {}
        self.anomaly_score: float = 0.0
        self.sample_values: list[Any] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "completeness": round(self.completeness, 4),
            "unique_count": self.unique_count,
            "uniqueness": round(self.uniqueness, 4),
            "most_common": [
                {"value": str(v), "count": c} for v, c in self.most_common
            ],
            "min_value": _safe_serialize(self.min_value),
            "max_value": _safe_serialize(self.max_value),
            "mean": _safe_round(self.mean),
            "median": _safe_round(self.median),
            "std_dev": _safe_round(self.std_dev),
            "percentiles": {k: round(v, 4) for k, v in self.percentiles.items()},
            "anomaly_score": round(self.anomaly_score, 4),
            "sample_values": [str(v) for v in self.sample_values],
        }


class DataProfiler:
    """
    Generate column-level statistics and schema information from a DataFrame.

    Typical usage::

        profiler = DataProfiler()
        profile = profiler.profile(df)
        schema = profiler.detect_schema(df)
        diff = profiler.compare_schemas(old_schema, new_schema)
    """

    def __init__(self, sample_size: int = 5, anomaly_zscore: float = 3.0) -> None:
        """
        Args:
            sample_size: Number of sample values to include per column.
            anomaly_zscore: Z-score threshold for flagging anomalies.
        """
        self.sample_size = sample_size
        self.anomaly_zscore = anomaly_zscore

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame, dataset_name: str = "unnamed") -> dict[str, Any]:
        """
        Profile every column in *df* and return a structured report.

        Returns:
            Dictionary with dataset-level metadata and a list of column profiles.
        """
        column_profiles: list[dict[str, Any]] = []
        for col in df.columns:
            cp = self._profile_column(df, col)
            column_profiles.append(cp.to_dict())

        report = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": column_profiles,
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "duplicate_row_count": int(df.duplicated().sum()),
        }
        logger.info(
            "Profiled '%s': %d rows x %d cols",
            dataset_name,
            len(df),
            len(df.columns),
        )
        return report

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        dtype_str = str(series.dtype)
        cp = ColumnProfile(col, dtype_str)
        cp.total_count = len(series)
        cp.null_count = int(series.isna().sum())
        cp.completeness = 1 - cp.null_count / cp.total_count if cp.total_count else 0
        non_null = series.dropna()
        cp.unique_count = int(non_null.nunique())
        cp.uniqueness = cp.unique_count / len(non_null) if len(non_null) else 0

        vc = non_null.value_counts().head(5)
        cp.most_common = list(zip(vc.index.tolist(), vc.values.tolist()))

        cp.sample_values = non_null.head(self.sample_size).tolist()

        if pd.api.types.is_numeric_dtype(series):
            cp.min_value = float(non_null.min()) if len(non_null) else None
            cp.max_value = float(non_null.max()) if len(non_null) else None
            cp.mean = float(non_null.mean()) if len(non_null) else None
            cp.median = float(non_null.median()) if len(non_null) else None
            cp.std_dev = float(non_null.std()) if len(non_null) > 1 else None
            if len(non_null) > 0:
                cp.percentiles = {
                    "p25": float(non_null.quantile(0.25)),
                    "p50": float(non_null.quantile(0.50)),
                    "p75": float(non_null.quantile(0.75)),
                    "p95": float(non_null.quantile(0.95)),
                    "p99": float(non_null.quantile(0.99)),
                }
            cp.anomaly_score = self._calculate_anomaly_score(non_null)
        elif pd.api.types.is_datetime64_any_dtype(series):
            cp.min_value = str(non_null.min()) if len(non_null) else None
            cp.max_value = str(non_null.max()) if len(non_null) else None
        else:
            if len(non_null) > 0:
                lengths = non_null.astype(str).str.len()
                cp.min_value = int(lengths.min())
                cp.max_value = int(lengths.max())
                cp.mean = float(lengths.mean())

        return cp

    def _calculate_anomaly_score(self, series: pd.Series) -> float:
        """Return fraction of values beyond the z-score threshold."""
        if len(series) < 3:
            return 0.0
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            return 0.0
        z_scores = np.abs((series - mean) / std)
        anomalies = (z_scores > self.anomaly_zscore).sum()
        return float(anomalies / len(series))

    # ------------------------------------------------------------------
    # Schema detection and comparison
    # ------------------------------------------------------------------

    def detect_schema(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Detect the schema (column name -> dtype string) of a DataFrame.
        """
        return {col: str(df[col].dtype) for col in df.columns}

    def compare_schemas(
        self,
        old_schema: dict[str, str],
        new_schema: dict[str, str],
    ) -> dict[str, Any]:
        """
        Compare two schemas and report structural differences.

        Returns a dict with keys: added, removed, type_changed, unchanged.
        """
        old_cols = set(old_schema.keys())
        new_cols = set(new_schema.keys())

        added = sorted(new_cols - old_cols)
        removed = sorted(old_cols - new_cols)
        common = old_cols & new_cols

        type_changed = []
        unchanged = []
        for col in sorted(common):
            if old_schema[col] != new_schema[col]:
                type_changed.append(
                    {"column": col, "old_type": old_schema[col], "new_type": new_schema[col]}
                )
            else:
                unchanged.append(col)

        return {
            "added": added,
            "removed": removed,
            "type_changed": type_changed,
            "unchanged": unchanged,
            "is_compatible": len(removed) == 0 and len(type_changed) == 0,
        }

    def __repr__(self) -> str:
        return f"DataProfiler(sample_size={self.sample_size})"


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_serialize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _safe_round(value: Optional[float], decimals: int = 4) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), decimals)

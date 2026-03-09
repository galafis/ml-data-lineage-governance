"""
Data quality validation engine.

Provides a rule-based framework for validating tabular data against
configurable quality expectations.  Each rule produces a granular
pass/fail result and optional details, aggregated into a quality report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RuleType(Enum):
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    RANGE_CHECK = "range_check"
    REGEX_MATCH = "regex_match"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    CUSTOM = "custom"


class RuleSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationRule:
    """Definition of a single data-quality validation rule."""

    name: str
    rule_type: RuleType
    column: str
    severity: RuleSeverity = RuleSeverity.ERROR
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule_type": self.rule_type.value,
            "column": self.column,
            "severity": self.severity.value,
            "parameters": self.parameters,
            "description": self.description,
        }


@dataclass
class RuleResult:
    """Outcome of a single validation rule execution."""

    rule_name: str
    rule_type: str
    column: str
    passed: bool
    severity: str
    total_rows: int
    failing_rows: int = 0
    failure_rate: float = 0.0
    details: str = ""
    sample_failures: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "column": self.column,
            "passed": self.passed,
            "severity": self.severity,
            "total_rows": self.total_rows,
            "failing_rows": self.failing_rows,
            "failure_rate": round(self.failure_rate, 4),
            "details": self.details,
            "sample_failures": self.sample_failures[:5],
        }


@dataclass
class QualityReport:
    """Aggregated validation report for a dataset."""

    dataset_name: str
    timestamp: str
    total_rules: int
    passed_rules: int
    failed_rules: int
    warning_rules: int
    overall_pass: bool
    results: list[dict[str, Any]]
    quality_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "warning_rules": self.warning_rules,
            "overall_pass": self.overall_pass,
            "quality_score": round(self.quality_score, 2),
            "results": self.results,
        }


class DataQualityValidator:
    """
    Configurable data-quality validation engine.

    Register rules via helper methods (``add_not_null_rule``, etc.),
    then call ``validate(df)`` to execute all rules against a DataFrame
    and receive a ``QualityReport``.
    """

    def __init__(self, dataset_name: str = "unnamed") -> None:
        self.dataset_name = dataset_name
        self._rules: list[ValidationRule] = []
        self._custom_validators: dict[str, Callable[[pd.DataFrame], RuleResult]] = {}

    # ------------------------------------------------------------------
    # Rule registration helpers
    # ------------------------------------------------------------------

    def add_not_null_rule(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
    ) -> None:
        self._rules.append(
            ValidationRule(
                name=f"not_null_{column}",
                rule_type=RuleType.NOT_NULL,
                column=column,
                severity=severity,
                description=description or f"Column '{column}' must not contain nulls",
            )
        )

    def add_unique_rule(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
    ) -> None:
        self._rules.append(
            ValidationRule(
                name=f"unique_{column}",
                rule_type=RuleType.UNIQUE,
                column=column,
                severity=severity,
                description=description or f"Column '{column}' must have unique values",
            )
        )

    def add_range_check(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
    ) -> None:
        params: dict[str, Any] = {}
        if min_value is not None:
            params["min_value"] = min_value
        if max_value is not None:
            params["max_value"] = max_value
        self._rules.append(
            ValidationRule(
                name=f"range_{column}",
                rule_type=RuleType.RANGE_CHECK,
                column=column,
                severity=severity,
                parameters=params,
                description=description
                or f"Column '{column}' must be in [{min_value}, {max_value}]",
            )
        )

    def add_regex_rule(
        self,
        column: str,
        pattern: str,
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
    ) -> None:
        self._rules.append(
            ValidationRule(
                name=f"regex_{column}",
                rule_type=RuleType.REGEX_MATCH,
                column=column,
                severity=severity,
                parameters={"pattern": pattern},
                description=description
                or f"Column '{column}' must match pattern '{pattern}'",
            )
        )

    def add_referential_integrity_rule(
        self,
        column: str,
        reference_values: list[Any],
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
    ) -> None:
        self._rules.append(
            ValidationRule(
                name=f"ref_integrity_{column}",
                rule_type=RuleType.REFERENTIAL_INTEGRITY,
                column=column,
                severity=severity,
                parameters={"reference_values": reference_values},
                description=description
                or f"Column '{column}' values must exist in reference set",
            )
        )

    def add_custom_rule(
        self,
        name: str,
        column: str,
        validator_fn: Callable[[pd.DataFrame], RuleResult],
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
    ) -> None:
        self._rules.append(
            ValidationRule(
                name=name,
                rule_type=RuleType.CUSTOM,
                column=column,
                severity=severity,
                description=description,
            )
        )
        self._custom_validators[name] = validator_fn

    # ------------------------------------------------------------------
    # Validation execution
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> QualityReport:
        """Execute all registered rules against *df* and return a report."""
        results: list[RuleResult] = []
        for rule in self._rules:
            if rule.column not in df.columns and rule.rule_type != RuleType.CUSTOM:
                results.append(
                    RuleResult(
                        rule_name=rule.name,
                        rule_type=rule.rule_type.value,
                        column=rule.column,
                        passed=False,
                        severity=rule.severity.value,
                        total_rows=len(df),
                        details=f"Column '{rule.column}' not found in DataFrame",
                    )
                )
                continue

            result = self._execute_rule(rule, df)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        errors = sum(
            1 for r in results if not r.passed and r.severity == RuleSeverity.ERROR.value
        )
        warnings = sum(
            1 for r in results if not r.passed and r.severity == RuleSeverity.WARNING.value
        )
        overall_pass = errors == 0
        score = (passed / len(results) * 100) if results else 100.0

        report = QualityReport(
            dataset_name=self.dataset_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_rules=len(results),
            passed_rules=passed,
            failed_rules=errors,
            warning_rules=warnings,
            overall_pass=overall_pass,
            results=[r.to_dict() for r in results],
            quality_score=score,
        )
        logger.info(
            "Quality report for '%s': score=%.1f%%, pass=%s",
            self.dataset_name,
            score,
            overall_pass,
        )
        return report

    # ------------------------------------------------------------------
    # Rule executors (private)
    # ------------------------------------------------------------------

    def _execute_rule(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        dispatch = {
            RuleType.NOT_NULL: self._check_not_null,
            RuleType.UNIQUE: self._check_unique,
            RuleType.RANGE_CHECK: self._check_range,
            RuleType.REGEX_MATCH: self._check_regex,
            RuleType.REFERENTIAL_INTEGRITY: self._check_referential,
            RuleType.CUSTOM: self._check_custom,
        }
        handler = dispatch.get(rule.rule_type)
        if handler is None:
            return RuleResult(
                rule_name=rule.name,
                rule_type=rule.rule_type.value,
                column=rule.column,
                passed=False,
                severity=rule.severity.value,
                total_rows=len(df),
                details=f"Unknown rule type: {rule.rule_type}",
            )
        return handler(rule, df)

    def _check_not_null(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        null_mask = df[rule.column].isna()
        failing = int(null_mask.sum())
        total = len(df)
        return RuleResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            column=rule.column,
            passed=failing == 0,
            severity=rule.severity.value,
            total_rows=total,
            failing_rows=failing,
            failure_rate=failing / total if total else 0,
            details=f"{failing}/{total} null values found",
            sample_failures=df.index[null_mask].tolist()[:5],
        )

    def _check_unique(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        dup_mask = df[rule.column].duplicated(keep=False)
        failing = int(dup_mask.sum())
        total = len(df)
        duplicated_values = df.loc[dup_mask, rule.column].unique().tolist()[:5]
        return RuleResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            column=rule.column,
            passed=failing == 0,
            severity=rule.severity.value,
            total_rows=total,
            failing_rows=failing,
            failure_rate=failing / total if total else 0,
            details=f"{failing}/{total} duplicate rows",
            sample_failures=duplicated_values,
        )

    def _check_range(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        col = pd.to_numeric(df[rule.column], errors="coerce")
        min_val = rule.parameters.get("min_value")
        max_val = rule.parameters.get("max_value")

        fail_mask = pd.Series(False, index=df.index)
        if min_val is not None:
            fail_mask = fail_mask | (col < min_val)
        if max_val is not None:
            fail_mask = fail_mask | (col > max_val)
        fail_mask = fail_mask | col.isna()

        failing = int(fail_mask.sum())
        total = len(df)
        return RuleResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            column=rule.column,
            passed=failing == 0,
            severity=rule.severity.value,
            total_rows=total,
            failing_rows=failing,
            failure_rate=failing / total if total else 0,
            details=f"{failing}/{total} values out of range [{min_val}, {max_val}]",
            sample_failures=col[fail_mask].dropna().tolist()[:5],
        )

    def _check_regex(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        pattern = rule.parameters.get("pattern", "")
        compiled = re.compile(pattern)
        col = df[rule.column].astype(str).fillna("")
        match_mask = col.apply(lambda v: bool(compiled.match(v)))
        fail_mask = ~match_mask
        failing = int(fail_mask.sum())
        total = len(df)
        return RuleResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            column=rule.column,
            passed=failing == 0,
            severity=rule.severity.value,
            total_rows=total,
            failing_rows=failing,
            failure_rate=failing / total if total else 0,
            details=f"{failing}/{total} values do not match /{pattern}/",
            sample_failures=col[fail_mask].tolist()[:5],
        )

    def _check_referential(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        ref_values = set(rule.parameters.get("reference_values", []))
        col = df[rule.column]
        fail_mask = ~col.isin(ref_values) & col.notna()
        failing = int(fail_mask.sum())
        total = len(df)
        orphans = col[fail_mask].unique().tolist()[:5]
        return RuleResult(
            rule_name=rule.name,
            rule_type=rule.rule_type.value,
            column=rule.column,
            passed=failing == 0,
            severity=rule.severity.value,
            total_rows=total,
            failing_rows=failing,
            failure_rate=failing / total if total else 0,
            details=f"{failing}/{total} orphan references found",
            sample_failures=orphans,
        )

    def _check_custom(self, rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
        fn = self._custom_validators.get(rule.name)
        if fn is None:
            return RuleResult(
                rule_name=rule.name,
                rule_type=rule.rule_type.value,
                column=rule.column,
                passed=False,
                severity=rule.severity.value,
                total_rows=len(df),
                details="Custom validator function not found",
            )
        return fn(df)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def rules(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._rules]

    def clear_rules(self) -> None:
        self._rules.clear()
        self._custom_validators.clear()

    def __repr__(self) -> str:
        return (
            f"DataQualityValidator(dataset={self.dataset_name!r}, "
            f"rules={len(self._rules)})"
        )

"""
Compliance engine for GDPR, LGPD, and data retention enforcement.

Detects personally identifiable information (PII) using pattern
matching, evaluates data against configurable compliance policies,
and generates structured compliance reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PIIType(Enum):
    EMAIL = "email"
    CPF = "cpf"
    CNPJ = "cnpj"
    PHONE_BR = "phone_br"
    PHONE_INTL = "phone_intl"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"
    SSN = "ssn"


class ComplianceFramework(Enum):
    GDPR = "gdpr"
    LGPD = "lgpd"
    HIPAA = "hipaa"
    CCPA = "ccpa"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ------------------------------------------------------------------
# PII detection patterns
# ------------------------------------------------------------------

PII_PATTERNS: dict[PIIType, re.Pattern] = {
    PIIType.EMAIL: re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
    ),
    PIIType.CPF: re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    PIIType.CNPJ: re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    PIIType.PHONE_BR: re.compile(
        r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}"
    ),
    PIIType.PHONE_INTL: re.compile(r"\+\d{1,3}[\s\-]?\d{6,14}"),
    PIIType.CREDIT_CARD: re.compile(
        r"\b(?:\d[ \-]*?){13,19}\b"
    ),
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# Heuristic column name indicators for PII
PII_COLUMN_HINTS: dict[PIIType, list[str]] = {
    PIIType.EMAIL: ["email", "e_mail", "e-mail", "correo", "mail"],
    PIIType.CPF: ["cpf", "documento", "tax_id"],
    PIIType.PHONE_BR: ["telefone", "phone", "celular", "tel"],
    PIIType.PHONE_INTL: ["phone", "mobile", "tel"],
    PIIType.CREDIT_CARD: ["card", "cartao", "credit", "cc_number"],
    PIIType.NAME: ["nome", "name", "full_name", "first_name", "last_name", "sobrenome"],
    PIIType.DATE_OF_BIRTH: ["nascimento", "birth", "dob", "data_nascimento", "birthdate"],
    PIIType.IP_ADDRESS: ["ip", "ip_address", "ip_addr"],
    PIIType.SSN: ["ssn", "social_security"],
}


@dataclass
class PIIFinding:
    """A single PII detection result."""

    column: str
    pii_type: str
    sample_count: int
    detection_method: str
    risk_level: str
    sample_matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "pii_type": self.pii_type,
            "sample_count": self.sample_count,
            "detection_method": self.detection_method,
            "risk_level": self.risk_level,
            "sample_matches": self.sample_matches[:3],
        }


@dataclass
class RetentionViolation:
    """A data retention policy violation."""

    column: str
    policy_name: str
    max_age_days: int
    oldest_record_days: int
    records_exceeding: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "policy_name": self.policy_name,
            "max_age_days": self.max_age_days,
            "oldest_record_days": self.oldest_record_days,
            "records_exceeding": self.records_exceeding,
            "description": self.description,
        }


@dataclass
class ComplianceReport:
    """Aggregated compliance assessment report."""

    dataset_name: str
    framework: str
    timestamp: str
    overall_compliant: bool
    risk_level: str
    pii_findings: list[dict[str, Any]]
    retention_violations: list[dict[str, Any]]
    recommendations: list[str]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "framework": self.framework,
            "timestamp": self.timestamp,
            "overall_compliant": self.overall_compliant,
            "risk_level": self.risk_level,
            "pii_findings": self.pii_findings,
            "retention_violations": self.retention_violations,
            "recommendations": self.recommendations,
            "summary": self.summary,
        }


class ComplianceEngine:
    """
    Evaluate data for regulatory compliance (GDPR, LGPD, etc.).

    Performs PII detection via regex and column-name heuristics,
    checks data retention policies, and produces an actionable
    compliance report.
    """

    def __init__(
        self,
        framework: ComplianceFramework = ComplianceFramework.LGPD,
        retention_policies: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Args:
            framework: Target regulatory framework.
            retention_policies: Mapping of date-column name to maximum
                retention age in days.
        """
        self.framework = framework
        self.retention_policies: dict[str, int] = retention_policies or {}

    # ------------------------------------------------------------------
    # PII detection
    # ------------------------------------------------------------------

    def detect_pii(self, df: pd.DataFrame) -> list[PIIFinding]:
        """Scan all columns for potential PII content."""
        findings: list[PIIFinding] = []
        for col in df.columns:
            col_findings = self._scan_column(df, col)
            findings.extend(col_findings)
        return findings

    def _scan_column(self, df: pd.DataFrame, col: str) -> list[PIIFinding]:
        results: list[PIIFinding] = []
        col_lower = col.lower()

        # 1) Column-name heuristic
        for pii_type, hints in PII_COLUMN_HINTS.items():
            if any(hint in col_lower for hint in hints):
                results.append(
                    PIIFinding(
                        column=col,
                        pii_type=pii_type.value,
                        sample_count=int(df[col].notna().sum()),
                        detection_method="column_name_heuristic",
                        risk_level=RiskLevel.MEDIUM.value,
                    )
                )

        # 2) Regex pattern matching on string columns
        if df[col].dtype == object or str(df[col].dtype) == "string":
            sample = df[col].dropna().astype(str)
            if len(sample) == 0:
                return results
            for pii_type, pattern in PII_PATTERNS.items():
                matches = sample[sample.apply(lambda v: bool(pattern.search(v)))]
                if len(matches) > 0:
                    already = any(
                        f.pii_type == pii_type.value and f.column == col
                        for f in results
                    )
                    risk = self._assess_risk(pii_type, len(matches), len(sample))
                    if already:
                        for f in results:
                            if f.pii_type == pii_type.value and f.column == col:
                                f.detection_method = "column_name_heuristic+regex"
                                f.risk_level = risk
                                f.sample_count = len(matches)
                                f.sample_matches = self._redact_samples(
                                    matches.head(3).tolist()
                                )
                    else:
                        results.append(
                            PIIFinding(
                                column=col,
                                pii_type=pii_type.value,
                                sample_count=len(matches),
                                detection_method="regex_pattern",
                                risk_level=risk,
                                sample_matches=self._redact_samples(
                                    matches.head(3).tolist()
                                ),
                            )
                        )
        return results

    @staticmethod
    def _assess_risk(pii_type: PIIType, match_count: int, total: int) -> str:
        high_risk = {PIIType.CPF, PIIType.CREDIT_CARD, PIIType.SSN, PIIType.CNPJ}
        if pii_type in high_risk:
            return RiskLevel.CRITICAL.value
        if match_count / total > 0.5:
            return RiskLevel.HIGH.value
        if match_count > 10:
            return RiskLevel.MEDIUM.value
        return RiskLevel.LOW.value

    @staticmethod
    def _redact_samples(values: list[str]) -> list[str]:
        """Partially redact sample PII values for the report."""
        redacted = []
        for v in values:
            if len(v) > 6:
                redacted.append(v[:3] + "*" * (len(v) - 6) + v[-3:])
            else:
                redacted.append("***")
        return redacted

    # ------------------------------------------------------------------
    # Retention policy
    # ------------------------------------------------------------------

    def check_retention(self, df: pd.DataFrame) -> list[RetentionViolation]:
        """Evaluate date columns against configured retention policies."""
        violations: list[RetentionViolation] = []
        now = pd.Timestamp.now(tz="UTC")

        for col, max_days in self.retention_policies.items():
            if col not in df.columns:
                continue
            dates = pd.to_datetime(df[col], errors="coerce", utc=True)
            valid = dates.dropna()
            if valid.empty:
                continue
            age_days = (now - valid).dt.days
            exceeding = (age_days > max_days).sum()
            oldest = int(age_days.max()) if len(age_days) else 0
            if exceeding > 0:
                violations.append(
                    RetentionViolation(
                        column=col,
                        policy_name=f"retention_{col}",
                        max_age_days=max_days,
                        oldest_record_days=oldest,
                        records_exceeding=int(exceeding),
                        description=(
                            f"{exceeding} records in '{col}' exceed the "
                            f"{max_days}-day retention limit"
                        ),
                    )
                )
        return violations

    # ------------------------------------------------------------------
    # Full compliance check
    # ------------------------------------------------------------------

    def run_compliance_check(
        self, df: pd.DataFrame, dataset_name: str = "unnamed"
    ) -> ComplianceReport:
        """
        Run PII detection and retention checks, then compile a report.
        """
        pii_findings = self.detect_pii(df)
        retention_violations = self.check_retention(df)

        recommendations = self._generate_recommendations(
            pii_findings, retention_violations
        )

        has_critical = any(f.risk_level == RiskLevel.CRITICAL.value for f in pii_findings)
        has_high = any(f.risk_level == RiskLevel.HIGH.value for f in pii_findings)
        has_retention = len(retention_violations) > 0

        if has_critical:
            risk = RiskLevel.CRITICAL.value
        elif has_high or has_retention:
            risk = RiskLevel.HIGH.value
        elif pii_findings:
            risk = RiskLevel.MEDIUM.value
        else:
            risk = RiskLevel.LOW.value

        compliant = not has_critical and not has_retention

        summary = {
            "total_columns_scanned": len(df.columns),
            "pii_columns_found": len({f.column for f in pii_findings}),
            "pii_types_found": list({f.pii_type for f in pii_findings}),
            "retention_violations_count": len(retention_violations),
        }

        report = ComplianceReport(
            dataset_name=dataset_name,
            framework=self.framework.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_compliant=compliant,
            risk_level=risk,
            pii_findings=[f.to_dict() for f in pii_findings],
            retention_violations=[v.to_dict() for v in retention_violations],
            recommendations=recommendations,
            summary=summary,
        )
        logger.info(
            "Compliance check for '%s' (%s): compliant=%s, risk=%s",
            dataset_name,
            self.framework.value,
            compliant,
            risk,
        )
        return report

    # ------------------------------------------------------------------
    # Recommendation generator
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        pii_findings: list[PIIFinding],
        retention_violations: list[RetentionViolation],
    ) -> list[str]:
        recs: list[str] = []

        pii_types_found = {f.pii_type for f in pii_findings}

        if PIIType.EMAIL.value in pii_types_found:
            recs.append(
                "Encrypt or hash email addresses at rest. "
                "Implement access controls for columns containing emails."
            )
        if PIIType.CPF.value in pii_types_found:
            recs.append(
                "CPF data requires encryption and strict access logging per LGPD Art. 46. "
                "Consider tokenization for analytics workloads."
            )
        if PIIType.CREDIT_CARD.value in pii_types_found:
            recs.append(
                "Credit card numbers must comply with PCI-DSS. "
                "Remove or tokenize card data immediately."
            )
        if PIIType.SSN.value in pii_types_found:
            recs.append(
                "Social Security Numbers are high-risk PII. "
                "Implement encryption, masking, and audit logging."
            )
        if PIIType.PHONE_BR.value in pii_types_found or PIIType.PHONE_INTL.value in pii_types_found:
            recs.append(
                "Phone numbers should be masked in non-production environments."
            )
        if PIIType.NAME.value in pii_types_found:
            recs.append(
                "Personal names may constitute PII under GDPR/LGPD. "
                "Ensure lawful basis for processing."
            )
        if PIIType.IP_ADDRESS.value in pii_types_found:
            recs.append(
                "IP addresses are considered personal data under GDPR. "
                "Anonymize where possible."
            )

        for violation in retention_violations:
            recs.append(
                f"Purge {violation.records_exceeding} records in "
                f"'{violation.column}' exceeding the {violation.max_age_days}-day "
                f"retention limit."
            )

        if self.framework == ComplianceFramework.LGPD and pii_findings:
            recs.append(
                "Under LGPD, maintain a Record of Processing Activities (ROPA) "
                "documenting the legal basis for each PII category."
            )
        elif self.framework == ComplianceFramework.GDPR and pii_findings:
            recs.append(
                "Under GDPR Art. 30, maintain records of processing activities. "
                "Conduct a Data Protection Impact Assessment (DPIA) if needed."
            )

        if not recs:
            recs.append("No compliance issues detected. Continue monitoring.")

        return recs

    def __repr__(self) -> str:
        return (
            f"ComplianceEngine(framework={self.framework.value}, "
            f"retention_policies={len(self.retention_policies)})"
        )

"""Tests for src.governance.compliance — PII detection and compliance checks."""

import pandas as pd
import pytest

from src.governance.compliance import (
    ComplianceEngine,
    ComplianceFramework,
    PIIType,
)


@pytest.fixture()
def pii_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_name": ["Alice Silva", "Bob Santos"],
            "email": ["alice@example.com", "bob@test.org"],
            "cpf": ["123.456.789-01", "987.654.321-00"],
            "phone": ["(11) 91234-5678", "(21) 98765-4321"],
            "notes": ["Regular customer", "VIP customer"],
        }
    )


class TestPIIDetection:

    def test_detects_email_by_regex(self, pii_df: pd.DataFrame):
        engine = ComplianceEngine()
        findings = engine.detect_pii(pii_df)
        email_findings = [f for f in findings if f.pii_type == PIIType.EMAIL.value]
        assert len(email_findings) > 0

    def test_detects_cpf_by_regex(self, pii_df: pd.DataFrame):
        engine = ComplianceEngine()
        findings = engine.detect_pii(pii_df)
        cpf_findings = [f for f in findings if f.pii_type == PIIType.CPF.value]
        assert len(cpf_findings) > 0

    def test_detects_phone_by_column_hint(self, pii_df: pd.DataFrame):
        engine = ComplianceEngine()
        findings = engine.detect_pii(pii_df)
        phone_findings = [f for f in findings if f.pii_type in (PIIType.PHONE_BR.value, PIIType.PHONE_INTL.value)]
        assert len(phone_findings) > 0

    def test_no_pii_in_clean_data(self):
        df = pd.DataFrame({"value": [1, 2, 3], "category": ["a", "b", "c"]})
        engine = ComplianceEngine()
        findings = engine.detect_pii(df)
        assert len(findings) == 0


class TestComplianceReport:

    def test_report_marks_non_compliant_with_critical_pii(self, pii_df: pd.DataFrame):
        engine = ComplianceEngine(framework=ComplianceFramework.LGPD)
        report = engine.run_compliance_check(pii_df, "pii_test")
        assert report.risk_level in ("critical", "high")

    def test_clean_data_is_compliant(self):
        df = pd.DataFrame({"metric": [1.0, 2.0], "label": ["x", "y"]})
        engine = ComplianceEngine(framework=ComplianceFramework.GDPR)
        report = engine.run_compliance_check(df, "clean_test")
        assert report.overall_compliant is True
        assert report.risk_level == "low"

    def test_recommendations_generated(self, pii_df: pd.DataFrame):
        engine = ComplianceEngine(framework=ComplianceFramework.LGPD)
        report = engine.run_compliance_check(pii_df, "recs_test")
        assert len(report.recommendations) > 0

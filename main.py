"""
ML Data Lineage & Governance Platform — End-to-End Demo

Demonstrates the full pipeline:
    1. Register datasets in the metadata catalog
    2. Track data transformations (raw -> clean -> features -> model)
    3. Validate data quality
    4. Profile data statistics
    5. Run compliance checks (PII detection, LGPD)
    6. Display lineage graph and impact analysis
    7. Simulate schema evolution and compatibility checks

Author: Gabriel Demetrios Lafis
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.governance.catalog import AssetType, MetadataCatalog
from src.governance.compliance import ComplianceEngine, ComplianceFramework
from src.lineage.graph import NodeType
from src.lineage.tracker import LineageTracker
from src.quality.profiler import DataProfiler
from src.quality.validator import DataQualityValidator, RuleSeverity
from src.schema.evolution import CompatibilityMode, SchemaEvolutionTracker


# ======================================================================
# Helpers
# ======================================================================

SEPARATOR = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def pretty(data: dict | list, indent: int = 2) -> None:
    print(json.dumps(data, indent=indent, default=str, ensure_ascii=False))


# ======================================================================
# 1. Generate synthetic data
# ======================================================================

def create_raw_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "name": [f"Customer {i}" for i in range(1, n + 1)],
            "email": [
                f"customer{i}@example.com" if i % 20 != 0 else None
                for i in range(1, n + 1)
            ],
            "cpf": [
                f"{np.random.randint(100,999)}.{np.random.randint(100,999)}."
                f"{np.random.randint(100,999)}-{np.random.randint(10,99)}"
                for _ in range(n)
            ],
            "phone": [
                f"(11) 9{np.random.randint(1000,9999)}-{np.random.randint(1000,9999)}"
                for _ in range(n)
            ],
            "age": np.random.randint(18, 80, size=n).tolist(),
            "income": np.round(np.random.lognormal(10, 1, size=n), 2).tolist(),
            "signup_date": [
                (now - timedelta(days=int(d))).isoformat()
                for d in np.random.randint(30, 1200, size=n)
            ],
            "region": np.random.choice(
                ["Southeast", "South", "Northeast", "North", "Central-West"],
                size=n,
            ).tolist(),
            "credit_score": np.random.randint(300, 900, size=n).tolist(),
        }
    )


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["email"] = df["email"].fillna("unknown@placeholder.com")
    df["age"] = df["age"].clip(18, 100)
    df["income"] = df["income"].clip(lower=0)
    return df


def engineer_features(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean[["customer_id", "age", "income", "credit_score", "region"]].copy()
    df["income_log"] = np.log1p(df["income"])
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "66+"],
    )
    df["credit_risk"] = np.where(df["credit_score"] < 500, "high", "low")
    return df


# ======================================================================
# Main demo
# ======================================================================

def main() -> None:
    print(textwrap.dedent("""\
        ╔══════════════════════════════════════════════════════════════════╗
        ║   ML Data Lineage & Governance Platform — End-to-End Demo      ║
        ╚══════════════════════════════════════════════════════════════════╝
    """))

    # ------------------------------------------------------------------
    # 1. Metadata Catalog
    # ------------------------------------------------------------------
    section("1. Metadata Catalog — Registering Assets")

    catalog = MetadataCatalog()

    catalog.register_dataset(
        asset_id="ds_raw_customers",
        name="Raw Customer Data",
        description="Unprocessed customer records ingested from CRM system",
        owner="data-engineering",
        tags=["raw", "customers", "crm"],
        schema={
            "customer_id": "int64",
            "name": "object",
            "email": "object",
            "cpf": "object",
            "phone": "object",
            "age": "int64",
            "income": "float64",
            "signup_date": "object",
            "region": "object",
            "credit_score": "int64",
        },
    )

    catalog.register_dataset(
        asset_id="ds_clean_customers",
        name="Cleaned Customer Data",
        description="Customer data after null imputation and outlier clipping",
        owner="data-engineering",
        tags=["clean", "customers"],
    )

    catalog.register_feature(
        asset_id="ft_customer_features",
        name="Customer Feature Set",
        description="Engineered features for credit risk modelling",
        owner="ml-engineering",
        tags=["features", "credit_risk", "ml"],
    )

    catalog.register_model(
        asset_id="mdl_credit_risk",
        name="Credit Risk Classifier",
        description="Gradient-boosted model for credit-risk prediction",
        owner="ml-engineering",
        tags=["model", "credit_risk", "gbm"],
        algorithm="GradientBoosting",
        framework="scikit-learn",
    )

    print("Catalog summary:")
    pretty(catalog.summary())

    print("\nSearch results for tag='customers':")
    for entry in catalog.search(tags=["customers"]):
        print(f"  - {entry}")

    # ------------------------------------------------------------------
    # 2. Lineage Tracking
    # ------------------------------------------------------------------
    section("2. Lineage Tracking — Building the DAG")

    tracker = LineageTracker()

    tracker.track_input("ds_raw_customers", "Raw Customer Data", node_type=NodeType.DATASET)

    tracker.track_transformation(
        source_ids=["ds_raw_customers"],
        target_id="ds_clean_customers",
        target_name="Cleaned Customer Data",
        transformation_name="clean_and_impute",
        target_type=NodeType.DATASET,
        parameters={"fill_email": "unknown@placeholder.com", "clip_age": [18, 100]},
        user="data-engineering",
        description="Null imputation and outlier clipping",
    )

    tracker.track_transformation(
        source_ids=["ds_clean_customers"],
        target_id="ft_customer_features",
        target_name="Customer Feature Set",
        transformation_name="feature_engineering",
        target_type=NodeType.FEATURE,
        parameters={"features": ["income_log", "age_group", "credit_risk"]},
        user="ml-engineering",
        description="Log-transform income, bin age, categorise credit risk",
    )

    tracker.track_model(
        model_id="mdl_credit_risk",
        name="Credit Risk Classifier",
        feature_ids=["ft_customer_features"],
        metadata={"algorithm": "GradientBoosting", "n_estimators": 200},
        user="ml-engineering",
    )

    tracker.track_transformation(
        source_ids=["mdl_credit_risk"],
        target_id="pred_credit_scores",
        target_name="Credit Score Predictions",
        transformation_name="model_inference",
        target_type=NodeType.PREDICTION,
        user="ml-engineering",
        description="Batch inference producing credit risk scores",
    )

    print("Lineage summary:")
    pretty(tracker.summary())

    print("\nLineage DOT graph:\n")
    print(tracker.export_dot("CreditRiskLineage"))

    # ------------------------------------------------------------------
    # 3. Data Quality Validation
    # ------------------------------------------------------------------
    section("3. Data Quality Validation")

    raw_df = create_raw_data()
    clean_df = clean_data(raw_df)

    validator = DataQualityValidator(dataset_name="raw_customers")
    validator.add_not_null_rule("customer_id")
    validator.add_not_null_rule("email", severity=RuleSeverity.WARNING)
    validator.add_unique_rule("customer_id")
    validator.add_range_check("age", min_value=0, max_value=120)
    validator.add_range_check("income", min_value=0)
    validator.add_range_check("credit_score", min_value=300, max_value=850)
    validator.add_regex_rule(
        "email",
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
        severity=RuleSeverity.WARNING,
    )
    validator.add_referential_integrity_rule(
        "region",
        ["Southeast", "South", "Northeast", "North", "Central-West"],
    )

    quality_report = validator.validate(raw_df)
    print("Quality report (raw data):")
    pretty(quality_report.to_dict())

    # ------------------------------------------------------------------
    # 4. Data Profiling
    # ------------------------------------------------------------------
    section("4. Data Profiling")

    profiler = DataProfiler()
    profile = profiler.profile(raw_df, dataset_name="raw_customers")
    print(f"Profiled {profile['row_count']} rows x {profile['column_count']} columns")
    print(f"Memory usage: {profile['memory_usage_bytes']:,} bytes")
    print(f"Duplicate rows: {profile['duplicate_row_count']}")
    print("\nColumn highlights:")
    for col_profile in profile["columns"][:4]:
        print(
            f"  {col_profile['name']:20s} | type={col_profile['dtype']:10s} "
            f"| completeness={col_profile['completeness']:.2%} "
            f"| unique={col_profile['unique_count']}"
        )

    # ------------------------------------------------------------------
    # 5. Compliance Checks (LGPD)
    # ------------------------------------------------------------------
    section("5. Compliance Checks — LGPD PII Detection")

    compliance_engine = ComplianceEngine(
        framework=ComplianceFramework.LGPD,
        retention_policies={"signup_date": 365},
    )
    compliance_report = compliance_engine.run_compliance_check(
        raw_df, dataset_name="raw_customers"
    )
    print("Compliance report:")
    pretty(compliance_report.to_dict())

    # ------------------------------------------------------------------
    # 6. Impact Analysis
    # ------------------------------------------------------------------
    section("6. Impact Analysis — What if 'Raw Customer Data' changes?")

    impact = tracker.impact_analysis("ds_raw_customers")
    print(f"Source node: {impact['source_node']}")
    print(f"Total affected nodes: {impact['total_affected']}")
    print(f"Affected by type: {impact['affected_by_type']}")
    print("Affected paths:")
    for path in impact["paths"]:
        print(f"  {' -> '.join(path)}")

    # ------------------------------------------------------------------
    # 7. Schema Evolution
    # ------------------------------------------------------------------
    section("7. Schema Evolution — Simulating Column Changes")

    schema_tracker = SchemaEvolutionTracker(
        dataset_id="ds_raw_customers",
        compatibility_mode=CompatibilityMode.BACKWARD,
    )

    v1_schema = {
        "customer_id": "int64",
        "name": "object",
        "email": "object",
        "cpf": "object",
        "phone": "object",
        "age": "int64",
        "income": "float64",
        "signup_date": "object",
        "region": "object",
        "credit_score": "int64",
    }
    schema_tracker.register_schema(v1_schema, "Initial schema from CRM import")

    v2_schema = {
        "customer_id": "int64",
        "full_name": "object",       # renamed from 'name'
        "email": "object",
        "cpf": "object",
        "phone": "object",
        "age": "int64",
        "income": "float64",
        "signup_date": "object",
        "region": "object",
        "credit_score": "int64",
        "loyalty_tier": "object",     # new column
    }
    schema_tracker.register_schema(v2_schema, "Renamed 'name' to 'full_name', added 'loyalty_tier'")

    print("Evolution timeline:")
    pretty(schema_tracker.get_evolution_timeline())

    compat = schema_tracker.check_compatibility(v2_schema, CompatibilityMode.BACKWARD)
    print(f"\nBackward compatibility: {compat.compatible}")
    print(f"Summary: {compat.summary}")
    if compat.breaking_changes:
        print("Breaking changes:")
        for ch in compat.breaking_changes:
            print(f"  - {ch['description']}")

    print("\nSuggested migrations:")
    for suggestion in schema_tracker.suggest_migrations(v2_schema):
        print(f"  {suggestion}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("Demo Complete")

    print(textwrap.dedent(f"""\
        Assets in catalog   : {len(catalog)}
        Lineage nodes       : {len(tracker.graph)}
        Quality score       : {quality_report.quality_score:.1f}%
        PII columns found   : {compliance_report.summary['pii_columns_found']}
        Compliance status   : {'COMPLIANT' if compliance_report.overall_compliant else 'NON-COMPLIANT'}
        Schema versions     : {schema_tracker.current_version}
    """))


if __name__ == "__main__":
    main()

"""Unit test suite for Stage 1 Polars data preparation and feature engineering."""

from datetime import date

import polars as pl
import pytest

from src.config import (
    ALL_24_PRODUCTS,
    NON_FEATURE_COLS,
    SELECTED_15_TARGETS,
    TARGET_TO_INDEX,
)
from src.stages.stage1_polars_features import (
    cap_categories_for_tabdpt,
    clean_pipeline,
    engineer_tabdpt_features,
    extract_test_targets,
)


@pytest.fixture
def sample_raw_data() -> pl.DataFrame:
    """Creates a synthetic raw DataFrame mimicking Santander train_ver2 schema."""
    data = {
        "snapshot_date": ["2016-01-28", "2016-02-28", "2016-03-28", "2016-01-28"],
        "customer_id": ["1001", "1001", "1001", "1002"],
        "employee_index": ["N", "NA", "N", None],
        "residence_country": ["ES", "ES", "ES", "FR"],
        "gender": ["H", "V", "NA", "V"],
        "age": [" 35", "35 ", "NA", "28"],
        "join_date": [
            "2015-01-15",
            "2015-01-15",
            "2015-01-15",
            None,
        ],  # row 4 has null join_date
        "is_new_customer": ["0", "0", "0", "1"],
        "seniority_months": ["12", "-5", "14", "1"],
        "customer_relation_primary": ["1", "1", "1", "1"],
        "last_date_primary_customer": ["NA", "NA", "NA", "NA"],
        "customer_type_month_start": ["1.0", "1", "NA", "1.0"],
        "relation_type_month_start": ["A", "I", "NA", "A"],
        "residence_status": ["S", "S", "N", "S"],
        "is_foreigner": ["N", "N", "S", "S"],
        "spouse_employee_index": ["NA", "N", "S", "N"],
        "entry_channel": ["KAT", "KAT", "KFC", "OTHER_CH"],
        "is_deceased": ["N", "N", "N", "N"],
        "address_type": ["1", "1", "1", "1"],
        "province_code": ["28", "28", "28", "08"],
        "province_name": ["MADRID", "MADRID", "MADRID", "BARCELONA"],
        "is_active": ["1", "1", "0", "1"],
        "gross_household_income": ["100000.0", "NA", "120000.0", "90000.0"],
        "customer_segment": [
            "02 - PARTICULARES",
            "NA",
            "02 - PARTICULARES",
            "01 - TOP",
        ],
    }
    # Initialize all 24 products with default 0 strings
    for p in ALL_24_PRODUCTS:
        data[p] = ["0", "0", "0", "0"]

    # Customer 1001 acquires current_account at t1, adds payroll_account at t2
    data["current_account"] = ["1", "1", "1", "0"]
    data["payroll_account"] = ["0", "1", "1", "0"]

    return pl.DataFrame(data)


def test_clean_pipeline(sample_raw_data: pl.DataFrame):
    """Validates type conversions, missing imputations, and join_date filtering."""
    df_clean = clean_pipeline(sample_raw_data)

    # Row with null join_date should be filtered out (4 rows -> 3 rows)
    assert df_clean.height == 3
    assert "1002" not in df_clean["customer_id"].to_list()

    # Dates parsed
    assert df_clean["snapshot_date"].dtype == pl.Date
    assert df_clean["join_date"].dtype == pl.Date

    # Demographics
    assert df_clean["gender"].to_list() == ["F", "M", "Unknown"]
    assert df_clean["age"].to_list() == [35, 35, None]
    assert df_clean["seniority_months"].to_list() == [12, 0, 14]
    assert df_clean["customer_type_month_start"].to_list() == ["1", "1", "UNKNOWN"]
    assert df_clean["relation_type_month_start"].to_list() == ["A", "I", "I"]

    # Binary flags
    assert df_clean["residence_status"].to_list() == [1, 1, 0]
    assert df_clean["is_foreigner"].to_list() == [0, 0, 1]

    # Income imputation: NA in row 1 for province 28 should be imputed via median
    row1_income = df_clean.filter(pl.col("snapshot_date") == date(2016, 2, 28))[
        "gross_household_income"
    ].item()
    assert row1_income == 110000.0  # Median of 100000 and 120000

    # Products cast to Int8
    for p in ALL_24_PRODUCTS:
        assert df_clean[p].dtype == pl.Int8


def test_extract_test_targets():
    """Validates ground truth target extraction by comparing April and May 2016."""
    apr_data = {
        "customer_id": ["c1", "c2"],
        "snapshot_date": [date(2016, 4, 28), date(2016, 4, 28)],
        "current_account": [1, 0],
        "credit_card": [0, 0],
    }
    may_data = {
        "customer_id": ["c1", "c2"],
        "snapshot_date": [date(2016, 5, 28), date(2016, 5, 28)],
        "current_account": [1, 1],  # c1 retained, c2 acquired (added!)
        "credit_card": [1, 0],  # c1 acquired (added!), c2 none
    }

    # Populate remaining products with 0
    for p in SELECTED_15_TARGETS:
        if p not in apr_data:
            apr_data[p] = [0, 0]
            may_data[p] = [0, 0]

    df_apr = pl.DataFrame(apr_data)
    df_may = pl.DataFrame(may_data)

    df_targets = extract_test_targets(df_may, df_apr)

    assert df_targets.height == 2
    c1 = df_targets.filter(pl.col("customer_id") == "c1").to_dicts()[0]
    c2 = df_targets.filter(pl.col("customer_id") == "c2").to_dicts()[0]

    # c1 added credit_card (TARGET_TO_INDEX["credit_card"])
    assert c1["target_class"] == TARGET_TO_INDEX["credit_card"]
    assert c1["added_products"] == ["credit_card"]
    assert c1["added_credit_card"] == 1
    assert c1["added_current_account"] == 0

    # c2 added current_account (TARGET_TO_INDEX["current_account"])
    assert c2["target_class"] == TARGET_TO_INDEX["current_account"]
    assert c2["added_products"] == ["current_account"]
    assert c2["added_current_account"] == 1


def test_cap_categories():
    """Validates top categorical capping."""
    df = pl.DataFrame(
        {
            "entry_channel": ["KAT", "KFC", "RARE_CHANNEL"],
            "residence_country": ["ES", "FR", "RARE_COUNTRY"],
            "province_code": ["28", None, "08"],
        }
    )
    capped = cap_categories_for_tabdpt(
        df, top_channels=["KAT", "KFC"], top_countries=["ES", "FR"]
    )
    assert capped["entry_channel"].to_list() == ["KAT", "KFC", "OTHER"]
    assert capped["residence_country"].to_list() == ["ES", "FR", "OTHER"]
    assert capped["province_code"].to_list() == ["28", "UNKNOWN", "08"]


def test_engineer_tabdpt_features():
    """Validates feature engineering logic, lags, bundles, and 16-class target."""
    rows = []
    # Build 3 months for customer c1
    for m in range(1, 4):
        row = {
            "customer_id": "c1",
            "snapshot_date": date(2016, m, 28),
            "join_date": date(2015, 1, 15),
            "last_date_primary_customer": None,
            "address_type": "1",
            "province_name": "MADRID",
            "province_code": "28",
            "gross_household_income": 100000.0,
            "age": 35,
            "seniority_months": 12 + m,
            "is_active": 1,
            "employee_index": "N",
            "residence_country": "ES",
            "gender": "F",
            "is_new_customer": 0,
            "customer_relation_primary": 1,
            "customer_type_month_start": "1",
            "relation_type_month_start": "A",
            "residence_status": 1,
            "is_foreigner": 0,
            "spouse_employee_index": 0,
            "entry_channel": "KAT",
            "is_deceased": 0,
            "customer_segment": "PARTICULARES",
        }
        for p in ALL_24_PRODUCTS:
            row[p] = 0
        if m >= 1:
            row["current_account"] = 1
        if m >= 2:
            row["payroll_account"] = 1
        if m == 3:
            row["credit_card"] = 1
        rows.append(row)

    df = pl.DataFrame(rows)
    df_features = engineer_tabdpt_features(df, explode_multi_targets=True)

    # First month m=1 is dropped because lag1 doesn't exist; m=2 and m=3 remain
    assert df_features.height == 2

    # Verify lag 1 holdings at m=3: holds current_account, payroll_account
    m3_feat = df_features.filter(
        pl.col("snapshot_date") == date(2016, 3, 28)
    ).to_dicts()[0]
    assert m3_feat["has_current_account_lag1"] == 1
    assert m3_feat["has_payroll_account_lag1"] == 1
    assert m3_feat["has_credit_card_lag1"] == 0

    # m=3 added credit_card
    assert m3_feat["target_class"] == TARGET_TO_INDEX["credit_card"]

    # Verify bundle and calendar features exist
    assert "payroll_bundle_lag1" in m3_feat
    assert "investment_bundle_lag1" in m3_feat
    assert "sin_month" in m3_feat
    assert "cos_month" in m3_feat
    assert "months_as_customer" in m3_feat

    # Total feature columns check
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    assert len(feature_cols) <= 100

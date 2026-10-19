"""
Customer Feature Extraction and Transformation Pipeline for TabDPT.

Provides pure functions to extract customer history from Parquet,
engineer 80 lag/delta/bundle/seasonal features, apply categorical encoding
and normalization, and zero-pad to 128 dimensions.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from src.config import (
    ALL_24_PRODUCTS,
    CATEGORICAL_MAPPINGS_JSON,
    CLEAN_TEST_PARQUET,
    CLEAN_TRAIN_PARQUET,
    CORE_7_PRODUCTS,
    CORE_10_PRODUCTS,
    NORMALIZATION_STATS_JSON,
    SELECTED_15_TARGETS,
)


def get_customer(id: str, df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Retrieve test evaluation snapshot for a customer."""
    if isinstance(df, pl.LazyFrame):
        return df.filter(pl.col("customer_id") == str(id)).collect()
    return df.filter(pl.col("customer_id") == str(id))


def get_customer_history(
    id: str, history: pl.LazyFrame | pl.DataFrame, n_history: int | None = None
) -> pl.DataFrame:
    """
    Retrieves historical snapshots for a customer from CLEAN_TRAIN_PARQUET.
    If n_history is specified, returns the most recent n_history snapshots.
    By default, returns all available history for the customer.
    """
    q = history.filter(pl.col("customer_id") == str(id)).sort("snapshot_date")
    if n_history is not None:
        q = q.tail(n_history)
    if isinstance(q, pl.LazyFrame):
        return q.collect()
    return q


def feature_engineering_pipeline(customer: pl.DataFrame) -> pl.DataFrame:
    """
    Engineers the exact extra predictive features matching Stage 1:
    - Lag 1 holdings for all 24 products
    - Lag 2 holdings for core 10 products
    - Acquisition/churn velocity deltas for core 7 products
    - Portfolio aggregations (total_products_lag1, total_products_lag2, delta_total_products, has_any_product_lag1)
    - Product bundle usage indices (payroll, investment, credit, savings bundles)
    - Payroll/pensions reactivation indicators (payroll_eligible, pensions_eligible)
    - Customer activity dynamics (is_active_lag1, is_active_lag2, activity_delta)
    - Socioeconomic ratios (log_income, income_to_province_ratio, months_as_customer, age_at_join)
    - Seasonality & calendar drivers (snap_month, sin_month, cos_month, 4 seasonal flags)
    """
    if customer.is_empty():
        return customer

    # 1. Sort chronologically per customer to compute lags
    df = customer.sort(["customer_id", "snapshot_date"])

    # 2. Holdings at lag 1 for all 24 Santander products
    lag1_exprs = [
        pl.col(c)
        .shift(1)
        .over("customer_id")
        .fill_null(0)
        .cast(pl.Int8)
        .alias(f"has_{c}_lag1")
        for c in ALL_24_PRODUCTS
        if c in df.columns
    ]
    df = df.with_columns(lag1_exprs)

    # 3. Holdings at lag 2 for 10 core products & Velocity deltas for 7 core products
    lag2_exprs = [
        pl.col(c)
        .shift(2)
        .over("customer_id")
        .fill_null(0)
        .cast(pl.Int8)
        .alias(f"has_{c}_lag2")
        for c in CORE_10_PRODUCTS
        if c in df.columns
    ]
    df = df.with_columns(lag2_exprs)

    delta_exprs = [
        (pl.col(f"has_{c}_lag1") - pl.col(f"has_{c}_lag2"))
        .cast(pl.Int32)
        .alias(f"delta_{c}")
        for c in CORE_7_PRODUCTS
    ]
    df = df.with_columns(delta_exprs)

    # 4. Portfolio aggregations & composite bundle indices
    all_24_lag1_sum = pl.sum_horizontal(
        [f"has_{c}_lag1" for c in ALL_24_PRODUCTS]
    ).cast(pl.Int32)
    core_10_lag2_sum = pl.sum_horizontal(
        [f"has_{c}_lag2" for c in CORE_10_PRODUCTS]
    ).cast(pl.Int32)

    df = df.with_columns(
        [
            all_24_lag1_sum.alias("total_products_lag1"),
            core_10_lag2_sum.alias("total_products_lag2"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("total_products_lag1") - pl.col("total_products_lag2"))
            .cast(pl.Int32)
            .alias("delta_total_products"),
            pl.when(pl.col("total_products_lag1") > 0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("has_any_product_lag1"),
            (
                (
                    pl.col("has_payroll_lag1")
                    + pl.col("has_payroll_account_lag1")
                    + pl.col("has_direct_debit_lag1")
                )
                / 3.0
            ).alias("payroll_bundle_lag1"),
            (
                (
                    pl.col("has_funds_lag1")
                    + pl.col("has_securities_lag1")
                    + pl.col("has_pensions_plan_lag1")
                )
                / 3.0
            ).alias("investment_bundle_lag1"),
            (
                (
                    pl.col("has_credit_card_lag1")
                    + pl.col("has_loans_lag1")
                    + pl.col("has_mortgage_lag1")
                )
                / 3.0
            ).alias("credit_bundle_lag1"),
            (
                (
                    pl.col("has_particular_account_lag1")
                    + pl.col("has_particular_plus_account_lag1")
                    + pl.col("has_more_particular_account_lag1")
                )
                / 3.0
            ).alias("savings_bundle_lag1"),
        ]
    )

    # 5. Payroll & Pensions Reactivation Signals
    df = df.with_columns(
        [
            pl.when(
                (pl.col("has_payroll_account_lag1") == 1)
                & (pl.col("has_payroll_lag1") == 0)
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("payroll_eligible"),
            pl.when(
                (pl.col("has_payroll_account_lag1") == 1)
                & (pl.col("has_pensions_lag1") == 0)
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("pensions_eligible"),
        ]
    )

    # 6. Customer activity dynamics
    df = df.with_columns(
        [
            pl.col("is_active")
            .shift(1)
            .over("customer_id")
            .fill_null(0)
            .cast(pl.Int32)
            .alias("is_active_lag1"),
            pl.col("is_active")
            .shift(2)
            .over("customer_id")
            .fill_null(0)
            .cast(pl.Int32)
            .alias("is_active_lag2"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("is_active_lag1") - pl.col("is_active_lag2"))
            .cast(pl.Int32)
            .alias("activity_delta")
        ]
    )

    # 7. Socioeconomic Context & Ratios
    prov_med_expr = pl.col("gross_household_income").median().over("province_code")
    months_between_expr = (
        (pl.col("snapshot_date").dt.year() - pl.col("join_date").dt.year()) * 12
        + (pl.col("snapshot_date").dt.month() - pl.col("join_date").dt.month())
        + (pl.col("snapshot_date").dt.day() - pl.col("join_date").dt.day()) / 31.0
    ).round(1)

    df = df.with_columns(
        [
            pl.col("gross_household_income").log1p().alias("log_income"),
            pl.when(prov_med_expr > 0)
            .then(pl.col("gross_household_income") / prov_med_expr)
            .otherwise(1.0)
            .alias("income_to_province_ratio"),
            months_between_expr.alias("months_as_customer"),
            (pl.col("age") - (pl.col("seniority_months") / 12.0))
            .round(1)
            .alias("age_at_join"),
        ]
    )

    # 8. Seasonality & Calendar Drivers
    snap_month = pl.col("snapshot_date").dt.month().cast(pl.Int32)
    df = df.with_columns(
        [
            snap_month.alias("snap_month"),
            (2 * math.pi * snap_month / 12.0).sin().round(4).alias("sin_month"),
            (2 * math.pi * snap_month / 12.0).cos().round(4).alias("cos_month"),
            pl.when(snap_month.is_in([4, 5, 6]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_tax_season"),
            pl.when(snap_month.is_in([11, 12]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_pension_season"),
            pl.when(snap_month.is_in([9, 10]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_academic_season"),
            pl.when(snap_month.is_in([6, 7]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_summer_season"),
        ]
    )

    # 9. Retain latest snapshot
    df = df.filter(
        pl.col("snapshot_date") == pl.col("snapshot_date").max().over("customer_id")
    )

    # 10. Drop raw products at time t to prevent leakage, plus non-feature metadata
    drop_cols = [c for c in ALL_24_PRODUCTS if c in df.columns]
    drop_cols.extend(
        [
            c
            for c in [
                "join_date",
                "last_date_primary_customer",
                "address_type",
                "province_name",
                "is_active",
                "gross_household_income",
            ]
            if c in df.columns
        ]
    )
    df = df.drop(drop_cols)

    return df


def encode_and_normalize(customer: pl.DataFrame) -> pl.DataFrame:
    """Applies frequency-based categorical encodings and global z-score scaling."""
    with open(CATEGORICAL_MAPPINGS_JSON) as f:
        mappings = dict(json.load(f))
    code = []
    for key, value in mappings.items():
        if key in customer.columns:
            code.append(
                pl.col(key).cast(pl.String).replace(value).cast(pl.Int32).alias(key)
            )
    customer = customer.with_columns(code)

    with open(NORMALIZATION_STATS_JSON) as f:
        stats = json.load(f)
    normalize = [
        ((pl.col(c) - stats["mean"]) / stats["std"]).cast(pl.Float32)
        for c in customer.columns
        if c not in ["snapshot_date", "customer_id"]
    ]
    customer = customer.with_columns(normalize)

    return customer.with_columns(
        pl.all().exclude("snapshot_date", "customer_id").clip(-10, 10)
    )


def get_already_held_mask(customer: pl.DataFrame) -> list[str]:
    """Returns list of products currently held by the customer at lag 1."""
    held = []
    for c in SELECTED_15_TARGETS:
        col = f"has_{c}_lag1"
        if col in customer.columns:
            val = customer.select(pl.col(col))[0, 0]
            if val and val > 0:
                held.append(c)
    return held


def prepare_query_vector(query_array: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Pads feature columns from native dimension (e.g. 80) up to 128 dimensions with zeros.
    Returns array of shape [1, 1, 128], dtype float32.
    """
    if isinstance(query_array, torch.Tensor):
        query_np = query_array.detach().cpu().numpy()
    else:
        query_np = np.asarray(query_array)

    if query_np.ndim == 1:
        query_np = query_np.reshape(1, -1)

    rows, cols = query_np.shape
    if cols < 128:
        pads = 128 - cols
        padding = np.zeros((rows, pads), dtype=np.float32)
        query_padded = np.hstack([query_np.astype(np.float32), padding])
    else:
        query_padded = query_np[:, :128].astype(np.float32)

    return query_padded.reshape(1, 1, 128)


def extract_customer_features_polars(
    customer_id: str,
    test_lf: pl.LazyFrame | None = None,
    train_lf: pl.LazyFrame | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """
    Complete Polars extraction and normalization pipeline for a single customer.
    Used for offline sync and online cache-miss fallback.
    """
    if test_lf is None:
        test_lf = pl.scan_parquet(str(CLEAN_TEST_PARQUET))
    if train_lf is None:
        train_lf = pl.scan_parquet(str(CLEAN_TRAIN_PARQUET))

    c_info = get_customer(customer_id, test_lf)
    c_hist = get_customer_history(customer_id, train_lf)

    if c_info.is_empty() and c_hist.is_empty():
        return np.zeros((1, 1, 128), dtype=np.float32), [], {"customer_id": str(customer_id), "unknown": True}

    customer = pl.concat([c_info, c_hist], how="vertical")
    engineered = feature_engineering_pipeline(customer)
    held = get_already_held_mask(engineered)
    processed = encode_and_normalize(engineered)

    feature_df = processed.drop("customer_id", "snapshot_date")
    raw_vector = feature_df.to_numpy()
    x_qry = prepare_query_vector(raw_vector)

    metadata: dict[str, Any] = {"customer_id": str(customer_id)}
    if not c_info.is_empty():
        row = c_info.to_dicts()[0]
        for k in [
            "age",
            "gender",
            "province_name",
            "customer_segment",
            "gross_household_income",
            "seniority_months",
        ]:
            if k in row:
                metadata[k] = row[k]

    return x_qry, held, metadata

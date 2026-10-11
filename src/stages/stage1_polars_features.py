"""
Stage 1: Polars Data Cleaning & TabDPT Feature Engineering Pipeline.

This module processes raw Santander Product Recommendation CSVs into cleaned,
feature-engineered Parquet datasets using high-performance, native Polars expressions.

Key Responsibilities:
1. Fast CSV ingestion, data sanitization, type casting, and missing value imputation.
2. Apply temporal split: Training (< 2016-05-28) and Test Evaluation (2016-05-28).
3. Compute and persist province-partitioned median incomes.
4. Cap high-cardinality categorical variables (<= 100 categories) without future leakage.
5. Engineer exactly 80 high-signal lag, velocity, bundle, and seasonal features.
6. Construct 16-class target (0 = do nothing, 1..15 = product additions).
7. Save Parquet datasets to data/processed/.
"""

import json
import math
import os

import polars as pl

from src.config import (
    ALL_24_PRODUCTS,
    COLUMN_MAPPING,
    CORE_7_PRODUCTS,
    CORE_10_PRODUCTS,
    NON_FEATURE_COLS,
    OTHER_9_PRODUCTS,
    PROVINCE_MEDIANS_JSON,
    RAW_TRAIN_CSV,
    SELECTED_15_TARGETS,
    TARGET_TO_INDEX,
    WRITE_CLEAN_TEST_PARQUET,
    WRITE_CLEAN_TRAIN_PARQUET,
    WRITE_FEATURES_PARQUET,
    WRITE_TEST_TARGETS_PARQUET,
    ensure_directories_exist,
)


def load_raw_data(train_csv_path: str | None = None) -> pl.LazyFrame:
    """
    Scans the raw Santander train CSV and renames columns from Spanish to English.

    Args:
        train_csv_path: Path to train_ver2.csv. Defaults to config RAW_TRAIN_CSV.

    Returns:
        Polars LazyFrame with standardized column names and null string handling.
    """
    csv_path = str(train_csv_path or RAW_TRAIN_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw train CSV not found at: {csv_path}")

    print(f"[Stage 1] Scanning raw CSV from: {csv_path}")

    # Read all columns as Utf8 initially or infer schema cleanly
    schema_overrides = {col: pl.String for col in COLUMN_MAPPING}
    lf = pl.scan_csv(
        csv_path,
        has_header=True,
        schema_overrides=schema_overrides,
        null_values=["NA", "nan", "NULL", ""],
    ).rename(COLUMN_MAPPING)

    return lf


def clean_pipeline(df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """
    Performs initial data sanitization, type casting, and missing value imputation.

    Transformations:
    - Standardize missing string representations and strip whitespace.
    - Parse dates (snapshot_date, join_date, last_date_primary_customer).
    - Standardize gender codes ('H' -> 'F', 'V' -> 'M', fallback 'Unknown').
    - Standardize customer status, relation, segment, and employee codes.
    - Convert 'S'/'N' character indicators to binary 1/0 int8.
    - Impute gross_household_income via province-level median, falling back to global median.
    - Cast all 24 product columns to binary Int8 (0 or 1).
    - Filter out rows missing join_date.
    """
    print("[Stage 1] Cleaning column types and imputing missing values...")

    if isinstance(df, pl.LazyFrame):
        df_collected = df.collect()
    else:
        df_collected = df

    # 1. Clean string columns: strip whitespace and treat 'NA', '', 'nan' as null
    string_cols = [
        col for col, dtype in df_collected.schema.items() if dtype == pl.String
    ]
    strip_exprs = [
        pl.when(
            pl.col(c).str.strip_chars().is_in(["NA", "", "nan", "NULL"])
            | pl.col(c).is_null()
        )
        .then(None)
        .otherwise(pl.col(c).str.strip_chars())
        .alias(c)
        for c in string_cols
    ]
    df_collected = df_collected.with_columns(strip_exprs)

    # 2. Date parsing
    date_exprs = [
        pl.col("snapshot_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("join_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("last_date_primary_customer").str.to_date("%Y-%m-%d", strict=False),
    ]

    # 3. Demographics and account metadata
    demographic_exprs = [
        pl.col("customer_id").cast(pl.String),
        pl.when(pl.col("gender") == "H")
        .then(pl.lit("F"))
        .when(pl.col("gender") == "V")
        .then(pl.lit("M"))
        .otherwise(pl.lit("Unknown"))
        .alias("gender"),
        pl.col("age").cast(pl.Int32, strict=False),
        pl.when(pl.col("seniority_months").cast(pl.Int32, strict=False) < 0)
        .then(0)
        .otherwise(pl.col("seniority_months").cast(pl.Int32, strict=False).fill_null(0))
        .alias("seniority_months"),
        pl.col("customer_type_month_start")
        .str.replace(r"\.0$", "")
        .fill_null("UNKNOWN")
        .alias("customer_type_month_start"),
        pl.col("relation_type_month_start")
        .fill_null("I")
        .alias("relation_type_month_start"),
        pl.col("customer_segment").fill_null("UNKNOWN").alias("customer_segment"),
        pl.col("employee_index").fill_null("N").alias("employee_index"),
    ]

    # 4. Spanish S/N character flags to binary 1/0
    binary_sn_cols = [
        "residence_status",
        "is_foreigner",
        "is_deceased",
        "spouse_employee_index",
    ]
    sn_exprs = [
        pl.when(pl.col(c).str.to_uppercase() == "S")
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias(c)
        for c in binary_sn_cols
    ]

    # 5. Integer flags with default 0 fallback
    int_flag_cols = ["is_new_customer", "customer_relation_primary", "is_active"]
    int_flag_exprs = [
        pl.col(c).cast(pl.Int32, strict=False).fill_null(0).alias(c)
        for c in int_flag_cols
    ]

    # 6. Clean 24 financial products to Int8
    product_exprs = [
        pl.col(target)
        .cast(pl.Int32, strict=False)
        .fill_null(0)
        .cast(pl.Int8)
        .alias(target)
        for target in ALL_24_PRODUCTS
    ]

    df_collected = df_collected.with_columns(
        date_exprs + demographic_exprs + sn_exprs + int_flag_exprs + product_exprs
    )

    # 7. Household income casting & province-partitioned median imputation
    df_collected = df_collected.with_columns(
        pl.col("gross_household_income")
        .cast(pl.Float64, strict=False)
        .alias("gross_household_income")
    )

    province_median = pl.col("gross_household_income").median().over("province_code")
    global_median = pl.col("gross_household_income").median()

    # Calculate and optionally persist province median lookup table for offline serving
    try:
        prov_table = (
            df_collected.filter(
                pl.col("province_code").is_not_null()
                & pl.col("gross_household_income").is_not_null()
            )
            .group_by("province_code")
            .agg(pl.col("gross_household_income").median().alias("med_inc"))
        )
        prov_dict = {
            str(row["province_code"]): float(row["med_inc"])
            for row in prov_table.iter_rows(named=True)
            if row["med_inc"] is not None
        }
        with open(PROVINCE_MEDIANS_JSON, "w") as f:
            json.dump(prov_dict, f, indent=2)
    except (OSError, ValueError, TypeError) as e:
        print(f"[Stage 1] Warning saving province medians JSON: {e}")

    df_collected = df_collected.with_columns(
        pl.coalesce(
            [pl.col("gross_household_income"), province_median, global_median]
        ).alias("gross_household_income")
    )

    # 8. Filter out entries with invalid account start dates
    df_collected = df_collected.filter(pl.col("join_date").is_not_null())

    return df_collected


def extract_test_targets(
    df_test: pl.DataFrame, df_train_apr: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes ground-truth target variables for the May 2016 test evaluation set
    by comparing May 2016 holdings against April 2016 holdings (lag 1).

    Produces:
    - customer_id
    - snapshot_date
    - target_class: Primary 16-class label (0 = do_nothing, 1..15 = first added product)
    - target_classes: Array of class IDs of all added products
    - added_products: Array of names of all added products
    - added_{col}: Binary indicator (1/0) for each of the 15 target products
    """
    df_eval = pl.concat([df_test, df_train_apr]).sort(["customer_id", "snapshot_date"])

    target_diff_exprs = []
    for c in SELECTED_15_TARGETS:
        lag_expr = pl.col(c).shift(1).over("customer_id").fill_null(0)
        is_added = (pl.col(c) == 1) & (lag_expr == 0)

        target_diff_exprs.append(
            pl.when(is_added).then(1).otherwise(0).cast(pl.Int8).alias(f"added_{c}")
        )
        target_diff_exprs.append(
            pl.when(is_added)
            .then(TARGET_TO_INDEX[c])
            .otherwise(None)
            .cast(pl.Int32)
            .alias(f"added_class_{c}")
        )
        target_diff_exprs.append(
            pl.when(is_added)
            .then(pl.lit(c))
            .otherwise(None)
            .cast(pl.String)
            .alias(f"added_name_{c}")
        )

    df_eval = df_eval.with_columns(target_diff_exprs)

    # Filter to May 2016 evaluation month
    may_date = pl.lit("2016-05-28").str.to_date("%Y-%m-%d")
    df_test_targets = df_eval.filter(pl.col("snapshot_date") == may_date)

    added_classes = [f"added_class_{c}" for c in SELECTED_15_TARGETS]
    added_names = [f"added_name_{c}" for c in SELECTED_15_TARGETS]
    added_binary_cols = [f"added_{c}" for c in SELECTED_15_TARGETS]

    df_test_targets = df_test_targets.with_columns(
        [
            pl.concat_list(added_classes).list.drop_nulls().alias("target_classes"),
            pl.concat_list(added_names).list.drop_nulls().alias("added_products"),
        ]
    )

    df_test_targets = df_test_targets.with_columns(
        pl.when(pl.col("target_classes").list.len() == 0)
        .then(0)
        .otherwise(pl.col("target_classes").list.get(0))
        .cast(pl.Int32)
        .alias("target_class")
    )

    return df_test_targets.select(
        [
            "customer_id",
            "snapshot_date",
            "target_class",
            "target_classes",
            "added_products",
        ]
        + added_binary_cols
    )


def cap_categories_for_tabdpt(
    df: pl.DataFrame,
    top_channels: list[str] | None = None,
    top_countries: list[str] | None = None,
) -> pl.DataFrame:
    """
    Limits high-cardinality categorical features to <= 100 categories for TabDPT:
    - entry_channel: Top 80 most frequent channels + 'OTHER'
    - residence_country: Top 30 most frequent countries + 'OTHER'
    - province_code: Coalesced with 'UNKNOWN' (<= 53 unique Spanish province codes)
    """
    top_channels = top_channels or []
    top_countries = top_countries or []

    return df.with_columns(
        [
            pl.when(pl.col("entry_channel").is_in(top_channels))
            .then(pl.col("entry_channel"))
            .otherwise(pl.lit("OTHER"))
            .alias("entry_channel"),
            pl.when(pl.col("residence_country").is_in(top_countries))
            .then(pl.col("residence_country"))
            .otherwise(pl.lit("OTHER"))
            .alias("residence_country"),
            pl.col("province_code").fill_null("UNKNOWN").alias("province_code"),
        ]
    )


def engineer_tabdpt_features(
    df: pl.DataFrame, explode_multi_targets: bool = True
) -> pl.DataFrame:
    """
    Constructs the 16-class product recommendation target and 80 predictive features.

    Features engineered:
    1. Lag 1 holdings: has_{product}_lag1 for all 24 Santander products.
    2. Added product indicators and 16-class target construction:
       - Class 0: Do nothing (no new product acquired at time t).
       - Class 1..15: Product added at time t that customer did not hold at t-1.
    3. Lag 2 holdings and velocity: delta_{product} = lag1 - lag2 for core 7 products.
    4. Portfolio aggregations: total_products_lag1, total_products_lag2, delta_total_products.
    5. Lifestyle bundle indices: payroll_bundle, investment_bundle, credit_bundle, savings_bundle.
    6. Activity dynamics: is_active_lag1, is_active_lag2, activity_delta.
    7. Socioeconomic ratios: log_income, income_to_province_ratio, months_as_customer, age_at_join.
    8. Seasonality & calendar drivers: sin_month, cos_month, seasonal indicator dummies.
    """
    print(
        "[Stage 1] Engineering lag features, portfolio bundles, and 16-class target in Polars..."
    )

    # Ensure chronological order per customer for shifting window functions
    df = df.sort(["customer_id", "snapshot_date"])

    # 1. Target Construction & 15 Target Holdings at lag 1
    lag1_target_exprs = []
    added_cols = []
    for c in SELECTED_15_TARGETS:
        has_lag1 = (
            pl.col(c)
            .shift(1)
            .over("customer_id")
            .fill_null(0)
            .cast(pl.Int8)
            .alias(f"has_{c}_lag1")
        )
        lag1_target_exprs.append(has_lag1)

        is_added = (pl.col(c) == 1) & (
            pl.col(c).shift(1).over("customer_id").fill_null(0) == 0
        )
        added_col = (
            pl.when(is_added)
            .then(TARGET_TO_INDEX[c])
            .otherwise(None)
            .cast(pl.Int32)
            .alias(f"added_{c}")
        )
        lag1_target_exprs.append(added_col)
        added_cols.append(f"added_{c}")

    # 2. Holdings at lag 1 for the other 9 ecosystem products
    lag1_other_exprs = [
        pl.col(c)
        .shift(1)
        .over("customer_id")
        .fill_null(0)
        .cast(pl.Int8)
        .alias(f"has_{c}_lag1")
        for c in OTHER_9_PRODUCTS
    ]

    df = df.with_columns(lag1_target_exprs + lag1_other_exprs)

    # 3. 16-Class Target Construction (0 = do_nothing, 1..15 = product added)
    df = df.with_columns(
        pl.concat_list(added_cols).list.drop_nulls().alias("added_classes")
    )

    if explode_multi_targets:
        df = df.with_columns(
            pl.when(pl.col("added_classes").list.len() == 0)
            .then(pl.lit([0]))
            .otherwise(pl.col("added_classes"))
            .alias("target_class")
        )
        df = df.explode("target_class", empty_as_null=True).with_columns(
            pl.col("target_class").cast(pl.Int32)
        )
    else:
        df = df.with_columns(
            pl.when(pl.col("added_classes").list.len() == 0)
            .then(0)
            .otherwise(pl.col("added_classes").list.get(0))
            .cast(pl.Int32)
            .alias("target_class")
        )

    df = df.drop(["added_classes"] + added_cols)

    # 4. Core holdings at lag 2 and velocity deltas (lag1 - lag2)
    lag2_exprs = [
        pl.col(c)
        .shift(2)
        .over("customer_id")
        .fill_null(0)
        .cast(pl.Int8)
        .alias(f"has_{c}_lag2")
        for c in CORE_10_PRODUCTS
    ]
    df = df.with_columns(lag2_exprs)

    delta_exprs = [
        (pl.col(f"has_{c}_lag1") - pl.col(f"has_{c}_lag2"))
        .cast(pl.Int32)
        .alias(f"delta_{c}")
        for c in CORE_7_PRODUCTS
    ]
    df = df.with_columns(delta_exprs)

    # 5. Portfolio aggregations & composite bundle indices
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

    portfolio_exprs = [
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
        # Reactivation indicators
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
        # Customer activity dynamics
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
    df = df.with_columns(portfolio_exprs)
    df = df.with_columns(
        (pl.col("is_active_lag1") - pl.col("is_active_lag2"))
        .cast(pl.Int32)
        .alias("activity_delta")
    )

    # 6. Socioeconomic Context & Ratios
    prov_med_expr = pl.col("gross_household_income").median().over("province_code")
    months_between_expr = (
        (pl.col("snapshot_date").dt.year() - pl.col("join_date").dt.year()) * 12
        + (pl.col("snapshot_date").dt.month() - pl.col("join_date").dt.month())
        + (pl.col("snapshot_date").dt.day() - pl.col("join_date").dt.day()) / 31.0
    ).round(1)

    socio_exprs = [
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
    df = df.with_columns(socio_exprs)

    # 7. Seasonality & Calendar Drivers
    snap_month = pl.col("snapshot_date").dt.month().cast(pl.Int32)
    calendar_exprs = [
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
    df = df.with_columns(calendar_exprs)

    # 8. Filter out customer's first snapshot month where lag 1 features do not exist
    df = df.with_columns(
        pl.col("snapshot_date").shift(1).over("customer_id").alias("prev_snapshot_date")
    )
    df = df.filter(pl.col("prev_snapshot_date").is_not_null()).drop(
        "prev_snapshot_date"
    )

    # 9. Drop raw target products at time t to prevent target leakage
    df = df.drop(ALL_24_PRODUCTS)

    # 10. Drop non-feature identification columns
    df = df.drop(
        [
            "join_date",
            "last_date_primary_customer",
            "address_type",
            "province_name",
            "is_active",
            "gross_household_income",
        ]
    )

    return df


def run_stage1():
    """
    Executes Stage 1 of the pipeline end-to-end using Polars:
    - Loads raw train data or reuses existing cleaned Parquets.
    - Applies cleaning and temporal partition.
    - Capping of high-cardinality categoricals.
    - Extracts TabDPT lag features and 16-class targets.
    - Saves Parquet datasets to data/processed/.
    """
    ensure_directories_exist()

    clean_train_exists = WRITE_CLEAN_TRAIN_PARQUET.is_dir() and any(
        WRITE_CLEAN_TRAIN_PARQUET.glob("*.parquet")
    )

    if clean_train_exists:
        print(
            f"[Stage 1] Pre-cleaned training dataset found at: {WRITE_CLEAN_TRAIN_PARQUET}"
        )
        df_train = pl.read_parquet(str(WRITE_CLEAN_TRAIN_PARQUET / "*.parquet"))
    else:
        lf_raw = load_raw_data()
        df_clean = clean_pipeline(lf_raw)

        print(
            "[Stage 1] Splitting dataset: Training (< 2016-05-28) and Test Evaluation (2016-05-28)..."
        )
        may_cutoff = pl.lit("2016-05-28").str.to_date("%Y-%m-%d")
        df_test_may2016 = df_clean.filter(pl.col("snapshot_date") == may_cutoff)
        df_train = df_clean.filter(pl.col("snapshot_date") < may_cutoff)

        print(
            f"[Stage 1] Writing cleaned evaluation test set to: {WRITE_CLEAN_TEST_PARQUET}"
        )
        WRITE_CLEAN_TEST_PARQUET.mkdir(parents=True, exist_ok=True)
        df_test_may2016.write_parquet(
            str(WRITE_CLEAN_TEST_PARQUET / "test_clean.parquet"), compression="snappy"
        )

        print(f"[Stage 1] Writing cleaned train set to: {WRITE_CLEAN_TRAIN_PARQUET}")
        WRITE_CLEAN_TRAIN_PARQUET.mkdir(parents=True, exist_ok=True)
        df_train.write_parquet(
            str(WRITE_CLEAN_TRAIN_PARQUET / "train_clean.parquet"), compression="snappy"
        )

    # Ensure ground-truth test targets are computed and saved separately
    test_targets_exists = WRITE_TEST_TARGETS_PARQUET.is_dir() and any(
        WRITE_TEST_TARGETS_PARQUET.glob("*.parquet")
    )
    if not test_targets_exists:
        print(
            f"[Stage 1] Extracting and saving isolated test targets to: {WRITE_TEST_TARGETS_PARQUET}..."
        )
        if "df_test_may2016" not in locals():
            df_test_may2016 = pl.read_parquet(
                str(WRITE_CLEAN_TEST_PARQUET / "*.parquet")
            )
        apr_cutoff = pl.lit("2016-04-28").str.to_date("%Y-%m-%d")
        df_apr = df_train.filter(pl.col("snapshot_date") == apr_cutoff)
        df_test_targets = extract_test_targets(df_test_may2016, df_apr)

        WRITE_TEST_TARGETS_PARQUET.mkdir(parents=True, exist_ok=True)
        df_test_targets.write_parquet(
            str(WRITE_TEST_TARGETS_PARQUET / "test_targets.parquet"),
            compression="snappy",
        )

    print("[Stage 1] Computing top channels and countries from training partition...")
    top_channels = (
        df_train.filter(pl.col("entry_channel").is_not_null())
        .group_by("entry_channel")
        .len()
        .sort("len", descending=True)
        .head(80)["entry_channel"]
        .to_list()
    )

    top_countries = (
        df_train.filter(pl.col("residence_country").is_not_null())
        .group_by("residence_country")
        .len()
        .sort("len", descending=True)
        .head(30)["residence_country"]
        .to_list()
    )

    df_train_capped = cap_categories_for_tabdpt(df_train, top_channels, top_countries)
    df_features = engineer_tabdpt_features(df_train_capped, explode_multi_targets=True)

    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    print(
        f"[Stage 1] Total Feature Columns: {len(feature_cols)} (Limit <= 100: {len(feature_cols) <= 100})"
    )

    print(f"[Stage 1] Saving engineered features to: {WRITE_FEATURES_PARQUET}")
    WRITE_FEATURES_PARQUET.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(
        str(WRITE_FEATURES_PARQUET / "features.parquet"), compression="snappy"
    )
    print("[Stage 1] Completed successfully in Polars!")


if __name__ == "__main__":
    run_stage1()

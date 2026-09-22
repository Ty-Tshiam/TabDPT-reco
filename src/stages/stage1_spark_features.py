"""
Stage 1: PySpark Data Cleaning & TabDPT Feature Engineering Pipeline.

This module processes raw Santander Product Recommendation CSVs into cleaned,
feature-engineered Parquet datasets.

Key Responsibilities:
1. Initialize local SparkSession with optimized memory configurations.
2. Clean raw demographics, impute missing values, and cast column datatypes.
3. Apply temporal split: Training (< 2016-05-28) and Test Evaluation (2016-05-28).
4. Cap high-cardinality categorical variables (<= 100 categories) without future leakage.
5. Engineer exactly 80 high-signal lag, velocity, bundle, and seasonal features.
6. Construct 16-class target (0 = do nothing, 1..15 = product additions).
7. Save partitioned Parquet datasets to data/processed/.
"""

import math
import os
from pathlib import Path
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType, ByteType, StringType

from src.config import (
    RAW_TRAIN_CSV,
    CLEAN_TRAIN_PARQUET,
    CLEAN_TEST_PARQUET,
    TEST_TARGETS_PARQUET,
    FEATURES_PARQUET,
    COLUMN_MAPPING,
    SELECTED_15_TARGETS,
    TARGET_TO_INDEX,
    OTHER_9_PRODUCTS,
    ALL_24_PRODUCTS,
    CORE_10_PRODUCTS,
    CORE_7_PRODUCTS,
    NON_FEATURE_COLS,
    ensure_directories_exist
)


def create_spark_session(app_name: str = "SantanderTabDPTPipeline", driver_memory: str = "6g") -> SparkSession:
    """
    Builds and returns an optimized PySpark session for single-node execution.

    Args:
        app_name: Name of the Spark application.
        driver_memory: Memory allocated to the driver and executor.

    Returns:
        Active SparkSession instance.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", driver_memory)
        .config("spark.local.dir", "/tmp/spark-local")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def load_raw_data(spark: SparkSession, train_csv_path: str = None):
    """
    Reads the raw Santander train CSV and renames columns from Spanish to English.

    Args:
        spark: Active SparkSession.
        train_csv_path: Path to train_ver2.csv. Defaults to config RAW_TRAIN_CSV.

    Returns:
        DataFrame with standardized column names.
    """
    csv_path = str(train_csv_path or RAW_TRAIN_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw train CSV not found at: {csv_path}")

    print(f"[Stage 1] Loading raw CSV from: {csv_path}")
    return (
        spark.read.csv(csv_path, header=True, inferSchema=False)
        .withColumnsRenamed(COLUMN_MAPPING)
    )


def clean_pipeline(df):
    """
    Performs initial data sanitization, type casting, and missing value imputation.

    Transformations:
    - Standardize missing string representations ('NA', '', 'nan', 'NULL') to None.
    - Parse dates (snapshot_date, join_date, last_date_primary_customer).
    - Standardize gender codes ('H' -> 'F', 'V' -> 'M').
    - Standardize customer status, relation, segment, and employee codes.
    - Convert 'S'/'N' character indicators to binary 1/0 bytes.
    - Impute gross_household_income via province-level median, falling back to global median.
    - Cast all 24 product columns to binary ByteType (0 or 1).
    - Filter out rows missing join_date.
    """
    print("[Stage 1] Cleaning column types and imputing missing values...")

    # Standardize string representations of missing values
    for col_name, col_type in df.dtypes:
        if col_type == "string":
            df = df.withColumn(
                col_name,
                F.when(F.trim(F.col(col_name)).isin("NA", "", "nan", "NULL"), None)
                .otherwise(F.trim(F.col(col_name)))
            )

    # Clean demographics and account metadata
    df = (
        df
        .withColumn("customer_id", F.col("customer_id").cast(StringType()))
        .withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd"))
        .withColumn("join_date", F.to_date(F.col("join_date"), "yyyy-MM-dd"))
        .withColumn("last_date_primary_customer", F.to_date(F.col("last_date_primary_customer"), "yyyy-MM-dd"))
        .withColumn(
            "gender",
            F.when(F.col("gender") == "H", "F")
            .when(F.col("gender") == "V", "M")
            .otherwise("Unknown")
        )
        .withColumn("age", F.col("age").cast(IntegerType()))
        .withColumn("seniority_months", F.col("seniority_months").cast(IntegerType()))
        .withColumn(
            "seniority_months",
            F.when(F.col("seniority_months") < 0, 0)
            .otherwise(F.coalesce(F.col("seniority_months"), F.lit(0)))
        )
        .withColumn("customer_type_month_start", F.regexp_replace(F.col("customer_type_month_start"), "\\.0$", ""))
        .withColumn("customer_type_month_start", F.coalesce(F.col("customer_type_month_start"), F.lit("UNKNOWN")))
        .withColumn("relation_type_month_start", F.coalesce(F.col("relation_type_month_start"), F.lit("I")))
        .withColumn("customer_segment", F.coalesce(F.col("customer_segment"), F.lit("UNKNOWN")))
        .withColumn("employee_index", F.coalesce(F.col("employee_index"), F.lit("N")))
    )

    # Convert Spanish S/N character flags to 1/0 binary indicators
    binary_sn_cols = ["residence_status", "is_foreigner", "is_deceased", "spouse_employee_index"]
    for c in binary_sn_cols:
        df = df.withColumn(
            c,
            F.when(F.upper(F.col(c)) == "S", 1).otherwise(0).cast(ByteType())
        )

    # Integer flags with default 0 fallback
    int_flag_cols = ["is_new_customer", "customer_relation_primary", "is_active"]
    for c in int_flag_cols:
        df = df.withColumn(c, F.coalesce(F.col(c).cast(IntegerType()), F.lit(0)))

    # Household income casting & province-partitioned median imputation
    df = df.withColumn("gross_household_income", F.col("gross_household_income").cast(DoubleType()))
    province_window = Window.partitionBy("province_code")
    global_window = Window.partitionBy()

    df = (
        df
        .withColumn("province_median_income", F.median("gross_household_income").over(province_window))
        .withColumn("global_median_income", F.median("gross_household_income").over(global_window))
        .withColumn(
            "gross_household_income",
            F.coalesce(F.col("gross_household_income"), F.col("province_median_income"), F.col("global_median_income"))
        )
        .drop("province_median_income", "global_median_income")
    )

    # Clean 24 financial target products to binary ByteType (0 or 1)
    for target in ALL_24_PRODUCTS:
        df = df.withColumn(target, F.coalesce(F.col(target).cast(IntegerType()), F.lit(0)).cast(ByteType()))

    # Filter out entries with invalid account start dates
    df = df.filter(F.col("join_date").isNotNull())
    return df


def extract_test_targets(df_test, df_train_apr):
    """
    Computes and isolates ground-truth target variables for the May 2016 test evaluation set
    by comparing May 2016 holdings against April 2016 holdings (lag 1).

    Produces:
    - customer_id
    - snapshot_date
    - target_class: Primary 16-class label (0 = do_nothing, 1..15 = first added product)
    - added_products: Array of names of all added products
    - added_{col}: Binary indicator (1/0) for each of the 15 target products
    """
    df_eval = df_test.unionByName(df_train_apr)
    cust_window = Window.partitionBy("customer_id").orderBy("snapshot_date")

    for col_name in SELECTED_15_TARGETS:
        lag_col = F.coalesce(F.lag(col_name, 1).over(cust_window), F.lit(0))
        df_eval = df_eval.withColumn(
            f"added_{col_name}",
            F.when((F.col(col_name) == 1) & (lag_col == 0), 1).otherwise(0).cast(ByteType())
        )
        df_eval = df_eval.withColumn(
            f"added_class_{col_name}",
            F.when(F.col(f"added_{col_name}") == 1, F.lit(TARGET_TO_INDEX[col_name])).otherwise(None)
        )
        df_eval = df_eval.withColumn(
            f"added_name_{col_name}",
            F.when(F.col(f"added_{col_name}") == 1, F.lit(col_name)).otherwise(None)
        )

    df_test_targets = df_eval.filter(F.col("snapshot_date") == "2016-05-28")

    added_classes = [f"added_class_{c}" for c in SELECTED_15_TARGETS]
    added_names = [f"added_name_{c}" for c in SELECTED_15_TARGETS]
    added_binary_cols = [f"added_{c}" for c in SELECTED_15_TARGETS]

    return (
        df_test_targets
        .withColumn("target_classes", F.array_compact(F.array(*added_classes)))
        .withColumn("added_products", F.array_compact(F.array(*added_names)))
        .withColumn(
            "target_class",
            F.when(F.size("target_classes") == 0, 0).otherwise(F.col("target_classes")[0]).cast(IntegerType())
        )
        .select(["customer_id", "snapshot_date", "target_class", "target_classes", "added_products"] + added_binary_cols)
    )


def cap_categories_for_tabdpt(df, top_channels=None, top_countries=None):
    """
    Limits high-cardinality categorical features to <= 100 categories for TabDPT:
    - entry_channel: Top 80 most frequent channels + 'OTHER'
    - residence_country: Top 30 most frequent countries + 'OTHER'
    - province_code: Coalesced with 'UNKNOWN' (<= 53 unique Spanish province codes)
    """
    return (
        df
        .withColumn(
            "entry_channel",
            F.when(F.col("entry_channel").isin(top_channels), F.col("entry_channel")).otherwise("OTHER")
        )
        .withColumn(
            "residence_country",
            F.when(F.col("residence_country").isin(top_countries), F.col("residence_country")).otherwise("OTHER")
        )
        .withColumn(
            "province_code",
            F.coalesce(F.col("province_code"), F.lit("UNKNOWN"))
        )
    )


def engineer_tabdpt_features(df, explode_multi_targets: bool = True):
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
    print("[Stage 1] Engineering lag features, portfolio bundles, and 16-class target...")
    cust_window = Window.partitionBy("customer_id").orderBy("snapshot_date")

    # 1. Target Construction & 15 Target Holdings at lag 1
    for col_name in SELECTED_15_TARGETS:
        df = df.withColumn(f"has_{col_name}_lag1", F.coalesce(F.lag(col_name, 1).over(cust_window), F.lit(0)).cast(ByteType()))
        df = df.withColumn(
            f"added_{col_name}",
            F.when(
                (F.col(col_name) == 1) & (F.col(f"has_{col_name}_lag1") == 0),
                F.lit(TARGET_TO_INDEX[col_name])
            ).otherwise(None)
        )

    # 2. Holdings at lag 1 for the other 9 ecosystem products
    for col_name in OTHER_9_PRODUCTS:
        df = df.withColumn(f"has_{col_name}_lag1", F.coalesce(F.lag(col_name, 1).over(cust_window), F.lit(0)).cast(ByteType()))

    # 3. 16-Class Target Construction (0 = do_nothing, 1..15 = product added)
    added_cols = [f"added_{c}" for c in SELECTED_15_TARGETS]
    df = df.withColumn("added_classes", F.array_compact(F.array(*added_cols)))

    if explode_multi_targets:
        df = df.withColumn(
            "target_class",
            F.when(F.size("added_classes") == 0, F.array(F.lit(0))).otherwise(F.col("added_classes"))
        )
        df = df.withColumn("target_class", F.explode("target_class"))
        df = df.withColumn("target_class", F.col("target_class").cast(IntegerType()))
    else:
        df = df.withColumn(
            "target_class",
            F.when(F.size("added_classes") == 0, 0).otherwise(F.col("added_classes")[0]).cast(IntegerType())
        )

    df = df.drop("added_classes", *added_cols)

    # 4. Core holdings at lag 2 and velocity deltas (lag1 - lag2)
    for col_name in CORE_10_PRODUCTS:
        df = df.withColumn(f"has_{col_name}_lag2", F.coalesce(F.lag(col_name, 2).over(cust_window), F.lit(0)).cast(ByteType()))

    for col_name in CORE_7_PRODUCTS:
        df = df.withColumn(
            f"delta_{col_name}",
            (F.col(f"has_{col_name}_lag1") - F.col(f"has_{col_name}_lag2")).cast(IntegerType())
        )

    # 5. Portfolio aggregations & composite bundle indices
    all_24_lag1 = [F.col(f"has_{c}_lag1") for c in ALL_24_PRODUCTS]
    core_10_lag2 = [F.col(f"has_{c}_lag2") for c in CORE_10_PRODUCTS]

    df = df.withColumn("total_products_lag1", sum(all_24_lag1).cast(IntegerType()))
    df = df.withColumn("total_products_lag2", sum(core_10_lag2).cast(IntegerType()))
    df = df.withColumn("delta_total_products", (F.col("total_products_lag1") - F.col("total_products_lag2")).cast(IntegerType()))
    df = df.withColumn("has_any_product_lag1", F.when(F.col("total_products_lag1") > 0, 1).otherwise(0).cast(ByteType()))

    # Product bundle usage indices
    df = df.withColumn(
        "payroll_bundle_lag1",
        (F.col("has_payroll_lag1") + F.col("has_payroll_account_lag1") + F.col("has_direct_debit_lag1")) / 3.0
    )
    df = df.withColumn(
        "investment_bundle_lag1",
        (F.col("has_funds_lag1") + F.col("has_securities_lag1") + F.col("has_pensions_plan_lag1")) / 3.0
    )
    df = df.withColumn(
        "credit_bundle_lag1",
        (F.col("has_credit_card_lag1") + F.col("has_loans_lag1") + F.col("has_mortgage_lag1")) / 3.0
    )
    df = df.withColumn(
        "savings_bundle_lag1",
        (F.col("has_particular_account_lag1") + F.col("has_particular_plus_account_lag1") + F.col("has_more_particular_account_lag1")) / 3.0
    )

    # Payroll & pensions reactivation indicators (holds payroll account but received no payroll/pension in lag 1)
    df = df.withColumn(
        "payroll_eligible",
        F.when((F.col("has_payroll_account_lag1") == 1) & (F.col("has_payroll_lag1") == 0), 1).otherwise(0).cast(ByteType())
    )
    df = df.withColumn(
        "pensions_eligible",
        F.when((F.col("has_payroll_account_lag1") == 1) & (F.col("has_pensions_lag1") == 0), 1).otherwise(0).cast(ByteType())
    )

    # Customer activity dynamics (separates active purchasers from dormant customers)
    df = df.withColumn("is_active_lag1", F.coalesce(F.lag("is_active", 1).over(cust_window), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("is_active_lag2", F.coalesce(F.lag("is_active", 2).over(cust_window), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("activity_delta", (F.col("is_active_lag1") - F.col("is_active_lag2")).cast(IntegerType()))

    # 6. Socioeconomic Context & Ratios
    prov_window = Window.partitionBy("province_code")
    df = df.withColumn("prov_med_inc", F.median("gross_household_income").over(prov_window))
    df = (
        df
        .withColumn("log_income", F.log1p("gross_household_income"))
        .withColumn(
            "income_to_province_ratio",
            F.when(F.col("prov_med_inc") > 0, F.col("gross_household_income") / F.col("prov_med_inc")).otherwise(1.0)
        )
        .withColumn("months_as_customer", F.round(F.months_between("snapshot_date", "join_date"), 1))
        .withColumn("age_at_join", F.round(F.col("age") - (F.col("seniority_months") / 12.0), 1))
        .drop("prov_med_inc")
    )

    # 7. Seasonality & Calendar Drivers
    df = (
        df
        .withColumn("snap_month", F.month("snapshot_date").cast(IntegerType()))
        .withColumn("sin_month", F.round(F.sin(2 * math.pi * F.col("snap_month") / 12.0), 4))
        .withColumn("cos_month", F.round(F.cos(2 * math.pi * F.col("snap_month") / 12.0), 4))
        .withColumn("is_tax_season", F.when(F.col("snap_month").isin(4, 5, 6), 1).otherwise(0).cast(ByteType()))
        .withColumn("is_pension_season", F.when(F.col("snap_month").isin(11, 12), 1).otherwise(0).cast(ByteType()))
        .withColumn("is_academic_season", F.when(F.col("snap_month").isin(9, 10), 1).otherwise(0).cast(ByteType()))
        .withColumn("is_summer_season", F.when(F.col("snap_month").isin(6, 7), 1).otherwise(0).cast(ByteType()))
    )

    # Filter out customer's first snapshot month where lag 1 features do not exist
    df = df.withColumn("prev_snapshot_date", F.lag("snapshot_date", 1).over(cust_window))
    df = df.filter(F.col("prev_snapshot_date").isNotNull()).drop("prev_snapshot_date")

    # Drop raw target products at time t to prevent target leakage
    df = df.drop(*ALL_24_PRODUCTS)

    # Drop non-feature identification columns
    df = df.drop("join_date", "last_date_primary_customer", "address_type", "province_name", "is_active", "gross_household_income")

    return df


def run_stage1(spark: SparkSession = None):
    """
    Executes Stage 1 of the pipeline end-to-end:
    - Loads raw train data.
    - Applies cleaning and temporal partition.
    - Capping of high-cardinality categoricals.
    - Extracts TabDPT lag features and 16-class targets.
    - Saves Parquet partitions to data/processed/.
    """
    ensure_directories_exist()
    should_stop_spark = False

    if spark is None:
        spark = create_spark_session()
        should_stop_spark = True

    try:
        # Load or read cleaned Parquet if already generated
        if CLEAN_TRAIN_PARQUET.exists():
            print(f"[Stage 1] Pre-cleaned training dataset found at: {CLEAN_TRAIN_PARQUET}")
            df_train = spark.read.parquet(str(CLEAN_TRAIN_PARQUET))
        else:
            df_raw = load_raw_data(spark)
            df_clean = clean_pipeline(df_raw)

            print("[Stage 1] Splitting dataset: Training (< 2016-05-28) and Test Evaluation (2016-05-28)...")
            df_test_may2016 = df_clean.filter(F.col("snapshot_date") == "2016-05-28")
            df_train = df_clean.filter(F.col("snapshot_date") < "2016-05-28")

            print(f"[Stage 1] Writing cleaned evaluation test set to: {CLEAN_TEST_PARQUET}")
            df_test_may2016.write.parquet(str(CLEAN_TEST_PARQUET), mode="overwrite")

            print(f"[Stage 1] Writing cleaned train set to: {CLEAN_TRAIN_PARQUET}")
            df_train.write.parquet(str(CLEAN_TRAIN_PARQUET), mode="overwrite")

        # Ensure ground-truth test targets are computed and saved separately
        test_targets_dir = str(TEST_TARGETS_PARQUET.parent)
        if not (Path(test_targets_dir).exists() and any(Path(test_targets_dir).glob("*.parquet"))):
            print(f"[Stage 1] Extracting and saving isolated test targets to: {test_targets_dir}...")
            if "df_test_may2016" not in locals():
                df_test_may2016 = spark.read.parquet(str(CLEAN_TEST_PARQUET))
            df_apr = df_train.filter(F.col("snapshot_date") == "2016-04-28")
            df_test_targets = extract_test_targets(df_test_may2016, df_apr)
            df_test_targets.write.parquet(test_targets_dir, mode="overwrite")

        print("[Stage 1] Computing top channels and countries from training partition...")
        top_channels = [
            row["entry_channel"] for row in df_train.filter(F.col("entry_channel").isNotNull())
            .groupBy("entry_channel")
            .count()
            .orderBy(F.desc("count"))
            .limit(80)
            .collect()
        ]
        top_countries = [
            row["residence_country"] for row in df_train.filter(F.col("residence_country").isNotNull())
            .groupBy("residence_country")
            .count()
            .orderBy(F.desc("count"))
            .limit(30)
            .collect()
        ]

        df_train_capped = cap_categories_for_tabdpt(df_train, top_channels, top_countries)
        df_features = engineer_tabdpt_features(df_train_capped, explode_multi_targets=True)

        feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
        print(f"[Stage 1] Total Feature Columns: {len(feature_cols)} (Limit <= 100: {len(feature_cols) <= 100})")

        print(f"[Stage 1] Saving engineered features to: {FEATURES_PARQUET}")
        df_features.write.parquet(str(FEATURES_PARQUET), mode="overwrite")
        print("[Stage 1] Completed successfully!")

    finally:
        if should_stop_spark:
            spark.stop()


if __name__ == "__main__":
    run_stage1()

import os
import math
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType, ByteType, StringType

# ==============================================================================
# Step 1: Initialize Spark Session
# ==============================================================================
spark = SparkSession.builder \
    .appName("SantanderTabDPTPipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "32") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Resolve data paths relative to this script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "santander-product-recommendation", "train_ver2.csv")
TEST_PATH = os.path.join(BASE_DIR, "santander-product-recommendation", "test_ver2.csv")
CLEAN_PARQUET_PATH = os.path.join(BASE_DIR, "output", "clean_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ==============================================================================
# Step 2: Define Column Mappings & Target Specifications
# ==============================================================================
COLUMN_MAPPING = {
    # Customer Demographics & Account Metadata
    "fecha_dato": "snapshot_date",
    "ncodpers": "customer_id",
    "ind_empleado": "employee_index",
    "pais_residencia": "residence_country",
    "sexo": "gender",
    "age": "age",
    "fecha_alta": "join_date",
    "ind_nuevo": "is_new_customer",
    "antiguedad": "seniority_months",
    "indrel": "customer_relation_primary",
    "ult_fec_cli_1t": "last_date_primary_customer",
    "indrel_1mes": "customer_type_month_start",
    "tiprel_1mes": "relation_type_month_start",
    "indresi": "residence_status",
    "indext": "is_foreigner",
    "conyuemp": "spouse_employee_index",
    "canal_entrada": "entry_channel",
    "indfall": "is_deceased",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "is_active",
    "renta": "gross_household_income",
    "segmento": "customer_segment",
    # Financial Target Products
    "ind_ahor_fin_ult1": "saving_account",
    "ind_aval_fin_ult1": "guarantees",
    "ind_cco_fin_ult1": "current_account",
    "ind_cder_fin_ult1": "derivative_account",
    "ind_cno_fin_ult1": "payroll_account",
    "ind_ctju_fin_ult1": "junior_account",
    "ind_ctma_fin_ult1": "more_particular_account",
    "ind_ctop_fin_ult1": "particular_account",
    "ind_ctpp_fin_ult1": "particular_plus_account",
    "ind_deco_fin_ult1": "short_term_deposits",
    "ind_deme_fin_ult1": "medium_term_deposits",
    "ind_dela_fin_ult1": "long_term_deposits",
    "ind_ecue_fin_ult1": "e_account",
    "ind_fond_fin_ult1": "funds",
    "ind_hip_fin_ult1": "mortgage",
    "ind_plan_fin_ult1": "pensions_plan",
    "ind_pres_fin_ult1": "loans",
    "ind_reca_fin_ult1": "taxes",
    "ind_tjcr_fin_ult1": "credit_card",
    "ind_valo_fin_ult1": "securities",
    "ind_viv_fin_ult1": "home_account",
    "ind_nomina_ult1": "payroll",
    "ind_nom_pens_ult1": "pensions",
    "ind_recibo_ult1": "direct_debit",
}

# The 15 active products selected for TabDPT modeling (Classes 1..15, Class 0 is 'do_nothing')
SELECTED_15_TARGETS = [
    "current_account",          # 1
    "payroll_account",          # 2
    "credit_card",              # 3
    "pensions",                 # 4
    "mortgage",                 # 5
    "securities",               # 6
    "particular_plus_account",  # 7
    "particular_account",       # 8
    "funds",                    # 9
    "direct_debit",             # 10
    "payroll",                  # 11
    "more_particular_account",  # 12
    "e_account",                # 13
    "long_term_deposits",       # 14
    "taxes"                     # 15
]

TARGET_TO_INDEX = {col_name: idx + 1 for idx, col_name in enumerate(SELECTED_15_TARGETS)}

# The other 9 products in Santander (tracked for lag1 ecosystem holdings)
OTHER_9_PRODUCTS = [
    "saving_account",
    "guarantees",
    "derivative_account",
    "junior_account",
    "short_term_deposits",
    "medium_term_deposits",
    "pensions_plan",
    "loans",
    "home_account"
]

ALL_24_PRODUCTS = SELECTED_15_TARGETS + OTHER_9_PRODUCTS

CORE_10_PRODUCTS = [
    "current_account", "payroll_account", "credit_card", "direct_debit", "payroll",
    "pensions", "particular_account", "e_account", "long_term_deposits", "taxes"
]

CORE_7_PRODUCTS = [
    "current_account", "payroll_account", "credit_card", "direct_debit",
    "payroll", "particular_account", "e_account"
]

TARGET_COLS = ALL_24_PRODUCTS


def load_data(train_path: str):
    """
    Step 3: Load raw CSV and rename columns.
    """
    df_train = (
        spark.read.csv(train_path, header=True, inferSchema=False)
        .withColumnsRenamed(COLUMN_MAPPING)
    )
    return df_train


def clean_pipeline(df):
    """
    Step 4-8: Initial cleaning, imputation, and type standardization.
    """
    # Standardize missing string values
    for col_name, col_type in df.dtypes:
        if col_type == "string":
            df = df.withColumn(
                col_name,
                F.when(F.trim(F.col(col_name)).isin("NA", "", "nan", "NULL"), None)
                .otherwise(F.trim(F.col(col_name)))
            )

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

    # Binary flags (S/N to 1/0)
    binary_sn_cols = ["residence_status", "is_foreigner", "is_deceased", "spouse_employee_index"]
    for c in binary_sn_cols:
        df = df.withColumn(
            c,
            F.when(F.upper(F.col(c)) == "S", 1).otherwise(0).cast(ByteType())
        )

    int_flag_cols = ["is_new_customer", "customer_relation_primary", "is_active"]
    for c in int_flag_cols:
        df = df.withColumn(c, F.coalesce(F.col(c).cast(IntegerType()), F.lit(0)))

    # Household income casting & median imputation
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

    # Clean 24 target products
    for target in ALL_24_PRODUCTS:
        df = df.withColumn(target, F.coalesce(F.col(target).cast(IntegerType()), F.lit(0)).cast(ByteType()))

    df = df.filter(F.col("join_date").isNotNull())
    return df


def cap_categories_for_tabdpt(df, top_channels=None, top_countries=None):
    """
    Enforces maximum 100 categories per column:
    - entry_channel: Top 80 + 'OTHER'
    - residence_country: Top 30 + 'OTHER'
    - province_code: coalesced with 'UNKNOWN' (<= 53 categories)
    """

    df = (
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
    return df


def engineer_tabdpt_features(df, explode_multi_targets: bool = True):
    """
    Builds the 16-class target (0 = do_nothing, 1..15 = products) and
    engineers exactly 80 high-signal features (<= 100 features total),
    all with <= 100 categories.
    """
    cust_window = Window.partitionBy("customer_id").orderBy("snapshot_date")

    # --------------------------------------------------------------------------
    # 1. Target Construction & 15 Target Holdings at lag1 (Masking Features)
    # --------------------------------------------------------------------------
    for col_name in SELECTED_15_TARGETS:
        df = df.withColumn(f"has_{col_name}_lag1", F.coalesce(F.lag(col_name, 1).over(cust_window), F.lit(0)).cast(ByteType()))
        df = df.withColumn(
            f"added_{col_name}",
            F.when(
                (F.col(col_name) == 1) & (F.col(f"has_{col_name}_lag1") == 0),
                F.lit(TARGET_TO_INDEX[col_name])
            ).otherwise(None)
        )

    # --------------------------------------------------------------------------
    # 2. Holdings at lag1 for the Other 9 Ecosystem Products
    # --------------------------------------------------------------------------
    for col_name in OTHER_9_PRODUCTS:
        df = df.withColumn(f"has_{col_name}_lag1", F.coalesce(F.lag(col_name, 1).over(cust_window), F.lit(0)).cast(ByteType()))

    # --------------------------------------------------------------------------
    # 3. 16-Class Target Construction (0 = do_nothing, 1..15 = product added)
    # --------------------------------------------------------------------------
    added_cols = [f"added_{c}" for c in SELECTED_15_TARGETS]
    df = df.withColumn("added_classes", F.array_compact(F.array(*added_cols)))

    if explode_multi_targets:
        # Explode multiple additions into separate training rows
        df = df.withColumn(
            "target_class",
            F.when(F.size("added_classes") == 0, F.array(F.lit(0))).otherwise(F.col("added_classes"))
        )
        df = df.withColumn("target_class", F.explode("target_class"))
        df = df.withColumn("target_class", F.col("target_class").cast(IntegerType()))
    else:
        # Primary single class (first added or 0)
        df = df.withColumn(
            "target_class",
            F.when(F.size("added_classes") == 0, 0).otherwise(F.col("added_classes")[0]).cast(IntegerType())
        )

    df = df.drop("added_classes", *added_cols)

    # --------------------------------------------------------------------------
    # 4. Core Holdings at lag2 & Velocity (lag1 - lag2)
    # --------------------------------------------------------------------------
    for col_name in CORE_10_PRODUCTS:
        df = df.withColumn(f"has_{col_name}_lag2", F.coalesce(F.lag(col_name, 2).over(cust_window), F.lit(0)).cast(ByteType()))

    for col_name in CORE_7_PRODUCTS:
        df = df.withColumn(
            f"delta_{col_name}",
            (F.col(f"has_{col_name}_lag1") - F.col(f"has_{col_name}_lag2")).cast(IntegerType())
        )

    # --------------------------------------------------------------------------
    # 5. Portfolio Aggregations & Activity Dynamics
    # --------------------------------------------------------------------------
    all_24_lag1 = [F.col(f"has_{c}_lag1") for c in ALL_24_PRODUCTS]
    core_10_lag2 = [F.col(f"has_{c}_lag2") for c in CORE_10_PRODUCTS]

    df = df.withColumn("total_products_lag1", sum(all_24_lag1).cast(IntegerType()))
    df = df.withColumn("total_products_lag2", sum(core_10_lag2).cast(IntegerType()))
    df = df.withColumn("delta_total_products", (F.col("total_products_lag1") - F.col("total_products_lag2")).cast(IntegerType()))
    df = df.withColumn("has_any_product_lag1", F.when(F.col("total_products_lag1") > 0, 1).otherwise(0).cast(ByteType()))

    # Composite bundles (lifestyle indices)
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

    # Activity transitions (strong predictor for separating class 0 from 1..15)
    df = df.withColumn("is_active_lag1", F.coalesce(F.lag("is_active", 1).over(cust_window), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("is_active_lag2", F.coalesce(F.lag("is_active", 2).over(cust_window), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("activity_delta", (F.col("is_active_lag1") - F.col("is_active_lag2")).cast(IntegerType()))

    # --------------------------------------------------------------------------
    # 6. Socioeconomic Context & Ratios
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 7. Seasonality & Calendar Drivers
    # --------------------------------------------------------------------------
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

    # Filter out initial snapshot per customer where lag1 does not exist
    df = df.withColumn("prev_snapshot_date", F.lag("snapshot_date", 1).over(cust_window))
    df = df.filter(F.col("prev_snapshot_date").isNotNull()).drop("prev_snapshot_date")

    # Drop raw target products at time t to prevent label leakage
    df = df.drop(*ALL_24_PRODUCTS)

    # Drop redundant non-feature columns""
    df = df.drop("join_date", "last_date_primary_customer", "address_type", "province_name", "is_active", "gross_household_income")

    return df


def subsample_do_nothing(df_train, fraction: float = 0.15, seed: int = 42):
    """
    Optional subsampling for TabDPT training:
    Preserves 100% of product acquisition rows (target_class > 0)
    and downsamples 'do_nothing' rows (target_class == 0).
    """
    df_pos = df_train.filter(F.col("target_class") > 0)
    df_neg = df_train.filter(F.col("target_class") == 0).sample(fraction=fraction, seed=seed)
    return df_pos.unionByName(df_neg)


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
if __name__ == "__main__":
    # Check if cleaned parquet already exists to avoid 20+ min CSV parsing
    if os.path.exists(CLEAN_PARQUET_PATH):
        print(f"Loading pre-cleaned dataset from: {CLEAN_PARQUET_PATH}")
        df_clean = spark.read.parquet(CLEAN_PARQUET_PATH)
    else:
        print(f"Loading raw train CSV from: {TRAIN_PATH}")
        df_raw = load_data(TRAIN_PATH)
        print("Applying cleaning pipeline...")
        df_clean = clean_pipeline(df_raw)

    # Temporal split: May 2016 (2016-05-28) is the evaluation/test set
    print("Splitting dataset into Training (< 2016-05-28) and Test (2016-05-28)...")
    df_test_may2016 = df_clean.filter(F.col("snapshot_date") == "2016-05-28")
    df_train = df_clean.filter(F.col("snapshot_date") < "2016-05-28")

    print("Capping high cardinality categoricals to <= 100 categories...")
    # Derive capping rules strictly from data prior to May 2016 to prevent temporal leakage

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

    df_train = cap_categories_for_tabdpt(df_train, top_channels, top_countries)

    print("Extracting TabDPT features and 16-class targets...")
    df_features = engineer_tabdpt_features(df_train, explode_multi_targets=True)

    non_feature_cols = ["customer_id", "snapshot_date", "target_class"]
    feature_cols = [c for c in df_features.columns if c not in non_feature_cols]

    print(f"Total Columns in Output: {len(df_features.columns)}")
    print(f"Total Predictive Features: {len(feature_cols)} (Limit <= 100: {len(feature_cols) <= 100})")

    test_out_path = os.path.join(OUTPUT_DIR, "tabdpt_test_may2016")
    train_out_path = os.path.join(OUTPUT_DIR, "tabdpt_train")
    features_out_path = os.path.join(OUTPUT_DIR, "features")

    print(f"Saving May 2016 evaluation set to: {test_out_path}")
    df_test_may2016.write.parquet(test_out_path, mode="overwrite")

    print(f"Saving training set to: {train_out_path}")
    df_train.write.parquet(train_out_path, mode="overwrite")

    print(f"Saving transformed features set to: {features_out_path}")
    df_features.write.parquet(features_out_path, mode="overwrite")

    print("Pipeline completed successfully!")

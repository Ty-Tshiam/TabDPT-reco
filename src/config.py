"""
Global Configuration & Path Definitions for TabDPT-reco Pipeline.

Centralizes:
1. Dynamic directory resolutions (ensuring paths work uniformly regardless of cwd).
2. Raw, processed, metadata, and tensor file locations.
3. Santander column schema definitions and 16-class product targets.
"""

from pathlib import Path

# ==============================================================================
# Path Definitions
# ==============================================================================

# Project Root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base Data Directory
DATA_DIR = PROJECT_ROOT / "data"

# 1. Raw Datasets (Raw CSVs and compressed archive)
RAW_DATA_DIR = DATA_DIR / "raw" / "santander-product-recommendation"
RAW_ZIP_PATH = RAW_DATA_DIR / "santander-product-recommendation.zip"
RAW_TRAIN_CSV = RAW_DATA_DIR / "train_ver2.csv"
RAW_TEST_CSV = RAW_DATA_DIR / "test_ver2.csv"

# 2. Processed Parquet Datasets across pipeline stages
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WRITE_CLEAN_TRAIN_PARQUET = PROCESSED_DATA_DIR / "train_clean"
CLEAN_TRAIN_PARQUET = PROCESSED_DATA_DIR / "train_clean" / "*.parquet"
WRITE_CLEAN_TEST_PARQUET = PROCESSED_DATA_DIR / "test_may2016_clean"
CLEAN_TEST_PARQUET = PROCESSED_DATA_DIR / "test_may2016_clean" / "*.parquet"
WRITE_TEST_TARGETS_PARQUET = PROCESSED_DATA_DIR / "test_targets"
TEST_TARGETS_PARQUET = PROCESSED_DATA_DIR / "test_targets" / "*.parquet"
WRITE_FEATURES_PARQUET = PROCESSED_DATA_DIR / "features"
FEATURES_PARQUET = PROCESSED_DATA_DIR / "features" / "*.parquet"
NORMALIZED_DATASET_PARQUET = PROCESSED_DATA_DIR / "normalized_dataset.parquet"

# 3. Metadata & Serialization Artifacts (Mappings, Normalization Stats)
METADATA_DIR = DATA_DIR / "metadata"
CATEGORICAL_MAPPINGS_JSON = METADATA_DIR / "categorical_mappings.json"
NORMALIZATION_STATS_JSON = METADATA_DIR / "normalization_stats.json"
PROVINCE_MEDIANS_JSON = METADATA_DIR / "province_median_incomes.json"

# 4. Final Tensors for TabDPT Model Training & Evaluation
TENSORS_DIR = DATA_DIR / "tensors"
CONTEXT_TENSOR_PATH = TENSORS_DIR / "context.pt"
Y_TENSOR_PATH = TENSORS_DIR / "y.pt"


def ensure_directories_exist():
    """Ensure all standard data subdirectories exist."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, TENSORS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Schema Definitions & Column Mappings
# ==============================================================================

# Santander raw Spanish column names mapped to standardized English names
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
    "current_account",          # Class 1
    "payroll_account",          # Class 2
    "credit_card",              # Class 3
    "pensions",                 # Class 4
    "mortgage",                 # Class 5
    "securities",               # Class 6
    "particular_plus_account",  # Class 7
    "particular_account",       # Class 8
    "funds",                    # Class 9
    "direct_debit",             # Class 10
    "payroll",                  # Class 11
    "more_particular_account",  # Class 12
    "e_account",                # Class 13
    "long_term_deposits",       # Class 14
    "taxes"                     # Class 15
]

# Mapping from product name to target integer class (1..15)
TARGET_TO_INDEX = {col_name: idx + 1 for idx, col_name in enumerate(SELECTED_15_TARGETS)}

# The other 9 products in Santander ecosystem (tracked for lag1 ecosystem holdings)
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

# All 24 Santander financial products
ALL_24_PRODUCTS = SELECTED_15_TARGETS + OTHER_9_PRODUCTS

# Core 10 products for lag2 tracking and portfolio size
CORE_10_PRODUCTS = [
    "current_account", "payroll_account", "credit_card", "direct_debit", "payroll",
    "pensions", "particular_account", "e_account", "long_term_deposits", "taxes"
]

# Core 7 products for delta (lag1 - lag2) acquisition/churn velocity
CORE_7_PRODUCTS = [
    "current_account", "payroll_account", "credit_card", "direct_debit",
    "payroll", "particular_account", "e_account"
]

# Columns that are categorical even if stored as numbers in raw data
EXPLICIT_CATEGORICAL_COLS = {"province_code", "customer_relation_primary"}

# Non-predictive identification & label columns to exclude from feature matrices
NON_FEATURE_COLS = ["customer_id", "snapshot_date", "target_class"]

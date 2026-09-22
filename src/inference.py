import math
import json
import datetime
import torch
import polars as pl
from tabdpt import TabDPTClassifier
from sklearn.metrics import accuracy_score

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)

try:
    from src.config import (
        CLEAN_TRAIN_PARQUET,
        CLEAN_TEST_PARQUET,
        TEST_TARGETS_PARQUET,
        SELECTED_15_TARGETS,
        TARGET_TO_INDEX,
        OTHER_9_PRODUCTS,
        ALL_24_PRODUCTS,
        CORE_10_PRODUCTS,
        CORE_7_PRODUCTS,
        CATEGORICAL_MAPPINGS_JSON,
        NORMALIZATION_STATS_JSON,
        CONTEXT_TENSOR_PATH,
        Y_TENSOR_PATH
    )
except ImportError:
    from config import (
        CLEAN_TRAIN_PARQUET,
        CLEAN_TEST_PARQUET,
        TEST_TARGETS_PARQUET,
        SELECTED_15_TARGETS,
        TARGET_TO_INDEX,
        OTHER_9_PRODUCTS,
        ALL_24_PRODUCTS,
        CORE_10_PRODUCTS,
        CORE_7_PRODUCTS,
        CATEGORICAL_MAPPINGS_JSON,
        NORMALIZATION_STATS_JSON,
        CONTEXT_TENSOR_PATH,
        Y_TENSOR_PATH
    )

df = pl.scan_parquet(str(CLEAN_TEST_PARQUET))
history = pl.scan_parquet(str(CLEAN_TRAIN_PARQUET))
test_targets = pl.scan_parquet(str(TEST_TARGETS_PARQUET.parent / "*.parquet"))
dummy_data = {"customer_id": "1166753"}
date = datetime.date(2016, 5, 28)


def get_customer(id: str, df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Retrieve test evaluation snapshot for a customer."""
    if isinstance(df, pl.LazyFrame):
        return df.filter(pl.col("customer_id") == id).collect()
    return df.filter(pl.col("customer_id") == id)


def get_customer_history(id: str, history: pl.LazyFrame | pl.DataFrame, n_history: int | None = None) -> pl.DataFrame:
    """
    Retrieves historical snapshots for a customer from CLEAN_TRAIN_PARQUET.
    If n_history is specified, returns the most recent n_history snapshots.
    By default, returns all available history for the customer.
    """
    q = history.filter(pl.col("customer_id") == id).sort("snapshot_date")
    if n_history is not None:
        q = q.tail(n_history)
    if isinstance(q, pl.LazyFrame):
        return q.collect()
    return q


def get_customer_targets(id: str, targets: pl.LazyFrame | pl.DataFrame = test_targets) -> pl.DataFrame:
    """Retrieve precomputed ground-truth targets (all added classes) without calculating in real time."""
    if isinstance(targets, pl.LazyFrame):
        return targets.filter(pl.col("customer_id") == id).collect()
    return targets.filter(pl.col("customer_id") == id)


def feature_engineering_pipeline(customer: pl.DataFrame) -> pl.DataFrame:
    """
    Engineers the exact extra predictive features matching Stage 1 (stage1_spark_features.py):
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
        pl.col(c).shift(1).over("customer_id").fill_null(0).cast(pl.Int8).alias(f"has_{c}_lag1")
        for c in ALL_24_PRODUCTS
        if c in df.columns
    ]
    df = df.with_columns(lag1_exprs)

    # 3. Holdings at lag 2 for 10 core products & Velocity deltas for 7 core products
    lag2_exprs = [
        pl.col(c).shift(2).over("customer_id").fill_null(0).cast(pl.Int8).alias(f"has_{c}_lag2")
        for c in CORE_10_PRODUCTS
        if c in df.columns
    ]
    df = df.with_columns(lag2_exprs)

    delta_exprs = [
        (pl.col(f"has_{c}_lag1") - pl.col(f"has_{c}_lag2")).cast(pl.Int32).alias(f"delta_{c}")
        for c in CORE_7_PRODUCTS
    ]
    df = df.with_columns(delta_exprs)

    # 4. Portfolio aggregations & composite bundle indices
    all_24_lag1_sum = pl.sum_horizontal([f"has_{c}_lag1" for c in ALL_24_PRODUCTS]).cast(pl.Int32)
    core_10_lag2_sum = pl.sum_horizontal([f"has_{c}_lag2" for c in CORE_10_PRODUCTS]).cast(pl.Int32)

    df = df.with_columns([
        all_24_lag1_sum.alias("total_products_lag1"),
        core_10_lag2_sum.alias("total_products_lag2"),
    ])

    df = df.with_columns([
        (pl.col("total_products_lag1") - pl.col("total_products_lag2")).cast(pl.Int32).alias("delta_total_products"),
        pl.when(pl.col("total_products_lag1") > 0).then(1).otherwise(0).cast(pl.Int8).alias("has_any_product_lag1"),
        ((pl.col("has_payroll_lag1") + pl.col("has_payroll_account_lag1") + pl.col("has_direct_debit_lag1")) / 3.0).alias("payroll_bundle_lag1"),
        ((pl.col("has_funds_lag1") + pl.col("has_securities_lag1") + pl.col("has_pensions_plan_lag1")) / 3.0).alias("investment_bundle_lag1"),
        ((pl.col("has_credit_card_lag1") + pl.col("has_loans_lag1") + pl.col("has_mortgage_lag1")) / 3.0).alias("credit_bundle_lag1"),
        ((pl.col("has_particular_account_lag1") + pl.col("has_particular_plus_account_lag1") + pl.col("has_more_particular_account_lag1")) / 3.0).alias("savings_bundle_lag1"),
    ])

    # 5. Payroll & Pensions Reactivation Signals (holds payroll account but had no deposit at lag 1)
    df = df.with_columns([
        pl.when((pl.col("has_payroll_account_lag1") == 1) & (pl.col("has_payroll_lag1") == 0))
        .then(1).otherwise(0).cast(pl.Int8).alias("payroll_eligible"),
        pl.when((pl.col("has_payroll_account_lag1") == 1) & (pl.col("has_pensions_lag1") == 0))
        .then(1).otherwise(0).cast(pl.Int8).alias("pensions_eligible"),
    ])

    # 6. Customer activity dynamics
    df = df.with_columns([
        pl.col("is_active").shift(1).over("customer_id").fill_null(0).cast(pl.Int32).alias("is_active_lag1"),
        pl.col("is_active").shift(2).over("customer_id").fill_null(0).cast(pl.Int32).alias("is_active_lag2"),
    ])
    df = df.with_columns([
        (pl.col("is_active_lag1") - pl.col("is_active_lag2")).cast(pl.Int32).alias("activity_delta")
    ])

    # 7. Socioeconomic Context & Ratios
    prov_med_expr = pl.col("gross_household_income").median().over("province_code")
    months_between_expr = (
        (pl.col("snapshot_date").dt.year() - pl.col("join_date").dt.year()) * 12
        + (pl.col("snapshot_date").dt.month() - pl.col("join_date").dt.month())
        + (pl.col("snapshot_date").dt.day() - pl.col("join_date").dt.day()) / 31.0
    ).round(1)

    df = df.with_columns([
        pl.col("gross_household_income").log1p().alias("log_income"),
        pl.when(prov_med_expr > 0)
        .then(pl.col("gross_household_income") / prov_med_expr)
        .otherwise(1.0)
        .alias("income_to_province_ratio"),
        months_between_expr.alias("months_as_customer"),
        (pl.col("age") - (pl.col("seniority_months") / 12.0)).round(1).alias("age_at_join"),
    ])

    # 8. Seasonality & Calendar Drivers
    snap_month = pl.col("snapshot_date").dt.month().cast(pl.Int32)
    df = df.with_columns([
        snap_month.alias("snap_month"),
        (2 * math.pi * snap_month / 12.0).sin().round(4).alias("sin_month"),
        (2 * math.pi * snap_month / 12.0).cos().round(4).alias("cos_month"),
        pl.when(snap_month.is_in([4, 5, 6])).then(1).otherwise(0).cast(pl.Int8).alias("is_tax_season"),
        pl.when(snap_month.is_in([11, 12])).then(1).otherwise(0).cast(pl.Int8).alias("is_pension_season"),
        pl.when(snap_month.is_in([9, 10])).then(1).otherwise(0).cast(pl.Int8).alias("is_academic_season"),
        pl.when(snap_month.is_in([6, 7])).then(1).otherwise(0).cast(pl.Int8).alias("is_summer_season"),
    ])

    # 9. Retain latest snapshot for inference
    df = df.filter(pl.col("snapshot_date") == pl.col("snapshot_date").max().over("customer_id"))

    # 10. Drop raw products at time t to prevent leakage, plus non-feature metadata
    drop_cols = [c for c in ALL_24_PRODUCTS if c in df.columns]
    drop_cols.extend([
        c for c in ["join_date", "last_date_primary_customer", "address_type",
                    "province_name", "is_active", "gross_household_income"]
        if c in df.columns
    ])
    df = df.drop(drop_cols)

    return df


def encode_and_normalize (customer):
    with open(CATEGORICAL_MAPPINGS_JSON) as f:
        mappings = dict(json.load(f))
    code = []
    for key, value in mappings.items():
        code.append(
            pl.col(key)
            .cast(pl.String)
            .replace(value)
            .cast(pl.Int32)
            .alias(key)
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
        pl.all().exclude("snapshot_date", "customer_id")
        .clip(-10,10)
    )

def get_already_held_mask(customer):
    held = []
    for c in SELECTED_15_TARGETS:
        col = f"has_{c}_lag1"
        if customer.select(pl.col(col))[0,0]:
            held.append(c)
    return held


def filter_valid_recommendations(
    candidate_ranked_products: list[str],
    already_held: list[str],
    top_k: int = 7
) -> list[str]:
    """
    Filters candidate recommendations to strictly exclude products already held by the customer.
    """
    return [p for p in candidate_ranked_products if p not in already_held][:top_k]


def evaluate_customer_recommendations(
    recommended_products: list[str],
    ground_truth_targets: dict
) -> dict:
    """
    Evaluates top-K recommendations against ground-truth additions (taking account of every class added).
    Computes hits, precision, recall, and average precision (AP for MAP@7).
    """
    true_additions = ground_truth_targets.get("added_products", [])
    if not true_additions:
        return {
            "true_added_products": [],
            "hits": [],
            "precision": 0.0,
            "recall": 1.0,
            "average_precision": 1.0 if not recommended_products else 0.0,
        }

    hits = [p for p in recommended_products if p in true_additions]
    score = 0.0
    num_hits = 0
    for i, p in enumerate(recommended_products):
        if p in true_additions:
            num_hits += 1
            score += num_hits / (i + 1.0)
    ap = score / min(len(true_additions), len(recommended_products)) if true_additions else 0.0

    return {
        "true_added_products": true_additions,
        "true_target_classes": ground_truth_targets.get("target_classes", []),
        "recommended_products": recommended_products,
        "hits": hits,
        "precision": round(len(hits) / len(recommended_products), 4) if recommended_products else 0.0,
        "recall": round(len(hits) / len(true_additions), 4) if true_additions else 0.0,
        "average_precision": round(ap, 4),
    }

def prepare_pass_through_tensors(query):
    context = torch.load(CONTEXT_TENSOR_PATH)
    y = torch.load(Y_TENSOR_PATH)

    rows, cols = query.shape
    pads = 128 - cols
    padding = torch.zeros((rows, pads), dtype = torch.bfloat16)
    query = torch.hstack([query, padding])

    return context.unsqueeze(0), query.unsqueeze(0), y.unsqueeze(0)
    
    

# ==============================================================================
# Pipeline Execution & Demonstration for Customer 1166753
# ==============================================================================
if __name__ == "__main__":
    print(f"[Inference] Fetching data for customer: {dummy_data['customer_id']}...")
    customer_info = get_customer(dummy_data["customer_id"], df)

    # Pull complete available customer history
    customer_history = get_customer_history(dummy_data["customer_id"], history)
    print(f"[Inference] Pulled {customer_history.height} historical snapshots (Total with test: {customer_history.height + customer_info.height})")

    customer = pl.concat([customer_info, customer_history], how="vertical")

    engineered_customer = feature_engineering_pipeline(customer)
    feature_cols = [c for c in engineered_customer.columns if c not in ["customer_id", "snapshot_date"]]
    print(f"[Inference] Engineered {len(feature_cols)} features (Total columns: {engineered_customer.width})")

    held = get_already_held_mask(engineered_customer)
    
    processed_customer = encode_and_normalize(engineered_customer)
    query = processed_customer.drop("customer_id", "snapshot_date").to_torch().to(torch.bfloat16)

    context, query, y_train = prepare_pass_through_tensors(query)

    context = context.to(dtype = torch.float32).squeeze(0).numpy()
    query = query.to(dtype = torch.float32).squeeze(0).numpy()
    y_train = y_train.to(dtype = torch.float32).squeeze(0).numpy()
        
    model = TabDPTClassifier()
    model.fit(context, y_train)
    y_pred = model.predict(
        query,
        n_ensembles = 8,
        temperature = 1,
        context_size = None,
        permute_classes=True,
        seed = 42
    )
    print(y_pred)
    target_info = get_customer_targets(dummy_data["customer_id"]).to_dicts()[0]
    print(target_info)






'''

    # 2. Precomputed Multi-Class Ground Truth
    target_info = get_customer_targets(dummy_data["customer_id"]).to_dicts()[0]
    print(f"[Ground Truth] Added Products (Every Class): {target_info['added_products']}")
    print(f"[Ground Truth] Added Target Classes: {target_info['target_classes']}")

    # 3. Example Evaluation: Simulated Model Prediction
    simulated_model_raw_rankings = ["payroll_account", "direct_debit", "pensions", "payroll", "credit_card", "e_account", "taxes"]
    valid_recommendations = filter_valid_recommendations(simulated_model_raw_rankings, already_held, top_k=7)
    print(f"\n[Model Output] Raw rankings: {simulated_model_raw_rankings}")
    print(f"[Filtered Recommendations] Valid top-7: {valid_recommendations}")

    eval_result = evaluate_customer_recommendations(valid_recommendations, target_info)
    print(f"\n[Evaluation Metrics against All Classes]:")
    for k, v in eval_result.items():
        print(f"  {k}: {v}")
'''
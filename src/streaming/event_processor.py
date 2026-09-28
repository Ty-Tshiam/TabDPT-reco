import os
import math
import json
import datetime
import torch
import time
import sys
import fakeredis
import numpy as np
from pathlib import Path
import polars as pl
import tritonclient.grpc as grpcclient

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)

project_root = Path.cwd().parent   # adjust depending on where the notebook is
sys.path.insert(0, str(project_root))

try:
    from src.config import (
        CLEAN_TRAIN_PARQUET,
        CLEAN_TEST_PARQUET,
        TEST_TARGETS_PARQUET,
        SELECTED_15_TARGETS,
        TARGET_TO_INDEX,
        INDEX_TO_TARGET,
        INDEX_TO_TARGET,
        OTHER_9_PRODUCTS,
        ALL_24_PRODUCTS,
        CORE_10_PRODUCTS,
        CORE_7_PRODUCTS,
        CATEGORICAL_MAPPINGS_JSON,
        NORMALIZATION_STATS_JSON,
        CONTEXT_TENSOR_PATH,
        Y_TENSOR_PATH,
        MODEL_CONFIG,
        KV_CACHE,
        MODEL_ARTIFACT,
        STREAM_CUSTOMER_EVENTS,
        CONSUMER_GROUP,
        get_redis_client,
        ACTION_RELEVANCE_MATRIX,
        STREAM_RECOMMENDATIONS,
        RECS_TTL_SECONDS
    )
except ImportError:
    from config import (
        CLEAN_TRAIN_PARQUET,
        CLEAN_TEST_PARQUET,
        TEST_TARGETS_PARQUET,
        SELECTED_15_TARGETS,
        TARGET_TO_INDEX,
        INDEX_TO_TARGET,
        INDEX_TO_TARGET,
        OTHER_9_PRODUCTS,
        ALL_24_PRODUCTS,
        CORE_10_PRODUCTS,
        CORE_7_PRODUCTS,
        CATEGORICAL_MAPPINGS_JSON,
        NORMALIZATION_STATS_JSON,
        CONTEXT_TENSOR_PATH,
        Y_TENSOR_PATH,
        MODEL_CONFIG,
        KV_CACHE,
        MODEL_ARTIFACT,
        STREAM_CUSTOMER_EVENTS,
        CONSUMER_GROUP,
        get_redis_client,
        ACTION_RELEVANCE_MATRIX,
        STREAM_RECOMMENDATIONS,
        RECS_TTL_SECONDS
    )

config = MODEL_CONFIG["settings"]


df = pl.scan_parquet(str(CLEAN_TEST_PARQUET))
history = pl.scan_parquet(str(CLEAN_TRAIN_PARQUET))
test_targets = pl.scan_parquet(str(TEST_TARGETS_PARQUET.parent / "*.parquet"))
#dummy_data = {"customer_id": "1166753"}
date = datetime.date(2016, 5, 28)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
classes = INDEX_TO_TARGET


r = get_redis_client()
try:
    r.xgroup_create(STREAM_CUSTOMER_EVENTS, CONSUMER_GROUP, id="0", mkstream=True)
except Exception as e:
    if "BUSYGROUP" not in str(e):
        raise

print(f"[Worker] Listening for events on {STREAM_CUSTOMER_EVENTS}...")

TRITON_URL = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
client = grpcclient.InferenceServerClient(url=TRITON_URL)


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

    # 9. Retain latest snapshot for Worker
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
    context = torch.load(CONTEXT_TENSOR_PATH, map_location=device)
    y_train = torch.load(Y_TENSOR_PATH, map_location=device)

    rows, cols = query.shape
    pads = 128 - cols
    padding = torch.zeros((rows, pads), dtype=dtype, device=device)
    query = torch.hstack([query.to(device, dtype=dtype), padding])

    context = context.to(device, dtype=dtype)
    y_train = y_train.to(device, dtype=torch.long)

    x = torch.cat([context, query], dim=0)
    x = x.unsqueeze(0)
    y_train = y_train.unsqueeze(0)
    return y_train, x, 0


def prepare_query(query):
    rows, cols = query.shape
    pads = 128 - cols
    padding = torch.zeros((rows, pads), dtype=dtype, device=device)
    query = torch.hstack([query.to(device, dtype=dtype), padding])
    return query.unsqueeze(0)
    
def format_predictions(probs, held):
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    probs = probs.squeeze()  # ensure shape is (16,)
    
    print(f'[Worker] Softmaxxed {probs}')

    pred_class = probs.argmax().item()
    print(f'[Worker] Predicted Class: {pred_class} ({classes[pred_class]})')

    vals, ind = torch.sort(probs, descending=True)
    print(f'[Worker] Sorted indices : {ind.tolist()}')
    sorted_probs = [classes[i] for i in ind.tolist()]
    recos = filter_valid_recommendations(sorted_probs, held, 7)
    print(f'[Worker] Filtered recommendations : {recos}')
    print(f'[Worker] The customer should buy this product : {recos[0]}')
    return recos

def product_relavance(reco, action) -> bool:
    return reco in ACTION_RELEVANCE_MATRIX.get(action, [])

def publish_reco(stream_payload: dict):
    new_msg_id = r.xadd(STREAM_RECOMMENDATIONS, stream_payload)
    print(f"[Worker] Pushed notification to {STREAM_RECOMMENDATIONS} (msg_id: {new_msg_id})")

    cache_key = f"recs:{stream_payload['customer_id']}"

    r.set(cache_key, json.dumps(stream_payload), ex = RECS_TTL_SECONDS)


# ==============================================================================
# Pipeline Execution & Demonstration for Customer 1166753
# ==============================================================================
if __name__ == "__main__":
    while True:
        try: 
            entries = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername="worker_1",
                streams={STREAM_CUSTOMER_EVENTS: ">"}, # ">" means only new, undelivered messages
                count=1,                               # read 1 event at a time for real-time
                block=500                              # wait up to 500ms before returning empty
            )
            if entries:
                start_time = time.time() 
                for stream_name, messages in entries:
                    for msg_id, payload in messages:
                        # 1. Extract your fields from the payload dict
                        customer_id = payload.get("customer_id")
                        action = payload.get("action")
                        timestamp = payload.get("timestamp")

                        print(f"[Worker] Received msg {msg_id} for customer {customer_id}: action={action}")

                        print(f"[Worker] Fetching data for customer: {customer_id}...")
                        customer_info = get_customer(customer_id, df)

                        # Pull complete available customer history
                        customer_history = get_customer_history(customer_id, history)
                        print(f"[Worker] Pulled {customer_history.height} historical snapshots (Total with test: {customer_history.height + customer_info.height})")

                        customer = pl.concat([customer_info, customer_history], how="vertical")

                        engineered_customer = feature_engineering_pipeline(customer)
                        feature_cols = [c for c in engineered_customer.columns if c not in ["customer_id", "snapshot_date"]]
                        print(f"[Worker] Engineered {len(feature_cols)} features (Total columns: {engineered_customer.width})")

                        held = get_already_held_mask(engineered_customer)
                        print(f"[Worker] Got mask {held}")
                        
                        processed_customer = encode_and_normalize(engineered_customer)
                        query = processed_customer.drop("customer_id", "snapshot_date").to_torch().to(torch.float32)
                        print(f'[Worker] Processed customer info')

                        x_qry = prepare_query(query.to(device, dtype=dtype)).to(torch.float32).cpu().numpy()
                        print(f'[Worker] Prepared query tensor with shape {x_qry.shape} and dtype {x_qry.dtype}')

                        target_info = get_customer_targets(customer_id).to_dicts()[0]
                        print(f'[Worker] Got target info \n {target_info}')

                        inputs = [grpcclient.InferInput("QUERY_FEATURES", [1, 1, 128], "FP32")]
                        inputs[0].set_data_from_numpy(x_qry)

                        outputs = [grpcclient.InferRequestedOutput("PROBABILITIES")]
                        response = client.infer(model_name="tabdpt", inputs=inputs, outputs=outputs)
                        probs = response.as_numpy("PROBABILITIES")
                        
                        recos = format_predictions(probs, held)
                        top_product = recos[0]
                        r.xack(STREAM_CUSTOMER_EVENTS, CONSUMER_GROUP, msg_id)

                        relavant = product_relavance(top_product, action)
                        update_time = now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                        stream_payload = {
                            "customer_id": str(customer_id),
                            "action": str(action),
                            "top_product": str(top_product),
                            "recommendations": json.dumps(recos),  # serialize list
                            "is_active_trigger": str(relavant).lower(),
                            "latency_ms": f"{(time.time() - start_time) * 1000:.2f}",
                            "timestamp": update_time,
                        }
                        publish_reco(stream_payload)

                        if relavant:
                            print(f"[Worker] 🔥 Active Trigger! Pushed to {STREAM_RECOMMENDATIONS} (Reco: {top_product})")
                        else:
                            print(f"[Worker] ⚪ Passive action '{action}'. Reco '{top_product}' suppressed (is_active_trigger=false).")

                        print(f'[Worker] Total pipeline: {time.time() - start_time:.4f}s') 
        except KeyboardInterrupt:
            print("[Worker] Stopping worker gracefully...")
            break
        except Exception as e:
            print(f"[Worker] Error processing stream: {e}")
            time.sleep(1)
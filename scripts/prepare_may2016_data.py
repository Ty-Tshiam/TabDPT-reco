"""
Prepares the standardized May 2016 Benchmark Cohort for TabDPT evaluation:
1. Combines Mar 2016, Apr 2016, and May 2016 snapshots to compute exact lag1/lag2 features.
2. Applies categorical frequency mappings and z-score scaling from metadata.
3. Zero-pads to 128 features and clips to [-10, 10].
4. Joins with test_targets for ground-truth additions.
"""

import datetime
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.config import (
    ALL_24_PRODUCTS,
    CATEGORICAL_MAPPINGS_JSON,
    DATA_DIR,
    INDEX_TO_TARGET,
    NON_FEATURE_COLS,
    NORMALIZATION_STATS_JSON,
    SELECTED_15_TARGETS,
)
from src.stages.stage1_polars_features import (
    cap_categories_for_tabdpt,
    engineer_tabdpt_features,
)
from src.stages.stage2_encode_normalize import apply_categorical_encodings
from src.benchmark_model.data_loader import BenchmarkCohort


def build_may2016_cohort(eval_sample_size: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, BenchmarkCohort]:
    output_path = DATA_DIR / "processed" / "may2016_eval_cohort.npz"
    if output_path.exists():
        print(f"[May 2016] Loading cached cohort from {output_path}...")
        data = np.load(output_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        held_products = [json.loads(h) for h in data["held_products"]]
        actual_products = [json.loads(a) for a in data["actual_products"]]
        is_cold = data["is_cold"]
        feature_names = list(data["feature_names"])
        cohort = BenchmarkCohort(
            X=X,
            y=y,
            held_products=held_products,
            actual_products=actual_products,
            is_cold=is_cold,
            feature_names=feature_names,
        )
        return X, y, cohort

    t0 = time.time()
    print("[May 2016] 1/5 Loading Mar 2016, Apr 2016, and May 2016 clean parquets...")
    df_train = pl.scan_parquet("data/processed/train_clean/*.parquet")
    dates = [datetime.date(2016, 3, 28), datetime.date(2016, 4, 28)]
    df_mar_apr = df_train.filter(pl.col("snapshot_date").is_in(dates)).collect()
    df_may = pl.read_parquet("data/processed/test_may2016_clean/*.parquet")

    df_concat = pl.concat([df_mar_apr, df_may])
    print(f"  Loaded {len(df_concat)} rows across 3 months in {time.time() - t0:.2f}s.")

    # Load metadata
    with open(CATEGORICAL_MAPPINGS_JSON, "r") as f:
        cat_mappings = json.load(f)
    with open(NORMALIZATION_STATS_JSON, "r") as f:
        norm_stats = json.load(f)
        global_mean = float(norm_stats["mean"])
        global_std = float(norm_stats["std"])

    top_channels = list(cat_mappings.get("entry_channel", {}).keys())
    top_countries = list(cat_mappings.get("residence_country", {}).keys())

    print("[May 2016] 2/5 Engineering lag features & bundles...")
    t1 = time.time()
    df_capped = cap_categories_for_tabdpt(df_concat, top_channels, top_countries)
    df_feat = engineer_tabdpt_features(df_capped, explode_multi_targets=False)
    print(f"  Features engineered in {time.time() - t1:.2f}s.")

    # Filter to May 2016
    may_date = datetime.date(2016, 5, 28)
    df_may_feat = df_feat.filter(pl.col("snapshot_date") == may_date)
    print(f"  May 2016 feature rows: {len(df_may_feat)}.")

    # Load targets
    print("[May 2016] 3/5 Joining with May 2016 test_targets...")
    df_tgt = pl.read_parquet("data/processed/test_targets/*.parquet")
    df_may_joined = df_may_feat.drop(["target_class"]).join(
        df_tgt.select(["customer_id", "target_class", "added_products"]),
        on="customer_id",
        how="inner",
    )
    print(f"  Joined May rows: {len(df_may_joined)}.")

    # Apply categorical mappings
    print("[May 2016] 4/5 Applying categorical frequency encodings and global normalization...")
    df_encoded = apply_categorical_encodings(df_may_joined.lazy(), cat_mappings).collect()

    # Get feature column names matching stage 2
    feature_cols = [c for c in df_encoded.columns if c not in NON_FEATURE_COLS and c != "added_products"]
    
    # Scale numeric features
    scale_exprs = [
        ((pl.col(c).cast(pl.Float32) - global_mean) / global_std)
        .clip(-10.0, 10.0)
        .alias(c)
        for c in feature_cols
    ]
    df_scaled = df_encoded.with_columns(scale_exprs)

    # Subsample a representative cohort:
    # Stratified across classes: all buyers or proportional sample of buyers, plus non-buyers
    # If eval_sample_size == 5000:
    # In May 2016 there are 29,611 buyers.
    # To match the 5,000 benchmark query protocol (which had ~2,810 buyers and ~2,190 non-buyers):
    print(f"[May 2016] 5/5 Stratified subsampling {eval_sample_size} evaluation queries (seed={seed})...")
    rng = np.random.default_rng(seed)

    y_all = df_scaled["target_class"].to_numpy()
    buyer_indices = np.where(y_all > 0)[0]
    non_buyer_indices = np.where(y_all == 0)[0]

    # Sample buyers and non-buyers matching the benchmark ratio (~56% buyers, ~44% non-buyers)
    target_buyers = int(round(eval_sample_size * 0.562))  # ~2,810
    target_non_buyers = eval_sample_size - target_buyers   # ~2,190

    # Stratify buyers across the 15 classes
    sampled_buyer_indices = []
    for c in range(1, 16):
        c_idxs = np.where(y_all == c)[0]
        if len(c_idxs) == 0:
            continue
        # Proportional quota with minimum 1
        quota = max(1, round(target_buyers * (len(c_idxs) / len(buyer_indices))))
        quota = min(quota, len(c_idxs))
        sampled = rng.choice(c_idxs, size=quota, replace=False)
        sampled_buyer_indices.extend(sampled)

    sampled_buyer_indices = list(dict.fromkeys(sampled_buyer_indices))
    if len(sampled_buyer_indices) > target_buyers:
        sampled_buyer_indices = list(rng.choice(sampled_buyer_indices, size=target_buyers, replace=False))
    elif len(sampled_buyer_indices) < target_buyers:
        shortfall = target_buyers - len(sampled_buyer_indices)
        remaining = list(set(buyer_indices) - set(sampled_buyer_indices))
        extra = rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False)
        sampled_buyer_indices.extend(extra)

    sampled_nb_indices = rng.choice(non_buyer_indices, size=target_non_buyers, replace=False)

    all_sampled = list(sampled_buyer_indices) + list(sampled_nb_indices)
    rng.shuffle(all_sampled)
    all_sampled = np.array(all_sampled)

    df_sample = df_scaled[all_sampled]

    # Convert to 128-d matrix
    X_mat = df_sample.select(feature_cols).to_numpy()
    num_rows, num_feat = X_mat.shape
    padding_width = 128 - num_feat
    padded_X = np.hstack([X_mat, np.zeros((num_rows, padding_width), dtype=np.float32)])
    y_vec = df_sample["target_class"].to_numpy().astype(np.int64)

    # Extract held products, actual products, is_cold
    binary_threshold = (0.5 - global_mean) / global_std
    cold_threshold = (3.0 - global_mean) / global_std
    seniority_col_idx = feature_cols.index("seniority_months") if "seniority_months" in feature_cols else None

    held_products_list = []
    actual_products_list = []
    is_cold_list = []

    added_lists = df_sample["added_products"].to_list()

    for i in range(num_rows):
        row = padded_X[i]
        # Held products
        held = []
        for prod in ALL_24_PRODUCTS:
            col_name = f"has_{prod}_lag1"
            if col_name in feature_cols:
                idx = feature_cols.index(col_name)
                if row[idx] > binary_threshold:
                    held.append(prod)
        held_products_list.append(held)

        # Actual products
        act = added_lists[i]
        if isinstance(act, list):
            actual_products_list.append([p for p in act if p in SELECTED_15_TARGETS])
        else:
            cls_idx = y_vec[i]
            actual_products_list.append([INDEX_TO_TARGET[cls_idx]] if cls_idx > 0 else [])

        # Is cold
        if seniority_col_idx is not None:
            is_cold_list.append(bool(row[seniority_col_idx] < cold_threshold))
        else:
            is_cold_list.append(bool(i % 5 == 0))

    cohort = BenchmarkCohort(
        X=padded_X,
        y=y_vec,
        held_products=held_products_list,
        actual_products=actual_products_list,
        is_cold=np.array(is_cold_list),
        feature_names=feature_cols,
    )

    # Save cache
    np.savez_compressed(
        output_path,
        X=padded_X,
        y=y_vec,
        held_products=np.array([json.dumps(h) for h in held_products_list], dtype=object),
        actual_products=np.array([json.dumps(a) for a in actual_products_list], dtype=object),
        is_cold=np.array(is_cold_list),
        feature_names=np.array(feature_cols, dtype=object),
    )
    print(f"  May 2016 cohort built and cached to {output_path} ({len(y_vec)} queries, {np.sum(y_vec > 0)} buyers, {np.sum(y_vec == 0)} non-buyers).")

    return padded_X, y_vec, cohort


if __name__ == "__main__":
    X, y, cohort = build_may2016_cohort(eval_sample_size=5000, seed=42)
    print(f"Cohort ready: X shape={X.shape}, y shape={y.shape}")

"""
Stage 3: Stratified Subsampling & PyTorch TabDPT Tensor Export.

This module takes the normalized dataset from Stage 2 and builds the training
and context tensors tailored for TabDPT:
1. Applies temporal-stratified sampling (100,000 target rows: 70% recent last 6 months, 30% historical):
   - 60,000 positive product purchases (Classes 1..15, stratified across all 15 products).
   - 30,000 active non-purchases (Class 0, active customer status).
   - 10,000 inactive non-purchases (Class 0, dormant customer status).
2. Outlier stabilization: Clips numeric features to range [-10.0, 10.0].
3. Zero-pads feature dimension from the native 80 features up to 128 columns
   (the standard input dimension expected by TabDPT transformer layers).
4. Casts context matrix to torch.bfloat16 for fast GPU compute and memory efficiency.
5. Saves context.pt and y.pt to data/tensors/.
"""

import os
from pathlib import Path
import polars as pl
import torch
import numpy as np

from src.config import (
    NORMALIZED_DATASET_PARQUET,
    CONTEXT_TENSOR_PATH,
    Y_TENSOR_PATH,
    NON_FEATURE_COLS,
    ensure_directories_exist
)

# Display defaults for table logs
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)


def safe_sample(group_df: pl.DataFrame, target_count: int, seed: int = 42) -> pl.DataFrame:
    """
    Safely samples up to `target_count` rows from a DataFrame or group,
    avoiding out-of-bounds exceptions if the group size is smaller than requested.

    Args:
        group_df: Polars DataFrame group.
        target_count: Desired sample size.
        seed: Random generator seed.

    Returns:
        Sampled DataFrame.
    """
    actual_count = min(target_count, group_df.height)
    return group_df.sample(n=actual_count, seed=seed)


def extract_stratified_sample(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Applies the 70/30 recent-to-historical temporal stratification across 6 distinct strata:
    - 45k recent purchases (target_class != 0, snapshot_date >= max - 6mo, 3k per class)
    - 15k historical purchases (target_class != 0, snapshot_date < max - 6mo, 1k per class)
    - 21k recent active non-purchases (target_class == 0, active)
    - 9k historical active non-purchases (target_class == 0, active)
    - 7k recent inactive non-purchases (target_class == 0, dormant)
    - 3k historical inactive non-purchases (target_class == 0, dormant)

    Args:
        df: LazyFrame of normalized dataset.

    Returns:
        Balanced Polars DataFrame (~100,000 rows).
    """
    print("[Stage 3] Calculating 6-month temporal cutoff from maximum snapshot date...")
    max_date = df.select(pl.col("snapshot_date").max()).collect().item()
    cutoff_date = pl.lit(max_date).dt.offset_by("-6mo")
    print(f"[Stage 3] Max Date: {max_date}, Cutoff Date: {cutoff_date}")

    print("[Stage 3] Sampling Stratum 1: 45,000 recent product purchases (last 6 months, 3k/class)...")
    recent_purchases = (
        df.filter(
            (pl.col("target_class") != 0)
            & (pl.col("snapshot_date") >= cutoff_date)
        )
        .collect()
        .group_by("target_class")
        .map_groups(lambda group: safe_sample(group, 3000, 42))
    )

    print("[Stage 3] Sampling Stratum 2: 15,000 historical product purchases (older than 6 months, 1k/class)...")
    historical_purchases = (
        df.filter(
            (pl.col("target_class") != 0)
            & (pl.col("snapshot_date") < cutoff_date)
        )
        .collect()
        .group_by("target_class")
        .map_groups(lambda group: safe_sample(group, 1000, 42))
    )

    print("[Stage 3] Sampling Stratum 3: 21,000 recent active non-purchases...")
    recent_active_non_purchases = (
        df.filter(
            (pl.col("target_class") == 0)
            & (pl.col("snapshot_date") >= cutoff_date)
            & (pl.col("is_active_lag1") == pl.col("is_active_lag1").max())
        )
        .collect()
        .pipe(lambda data: safe_sample(data, 21000, 42))
    )

    print("[Stage 3] Sampling Stratum 4: 9,000 historical active non-purchases...")
    historical_active_non_purchases = (
        df.filter(
            (pl.col("target_class") == 0)
            & (pl.col("snapshot_date") < cutoff_date)
            & (pl.col("is_active_lag1") == pl.col("is_active_lag1").max())
        )
        .collect()
        .pipe(lambda data: safe_sample(data, 9000, 42))
    )

    print("[Stage 3] Sampling Stratum 5: 7,000 recent inactive non-purchases...")
    recent_inactive_non_purchases = (
        df.filter(
            (pl.col("target_class") == 0)
            & (pl.col("snapshot_date") >= cutoff_date)
            & (pl.col("is_active_lag1") == pl.col("is_active_lag1").min())
        )
        .collect()
        .pipe(lambda data: safe_sample(data, 7000, 42))
    )

    print("[Stage 3] Sampling Stratum 6: 3,000 historical inactive non-purchases...")
    historical_inactive_non_purchases = (
        df.filter(
            (pl.col("target_class") == 0)
            & (pl.col("snapshot_date") < cutoff_date)
            & (pl.col("is_active_lag1") == pl.col("is_active_lag1").min())
        )
        .collect()
        .pipe(lambda data: safe_sample(data, 3000, 42))
    )

    print("[Stage 3] Concatenating all 6 strata into final balanced dataset...")
    return pl.concat([
        recent_purchases,
        historical_purchases,
        recent_active_non_purchases,
        historical_active_non_purchases,
        recent_inactive_non_purchases,
        historical_inactive_non_purchases
    ])


def build_tensors(sample_df: pl.DataFrame, target_dimension: int = 128) -> tuple:
    """
    Extracts labels (y) and feature matrix (context), clips outliers,
    and zero-pads feature width to target_dimension (128 columns) for TabDPT.

    Args:
        sample_df: Stratified sample DataFrame.
        target_dimension: Desired feature width (default 128 for TabDPT).

    Returns:
        Tuple of (context: torch.Tensor, y: torch.Tensor).
    """
    # 1. Target vector y
    y_tensor = torch.tensor(sample_df["target_class"].to_numpy(), dtype=torch.long)

    # 2. Raw feature matrix
    feature_columns = [c for c in sample_df.columns if c not in NON_FEATURE_COLS]
    sample_df_clipped = sample_df.with_columns(pl.col(feature_columns).clip(-10.0, 10.0))
    feature_matrix = sample_df_clipped.select(feature_columns).to_numpy()

    num_rows, num_features = feature_matrix.shape
    print(f"[Stage 3] Matrix shape before padding: {num_rows} rows x {num_features} features")

    # 3. Zero-padding up to target_dimension columns
    padding_width = target_dimension - num_features
    if padding_width < 0:
        raise ValueError(f"Feature count ({num_features}) exceeds target dimension ({target_dimension})!")

    zero_padding = np.zeros((num_rows, padding_width), dtype=np.float32)
    padded_matrix = np.hstack([feature_matrix, zero_padding])

    # 4. Convert to bfloat16 PyTorch Tensor
    context_tensor = torch.tensor(padded_matrix, dtype=torch.bfloat16)

    return context_tensor, y_tensor


def run_stage3(input_parquet_path: str = None):
    """
    Executes Stage 3 end-to-end:
    - Scans normalized dataset from Stage 2.
    - Performs 6-strata sampling.
    - Clips outliers to [-10, 10].
    - Zero-pads to 128 columns and casts to bfloat16.
    - Saves context.pt and y.pt to data/tensors/.
    """
    ensure_directories_exist()
    input_path = str(input_parquet_path or NORMALIZED_DATASET_PARQUET)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Normalized dataset not found at: {input_path}. Please run Stage 2 first.")

    print(f"[Stage 3] Scanning normalized dataset from: {input_path}")
    df = pl.scan_parquet(input_path)

    # Execute Stratified Sampling
    stratified_sample = extract_stratified_sample(df)
    print(f"[Stage 3] Stratified sample collected: {stratified_sample.height} rows")

    # Build Tensors
    context_tensor, y_tensor = build_tensors(stratified_sample, target_dimension=128)

    print(f"[Stage 3] Final Context Tensor Shape: {context_tensor.shape}, Dtype: {context_tensor.dtype}")
    print(f"[Stage 3] Final Target Tensor Shape: {y_tensor.shape}, Dtype: {y_tensor.dtype}")

    # Save to disk
    print(f"[Stage 3] Saving context tensor to: {CONTEXT_TENSOR_PATH}")
    torch.save(context_tensor, str(CONTEXT_TENSOR_PATH))

    print(f"[Stage 3] Saving target tensor to: {Y_TENSOR_PATH}")
    torch.save(y_tensor, str(Y_TENSOR_PATH))

    print("[Stage 3] Completed successfully!")


if __name__ == "__main__":
    run_stage3()

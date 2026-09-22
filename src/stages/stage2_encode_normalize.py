"""
Stage 2: Categorical Frequency Encoding & Global Feature Normalization.

This module processes the engineered Parquet features output from Stage 1:
1. Identifies categorical columns (String/Categorical types plus explicit categoricals).
2. Derives frequency-ranked integer dictionaries (mapping most frequent category to 0).
3. Applies safe default mappings for unseen categories.
4. Computes a global dataset-level mean and standard deviation across all feature columns.
5. Standardizes features via z-score scaling: (x - global_mean) / global_std.
6. Serializes metadata (categorical_mappings.json, normalization_stats.json) to data/metadata/.
7. Sinks the fully encoded and scaled dataset to data/processed/normalized_dataset.parquet.
"""

import os
import json
from pathlib import Path
import polars as pl
import numpy as np

from src.config import (
    FEATURES_PARQUET,
    NORMALIZED_DATASET_PARQUET,
    CATEGORICAL_MAPPINGS_JSON,
    NORMALIZATION_STATS_JSON,
    EXPLICIT_CATEGORICAL_COLS,
    NON_FEATURE_COLS,
    ensure_directories_exist
)

# Configure Polars display defaults for interactive debugging
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(5)


def build_categorical_frequency_mappings(categorical_columns: list, df: pl.LazyFrame) -> dict:
    """
    Computes frequency-ordered integer encodings for each categorical feature.
    The most frequent category receives index 0, the next receives 1, and so on.

    Args:
        categorical_columns: List of column names to encode.
        df: Polars LazyFrame containing the data.

    Returns:
        Dictionary of {column_name: {category_value: integer_index}}.
    """
    mappings = {}

    for col in categorical_columns:
        # Group by category, count frequency, and sort descending
        category_counts = (
            df.select(pl.col(col))
            .group_by(col)
            .len()
            .sort("len", descending=True)
            .collect()
        )

        mappings[col] = {}
        for rank_idx in range(len(category_counts)):
            category_val = category_counts[rank_idx, 0]
            mappings[col][category_val] = rank_idx

    return mappings


def apply_categorical_encodings(df: pl.LazyFrame, categorical_mappings: dict) -> pl.LazyFrame:
    """
    Applies categorical mapping expressions with robust fallbacks for unseen values.

    Args:
        df: Polars LazyFrame to transform.
        categorical_mappings: Dictionary of categorical mappings.

    Returns:
        Transformed LazyFrame with columns replaced by integer indices.
    """
    encoding_expressions = []

    for col_name, mapping in categorical_mappings.items():
        # Determine fallback index for unseen categories (UNKNOWN, OTHER, or max_idx + 1)
        fallback_index = mapping.get("UNKNOWN", mapping.get("OTHER", max(mapping.values()) + 1))

        # Detect whether mapping keys are integers or strings to ensure correct type casting
        sample_key = next(iter(mapping.keys()))
        cast_type = pl.Int32 if isinstance(sample_key, int) else pl.String

        encoding_expressions.append(
            pl.col(col_name)
            .cast(cast_type)
            .replace(mapping, default=fallback_index)
            .cast(pl.Int32)
            .alias(col_name)
        )

    return df.with_columns(encoding_expressions)


def compute_global_normalization_stats(df: pl.LazyFrame, feature_columns: list) -> tuple:
    """
    Flattens all numeric feature values across the dataset to compute a single
    global mean and standard deviation (used by TabDPT transformers).

    Args:
        df: LazyFrame of data.
        feature_columns: List of feature column names to include in scaling.

    Returns:
        Tuple of (global_mean: float, global_std: float).
    """
    print("[Stage 2] Collecting and flattening feature matrix to compute global mean and std...")
    feature_matrix_flattened = (
        df.select(feature_columns)
        .collect()
        .to_numpy()
        .flatten()
    )

    global_mean = float(feature_matrix_flattened.mean())
    global_std = float(feature_matrix_flattened.std())

    return global_mean, global_std


def run_stage2(input_parquet_path: str = None):
    """
    Executes Stage 2 end-to-end:
    - Scans features Parquet.
    - Encodes categoricals to frequency indices.
    - Calculates and applies global z-score normalization.
    - Saves mappings and normalization metadata.
    - Sinks prepared dataset to Parquet.
    """
    ensure_directories_exist()
    input_path = input_parquet_path or (
        str(FEATURES_PARQUET / "*.parquet") if FEATURES_PARQUET.is_dir() else str(FEATURES_PARQUET)
    )

    print(f"[Stage 2] Scanning features from: {input_path}")
    df = pl.scan_parquet(input_path)

    # 1. Identify Categorical Columns
    schema = df.collect_schema()
    categorical_columns = []

    for col_name, dtype in schema.items():
        if col_name in NON_FEATURE_COLS:
            continue
        if dtype in (pl.String, pl.Categorical) or col_name in EXPLICIT_CATEGORICAL_COLS:
            categorical_columns.append(col_name)

    print(f"[Stage 2] Discovered {len(categorical_columns)} categorical features to encode: {categorical_columns}")

    # 2. Build Frequency Mappings
    categorical_mappings = build_categorical_frequency_mappings(categorical_columns, df)

    # Santander specific relation code override
    categorical_mappings["customer_relation_primary"] = {1: 0, 99: 1, 0: 2}

    # 3. Apply Encodings
    df_encoded = apply_categorical_encodings(df, categorical_mappings)

    # 4. Global Normalization
    all_column_names = schema.names()
    feature_columns = [c for c in all_column_names if c not in NON_FEATURE_COLS]

    global_mean, global_std = compute_global_normalization_stats(df_encoded, feature_columns)
    print(f"[Stage 2] Global Normalization - Mean: {global_mean:.6f}, Std: {global_std:.6f}")

    df_normalized = df_encoded.with_columns(
        (pl.col(feature_columns) - global_mean) / global_std
    )

    # 5. Serialize Metadata
    print(f"[Stage 2] Saving categorical mappings to: {CATEGORICAL_MAPPINGS_JSON}")
    with open(CATEGORICAL_MAPPINGS_JSON, "w") as f:
        json.dump(categorical_mappings, f, indent=2)

    normalization_stats = {
        "mean": global_mean,
        "std": global_std
    }
    print(f"[Stage 2] Saving normalization stats to: {NORMALIZATION_STATS_JSON}")
    with open(NORMALIZATION_STATS_JSON, "w") as f:
        json.dump(normalization_stats, f, indent=2)

    # 6. Sink Normalized Dataset Parquet
    print(f"[Stage 2] Sinking normalized dataset to: {NORMALIZED_DATASET_PARQUET}")
    df_normalized.sink_parquet(str(NORMALIZED_DATASET_PARQUET))
    print("[Stage 2] Completed successfully!")


if __name__ == "__main__":
    run_stage2()

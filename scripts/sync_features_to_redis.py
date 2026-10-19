"""
Batch Feature Store Materialization Script.

Precomputes and syncs customer feature vectors and metadata into Redis Feature Store:
- Eliminates live Polars Parquet scanning and window calculations during serving.
- Materializes 128-dim normalized tensors, held products, and demographic metadata.
- Pipelined batch ingestion for high throughput.

Usage:
    # Sync curated demo archetypes (< 1 second)
    python scripts/sync_features_to_redis.py --subset archetypes

    # Sync all test evaluation customers (or top N)
    python scripts/sync_features_to_redis.py --subset test --limit 5000

    # Inspect a customer's stored features in Redis
    python scripts/sync_features_to_redis.py --inspect 1166753

    # Check feature store status
    python scripts/sync_features_to_redis.py --stats
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from src.config import (
    CLEAN_TEST_PARQUET,
    CLEAN_TRAIN_PARQUET,
    SELECTED_15_TARGETS,
)
from src.features.extractor import extract_customer_features_polars
from src.features.feature_store import RedisFeatureStore

CURATED_ARCHETYPES = [
    "106596",
    "1273930",
    "663201",
    "392818",
    "374737",
    "1542491",
    "1221443",
    "1034229",
    "1009063",
    "1002447",
    "1002271",
    "1166753",
    "1011230",
    "1000008",
    "658033",
]


def sync_customer(
    customer_id: str,
    test_lf: pl.LazyFrame,
    train_lf: pl.LazyFrame,
    store: RedisFeatureStore,
) -> bool:
    """Computes and stores features for a single customer."""
    x_qry, held, metadata = extract_customer_features_polars(customer_id, test_lf, train_lf)
    if metadata.get("unknown"):
        return False

    store.set_customer_features(
        customer_id=customer_id,
        vector_128=x_qry,
        held_products=held,
        metadata=metadata,
    )
    return True


def sync_archetypes(store: RedisFeatureStore):
    """Syncs the 15 curated demo personas."""
    print(f"\n[Materializer] Syncing {len(CURATED_ARCHETYPES)} curated archetypes to Redis...")
    t0 = time.time()
    test_lf = pl.scan_parquet(str(CLEAN_TEST_PARQUET))
    train_lf = pl.scan_parquet(str(CLEAN_TRAIN_PARQUET))

    synced = 0
    for cid in CURATED_ARCHETYPES:
        if sync_customer(cid, test_lf, train_lf, store):
            synced += 1
            print(f"  [OK] Customer {cid} materialized")
        else:
            print(f"  [SKIP] Customer {cid} not found in Parquet partitions")

    elapsed = time.time() - t0
    print(f"[Materializer] Successfully synced {synced}/{len(CURATED_ARCHETYPES)} archetypes in {elapsed:.2f}s!")


def sync_test_customers(store: RedisFeatureStore, limit: int | None = None, batch_size: int = 500):
    """
    Syncs test evaluation customers into Redis Feature Store.
    """
    print(f"\n[Materializer] Scanning test evaluation customers from: {CLEAN_TEST_PARQUET}...")
    test_lf = pl.scan_parquet(str(CLEAN_TEST_PARQUET))
    train_lf = pl.scan_parquet(str(CLEAN_TRAIN_PARQUET))

    cids_df = test_lf.select(pl.col("customer_id").unique()).collect()
    cids = cids_df["customer_id"].to_list()
    if limit:
        cids = cids[:limit]

    total = len(cids)
    print(f"[Materializer] Materializing {total} customers into Redis Feature Store...")
    t0 = time.time()

    synced = 0
    for idx, cid in enumerate(cids, 1):
        if sync_customer(cid, test_lf, train_lf, store):
            synced += 1
        if idx % 100 == 0 or idx == total:
            print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%) in {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    print(f"[Materializer] Successfully synced {synced} customers in {elapsed:.2f}s ({synced/max(elapsed, 0.001):.1f} cust/sec)!")


def inspect_customer(store: RedisFeatureStore, customer_id: str):
    """Displays feature store payload for a given customer."""
    print(f"\n--- Inspecting Customer '{customer_id}' in Redis Feature Store ---")
    exists = store.exists(customer_id)
    print(f"Exists in Redis: {exists}")
    if not exists:
        return

    vector, held, metadata = store.get_customer_features(customer_id, enable_fallback=False)
    print(f"Feature Vector Shape: {vector.shape}, Dtype: {vector.dtype}")
    print(f"Non-zero feature count: {np.count_nonzero(vector)}/128")
    print(f"Sample values (first 10): {np.round(vector.flatten()[:10], 4).tolist()}")
    print(f"Held Products: {held}")
    print(f"Metadata: {metadata}")


def main():
    parser = argparse.ArgumentParser(description="Sync Customer Features into Redis Feature Store")
    parser.add_argument(
        "--subset",
        choices=["archetypes", "test"],
        default="archetypes",
        help="Subset of customers to sync (default: archetypes)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of test customers to sync",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Inspect a specific customer ID from Redis Feature Store",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print diagnostic stats of the Redis Feature Store",
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Clear all customer feature keys from Redis",
    )
    args = parser.parse_args()

    store = RedisFeatureStore()

    if args.flush:
        keys = store.r.keys("feature_store:customer:*")
        if keys:
            store.r.delete(*keys)
            print(f"[Materializer] Cleared {len(keys)} feature store keys from Redis.")
        else:
            print("[Materializer] No feature store keys found.")
        return

    if args.inspect:
        inspect_customer(store, args.inspect)
        return

    if args.stats:
        print("\n--- Redis Feature Store Stats ---")
        for k, v in store.get_stats().items():
            print(f"  {k}: {v}")
        return

    if args.subset == "archetypes":
        sync_archetypes(store)
    elif args.subset == "test":
        sync_test_customers(store, limit=args.limit)


if __name__ == "__main__":
    main()

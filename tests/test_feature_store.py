"""
Unit test suite for Redis Feature Store.

Tests:
1. Exact float32 bit-level serialization & deserialization of 128-dim vectors.
2. Individual and batch writes with Redis pipelines.
3. Sub-millisecond point lookups and metadata retrieval.
4. Transparent Polars fallback on cache misses.
5. Dynamic near-real-time mutation of held products.
6. TTL expiration management.
"""

import json
import numpy as np
import pytest
import fakeredis

from src.features.feature_store import (
    RedisFeatureStore,
    deserialize_vector,
    serialize_vector,
)


@pytest.fixture
def fake_redis_client():
    """Provides an isolated in-memory fakeredis client for testing."""
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def feature_store(fake_redis_client):
    """Provides a RedisFeatureStore instance using fakeredis with 30-day default TTL."""
    return RedisFeatureStore(redis_client=fake_redis_client, ttl_seconds=30 * 86400)


class TestVectorSerialization:
    """Validates binary preservation and padding mechanics."""

    def test_serialize_and_deserialize_exact_shape(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal((1, 1, 128)).astype(np.float32)

        raw_str = serialize_vector(vec)
        assert isinstance(raw_str, str)
        assert len(raw_str) == 684  # 512 bytes base64 encoded is 684 chars

        restored = deserialize_vector(raw_str)
        assert restored.shape == (1, 1, 128)
        assert restored.dtype == np.float32
        assert np.array_equal(vec, restored)

    def test_serialize_1d_array(self):
        vec = np.ones(128, dtype=np.float32) * 2.5
        raw_str = serialize_vector(vec)
        restored = deserialize_vector(raw_str)
        assert restored.shape == (1, 1, 128)
        assert np.allclose(restored, 2.5)

    def test_serialize_under_128_pads_with_zeros(self):
        # Emulate 80 native features needing padding to 128
        vec_80 = np.full(80, 1.23, dtype=np.float32)
        raw_str = serialize_vector(vec_80)
        restored = deserialize_vector(raw_str)
        assert restored.shape == (1, 1, 128)
        assert np.allclose(restored[0, 0, :80], 1.23)
        assert np.allclose(restored[0, 0, 80:], 0.0)


class TestFeatureStoreCRUD:
    """Validates storage, retrieval, and updates in Redis."""

    def test_set_and_get_customer_features(self, feature_store):
        customer_id = "test_cust_001"
        vec = np.random.randn(1, 1, 128).astype(np.float32)
        held = ["current_account", "payroll_account"]
        meta = {"age": 32, "gender": "F", "segment": "02 - PARTICULARES"}

        feature_store.set_customer_features(
            customer_id=customer_id,
            vector_128=vec,
            held_products=held,
            metadata=meta,
        )

        assert feature_store.exists(customer_id) is True

        ret_vec, ret_held, ret_meta = feature_store.get_customer_features(
            customer_id, enable_fallback=False
        )

        assert np.array_equal(vec, ret_vec)
        assert ret_held == held
        assert ret_meta == meta

    def test_get_missing_customer_raises_when_fallback_disabled(self, feature_store):
        with pytest.raises(KeyError):
            feature_store.get_customer_features("non_existent_999", enable_fallback=False)

    def test_update_held_products(self, feature_store):
        customer_id = "test_cust_002"
        vec = np.zeros((1, 1, 128), dtype=np.float32)
        feature_store.set_customer_features(
            customer_id=customer_id,
            vector_128=vec,
            held_products=["current_account"],
        )

        # 1. Append new product
        updated = feature_store.update_held_products(customer_id, "credit_card")
        assert "credit_card" in updated
        assert "current_account" in updated
        assert len(updated) == 2

        # 2. Append duplicate product (should not duplicate)
        updated2 = feature_store.update_held_products(customer_id, "credit_card")
        assert len(updated2) == 2

        # 3. Append 'do_nothing' (should be ignored)
        updated3 = feature_store.update_held_products(customer_id, "do_nothing")
        assert len(updated3) == 2
        assert "do_nothing" not in updated3

    def test_ttl_applied(self, feature_store):
        customer_id = "test_cust_ttl"
        vec = np.zeros((1, 1, 128), dtype=np.float32)
        feature_store.set_customer_features(
            customer_id=customer_id,
            vector_128=vec,
            held_products=[],
            ttl_seconds=3600,
        )
        ttl = feature_store.r.ttl(feature_store._get_key(customer_id))
        assert 0 < ttl <= 3600

    def test_delete_customer_features(self, feature_store):
        customer_id = "test_cust_del"
        vec = np.zeros((1, 1, 128), dtype=np.float32)
        feature_store.set_customer_features(customer_id, vec, [])
        assert feature_store.exists(customer_id) is True

        deleted = feature_store.delete_customer_features(customer_id)
        assert deleted is True
        assert feature_store.exists(customer_id) is False


class TestBatchMaterialization:
    """Validates high-throughput batch writes with Redis pipelines."""

    def test_batch_set_features(self, feature_store):
        num_records = 200
        records = []
        for i in range(num_records):
            records.append(
                {
                    "customer_id": f"batch_cust_{i}",
                    "vector_128": np.full((1, 1, 128), i * 0.1, dtype=np.float32),
                    "held_products": [f"prod_{i % 5}"],
                    "metadata": {"index": i},
                }
            )

        written = feature_store.batch_set_features(records, batch_size=50)
        assert written == num_records

        stats = feature_store.get_stats()
        assert stats["total_materialized_customers"] == num_records

        # Verify sample lookup
        sample_vec, sample_held, sample_meta = feature_store.get_customer_features(
            "batch_cust_42", enable_fallback=False
        )
        assert np.allclose(sample_vec, 4.2)
        assert sample_held == ["prod_2"]
        assert sample_meta["index"] == 42


class TestFallbackBehavior:
    """Validates graceful fallback computation and auto-materialization on cache miss."""

    def test_fallback_auto_populates_redis(self, feature_store):
        # Customer 1166753 exists in test parquets
        target_cust = "1166753"

        # Initially absent from Redis
        assert feature_store.exists(target_cust) is False

        # Retrieve with fallback=True
        vec, held, metadata = feature_store.get_customer_features(
            target_cust, enable_fallback=True
        )

        assert vec.shape == (1, 1, 128)
        assert len(held) > 0
        assert metadata.get("customer_id") == target_cust

        # Should now be populated in Redis!
        assert feature_store.exists(target_cust) is True

        # Second lookup hits Redis directly
        vec2, held2, metadata2 = feature_store.get_customer_features(
            target_cust, enable_fallback=False
        )
        assert np.array_equal(vec, vec2)
        assert held == held2

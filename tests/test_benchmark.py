"""Unit test suite for TabDPT benchmark models and evaluation metrics."""

import numpy as np
import pytest

from src.benchmark_model.base import BaseRecommender
from src.benchmark_model.data_loader import (
    load_context_data,
    prepare_benchmark_datasets,
)
from src.benchmark_model.metrics import (
    LatencyProfiler,
    compute_alert_distribution,
    compute_ap_at_k,
    compute_buyer_accuracy_breakdown,
    compute_catalog_gini,
    compute_ece,
    compute_js_divergence,
    compute_map_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_psi,
    compute_recall_and_hit_rate_at_k,
)
from src.benchmark_model.models.real_mlp import RealMLPRecommender
from src.benchmark_model.models.tree_models import (
    CatBoostRecommender,
    LightGBMRecommender,
)


class TestMetrics:
    """Validate mathematical correctness of all ranking, calibration, and system metrics."""

    def test_compute_ap_at_k_perfect(self):
        recs = ["credit_card", "payroll_account"]
        actual = ["credit_card", "payroll_account"]
        ap = compute_ap_at_k(recs, actual, k=7)
        assert pytest.approx(ap, rel=1e-4) == 1.0

    def test_compute_ap_at_k_partial_and_cutoff(self):
        # 1 hit at position 2 (1-indexed) out of 1 actual item
        recs = ["pensions", "credit_card", "taxes"]
        actual = ["credit_card"]
        ap = compute_ap_at_k(recs, actual, k=7)
        # hits: 0 at i=0, 1 at i=1 -> score = (1/2) / min(1, 7) = 0.5
        assert pytest.approx(ap, rel=1e-4) == 0.5

    def test_compute_ap_at_k_empty(self):
        assert compute_ap_at_k(["credit_card"], [], k=7) == 0.0

    def test_compute_map_at_k(self):
        all_recs = [["credit_card"], ["pensions", "credit_card"]]
        all_actual = [["credit_card"], ["credit_card"]]
        # user 1: 1.0; user 2: 0.5 -> mean = 0.75
        map_val = compute_map_at_k(all_recs, all_actual, k=7)
        assert pytest.approx(map_val, rel=1e-4) == 0.75

    def test_compute_ndcg_at_k(self):
        # Position 1 hit: DCG = 1/log2(2) = 1.0; IDCG = 1.0 -> NDCG = 1.0
        recs = [["credit_card"]]
        actual = [["credit_card"]]
        assert pytest.approx(compute_ndcg_at_k(recs, actual, k=4), rel=1e-4) == 1.0

        # Position 2 hit: DCG = 1/log2(3) = 0.6309; IDCG = 1/log2(2) = 1.0 -> NDCG = 0.6309
        recs = [["taxes", "credit_card"]]
        actual = [["credit_card"]]
        ndcg = compute_ndcg_at_k(recs, actual, k=4)
        assert pytest.approx(ndcg, rel=1e-3) == 0.6309

    def test_compute_mrr(self):
        recs = [["taxes", "credit_card"], ["funds", "mortgage"]]
        actual = [["credit_card"], ["loans"]]
        # user 1 has first hit at rank 2 -> RR = 0.5; user 2 has no hit -> RR = 0.0
        mrr = compute_mrr(recs, actual, k=7)
        assert pytest.approx(mrr, rel=1e-4) == 0.25

    def test_compute_recall_and_hit_rate(self):
        recs = [["credit_card", "taxes"], ["funds", "mortgage"]]
        actual = [["credit_card", "pensions"], ["loans"]]
        # user 1: matched 1 out of 2 (recall 0.5, hit 1); user 2: matched 0 (recall 0, hit 0)
        rec, hr = compute_recall_and_hit_rate_at_k(recs, actual, k=7)
        assert pytest.approx(rec, rel=1e-4) == 0.25
        assert pytest.approx(hr, rel=1e-4) == 0.5

    def test_compute_ece(self):
        # 4 samples with 100% confidence matching truth -> ECE should be 0.0
        y_true = np.array([0, 1, 2, 3])
        y_probs = np.zeros((4, 16))
        for i in range(4):
            y_probs[i, i] = 1.0

        ece = compute_ece(y_true, y_probs, n_bins=5)
        assert pytest.approx(ece, abs=1e-4) == 0.0

    def test_compute_psi(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        assert pytest.approx(compute_psi(p, q), abs=1e-5) == 0.0

        q_drift = np.array([0.6, 0.2, 0.1, 0.1])
        assert compute_psi(p, q_drift) > 0.1

    def test_compute_js_divergence(self):
        # Identical distributions -> 0 divergence and 0 distance
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        res = compute_js_divergence(p, q)
        assert res["js_divergence"] == pytest.approx(0.0, abs=1e-4)
        assert res["js_distance"] == pytest.approx(0.0, abs=1e-4)

        # Orthogonal distributions -> maximum divergence = 1.0 (with base 2)
        p_ortho = np.array([1.0, 0.0])
        q_ortho = np.array([0.0, 1.0])
        res_ortho = compute_js_divergence(p_ortho, q_ortho)
        assert res_ortho["js_divergence"] == pytest.approx(1.0, abs=1e-4)
        assert res_ortho["js_distance"] == pytest.approx(1.0, abs=1e-4)

        # Asymmetric / moderate drift
        p_norm = np.array([0.4, 0.3, 0.2, 0.1])
        q_norm = np.array([0.1, 0.2, 0.3, 0.4])
        res_norm = compute_js_divergence(p_norm, q_norm)
        assert 0.0 < res_norm["js_divergence"] < 1.0
        assert pytest.approx(res_norm["js_distance"], abs=1e-2) == np.sqrt(
            res_norm["js_divergence"]
        )

    def test_compute_catalog_gini(self):
        catalog = ["prod_a", "prod_b", "prod_c", "prod_d"]
        # Perfectly uniform recommendations -> Gini = 0.0
        uniform_recs = [
            ["prod_a", "prod_b"],
            ["prod_c", "prod_d"],
            ["prod_a", "prod_c"],
            ["prod_b", "prod_d"],
        ]
        assert compute_catalog_gini(uniform_recs, catalog) == pytest.approx(
            0.0, abs=1e-4
        )

        # Heavily concentrated / skewed on prod_a -> Gini close to maximum inequality (1 - 1/n)
        skewed_recs = [["prod_a"] for _ in range(100)]
        gini_skewed = compute_catalog_gini(skewed_recs, catalog)
        # For 4 items with 1 having all mass: (4-1)/4 = 0.75
        assert gini_skewed == pytest.approx(0.75, abs=1e-2)

        # Empty recs -> 0.0
        assert compute_catalog_gini([], catalog) == 0.0

    def test_compute_alert_distribution(self):
        # 3 instances: 2 class 0 (silent), 1 class 3 (active)
        preds = np.array([0, 3, 0])
        alerts = compute_alert_distribution(preds)
        assert alerts["active_alerts_pct"] == pytest.approx(33.33, abs=0.1)
        assert alerts["silent_alerts_pct"] == pytest.approx(66.67, abs=0.1)

    def test_compute_buyer_accuracy_breakdown(self):
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 3])
        breakdown = compute_buyer_accuracy_breakdown(y_true, y_pred)
        # Non-buyers: indices 0, 1 (true=0,0). pred is 0, 1 -> 1/2 = 50.0%
        assert breakdown["non_buyers_accuracy"] == 50.0
        # Buyers: indices 2, 3 (true=1,2). pred is 1, 3 -> 1/2 = 50.0%
        assert breakdown["active_buyers_top1_acc"] == 50.0
        # Overall: 2/4 = 50.0%
        assert breakdown["overall_top1_acc"] == 50.0

    def test_latency_profiler(self):
        profiler = LatencyProfiler()
        profiler.record(0.010, batch_size=1)  # 10ms
        profiler.record(0.020, batch_size=1)  # 20ms
        summary = profiler.summary()
        assert summary["mean_ms"] == pytest.approx(15.0, abs=0.1)
        assert summary["p50_ms"] == pytest.approx(15.0, abs=0.1)
        assert summary["p99_ms"] > 0.0


class TestModels:
    """Validate model contracts: fit, predict_proba, recommend."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 20)).astype(np.float32)
        y = rng.integers(0, 16, size=60).astype(np.int64)
        return X, y

    def test_base_recommender_filtering(self):
        class DummyRecommender(BaseRecommender):
            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                return np.ones((len(X), 16)) / 16.0

        model = DummyRecommender("Dummy", "Test")
        probs = np.zeros((1, 16))
        # Give class 1 (current_account) high prob, class 2 (payroll_account) second high
        probs[0, 1] = 0.6
        probs[0, 2] = 0.3
        probs[0, 3] = 0.1

        # Customer already holds current_account
        recs = model.recommend(probs, held_products=[["current_account"]], top_k=2)
        assert "current_account" not in recs[0]
        assert recs[0][0] == "payroll_account"

    def test_real_mlp_smoke(self, synthetic_data):
        X, y = synthetic_data
        model = RealMLPRecommender(hidden_dim=32, num_blocks=1, epochs=2, batch_size=32)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 16)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    def test_lightgbm_smoke(self, synthetic_data):
        X, y = synthetic_data
        model = LightGBMRecommender(n_estimators=5, num_leaves=7)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 16)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    def test_catboost_smoke(self, synthetic_data):
        X, y = synthetic_data
        model = CatBoostRecommender(iterations=5, depth=3)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 16)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    def test_tabdpt_wrapper_smoke(self, synthetic_data):
        from src.benchmark_model.models.tabdpt_wrapper import (
            TabDPTKVCacheRecommender,
            TabDPTVanillaRecommender,
        )

        X, y = synthetic_data
        kv_model = TabDPTKVCacheRecommender()
        kv_model.fit(X, y)
        probs_kv = kv_model.predict_proba(X[:5])
        assert probs_kv.shape == (5, 16)
        assert np.allclose(np.sum(probs_kv, axis=1), 1.0, atol=1e-4)

        vanilla_model = TabDPTVanillaRecommender()
        vanilla_model.fit(X, y)
        probs_vanilla = vanilla_model.predict_proba(X[:5])
        assert probs_vanilla.shape == (5, 16)
        assert np.allclose(np.sum(probs_vanilla, axis=1), 1.0, atol=1e-4)


class TestDataLoader:
    """Validate tensor loading and cohort partitioning."""

    def test_context_data_loading(self):
        X, y = load_context_data()
        assert X.shape[0] in (73968, 91313)
        assert X.shape[1] == 128
        assert y.shape[0] in (73968, 91313)

    def test_prepare_benchmark_datasets(self):
        X_tr, y_tr, cohort = prepare_benchmark_datasets(
            train_ratio=0.8, eval_sample_size=100, seed=42
        )
        assert len(X_tr) > 0
        assert len(y_tr) > 0
        assert len(cohort.X) == 100
        assert len(cohort.y) == 100
        assert len(cohort.held_products) == 100
        assert len(cohort.actual_products) == 100
        assert len(cohort.is_cold) == 100
        assert cohort.is_cold.dtype == bool


class TestMLflowLogger:
    """Validate MLflow experiment tracking and model registration."""

    def test_mlflow_logging_and_registry(self, tmp_path):
        from src.benchmark_model.mlflow_logger import (
            log_model_run_to_mlflow,
            register_benchmark_models,
            setup_mlflow,
        )

        test_db = tmp_path / "test_mlflow.db"
        tracking_uri = f"sqlite:///{test_db}"
        setup_mlflow(tracking_uri=tracking_uri, experiment_name="Test-Experiment")

        sample_record = {
            "model_name": "TabDPT (KV-Cached)",
            "family": "Tabular Foundation Model",
            "train_time_sec": 0.0,
            "map7": 0.5910,
            "hit_rate7": 99.0,
            "catalog_gini": 0.2840,
            "js_dist": 0.0616,
            "psi": 0.0315,
            "latency_p99_ms": 22.40,
        }

        run_id = log_model_run_to_mlflow(sample_record)
        assert run_id is not None
        sample_record["mlflow_run_id"] = run_id

        versions = register_benchmark_models(
            [sample_record],
            tracking_uri=tracking_uri,
            registered_model_name="Test-TabDPT",
        )
        assert "champion" in versions
        assert versions["champion"] == "1"


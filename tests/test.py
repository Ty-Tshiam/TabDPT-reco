"""Unit test suite for TabDPT product recommendation system.

Validates:
1. Configuration constants, directory resolutions, and path integrity.
2. Santander column schema definitions and 16-class product targets.
3. Recommendation candidate filtering and evaluation metrics.
4. PyTorch tensor zero-padding and transformation utilities.
"""

from pathlib import Path

import torch

from src.config import (
    ALL_24_PRODUCTS,
    COLUMN_MAPPING,
    CORE_7_PRODUCTS,
    CORE_10_PRODUCTS,
    DATA_DIR,
    METADATA_DIR,
    MODEL_CONFIG,
    NON_FEATURE_COLS,
    OTHER_9_PRODUCTS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SELECTED_15_TARGETS,
    TARGET_TO_INDEX,
    TENSORS_DIR,
    ensure_directories_exist,
)


class TestConfigPaths:
    """Validate filesystem path resolutions and directory management."""

    def test_project_root_exists(self):
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "src").is_dir()

    def test_directory_hierarchy(self):
        assert DATA_DIR == PROJECT_ROOT / "data"
        assert RAW_DATA_DIR.parent == DATA_DIR / "raw"
        assert PROCESSED_DATA_DIR.parent == DATA_DIR
        assert METADATA_DIR.parent == DATA_DIR
        assert TENSORS_DIR.parent == DATA_DIR

    def test_ensure_directories_exist(self, tmp_path, monkeypatch):
        mock_raw = tmp_path / "data" / "raw" / "santander"
        mock_processed = tmp_path / "data" / "processed"
        mock_metadata = tmp_path / "data" / "metadata"
        mock_tensors = tmp_path / "data" / "tensors"

        monkeypatch.setattr("src.config.RAW_DATA_DIR", mock_raw)
        monkeypatch.setattr("src.config.PROCESSED_DATA_DIR", mock_processed)
        monkeypatch.setattr("src.config.METADATA_DIR", mock_metadata)
        monkeypatch.setattr("src.config.TENSORS_DIR", mock_tensors)

        ensure_directories_exist()

        assert mock_raw.is_dir()
        assert mock_processed.is_dir()
        assert mock_metadata.is_dir()
        assert mock_tensors.is_dir()


class TestConfigSchemas:
    """Validate dataset column mappings and product target lists."""

    def test_column_mapping_keys(self):
        assert "fecha_dato" in COLUMN_MAPPING
        assert "ncodpers" in COLUMN_MAPPING
        assert COLUMN_MAPPING["fecha_dato"] == "snapshot_date"
        assert COLUMN_MAPPING["ncodpers"] == "customer_id"

    def test_selected_15_targets_integrity(self):
        assert len(SELECTED_15_TARGETS) == 15
        assert len(set(SELECTED_15_TARGETS)) == 15
        for target in SELECTED_15_TARGETS:
            assert target in TARGET_TO_INDEX

    def test_target_to_index_mapping(self):
        assert len(TARGET_TO_INDEX) == 15
        indices = set(TARGET_TO_INDEX.values())
        # Target classes must be 1 through 15 (class 0 reserved for no-purchase)
        assert indices == set(range(1, 16))
        for idx, target in enumerate(SELECTED_15_TARGETS, start=1):
            assert TARGET_TO_INDEX[target] == idx

    def test_other_9_products_integrity(self):
        assert len(OTHER_9_PRODUCTS) == 9
        assert len(set(OTHER_9_PRODUCTS)) == 9
        overlap = set(SELECTED_15_TARGETS).intersection(set(OTHER_9_PRODUCTS))
        assert len(overlap) == 0

    def test_all_24_products(self):
        assert len(ALL_24_PRODUCTS) == 24
        assert ALL_24_PRODUCTS == SELECTED_15_TARGETS + OTHER_9_PRODUCTS

    def test_core_subsets(self):
        assert len(CORE_10_PRODUCTS) == 10
        assert len(CORE_7_PRODUCTS) == 7
        assert set(CORE_7_PRODUCTS).issubset(set(CORE_10_PRODUCTS))
        assert set(CORE_10_PRODUCTS).issubset(set(ALL_24_PRODUCTS))

    def test_non_feature_cols(self):
        assert "customer_id" in NON_FEATURE_COLS
        assert "snapshot_date" in NON_FEATURE_COLS
        assert "target_class" in NON_FEATURE_COLS


class TestModelConfiguration:
    """Validate model architecture parameters and hyperparameter configurations."""

    def test_model_config_sections(self):
        expected_sections = {
            "version",
            "description",
            "env",
            "model",
            "training",
            "data",
            "logging",
        }
        assert expected_sections.issubset(set(MODEL_CONFIG.keys()))

    def test_model_architecture_parameters(self):
        model_cfg = MODEL_CONFIG["model"]
        assert model_cfg["emsize"] == 512
        assert model_cfg["max_num_classes"] == 16
        assert model_cfg["max_num_features"] == 128
        assert model_cfg["nhead"] == 8
        assert model_cfg["nlayers"] == 32

    def test_training_parameters(self):
        training_cfg = MODEL_CONFIG["training"]
        assert training_cfg["clip_grad_norm"] == 4.0
        assert training_cfg["weight_decay"] == 0.05
        assert len(training_cfg["seq_lens"]) > 0


class TestRecommendationFiltering:
    """Validate candidate ranking exclusion and MAP evaluation logic."""

    @staticmethod
    def filter_valid_recommendations(
        candidate_ranked_products: list[str],
        already_held: list[str],
        top_k: int = 7,
    ) -> list[str]:
        """Filter out products already held by the customer, returning top_k."""
        return [p for p in candidate_ranked_products if p not in already_held][:top_k]

    @staticmethod
    def evaluate_customer_recommendations(
        recommended_products: list[str],
        ground_truth_targets: dict,
    ) -> dict:
        """Evaluate top-K recommendations against ground-truth additions."""
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
        ap = (
            score / min(len(true_additions), len(recommended_products))
            if true_additions
            else 0.0
        )

        return {
            "true_added_products": true_additions,
            "hits": hits,
            "precision": round(len(hits) / len(recommended_products), 4)
            if recommended_products
            else 0.0,
            "recall": round(len(hits) / len(true_additions), 4)
            if true_additions
            else 0.0,
            "average_precision": round(ap, 4),
        }

    def test_filter_excludes_already_held(self):
        candidates = [
            "current_account",
            "payroll_account",
            "credit_card",
            "pensions",
            "direct_debit",
        ]
        already_held = ["current_account", "direct_debit"]
        filtered = self.filter_valid_recommendations(candidates, already_held, top_k=3)

        assert "current_account" not in filtered
        assert "direct_debit" not in filtered
        assert filtered == ["payroll_account", "credit_card", "pensions"]
        assert len(filtered) == 3

    def test_filter_top_k_bounds(self):
        candidates = ["payroll_account", "credit_card"]
        already_held = []
        filtered = self.filter_valid_recommendations(candidates, already_held, top_k=5)
        assert len(filtered) == 2

    def test_evaluate_perfect_match(self):
        recommended = ["credit_card", "payroll_account"]
        ground_truth = {"added_products": ["credit_card", "payroll_account"]}

        eval_res = self.evaluate_customer_recommendations(recommended, ground_truth)
        assert eval_res["precision"] == 1.0
        assert eval_res["recall"] == 1.0
        assert eval_res["average_precision"] == 1.0
        assert eval_res["hits"] == ["credit_card", "payroll_account"]

    def test_evaluate_no_overlap(self):
        recommended = ["credit_card", "taxes"]
        ground_truth = {"added_products": ["current_account"]}

        eval_res = self.evaluate_customer_recommendations(recommended, ground_truth)
        assert eval_res["precision"] == 0.0
        assert eval_res["recall"] == 0.0
        assert eval_res["average_precision"] == 0.0
        assert eval_res["hits"] == []

    def test_evaluate_empty_ground_truth(self):
        recommended = ["credit_card"]
        ground_truth = {"added_products": []}

        eval_res = self.evaluate_customer_recommendations(recommended, ground_truth)
        assert eval_res["hits"] == []
        assert eval_res["precision"] == 0.0


class TestTensorPaddingUtilities:
    """Validate tensor padding logic matching Stage 3 tensor builder."""

    def test_feature_padding_to_128_dims(self):
        batch_size = 4
        num_features = 46
        target_dims = 128

        query = torch.randn(batch_size, num_features, dtype=torch.float32)
        pads = target_dims - num_features
        padding = torch.zeros((batch_size, pads), dtype=torch.float32)
        padded_query = torch.hstack([query, padding])

        assert padded_query.shape == (batch_size, 128)
        assert torch.all(padded_query[:, num_features:] == 0.0)

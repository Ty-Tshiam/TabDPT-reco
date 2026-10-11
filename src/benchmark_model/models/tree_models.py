"""
Tree-Based Recommenders: LightGBM and CatBoost Baselines.

Implements high-performance gradient boosted decision trees for 16-class product recommendation.
"""

import time
import numpy as np
from src.benchmark_model.base import BaseRecommender


class LightGBMRecommender(BaseRecommender):
    """LightGBM Multi-class Recommender."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        super().__init__(name="LightGBM", family="Gradient Boosted Trees")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMRecommender":
        import lightgbm as lgb

        start_time = time.time()
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=16,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model.fit(X, y)
        self.training_time_sec = round(time.time() - start_time, 2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM model must be fitted before predict_proba.")

        probs = self.model.predict_proba(X)
        # Ensure (N, 16) shape in case any class was absent in a tiny subset
        if probs.shape[1] < 16:
            full_probs = np.zeros((X.shape[0], 16), dtype=np.float32)
            classes_present = self.model.classes_
            for idx, c in enumerate(classes_present):
                if c < 16:
                    full_probs[:, c] = probs[:, idx]
            return full_probs
        return probs


class CatBoostRecommender(BaseRecommender):
    """CatBoost Multi-class Recommender."""

    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.08,
        depth: int = 6,
        random_seed: int = 42,
    ):
        super().__init__(name="CatBoost", family="Gradient Boosted Trees")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostRecommender":
        from catboost import CatBoostClassifier

        start_time = time.time()
        self.model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.random_seed,
            thread_count=-1,
            verbose=False,
        )
        self.model.fit(X, y)
        self.training_time_sec = round(time.time() - start_time, 2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatBoost model must be fitted before predict_proba.")

        probs = self.model.predict_proba(X)
        if probs.shape[1] < 16:
            full_probs = np.zeros((X.shape[0], 16), dtype=np.float32)
            classes_present = self.model.classes_
            for idx, c in enumerate(classes_present):
                if c < 16:
                    full_probs[:, int(c)] = probs[:, idx]
            return full_probs
        return probs

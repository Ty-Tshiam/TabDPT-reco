"""
Base Recommender Interface for TabDPT Benchmarking.

Defines the contract for all candidate models (RealMLP, LightGBM, CatBoost, TabDPT):
1. fit(X, y): Train or fit on feature matrix and target classes (0..15).
2. predict_proba(X): Output (N, 16) probability distribution across classes.
3. recommend(probs, held_products, top_k): Candidate filtering and ranking.
4. save(path) / load(path): Model persistence.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.config import INDEX_TO_TARGET


class BaseRecommender(ABC):
    """Abstract base class for recommendation models in TabDPT benchmark suite."""

    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family
        self.classes = INDEX_TO_TARGET
        self.training_time_sec: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRecommender":
        """
        Fit model on training features X and targets y (classes 0..15).
        
        Args:
            X: 2D NumPy array of shape (N, num_features).
            y: 1D NumPy array of shape (N,) with integers 0..15.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted probabilities for each of the 16 classes.
        
        Args:
            X: 2D NumPy array of shape (N, num_features).
            
        Returns:
            2D NumPy array of shape (N, 16) with row sums close to 1.0.
        """

    def recommend(
        self,
        probs: np.ndarray,
        held_products: list[list[str]],
        top_k: int = 7,
    ) -> list[list[str]]:
        """
        Rank candidate products excluding already-held products at lag 1.
        
        Args:
            probs: 2D NumPy array of shape (N, 16) probabilities.
            held_products: List of lists of already-held product names per customer.
            top_k: Number of recommendations to produce per customer.
            
        Returns:
            List of lists of top_k recommended product names.
        """
        recs = []
        num_samples = probs.shape[0]

        for i in range(num_samples):
            held_set = set(held_products[i]) if i < len(held_products) else set()
            sample_probs = probs[i]

            # Sort candidate classes 1..15 by probability descending (class 0 is hold/status-quo)
            # We rank product classes 1..15:
            ranked_indices = np.argsort(sample_probs[1:])[::-1] + 1
            ranked_products = [self.classes[idx] for idx in ranked_indices]

            # Filter out already held products
            filtered = [p for p in ranked_products if p not in held_set][:top_k]
            recs.append(filtered)

        return recs

    def save(self, path: str) -> None:
        """Persist model state to disk (optional fallback)."""

    def load(self, path: str) -> "BaseRecommender":
        """Load model state from disk (optional fallback)."""
        return self

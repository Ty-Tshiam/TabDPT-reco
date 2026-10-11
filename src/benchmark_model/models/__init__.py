"""Benchmark Models Zoo."""

from src.benchmark_model.models.real_mlp import RealMLPRecommender
from src.benchmark_model.models.tabdpt_wrapper import TabDPTRecommender
from src.benchmark_model.models.tree_models import (
    CatBoostRecommender,
    LightGBMRecommender,
)

__all__ = [
    "CatBoostRecommender",
    "LightGBMRecommender",
    "RealMLPRecommender",
    "TabDPTRecommender",
]

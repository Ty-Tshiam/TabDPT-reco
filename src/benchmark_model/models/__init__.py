"""Benchmark Models Zoo."""

from src.benchmark_model.models.real_mlp import RealMLPRecommender
from src.benchmark_model.models.tabdpt_wrapper import (
    TabDPTKVCacheRecommender,
    TabDPTRecommender,
    TabDPTVanillaRecommender,
)
from src.benchmark_model.models.tree_models import (
    CatBoostRecommender,
    LightGBMRecommender,
)

__all__ = [
    "RealMLPRecommender",
    "LightGBMRecommender",
    "CatBoostRecommender",
    "TabDPTRecommender",
    "TabDPTVanillaRecommender",
    "TabDPTKVCacheRecommender",
]

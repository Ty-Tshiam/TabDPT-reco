"""
Export raw logits, probabilities, and test cohort ground-truth for split_uniform_classes.
Enables offline, instant experimentation with distribution gating and confidence meters (0 GPU passes).
"""

import json
import time
from pathlib import Path
import numpy as np
import torch

from src.config import DATA_DIR
from src.benchmark_model.data_loader import prepare_benchmark_datasets
from scripts.experiment_context_splits import load_tabdpt_model, sample_split_from_pool
from scripts.experiment_confidence_meter import get_raw_logits_and_probs


def export_uniform_cache(n_context: int = 5000, eval_samples: int = 5000, device: str = "cuda"):
    output_dir = DATA_DIR / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "uniform_logits_cache.npz"

    print("================================================================================")
    print("  Exporting split_uniform_classes Logits & Test Cohort Cache")
    print("================================================================================")
    print(f"  Device:         {device}")
    print(f"  Context Budget: {n_context} samples")
    print(f"  Test Queries:   {eval_samples}")
    print(f"  Output Cache:   {cache_path}")
    print("================================================================================\n")

    # 1. Load data
    print(f"[1/4] Preparing benchmark datasets ({eval_samples} queries)...")
    X_train_pool, y_train_pool, test_cohort = prepare_benchmark_datasets(
        train_ratio=0.8,
        eval_sample_size=eval_samples,
        seed=42,
    )

    # 2. Sample context
    print(f"[2/4] Sampling split_uniform_classes context ({n_context} samples)...")
    X_ctx, y_ctx = sample_split_from_pool(
        X_pool=X_train_pool,
        y_pool=y_train_pool,
        feature_names=test_cohort.feature_names,
        split_name="split_uniform_classes",
        n_samples=n_context,
        seed=42,
    )

    # 3. Model Inference
    print(f"[3/4] Initializing model and running batched inference on {device}...")
    t0 = time.time()
    model = load_tabdpt_model(device=device)
    logits, probs = get_raw_logits_and_probs(
        model=model,
        X_ctx_np=X_ctx,
        y_ctx_np=y_ctx,
        X_test_np=test_cohort.X,
        device=device,
        batch_size=250,
    )
    print(f"[3/4] Inference finished in {time.time() - t0:.2f}s! Logits shape: {logits.shape}")

    # 4. Save cache
    print(f"[4/4] Saving cache to {cache_path}...")
    np.savez_compressed(
        cache_path,
        logits=logits,
        probs=probs,
        y_test=test_cohort.y,
        is_cold=test_cohort.is_cold,
        held_products=np.array([json.dumps(h) for h in test_cohort.held_products], dtype=object),
        actual_products=np.array([json.dumps(a) for a in test_cohort.actual_products], dtype=object),
    )
    print(f"[Done] Cache exported successfully! File size: {cache_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    export_uniform_cache(device=device)

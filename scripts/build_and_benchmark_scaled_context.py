"""
Scaled Context Benchmark (<= 100k rows):
Builds a strictly balanced 73,600-row context tensor from normalized_dataset.parquet:
- 4,600 rows per class across all 16 classes (Classes 0..15).
- No oversampling, no undersampling (capped at bottleneck class mortgage = 4,623).
- Preserves winning natural Class 0 distribution (~25% dormant / ~75% active).
- Evaluates on May 2016 Ground-Truth Holdout (5,000 queries) on NVIDIA RTX 3090.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from tabdpt.model import TabDPTModel
from src.config import (
    CONTEXT_TENSOR_PATH,
    DATA_DIR,
    INDEX_TO_TARGET,
    NON_FEATURE_COLS,
    NORMALIZED_DATASET_PARQUET,
    SELECTED_15_TARGETS,
    Y_TENSOR_PATH,
)
from src.benchmark_model.data_loader import BenchmarkCohort, prepare_benchmark_datasets
from scripts.experiment_context_splits import load_tabdpt_model, sample_split_from_pool
from scripts.experiment_frankenstein import (
    run_gpu_forward_pass,
    train_lgbm_gatekeeper,
)
from scripts.experiment_distribution_gating import (
    compute_ap_at_k,
    precompute_rankings,
)
from scripts.experiment_may2016_benchmark import (
    evaluate_pipeline_config,
    evaluate_three_tier_metrics,
    load_may2016_cohort,
)


def build_scaled_context_tensor(samples_per_class: int = 4600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    tensor_path = DATA_DIR / "tensors" / f"context_scaled_{samples_per_class * 16}.pt"
    y_path = DATA_DIR / "tensors" / f"y_scaled_{samples_per_class * 16}.pt"

    if tensor_path.exists() and y_path.exists():
        print(f"[Scaled Context] Loading cached tensor from {tensor_path}...")
        X_ctx = torch.load(tensor_path, map_location="cpu").to(torch.float32).numpy()
        y_ctx = torch.load(y_path, map_location="cpu").to(torch.long).numpy()
        return X_ctx, y_ctx

    t0 = time.time()
    print(f"[Scaled Context] Sampling {samples_per_class} rows per class from normalized_dataset.parquet...")
    df = pl.scan_parquet(str(NORMALIZED_DATASET_PARQUET))

    schema = df.collect_schema()
    feature_cols = [c for c in schema.names() if c not in NON_FEATURE_COLS and c != "added_products"]

    sampled_dfs = []
    for c in range(16):
        df_c = df.filter(pl.col("target_class") == c).collect()
        df_c_sample = df_c.sample(n=samples_per_class, seed=seed)
        sampled_dfs.append(df_c_sample)

    df_all = pl.concat(sampled_dfs)
    print(f"  Sampled {len(df_all)} rows across 16 classes in {time.time() - t0:.2f}s.")

    X_mat = df_all.select(feature_cols).to_numpy()
    X_mat = np.clip(X_mat, -10.0, 10.0)
    padding = np.zeros((len(df_all), 128 - X_mat.shape[1]), dtype=np.float32)
    X_padded = np.hstack([X_mat, padding])
    y_vec = df_all["target_class"].to_numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df_all))
    X_padded = X_padded[perm]
    y_vec = y_vec[perm]

    # Save tensors
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(X_padded, dtype=torch.bfloat16), tensor_path)
    torch.save(torch.tensor(y_vec, dtype=torch.long), y_path)
    print(f"  Saved scaled context tensors to {tensor_path} ({len(y_vec)} rows).")

    return X_padded, y_vec


def run_scaled_benchmark(device: str = "cuda"):
    output_dir = DATA_DIR / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("  SCALED CONTEXT BENCHMARK (73,600 ROWS) VS. 5K BASELINE & UNIFORM")
    print("================================================================================")
    print(f"  Device:          {device}")
    print(f"  Cohort Date:     May 28, 2016 (Holdout)")
    print(f"  Cohort Size:     5,000 queries")
    print(f"  Scaled Context:  73,600 rows (4,600 rows/class, 16 classes strictly uniform)")
    print("================================================================================\n")

    # 1. Load May 2016 Cohort
    print("[1/5] Loading May 2016 evaluation cohort...")
    X_test, y_test, test_cohort = load_may2016_cohort()
    print(f"  May 2016 cohort loaded: {len(y_test)} queries.\n")

    # 2. Build / Load Scaled 73.6k Context
    print("[2/5] Building strictly balanced 73,600-row context tensor...")
    X_ctx_scaled, y_ctx_scaled = build_scaled_context_tensor(samples_per_class=4600, seed=42)
    print(f"  Context ready: shape={X_ctx_scaled.shape}, classes={len(np.unique(y_ctx_scaled))}.\n")

    # 3. Train LightGBM Gatekeeper on CPU
    print("[3/5] Training LightGBM Gatekeeper on CPU...")
    X_train_pool, y_train_pool, _ = prepare_benchmark_datasets(
        train_ratio=0.8, eval_sample_size=5000, seed=42
    )
    lgbm_model = train_lgbm_gatekeeper(X_train_pool, y_train_pool, seed=42)
    p_buy_lgbm = lgbm_model.predict_proba(test_cohort.X[:, :80])[:, 1]
    y_test_bin = (test_cohort.y > 0).astype(int)
    test_auc = float(roc_auc_score(y_test_bin, p_buy_lgbm))
    print(f"  LightGBM ROC-AUC on May 2016: {test_auc:.4f}\n")

    # 4. Initialize TabDPT Model and Forward Pass on Scaled Context
    print(f"[4/5] Initializing TabDPTModel and executing forward pass with 73,600 context rows on {device}...")
    model = load_tabdpt_model(device=device)

    t0 = time.time()
    logits_scaled, probs_scaled, t_infer = run_gpu_forward_pass(
        model=model,
        X_ctx=X_ctx_scaled,
        y_ctx=y_ctx_scaled,
        X_test=test_cohort.X,
        device=device,
        batch_size=250,
    )
    print(f"  Scaled 73.6k context forward pass completed in {t_infer:.2f}s! Logits shape: {logits_scaled.shape}\n")

    # 5. Evaluate Scaled Context Configurations
    print("[5/5] Evaluating Scaled 73.6k Context against 5k Baseline & 5k Uniform...")
    base_recs_scaled = precompute_rankings(probs_scaled, test_cohort.held_products)
    preds_raw_scaled = np.argmax(probs_scaled, axis=1)

    eval_results = []

    # A. Scaled Context Raw Argmax
    res_scaled_raw = evaluate_three_tier_metrics(
        top1_preds=preds_raw_scaled,
        active_recs=base_recs_scaled,
        base_recs=base_recs_scaled,
        y_test=test_cohort.y,
        actual_products=test_cohort.actual_products,
    )
    res_scaled_raw["config_label"] = "Scaled Context 73.6k (Raw Argmax, No Gate)"
    res_scaled_raw["gpu_passes_saved_pct"] = 0.0
    eval_results.append(res_scaled_raw)

    # B. Scaled Context + Pure Gate 1C
    res_scaled_g1c = evaluate_pipeline_config(
        p_buy_lgbm=None,
        gamma=None,
        probs_tabdpt=probs_scaled,
        theta_noisy=5.0,
        theta_clean=1.2,
        cohort=test_cohort,
    )
    res_scaled_g1c["config_label"] = "Scaled Context 73.6k + Pure Gate 1C"
    eval_results.append(res_scaled_g1c)

    # C. Scaled Context + Pipeline (gamma=0.30)
    res_scaled_p30 = evaluate_pipeline_config(
        p_buy_lgbm=p_buy_lgbm,
        gamma=0.30,
        probs_tabdpt=probs_scaled,
        theta_noisy=5.0,
        theta_clean=1.2,
        cohort=test_cohort,
    )
    res_scaled_p30["config_label"] = "Scaled Context 73.6k + Pipeline (gamma=0.30) [Rec]"
    eval_results.append(res_scaled_p30)

    # D. Scaled Context + Pipeline (gamma=0.50)
    res_scaled_p50 = evaluate_pipeline_config(
        p_buy_lgbm=p_buy_lgbm,
        gamma=0.50,
        probs_tabdpt=probs_scaled,
        theta_noisy=5.0,
        theta_clean=1.2,
        cohort=test_cohort,
    )
    res_scaled_p50["config_label"] = "Scaled Context 73.6k + Pipeline (gamma=0.50) [High Hold]"
    eval_results.append(res_scaled_p50)

    # Load 5k benchmarks from previous run for direct comparison
    prev_json = output_dir / "may2016_benchmark_results.json"
    if prev_json.exists():
        with open(prev_json, "r") as f:
            prev_data = json.load(f)
            for r in prev_data.get("results", []):
                if any(k in r["config_label"] for k in ["Original Baseline", "Initial Uniform (split_uniform_classes, raw)", "Initial Uniform + Pipeline (gamma=0.30"]):
                    eval_results.insert(0, r)

    print("\nBenchmark Results Summary on May 2016 Holdout:")
    print(f"{'Configuration':<55} | {'Hold':<6} | {'Sniper':<7} | {'Overall':<7} | {'GPU Saved':<9} | {'Cond MAP@7':<10} | {'Uncond MAP':<10}")
    print("-" * 125)
    for r in eval_results:
        print(f"{r['config_label']:<55} | {r['non_buyer_top1_acc']:5.1f}% | {r['buyer_top1_acc']:6.1f}% | {r['overall_top1_acc']:6.1f}% | {r['gpu_passes_saved_pct']:8.1f}% | {r['conditional_alert_map7']:10.4f} | {r['unconditional_cached_map7']:10.4f}")

    # Save to JSON
    json_path = output_dir / "scaled_context_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cohort": "May 28, 2016 Holdout (5,000 queries)",
                "context_size": len(y_ctx_scaled),
                "samples_per_class": 4600,
                "gpu_infer_time_s": t_infer,
                "results": eval_results,
            },
            f,
            indent=2,
        )

    # Save Markdown Report
    generate_scaled_report(eval_results, output_dir / "scaled_context_benchmark_results.md", len(y_ctx_scaled), t_infer)
    print(f"\n[Done] Benchmark complete! Results saved to {json_path} and {output_dir / 'scaled_context_benchmark_results.md'}.")


def generate_scaled_report(results: list[dict], out_path: Path, n_ctx: int, infer_time: float):
    lines = [
        f"# Scaled Context Benchmark ({n_ctx:,} Rows): May 2016 Holdout",
        "",
        f"**Context Design**: Strictly uniform {n_ctx:,} rows across 16 classes (4,600 rows/class).",
        "**Class Capping**: Bottleneck class (`mortgage` = 4,623 available rows). All 15 other classes capped to match.",
        f"**Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM). Inference time: {infer_time:.2f}s.",
        "",
        "---",
        "",
        "## 1. Head-to-Head Performance Matrix",
        "",
        "| Architecture / Configuration | Non-Buyer Hold | Buyer Top-1 Sniper | Overall Top-1 Acc | GPU Passes Saved % | **Conditional Alert MAP@7** | Unconditional Cached MAP@7 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lines.append(
            f"| **{r['config_label']}** | **{r['non_buyer_top1_acc']}%** | **{r['buyer_top1_acc']}%** | **{r['overall_top1_acc']}%** | **{r['gpu_passes_saved_pct']}%** | **{r['conditional_alert_map7']:.4f}** | {r['unconditional_cached_map7']:.4f} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_scaled_benchmark(device=device)

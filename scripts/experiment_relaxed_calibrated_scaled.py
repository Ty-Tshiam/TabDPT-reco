"""
Relaxed LightGBM + Tightened Gate 1C on Maximum Scaled Context (73,968 rows):
Context: Exactly 16 x 4,623 = 73,968 rows (100% of available mortgage rows, perfectly balanced).

Explores the new calibrated operating regime on May 2016 holdout:
1. Relaxed LightGBM (gamma in {0.15, 0.20, 0.25, 0.30}) to minimize false negatives on true buyers.
2. Tightened Gate 1C (theta_clean in {1.5, 1.8, 2.0}, theta_noisy in {5.0, 6.0, 7.0}) to exploit
   TabDPT's superior 61.5% in-context sniper accuracy.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from tabdpt.model import TabDPTModel
from src.config import (
    DATA_DIR,
    INDEX_TO_TARGET,
    SELECTED_15_TARGETS,
)
from src.benchmark_model.data_loader import BenchmarkCohort, prepare_benchmark_datasets
from scripts.experiment_context_splits import load_tabdpt_model
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


def run_relaxed_calibrated_experiment(device: str = "cuda"):
    output_dir = DATA_DIR / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("  MAXIMUM SCALED CONTEXT (73,968 ROWS) & RELAXED/TIGHTENED CALIBRATION")
    print("================================================================================")
    print(f"  Device:         {device}")
    print(f"  Cohort:         May 28, 2016 Holdout (5,000 queries)")
    print(f"  Context Size:   73,968 rows (16 x 4,623 rows strictly uniform)")
    print("================================================================================\n")

    # 1. Load May 2016 Cohort
    print("[1/5] Loading May 2016 evaluation cohort...")
    X_test, y_test, test_cohort = load_may2016_cohort()
    print(f"  May 2016 cohort loaded: {len(y_test)} queries.\n")

    # 2. Load 73,968 Context Tensor
    print("[2/5] Loading 73,968-row context tensor...")
    ctx_path = DATA_DIR / "tensors" / "context.pt"
    if not ctx_path.exists():
        ctx_path = DATA_DIR / "tensors" / "context_scaled_73968.pt"
    y_ctx_path = DATA_DIR / "tensors" / "y.pt"
    if not y_ctx_path.exists():
        y_ctx_path = DATA_DIR / "tensors" / "y_scaled_73968.pt"
    X_ctx = torch.load(ctx_path, map_location="cpu").to(torch.float32).numpy()
    y_ctx = torch.load(y_ctx_path, map_location="cpu").to(torch.long).numpy()
    print(f"  Context loaded: shape={X_ctx.shape}, classes={len(np.unique(y_ctx))}.\n")

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

    # 4. Forward Pass on RTX 3090
    print(f"[4/5] Executing forward pass with 73,968 context rows on {device}...")
    model = load_tabdpt_model(device=device)
    logits_scaled, probs_scaled, t_infer = run_gpu_forward_pass(
        model=model,
        X_ctx=X_ctx,
        y_ctx=y_ctx,
        X_test=test_cohort.X,
        device=device,
        batch_size=250,
    )
    print(f"  73.9k context forward pass completed in {t_infer:.2f}s! Logits shape: {logits_scaled.shape}\n")

    # 5. Sweep Calibration Matrix
    print("[5/5] Sweeping Relaxed LightGBM + Tightened Gate 1C configurations...")
    base_recs_scaled = precompute_rankings(probs_scaled, test_cohort.held_products)
    preds_raw = np.argmax(probs_scaled, axis=1)

    eval_results = []

    # Benchmark references
    # Raw Argmax
    raw_res = evaluate_three_tier_metrics(
        top1_preds=preds_raw,
        active_recs=base_recs_scaled,
        base_recs=base_recs_scaled,
        y_test=test_cohort.y,
        actual_products=test_cohort.actual_products,
    )
    raw_res["config_label"] = "73.9k Context (Raw Argmax, No Gate)"
    raw_res["gpu_passes_saved_pct"] = 0.0
    eval_results.append(raw_res)

    # Sweep grid
    sweep_configs = [
        # (label, gamma, theta_noisy, theta_clean)
        # Previous sweet spot
        ("73.9k Context (gamma=0.30, theta_n=5.0, theta_c=1.2)", 0.30, 5.0, 1.2),
        # User proposed: Relaxed gamma=0.15 + Tightened Gate 1C
        ("73.9k Context (gamma=0.15, theta_n=6.0, theta_c=1.8)", 0.15, 6.0, 1.8),
        ("73.9k Context (gamma=0.15, theta_n=7.0, theta_c=2.0)", 0.15, 7.0, 2.0),
        # User proposed: Relaxed gamma=0.20 + Tightened Gate 1C
        ("73.9k Context (gamma=0.20, theta_n=5.0, theta_c=1.5)", 0.20, 5.0, 1.5),
        ("73.9k Context (gamma=0.20, theta_n=6.0, theta_c=1.8) [Candidate A]", 0.20, 6.0, 1.8),
        ("73.9k Context (gamma=0.20, theta_n=7.0, theta_c=2.0) [Candidate B]", 0.20, 7.0, 2.0),
        # Relaxed gamma=0.25 + Tightened Gate 1C
        ("73.9k Context (gamma=0.25, theta_n=6.0, theta_c=1.8)", 0.25, 6.0, 1.8),
        ("73.9k Context (gamma=0.25, theta_n=7.0, theta_c=2.0)", 0.25, 7.0, 2.0),
        # Standard gamma=0.30 with tightened theta
        ("73.9k Context (gamma=0.30, theta_n=6.0, theta_c=1.8) [Candidate C]", 0.30, 6.0, 1.8),
        ("73.9k Context (gamma=0.30, theta_n=7.0, theta_c=2.0)", 0.30, 7.0, 2.0),
        # Ultra Hold gamma=0.40
        ("73.9k Context (gamma=0.40, theta_n=6.0, theta_c=1.8)", 0.40, 6.0, 1.8),
    ]

    for label, gamma, th_noisy, th_clean in sweep_configs:
        r = evaluate_pipeline_config(
            p_buy_lgbm=p_buy_lgbm,
            gamma=gamma,
            probs_tabdpt=probs_scaled,
            theta_noisy=th_noisy,
            theta_clean=th_clean,
            cohort=test_cohort,
        )
        r["config_label"] = label
        eval_results.append(r)

    # Insert previous baseline for direct comparison
    eval_results.insert(0, {
        "config_label": "5k Original Baseline (baseline_default, 40% C0)",
        "non_buyer_top1_acc": 95.34,
        "buyer_top1_acc": 39.72,
        "overall_top1_acc": 64.08,
        "gpu_passes_saved_pct": 0.0,
        "conditional_alert_map7": 0.8522,
        "unconditional_cached_map7": 0.7284,
        "coverage_weighted_map7": 0.4412,
        "alert_fired_pct": 31.1,
    })

    print(f"\n{'Configuration':<62} | {'Hold':<6} | {'Sniper':<7} | {'Overall':<7} | {'GPU Saved':<9} | {'Cond MAP@7':<10} | {'Uncond MAP':<10}")
    print("-" * 135)
    for r in eval_results:
        print(f"{r['config_label']:<62} | {r['non_buyer_top1_acc']:5.1f}% | {r['buyer_top1_acc']:6.1f}% | {r['overall_top1_acc']:6.1f}% | {r['gpu_passes_saved_pct']:8.1f}% | {r['conditional_alert_map7']:10.4f} | {r['unconditional_cached_map7']:10.4f}")

    # Save to JSON
    json_path = output_dir / "max_scaled_calibration_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cohort": "May 28, 2016 Holdout (5,000 queries)",
                "context_size": len(y_ctx),
                "samples_per_class": 4623,
                "gpu_infer_time_s": t_infer,
                "results": eval_results,
            },
            f,
            indent=2,
        )

    # Save Markdown Report
    generate_calibration_report(eval_results, output_dir / "max_scaled_calibration_results.md", len(y_ctx), t_infer)
    print(f"\n[Done] Experiments completed! Results saved to {json_path} and {output_dir / 'max_scaled_calibration_results.md'}.")


def generate_calibration_report(results: list[dict], out_path: Path, n_ctx: int, infer_time: float):
    lines = [
        f"# Maximum Scaled Context Benchmark ({n_ctx:,} Rows) & Calibration Frontier",
        "",
        f"**Context Setup**: 16 classes x 4,623 rows strictly uniform = {n_ctx:,} rows.",
        "**Bottleneck Class**: 100% of all available `mortgage` rows in Santander history.",
        f"**Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM). Inference: {infer_time:.2f}s.",
        "",
        "---",
        "",
        "## 1. Executive Performance Table",
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
    run_relaxed_calibrated_experiment(device=device)

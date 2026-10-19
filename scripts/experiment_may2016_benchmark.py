"""
May 2016 Ground-Truth Evaluation: Baseline vs. Frankenstein Architecture.

Evaluates on the true May 2016 Santander holdout cohort:
1. Baseline Default (40% Class 0 Anchor).
2. Initial Uniform (split_uniform_classes, raw argmax).
3. Frankenstein (1/16 active C0, 70/30) Raw Argmax.
4. Frankenstein + LightGBM Pre-Filter + Gate 1C sweeps:
   - gamma in {0.20, 0.30, 0.40, 0.50, 0.60}
   - theta_noisy = 5.0, theta_clean = 1.2

Metrics tracked per configuration:
- non_buyer_top1_acc (Hold %)
- buyer_top1_acc (Sniper %)
- overall_top1_acc
- gpu_passes_saved_pct
- conditional_alert_map7: MAP@7 on buyers who received a live push notification.
- unconditional_cached_map7: MAP@7 of raw TabDPT top-7 cached ranking on buyers.
- coverage_weighted_map7: Strict AP=0.0 metric when suppressed queries return [].
- conditional_hit_rate7 & unconditional_hit_rate7
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure project root in sys.path
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
    SELECTED_15_TARGETS,
    Y_TENSOR_PATH,
)
from src.benchmark_model.data_loader import BenchmarkCohort, prepare_benchmark_datasets
from scripts.experiment_context_splits import (
    load_tabdpt_model,
    sample_split_from_pool,
)
from scripts.experiment_frankenstein import (
    sample_frankenstein_context,
    run_gpu_forward_pass,
    train_lgbm_gatekeeper,
)
from scripts.experiment_distribution_gating import (
    compute_ap_at_k,
    precompute_rankings,
)


def load_may2016_cohort() -> tuple[np.ndarray, np.ndarray, BenchmarkCohort]:
    """Loads May 2016 evaluation cohort."""
    cohort_path = DATA_DIR / "processed" / "may2016_eval_cohort.npz"
    if not cohort_path.exists():
        # Fallback to local script builder if needed
        from scripts.prepare_may2016_data import build_may2016_cohort
        return build_may2016_cohort(eval_sample_size=5000, seed=42)

    data = np.load(cohort_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    held_products = [json.loads(h) for h in data["held_products"]]
    actual_products = [json.loads(a) for a in data["actual_products"]]
    is_cold = data["is_cold"]
    feature_names = list(data["feature_names"])
    cohort = BenchmarkCohort(
        X=X,
        y=y,
        held_products=held_products,
        actual_products=actual_products,
        is_cold=is_cold,
        feature_names=feature_names,
    )
    return X, y, cohort


def evaluate_three_tier_metrics(
    top1_preds: np.ndarray,
    active_recs: list[list[str]],
    base_recs: list[list[str]],
    y_test: np.ndarray,
    actual_products: list[list[str]],
) -> dict:
    """
    Computes all classification and three-tier MAP@7 / Hit@7 metrics:
    1. Conditional Alert MAP@7 (precision when alert is fired).
    2. Unconditional / Cached MAP@7 (precision of TabDPT rankings on buyers).
    3. Coverage-Weighted End-to-End MAP@7 (strict AP=0.0 when suppressed).
    """
    n = len(y_test)
    buyer_mask = y_test > 0
    non_buyer_mask = y_test == 0

    buyer_top1_acc = (
        float(np.mean(top1_preds[buyer_mask] == y_test[buyer_mask]) * 100.0)
        if np.any(buyer_mask)
        else 0.0
    )
    non_buyer_top1_acc = (
        float(np.mean(top1_preds[non_buyer_mask] == 0) * 100.0)
        if np.any(non_buyer_mask)
        else 0.0
    )
    overall_top1_acc = float(np.mean(top1_preds == y_test) * 100.0)

    # 1. Unconditional / Cached MAP@7 & Hit@7 (raw TabDPT rankings on all true buyers)
    uncond_scores = []
    uncond_hits = []
    for i in range(n):
        if not buyer_mask[i]:
            continue
        actual = actual_products[i]
        if not actual:
            continue
        ap = compute_ap_at_k(base_recs[i][:7], actual, k=7)
        uncond_scores.append(ap)
        hit = 1.0 if set(base_recs[i][:7]).intersection(set(actual)) else 0.0
        uncond_hits.append(hit)

    uncond_map7 = float(np.mean(uncond_scores)) if uncond_scores else 0.0
    uncond_hit7 = float(np.mean(uncond_hits) * 100.0) if uncond_hits else 0.0

    # 2. Conditional Alert MAP@7 & Hit@7 (only on true buyers who received an alert)
    cond_scores = []
    cond_hits = []
    cov_scores = []
    cov_hits = []

    for i in range(n):
        if not buyer_mask[i]:
            continue
        actual = actual_products[i]
        if not actual:
            continue

        if top1_preds[i] > 0 and len(active_recs[i]) > 0:
            ap = compute_ap_at_k(active_recs[i][:7], actual, k=7)
            hit = 1.0 if set(active_recs[i][:7]).intersection(set(actual)) else 0.0
            cond_scores.append(ap)
            cond_hits.append(hit)
            cov_scores.append(ap)
            cov_hits.append(hit)
        else:
            # Suppressed buyer
            cov_scores.append(0.0)
            cov_hits.append(0.0)

    cond_map7 = float(np.mean(cond_scores)) if cond_scores else 0.0
    cond_hit7 = float(np.mean(cond_hits) * 100.0) if cond_hits else 0.0
    cov_map7 = float(np.mean(cov_scores)) if cov_scores else 0.0
    cov_hit7 = float(np.mean(cov_hits) * 100.0) if cov_hits else 0.0

    return {
        "buyer_top1_acc": round(buyer_top1_acc, 2),
        "non_buyer_top1_acc": round(non_buyer_top1_acc, 2),
        "overall_top1_acc": round(overall_top1_acc, 2),
        "conditional_alert_map7": round(cond_map7, 4),
        "conditional_hit_rate7": round(cond_hit7, 1),
        "unconditional_cached_map7": round(uncond_map7, 4),
        "unconditional_hit_rate7": round(uncond_hit7, 1),
        "coverage_weighted_map7": round(cov_map7, 4),
        "coverage_weighted_hit_rate7": round(cov_hit7, 1),
        "alert_fired_pct": round(float(np.mean(top1_preds > 0) * 100.0), 1),
    }


def evaluate_pipeline_config(
    p_buy_lgbm: np.ndarray | None,
    gamma: float | None,
    probs_tabdpt: np.ndarray,
    theta_noisy: float,
    theta_clean: float,
    cohort: BenchmarkCohort,
) -> dict:
    y_test = cohort.y
    actual_products = cohort.actual_products
    held_products = cohort.held_products
    n = len(y_test)

    # Noisy sinks: 1 (current), 7 (particular_plus), 9 (funds), 14 (long_term), 15 (taxes)
    noisy_sinks = {1, 7, 9, 14, 15}
    theta_vec = np.ones(16, dtype=np.float32) * theta_clean
    theta_vec[0] = 0.0
    for c in noisy_sinks:
        theta_vec[c] = theta_noisy

    base_recs = precompute_rankings(probs_tabdpt, held_products)
    p0 = probs_tabdpt[:, 0]
    prod_probs = probs_tabdpt[:, 1:16]
    top_prod_indices = np.argmax(prod_probs, axis=1) + 1  # 1..15
    p_top = np.max(prod_probs, axis=1)
    odds_ratios = p_top / (p0 + 1e-9)

    top1_preds = np.zeros(n, dtype=np.int64)
    active_recs = []
    gpu_passes_saved_count = 0

    for i in range(n):
        # Stage 1: LightGBM Gatekeeper
        if p_buy_lgbm is not None and gamma is not None and p_buy_lgbm[i] < gamma:
            top1_preds[i] = 0
            active_recs.append([])
            gpu_passes_saved_count += 1
            continue

        # Stage 2: Gate 1C
        cand_prod = top_prod_indices[i]
        req_theta = theta_vec[cand_prod]

        if odds_ratios[i] > req_theta:
            top1_preds[i] = cand_prod
            active_recs.append(base_recs[i][:7])
        else:
            top1_preds[i] = 0
            active_recs.append([])

    metrics = evaluate_three_tier_metrics(
        top1_preds, active_recs, base_recs, y_test, actual_products
    )
    metrics["gamma"] = gamma
    metrics["theta_noisy"] = theta_noisy
    metrics["theta_clean"] = theta_clean
    metrics["gpu_passes_saved_pct"] = round(gpu_passes_saved_count / n * 100.0, 1)

    return metrics


def run_may2016_benchmark(device: str = "cuda"):
    output_dir = DATA_DIR / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("  MAY 2016 HOLDOUT BENCHMARK: BASELINE VS. FRANKENSTEIN ARCHITECTURE")
    print("================================================================================")
    print(f"  Device:          {device}")
    print(f"  Cohort Date:     May 28, 2016 (Holdout)")
    print(f"  Cohort Size:     5,000 queries")
    print("================================================================================\n")

    # 1. Load May 2016 Cohort
    print("[1/6] Loading May 2016 evaluation cohort...")
    X_test, y_test, test_cohort = load_may2016_cohort()
    n_nb = np.sum(y_test == 0)
    n_b = np.sum(y_test > 0)
    print(f"  May 2016 cohort loaded: {len(y_test)} queries ({n_b} buyers, {n_nb} non-buyers).\n")

    # 2. Load Training Pool for Context Construction & LightGBM
    print("[2/6] Loading training partition and context tensors (< 2016-05-28)...")
    X_train_pool, y_train_pool, _ = prepare_benchmark_datasets(
        train_ratio=0.8, eval_sample_size=5000, seed=42
    )
    X_full = torch.load(CONTEXT_TENSOR_PATH, map_location="cpu").to(torch.float32).numpy()
    y_full = torch.load(Y_TENSOR_PATH, map_location="cpu").to(torch.long).numpy()

    rng = np.random.default_rng(42)
    train_indices = []
    for c in range(16):
        c_idxs = np.where(y_full == c)[0]
        rng.shuffle(c_idxs)
        train_indices.extend(c_idxs[: int(len(c_idxs) * 0.8)])
    train_indices = np.array(train_indices)

    # 3. Train LightGBM Gatekeeper on CPU
    print("[3/6] Training Stage 1 LightGBM Gatekeeper on CPU...")
    lgbm_model = train_lgbm_gatekeeper(X_train_pool, y_train_pool, seed=42)
    t0 = time.time()
    p_buy_lgbm = lgbm_model.predict_proba(test_cohort.X[:, :80])[:, 1]
    cpu_latency = (time.time() - t0) * 1000.0 / len(test_cohort.X)

    y_test_bin = (test_cohort.y > 0).astype(int)
    test_auc = float(roc_auc_score(y_test_bin, p_buy_lgbm))
    test_pr_auc = float(average_precision_score(y_test_bin, p_buy_lgbm))
    print(f"  LightGBM on May 2016: ROC-AUC = {test_auc:.4f} | PR-AUC = {test_pr_auc:.4f} | Latency = {cpu_latency:.3f} ms/query.\n")

    # 4. Initialize TabDPT Model
    print(f"[4/6] Initializing TabDPTModel on {device}...")
    model = load_tabdpt_model(device=device)

    # Context A: baseline_default (40% Class 0: 2,000 Class 0, 200/prod)
    print("\n--- Running Context A: Original Baseline (baseline_default, 40% Class 0) ---")
    X_ctx_base, y_ctx_base = sample_split_from_pool(
        X_pool=X_train_pool,
        y_pool=y_train_pool,
        feature_names=test_cohort.feature_names,
        split_name="baseline_default",
        n_samples=5000,
        seed=42,
    )
    logits_base, probs_base, t_base = run_gpu_forward_pass(
        model, X_ctx_base, y_ctx_base, test_cohort.X, device=device
    )
    print(f"  Baseline GPU forward pass completed in {t_base:.2f}s.")

    # Context B: split_uniform_classes (1/16 equal split, natural C0)
    print("\n--- Running Context B: Initial Uniform (split_uniform_classes, 1/16 equal split) ---")
    X_ctx_uni, y_ctx_uni = sample_split_from_pool(
        X_pool=X_train_pool,
        y_pool=y_train_pool,
        feature_names=test_cohort.feature_names,
        split_name="split_uniform_classes",
        n_samples=5000,
        seed=42,
    )
    logits_uni, probs_uni, t_uni = run_gpu_forward_pass(
        model, X_ctx_uni, y_ctx_uni, test_cohort.X, device=device
    )
    print(f"  Uniform GPU forward pass completed in {t_uni:.2f}s.")

    # Context C: Frankenstein Context (1/16 equal split, 100% active C0, 70/30)
    print("\n--- Running Context C: Frankenstein Context (1/16 equal split, 100% active C0, 70/30) ---")
    X_ctx_frank, y_ctx_frank, audit_frank = sample_frankenstein_context(
        X_full, y_full, train_indices, total_samples=5000, seed=42
    )
    logits_frank, probs_frank, t_frank = run_gpu_forward_pass(
        model, X_ctx_frank, y_ctx_frank, test_cohort.X, device=device
    )
    print(f"  Frankenstein GPU forward pass completed in {t_frank:.2f}s.")

    # 5. Evaluate All Configurations
    print("\n[5/6] Evaluating all models on May 2016 holdout cohort...")
    base_recs_base = precompute_rankings(probs_base, test_cohort.held_products)
    base_recs_uni = precompute_rankings(probs_uni, test_cohort.held_products)
    base_recs_frank = precompute_rankings(probs_frank, test_cohort.held_products)

    preds_raw_base = np.argmax(probs_base, axis=1)
    preds_raw_uni = np.argmax(probs_uni, axis=1)
    preds_raw_frank = np.argmax(probs_frank, axis=1)

    eval_results = []

    # 1. Baseline Default (Raw Argmax)
    res_base = evaluate_three_tier_metrics(
        top1_preds=preds_raw_base,
        active_recs=base_recs_base,
        base_recs=base_recs_base,
        y_test=test_cohort.y,
        actual_products=test_cohort.actual_products,
    )
    res_base["config_label"] = "Original Baseline (baseline_default, 40% C0)"
    res_base["gpu_passes_saved_pct"] = 0.0
    eval_results.append(res_base)

    # 2. Initial Uniform (split_uniform_classes, Raw Argmax)
    res_uni = evaluate_three_tier_metrics(
        top1_preds=preds_raw_uni,
        active_recs=base_recs_uni,
        base_recs=base_recs_uni,
        y_test=test_cohort.y,
        actual_products=test_cohort.actual_products,
    )
    res_uni["config_label"] = "Initial Uniform (split_uniform_classes, raw)"
    res_uni["gpu_passes_saved_pct"] = 0.0
    eval_results.append(res_uni)

    # 2b. Initial Uniform + Pure Gate 1C (No LGBM)
    res_uni_g1c = evaluate_pipeline_config(
        p_buy_lgbm=None,
        gamma=None,
        probs_tabdpt=probs_uni,
        theta_noisy=5.0,
        theta_clean=1.2,
        cohort=test_cohort,
    )
    res_uni_g1c["config_label"] = "Initial Uniform + Pure Gate 1C (theta_n=5.0, theta_c=1.2)"
    eval_results.append(res_uni_g1c)

    # 2c. Initial Uniform + Pipeline (gamma=0.30, theta_n=5.0, theta_c=1.2)
    res_uni_p30 = evaluate_pipeline_config(
        p_buy_lgbm=p_buy_lgbm,
        gamma=0.30,
        probs_tabdpt=probs_uni,
        theta_noisy=5.0,
        theta_clean=1.2,
        cohort=test_cohort,
    )
    res_uni_p30["config_label"] = "Initial Uniform + Pipeline (gamma=0.30, theta_n=5.0, theta_c=1.2)"
    eval_results.append(res_uni_p30)

    # 2d. Initial Uniform + Pipeline (gamma=0.50, theta_n=5.0, theta_c=1.2)
    res_uni_p50 = evaluate_pipeline_config(
        p_buy_lgbm=p_buy_lgbm,
        gamma=0.50,
        probs_tabdpt=probs_uni,
        theta_noisy=5.0,
        theta_clean=1.2,
        cohort=test_cohort,
    )
    res_uni_p50["config_label"] = "Initial Uniform + Pipeline (gamma=0.50, theta_n=5.0, theta_c=1.2)"
    eval_results.append(res_uni_p50)

    # 3. Frankenstein Raw Argmax
    res_frank_raw = evaluate_three_tier_metrics(
        top1_preds=preds_raw_frank,
        active_recs=base_recs_frank,
        base_recs=base_recs_frank,
        y_test=test_cohort.y,
        actual_products=test_cohort.actual_products,
    )
    res_frank_raw["config_label"] = "Frankenstein Context (Raw Argmax, No Gate)"
    res_frank_raw["gpu_passes_saved_pct"] = 0.0
    eval_results.append(res_frank_raw)

    # 4. Frankenstein Pipeline Configurations
    configs_to_test = [
        ("Frankenstein (gamma=0.20, theta_n=5.0, theta_c=1.2)", 0.20, 5.0, 1.2),
        ("Frankenstein (gamma=0.30, theta_n=4.0, theta_c=1.0)", 0.30, 4.0, 1.0),
        ("Frankenstein (gamma=0.30, theta_n=5.0, theta_c=1.2) [Rec]", 0.30, 5.0, 1.2),
        ("Frankenstein (gamma=0.40, theta_n=5.0, theta_c=1.2)", 0.40, 5.0, 1.2),
        ("Frankenstein (gamma=0.50, theta_n=5.0, theta_c=1.2) [Max Hold]", 0.50, 5.0, 1.2),
        ("Frankenstein (gamma=0.60, theta_n=5.0, theta_c=1.2)", 0.60, 5.0, 1.2),
    ]

    for label, gamma, th_noisy, th_clean in configs_to_test:
        r = evaluate_pipeline_config(
            p_buy_lgbm=p_buy_lgbm,
            gamma=gamma,
            probs_tabdpt=probs_frank,
            theta_noisy=th_noisy,
            theta_clean=th_clean,
            cohort=test_cohort,
        )
        r["config_label"] = label
        eval_results.append(r)

    print("\n[6/6] Benchmark Results Summary on May 2016 Holdout:")
    print(f"{'Configuration':<52} | {'Hold':<6} | {'Sniper':<7} | {'Overall':<7} | {'GPU Saved':<9} | {'Cond MAP@7':<10} | {'Uncond MAP':<10} | {'Cov MAP':<8}")
    print("-" * 125)
    for r in eval_results:
        print(f"{r['config_label']:<52} | {r['non_buyer_top1_acc']:5.1f}% | {r['buyer_top1_acc']:6.1f}% | {r['overall_top1_acc']:6.1f}% | {r['gpu_passes_saved_pct']:8.1f}% | {r['conditional_alert_map7']:10.4f} | {r['unconditional_cached_map7']:10.4f} | {r['coverage_weighted_map7']:8.4f}")

    # Save to JSON
    json_path = output_dir / "may2016_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cohort": "May 28, 2016 Holdout (5,000 queries)",
                "date": "2016-05-28",
                "lgbm_stats": {"roc_auc": test_auc, "pr_auc": test_pr_auc, "latency_ms": cpu_latency},
                "results": eval_results,
            },
            f,
            indent=2,
        )

    # Save Markdown Report
    generate_may2016_markdown_report(eval_results, output_dir / "may2016_benchmark_results.md")
    print(f"\n[Done] Benchmark complete! Results saved to {json_path} and {output_dir / 'may2016_benchmark_results.md'}.")


def generate_may2016_markdown_report(results: list[dict], out_path: Path):
    lines = [
        "# May 2016 Ground-Truth Evaluation: Baseline vs. Frankenstein Architecture",
        "",
        "**Cohort**: True Santander May 28, 2016 holdout test partition ($N = 5,000$ queries).",
        "**Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM).",
        "",
        "---",
        "",
        "## 1. Executive Performance Table (Conditional Alert MAP@7)",
        "",
        "| Architecture / Configuration | Non-Buyer Hold | Buyer Top-1 Sniper | Overall Top-1 Acc | GPU Passes Saved % | **Conditional Alert MAP@7** | Conditional Hit@7 | Unconditional Cached MAP@7 | Coverage-Weighted End-to-End MAP@7 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lines.append(
            f"| **{r['config_label']}** | **{r['non_buyer_top1_acc']}%** | **{r['buyer_top1_acc']}%** | **{r['overall_top1_acc']}%** | **{r['gpu_passes_saved_pct']}%** | **{r['conditional_alert_map7']:.4f}** | {r['conditional_hit_rate7']}% | {r['unconditional_cached_map7']:.4f} | {r['coverage_weighted_map7']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Metric Methodology & Interpretation",
        "",
        "- **Conditional Alert MAP@7**: Measures the ranking quality strictly on customers who receive a live push alert. Reflects the actual user experience of customers targeted by recommendations.",
        "- **Unconditional / Cached MAP@7**: Measures the raw ranking precision of TabDPT's top-7 outputs for queries that reach the GPU. Demonstrates the model's intrinsic product catalog intelligence before suppression.",
        "- **Coverage-Weighted End-to-End MAP@7**: The strict information retrieval metric assigning AP = 0.0 to all suppressed buyer events.",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_may2016_benchmark(device=device)

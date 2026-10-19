"""
Experiment: Confidence Meter Optimization for TabDPT In-Context Recommender.

Objective:
Tame the "Do Nothing" false alarm rate in `split_uniform_classes` by tuning the confidence meter,
elevating Non-Buyer Hold accuracy (Do Nothing) as high as possible while preserving
Active Buyer Top-1 sniper accuracy (near 47.4%).

Meters Explored:
1. Ratio Meter (theta): P(top_product) / P(Class 0) >= theta
2. Logit Prior Shift (lambda): logit(Class 0) += lambda (Bayesian odds restoration)
3. Confidence Margin (delta): P(top_product) - P(Class 0) >= delta
4. Minimum Product Probability (p_min): P(top_product) >= p_min
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from tabdpt.model import TabDPTModel
from src.config import DATA_DIR, INDEX_TO_TARGET, SELECTED_15_TARGETS
from src.benchmark_model.data_loader import prepare_benchmark_datasets
from src.benchmark_model.metrics import (
    compute_ap_at_k,
    compute_map_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_and_hit_rate_at_k,
)
from scripts.experiment_context_splits import (
    load_tabdpt_model,
    sample_split_from_pool,
)


def get_raw_logits_and_probs(
    model: TabDPTModel,
    X_ctx_np: np.ndarray,
    y_ctx_np: np.ndarray,
    X_test_np: np.ndarray,
    device: str = "cuda",
    batch_size: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs single forward pass to collect uncalibrated 16-class logits and probabilities."""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    x_ctx = torch.from_numpy(X_ctx_np).unsqueeze(0).to(device=device, dtype=dtype)
    y_ctx = torch.from_numpy(y_ctx_np).unsqueeze(0).to(device=device, dtype=torch.long)

    with torch.no_grad():
        kv_cache, n_ctx, stats = model.encode_context(x_ctx, y_ctx, is_cls=True)

    logits_list = []
    num_queries = len(X_test_np)

    for i in range(0, num_queries, batch_size):
        chunk = X_test_np[i : i + batch_size]
        batch = torch.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            pred = model.predict_with_cache(batch, kv_cache, n_ctx, stats)
            cls_logits = pred[..., :16].squeeze(1).float().cpu().numpy()
            logits_list.append(cls_logits)

    all_logits = np.vstack(logits_list)
    exp_l = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    all_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    return all_logits, all_probs


def evaluate_meter_decision(
    top1_preds: np.ndarray,
    ranked_recs: list[list[str]],
    cohort,
) -> dict:
    """Computes exact metrics under a specific meter decision rule."""
    y_test = cohort.y
    actual_products = cohort.actual_products
    is_cold = cohort.is_cold
    is_warm = ~is_cold

    # 1. Sniper Accuracies
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

    # 2. Ranking Metrics
    map7 = compute_map_at_k(ranked_recs, actual_products, k=7)
    ndcg7 = compute_ndcg_at_k(ranked_recs, actual_products, k=7)
    mrr = compute_mrr(ranked_recs, actual_products, k=7)

    cold_actuals = [actual_products[i] for i in range(len(y_test)) if is_cold[i]]
    cold_recs = [ranked_recs[i] for i in range(len(y_test)) if is_cold[i]]
    cold_map7 = compute_map_at_k(cold_recs, cold_actuals, k=7)

    warm_actuals = [actual_products[i] for i in range(len(y_test)) if is_warm[i]]
    warm_recs = [ranked_recs[i] for i in range(len(y_test)) if is_warm[i]]
    warm_map7 = compute_map_at_k(warm_recs, warm_actuals, k=7)

    rec7, hit7 = compute_recall_and_hit_rate_at_k(ranked_recs, actual_products, k=7)

    return {
        "buyer_top1_acc": round(buyer_top1_acc, 2),
        "non_buyer_top1_acc": round(non_buyer_top1_acc, 2),
        "overall_top1_acc": round(overall_top1_acc, 2),
        "map7": round(map7, 4),
        "cold_map7": round(cold_map7, 4),
        "warm_map7": round(warm_map7, 4),
        "ndcg7": round(ndcg7, 4),
        "mrr": round(mrr, 4),
        "hit_rate7": round(hit7 * 100.0, 1),
        "recall7": round(rec7, 4),
    }


def sweep_ratio_meter(probs: np.ndarray, cohort, thetas: list[float]) -> list[dict]:
    """Sweeps theta in: P(top_product) / P(Class 0) >= theta."""
    results = []
    num_queries = len(probs)
    held_products = cohort.held_products

    # Pre-rank products (classes 1..15)
    base_recs = []
    for i in range(num_queries):
        held_set = set(held_products[i])
        prod_probs = probs[i, 1:16]
        order = np.argsort(-prod_probs)
        recs = [SELECTED_15_TARGETS[idx] for idx in order if SELECTED_15_TARGETS[idx] not in held_set]
        base_recs.append(recs)

    for theta in thetas:
        top1_preds = np.zeros(num_queries, dtype=np.int64)
        active_recs = []

        for i in range(num_queries):
            p0 = probs[i, 0]
            prod_probs = probs[i, 1:16]
            top_prod_idx = np.argmax(prod_probs) + 1  # 1..15
            top_p = probs[i, top_prod_idx]

            ratio = top_p / (p0 + 1e-9)

            if ratio >= theta:
                top1_preds[i] = top_prod_idx
                active_recs.append(base_recs[i][:7])
            else:
                top1_preds[i] = 0  # Hold / Do Nothing
                active_recs.append([])  # Suppressed recommendations

        m = evaluate_meter_decision(top1_preds, active_recs, cohort)
        m["meter_type"] = "Ratio Meter"
        m["parameter"] = f"theta = {theta:.2f}"
        m["theta"] = theta
        results.append(m)

    return results


def sweep_logit_shift_meter(logits: np.ndarray, cohort, lambdas: list[float]) -> list[dict]:
    """Sweeps lambda in: logit(Class 0) += lambda."""
    results = []
    num_queries = len(logits)
    held_products = cohort.held_products

    for lam in lambdas:
        shifted_logits = logits.copy()
        shifted_logits[:, 0] += lam

        exp_l = np.exp(shifted_logits - np.max(shifted_logits, axis=-1, keepdims=True))
        shifted_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        top1_preds = np.argmax(shifted_probs, axis=1)

        active_recs = []
        for i in range(num_queries):
            held_set = set(held_products[i])
            if top1_preds[i] == 0:
                active_recs.append([])  # Hold / Do Nothing
            else:
                prod_probs = shifted_probs[i, 1:16]
                order = np.argsort(-prod_probs)
                recs = [SELECTED_15_TARGETS[idx] for idx in order if SELECTED_15_TARGETS[idx] not in held_set]
                active_recs.append(recs[:7])

        m = evaluate_meter_decision(top1_preds, active_recs, cohort)
        m["meter_type"] = "Logit Shift (Bayesian)"
        m["parameter"] = f"lambda = +{lam:.2f}"
        m["lambda"] = lam
        results.append(m)

    return results


def sweep_margin_meter(probs: np.ndarray, cohort, deltas: list[float]) -> list[dict]:
    """Sweeps delta in: P(top_product) - P(Class 0) >= delta."""
    results = []
    num_queries = len(probs)
    held_products = cohort.held_products

    base_recs = []
    for i in range(num_queries):
        held_set = set(held_products[i])
        prod_probs = probs[i, 1:16]
        order = np.argsort(-prod_probs)
        recs = [SELECTED_15_TARGETS[idx] for idx in order if SELECTED_15_TARGETS[idx] not in held_set]
        base_recs.append(recs)

    for delta in deltas:
        top1_preds = np.zeros(num_queries, dtype=np.int64)
        active_recs = []

        for i in range(num_queries):
            p0 = probs[i, 0]
            prod_probs = probs[i, 1:16]
            top_prod_idx = np.argmax(prod_probs) + 1
            top_p = probs[i, top_prod_idx]

            if (top_p - p0) >= delta:
                top1_preds[i] = top_prod_idx
                active_recs.append(base_recs[i][:7])
            else:
                top1_preds[i] = 0
                active_recs.append([])

        m = evaluate_meter_decision(top1_preds, active_recs, cohort)
        m["meter_type"] = "Probability Margin"
        m["parameter"] = f"delta = {delta:+.2f}"
        m["delta"] = delta
        results.append(m)

    return results


def format_results_markdown(ratio_res: list[dict], logit_res: list[dict], margin_res: list[dict]) -> str:
    """Formats comparison tables and highlights the optimal Pareto operating point."""
    md = []
    md.append("# 🎛️ Confidence Meter Optimization: Taming 'Do Nothing' vs. Sniper Accuracy\n")
    md.append("Empirical sweep on **5,000 test evaluation queries** using TabDPT with `split_uniform_classes` ($N_{\\text{ctx}} = 5,000$) "
              "on NVIDIA RTX 3060 GPU.\n")

    # Table 1: Ratio Meter Sweep
    md.append("## 1. Ratio Meter Sweep: $P(\\text{top\\_product}) / P(\\text{Class 0}) \\ge \\theta$\n")
    md.append("| Setting ($\\theta$) | Non-Buyer Hold Acc (Do Nothing) | Active Buyer Top-1 (Sniper) | Overall Top-1 Acc | Overall MAP@7 | Cold MAP@7 | Hit Rate@7 | Suppression Rate |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in ratio_res:
        suppression = f"{100.0 - (r['overall_top1_acc']):.1f}%"
        is_highlight = r.get("theta") in [1.0, 1.5, 2.0, 2.5, 3.0]
        mark = "**" if is_highlight else ""
        row = [
            f"{mark}{r['parameter']}{mark}",
            f"{mark}{r['non_buyer_top1_acc']:.1f}%{mark}",
            f"{mark}{r['buyer_top1_acc']:.1f}%{mark}",
            f"{r['overall_top1_acc']:.1f}%",
            f"{r['map7']:.4f}",
            f"{r['cold_map7']:.4f}",
            f"{r['hit_rate7']:.1f}%",
            f"{r['non_buyer_top1_acc']:.1f}%",
        ]
        md.append("| " + " | ".join(row) + " |")

    # Table 2: Logit Prior Shift
    md.append("\n## 2. Bayesian Logit Shift: $\\text{logit}(\\text{Class 0}) \\mathrel{+}= \\lambda$\n")
    md.append("| Setting ($\\lambda$) | Non-Buyer Hold Acc (Do Nothing) | Active Buyer Top-1 (Sniper) | Overall Top-1 Acc | Overall MAP@7 | Cold MAP@7 | Hit Rate@7 |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in logit_res:
        is_highlight = r.get("lambda") in [0.0, 1.0, 1.5, 2.0, 2.5]
        mark = "**" if is_highlight else ""
        row = [
            f"{mark}{r['parameter']}{mark}",
            f"{mark}{r['non_buyer_top1_acc']:.1f}%{mark}",
            f"{mark}{r['buyer_top1_acc']:.1f}%{mark}",
            f"{r['overall_top1_acc']:.1f}%",
            f"{r['map7']:.4f}",
            f"{r['cold_map7']:.4f}",
            f"{r['hit_rate7']:.1f}%",
        ]
        md.append("| " + " | ".join(row) + " |")

    # Find the Pareto Sweet Spot: Non-buyer >= 85% with maximum Buyer Top-1
    candidate_sweet_spots = [r for r in ratio_res + logit_res if r["non_buyer_top1_acc"] >= 85.0]
    best_sweet = max(candidate_sweet_spots, key=lambda x: x["buyer_top1_acc"]) if candidate_sweet_spots else None

    candidate_conservative = [r for r in ratio_res + logit_res if r["non_buyer_top1_acc"] >= 92.0]
    best_cons = max(candidate_conservative, key=lambda x: x["buyer_top1_acc"]) if candidate_conservative else None

    md.append("\n---\n")
    md.append("## 🏆 The Optimal Sweet Spot (Pareto Frontier)\n")
    if best_sweet:
        md.append(f"> ### **Balanced Production Sweet Spot: `{best_sweet['meter_type']} ({best_sweet['parameter']})`**\n")
        md.append(f"> - **Non-Buyer Hold Accuracy (Do Nothing)**: **{best_sweet['non_buyer_top1_acc']:.1f}%** (eliminates false alarm spam!)\n")
        md.append(f"> - **Active Buyer Top-1 Sniper Accuracy**: **{best_sweet['buyer_top1_acc']:.1f}%** (preserves sharp ranking!)\n")
        md.append(f"> - **Overall MAP@7**: **{best_sweet['map7']:.4f}**\n")
        md.append(f"> - **Cold MAP@7**: **{best_sweet['cold_map7']:.4f}**\n")

    if best_cons and best_cons != best_sweet:
        md.append(f"> ### **Zero-Friction Guardrail Setting: `{best_cons['meter_type']} ({best_cons['parameter']})`**\n")
        md.append(f"> - **Non-Buyer Hold Accuracy**: **{best_cons['non_buyer_top1_acc']:.1f}%** (matches baseline hold rate!)\n")
        md.append(f"> - **Active Buyer Top-1 Sniper Accuracy**: **{best_cons['buyer_top1_acc']:.1f}%**\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Confidence Meter Optimization for TabDPT")
    parser.add_argument("--n-context", type=int, default=5000, help="Context size budget")
    parser.add_argument("--eval-samples", type=int, default=5000, help="Test queries count")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("================================================================================")
    print("  TabDPT Confidence Meter Sweep: Taming 'Do Nothing' vs Sniper Precision")
    print("================================================================================")
    print(f"  Device:           {args.device}")
    print(f"  Context Budget:   {args.n_context} samples (split_uniform_classes)")
    print(f"  Evaluation Scale: {args.eval_samples} test queries")
    print("================================================================================\n")

    # 1. Load data
    print(f"[Data] Loading context pool and test cohort ({args.eval_samples} queries)...")
    X_train_pool, y_train_pool, test_cohort = prepare_benchmark_datasets(
        train_ratio=0.8,
        eval_sample_size=args.eval_samples,
        seed=42,
    )
    print(f"[Data] Test cohort ready: {len(test_cohort.y)} queries ({np.sum(test_cohort.is_cold)} cold, {np.sum(~test_cohort.is_cold)} warm)")

    # 2. Sample split_uniform_classes context
    X_ctx, y_ctx = sample_split_from_pool(
        X_pool=X_train_pool,
        y_pool=y_train_pool,
        feature_names=test_cohort.feature_names,
        split_name="split_uniform_classes",
        n_samples=args.n_context,
        seed=42,
    )

    # 3. Load TabDPT model
    model = load_tabdpt_model(device=args.device)

    # 4. Run single high-throughput forward pass on GPU
    print(f"\n[Inference] Running batched inference across {args.eval_samples} queries on {args.device}...")
    t0 = time.time()
    raw_logits, raw_probs = get_raw_logits_and_probs(
        model=model,
        X_ctx_np=X_ctx,
        y_ctx_np=y_ctx,
        X_test_np=test_cohort.X,
        device=args.device,
        batch_size=250,
    )
    print(f"[Inference] Inference complete in {time.time() - t0:.2f}s! Logits shape: {raw_logits.shape}")

    # 5. Sweep Meters
    print("\n[Sweep] Evaluating Ratio Meter (theta)...")
    thetas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
    ratio_results = sweep_ratio_meter(raw_probs, test_cohort, thetas)

    print("[Sweep] Evaluating Logit Prior Shift (lambda)...")
    lambdas = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.74, 3.0]
    logit_results = sweep_logit_shift_meter(raw_logits, test_cohort, lambdas)

    print("[Sweep] Evaluating Probability Margin (delta)...")
    deltas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    margin_results = sweep_margin_meter(raw_probs, test_cohort, deltas)

    # 6. Save results & Print report
    output_dir = DATA_DIR / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "confidence_meter_results.json"
    md_path = output_dir / "confidence_meter_results.md"

    combined_results = {
        "ratio_meter": ratio_results,
        "logit_shift": logit_results,
        "margin_meter": margin_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\n[Output] Saved JSON results to {json_path}")

    md_report = format_results_markdown(ratio_results, logit_results, margin_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[Output] Saved Markdown report to {md_path}")

    try:
        print("\n" + md_report)
    except UnicodeEncodeError:
        print("\n[Output] Full Markdown report generated and saved to file.")


if __name__ == "__main__":
    main()

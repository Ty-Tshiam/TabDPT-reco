"""
Benchmark Runner CLI for TabDPT & Baseline Models.

Executes benchmark across RealMLP, LightGBM, CatBoost, and TabDPT.
Outputs formatted Markdown tables ready for README.md and persists JSON results.

Usage:
    python src/benchmark_model/run_benchmark.py --eval-samples 2000
    python src/benchmark_model/run_benchmark.py --models lgbm catboost realmlp tabdpt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_model.data_loader import prepare_benchmark_datasets
from src.benchmark_model.evaluator import BenchmarkEvaluator
from src.benchmark_model.models.real_mlp import RealMLPRecommender
from src.benchmark_model.models.tabdpt_wrapper import TabDPTRecommender
from src.benchmark_model.models.tree_models import (
    CatBoostRecommender,
    LightGBMRecommender,
)
from src.config import DATA_DIR


def format_readme_tables(results: list[dict]) -> str:
    """Format benchmark results into publication-ready GitHub Markdown tables."""
    md = []
    md.append("## Recommendation & Ranking Benchmarks: TabDPT vs. Baselines\n")
    md.append(
        "Empirical benchmark evaluating **TabDPT** (in-context tabular foundation model) "
        "against gradient boosting (**LightGBM**, **CatBoost**) and modern tabular deep learning (**RealMLP**) "
        "on the Santander Product Recommendation dataset.\n"
    )

    # Table 1: Ranking & Cold-Start Breakdown
    md.append("### 1. Ranking Performance & Cold-Start Resilience")
    md.append(
        "> **Note on Tenure Segmentation**: *Cold* customers are defined as `< 90 days` of account history; "
        "*Warm* customers have `≥ 90 days` of history.\n"
    )
    md.append(
        "| Model | Family | Warm MAP@7 | Cold MAP@7 | Overall MAP@7 | Warm MAP@4 | Cold MAP@4 | Overall MAP@4 | NDCG@4 | NDCG@5 | NDCG@7 | MRR |"
    )
    md.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        md.append(
            f"| **{r['model_name']}** | {r['family']} | "
            f"**{r['warm_map7']:.4f}** | **{r['cold_map7']:.4f}** | **{r['map7']:.4f}** | "
            f"{r['warm_map4']:.4f} | {r['cold_map4']:.4f} | {r['map4']:.4f} | "
            f"{r['ndcg4']:.4f} | {r['ndcg5']:.4f} | {r['ndcg7']:.4f} | {r['mrr']:.4f} |"
        )
    md.append("\n")

    # Table 2: Coverage, Calibration & Alert Dynamics
    md.append("### 2. Coverage, Calibration & Production Alert Dynamics\n")
    md.append(
        "| Model | Recall@3 | Hit Rate@3 | Recall@4 | Hit Rate@4 | Recall@7 | Hit Rate@7 | ECE (Calib) | PSI (Drift) | Active Alerts % | Silent Alerts % |"
    )
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        md.append(
            f"| **{r['model_name']}** | "
            f"{r['recall3']:.4f} | {r['hit_rate3']:.1f}% | "
            f"{r['recall4']:.4f} | {r['hit_rate4']:.1f}% | "
            f"{r['recall7']:.4f} | {r['hit_rate7']:.1f}% | "
            f"{r['ece']:.4f} | {r['psi']:.4f} | "
            f"{r['active_alerts_pct']:.1f}% | {r['silent_alerts_pct']:.1f}% |"
        )
    md.append("\n")

    # Table 3: Accuracy Breakdown & Inference Latency
    md.append("### 3. Accuracy Breakdown & System Latency Profile\n")
    md.append(
        "| Model | Active Buyers Top-1 | Non-Buyers (Hold) Acc | Overall Top-1 Acc | Training Time | Latency p50 | Latency p95 | Latency p99 |"
    )
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        md.append(
            f"| **{r['model_name']}** | "
            f"{r['buyer_top1_acc']:.1f}% | {r['non_buyer_acc']:.1f}% | {r['overall_top1_acc']:.1f}% | "
            f"{r['train_time_sec']:.1f}s | {r['latency_p50_ms']:.2f} ms | {r['latency_p95_ms']:.2f} ms | **{r['latency_p99_ms']:.2f} ms** |"
        )
    md.append("\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Run TabDPT & Baseline Recommender Benchmarks"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "catboost", "realmlp", "tabdpt"],
        choices=["lgbm", "catboost", "realmlp", "tabdpt", "all"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=2000,
        help="Number of test evaluation samples (default: 2000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR / "benchmark_output"),
        help="Directory to save benchmark reports and JSON",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" TabDPT vs. RealMLP, LightGBM, and CatBoost Recommendation Benchmark")
    print("=" * 80)

    # 1. Load Data
    print(
        f"\n[1/3] Loading dataset and partitioning test cohort (eval samples: {args.eval_samples})..."
    )
    X_train, y_train, test_cohort = prepare_benchmark_datasets(
        train_ratio=args.train_ratio,
        eval_sample_size=args.eval_samples,
    )
    print(
        f"[1/3] Train set: {X_train.shape[0]} rows x {X_train.shape[1]} features | "
        f"Test cohort: {test_cohort.X.shape[0]} rows (Cold: {np.sum(test_cohort.is_cold)}, Warm: {np.sum(~test_cohort.is_cold)})"
    )

    # 2. Instantiate Models
    selected_keys = set(args.models)
    if "all" in selected_keys:
        selected_keys = {"lgbm", "catboost", "realmlp", "tabdpt"}

    model_instances = []
    if "lgbm" in selected_keys:
        model_instances.append(
            LightGBMRecommender(n_estimators=100, learning_rate=0.08)
        )
    if "catboost" in selected_keys:
        model_instances.append(CatBoostRecommender(iterations=100, learning_rate=0.08))
    if "realmlp" in selected_keys:
        model_instances.append(
            RealMLPRecommender(hidden_dim=256, num_blocks=3, epochs=12, batch_size=512)
        )
    if "tabdpt" in selected_keys:
        model_instances.append(TabDPTRecommender())

    # 3. Evaluate Models
    print(f"\n[2/3] Executing benchmark across {len(model_instances)} models...")
    evaluator = BenchmarkEvaluator(model_instances)
    results = evaluator.evaluate(X_train, y_train, test_cohort, latency_samples=50)

    # 4. Generate Reports
    print("\n[3/3] Generating README Markdown tables & JSON records...")
    readme_content = format_readme_tables(results)

    # Save Markdown
    readme_path = output_dir / "benchmark_readme.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[3/3] Wrote README report to: {readme_path}")

    # Save JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[3/3] Wrote JSON artifacts to: {json_path}")

    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS (README.MD FORMAT)")
    print("=" * 80 + "\n")
    print(readme_content)


if __name__ == "__main__":
    main()

"""
Experiment: LightGBM as a Reranker for TabDPT.

Evaluates whether using LightGBM as a second-stage reranker for TabDPT candidate outputs
combines TabDPT's high candidate recall and catalog diversity with LightGBM's sharp
top-rank discrimination (MAP@7).
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Add project root and TabDPT submodule to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from tabdpt.model import TabDPTModel
from src.config import (
    MODEL_CONFIG,
    SELECTED_15_TARGETS,
    INDEX_TO_TARGET,
)
from src.benchmark_model.data_loader import prepare_benchmark_datasets
from src.benchmark_model.models.tree_models import LightGBMRecommender
from src.benchmark_model.metrics import (
    compute_map_at_k,
    compute_ndcg_at_k,
    compute_mrr,
    compute_recall_and_hit_rate_at_k,
    compute_catalog_gini,
    compute_buyer_accuracy_breakdown,
    compute_psi,
)


def load_tabdpt_model_cpu():
    """Instantiate and load pre-trained TabDPT weights on CPU."""
    cfg = dict(MODEL_CONFIG["settings"])
    cfg["use_flash"] = False
    model = TabDPTModel(**cfg)
    
    weights_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--Layer6--TabDPT" / "snapshots" / "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a" / "tabdpt1_3.safetensors"
    if not weights_path.exists():
        from huggingface_hub import hf_hub_download
        weights_path = Path(hf_hub_download(repo_id="Layer6/TabDPT", filename="tabdpt1_3.safetensors"))
    
    model.load_state_dict(load_file(str(weights_path)))
    model.eval()
    return model


def get_tabdpt_probabilities(model, X_train, y_train, X_test, n_context=2000, batch_size=250):
    """Compute TabDPT class probabilities using in-context KV-cache."""
    print(f"  [TabDPT] Encoding {n_context} in-context samples on CPU...")
    t0 = time.time()
    
    # Stratified subsample of context
    rng = np.random.default_rng(42)
    ctx_indices = []
    for c in range(16):
        c_idxs = np.where(y_train == c)[0]
        take = max(1, int(n_context * (len(c_idxs) / len(y_train))))
        ctx_indices.extend(rng.choice(c_idxs, size=min(take, len(c_idxs)), replace=False))
    ctx_indices = np.array(ctx_indices[:n_context])
    
    x_ctx = torch.from_numpy(X_train[ctx_indices]).unsqueeze(0).float()
    y_ctx = torch.from_numpy(y_train[ctx_indices]).unsqueeze(0).long()
    
    with torch.no_grad():
        kv_cache, n_ctx, stats = model.encode_context(x_ctx, y_ctx, is_cls=True)
    print(f"  [TabDPT] Context encoded in {time.time() - t0:.2f}s")
    
    print(f"  [TabDPT] Running inference on {len(X_test)} queries (batch size {batch_size})...")
    t1 = time.time()
    probs_list = []
    for i in range(0, len(X_test), batch_size):
        batch = torch.from_numpy(X_test[i : i + batch_size]).unsqueeze(0).float()
        with torch.no_grad():
            logits = model.predict_with_cache(batch, kv_cache, n_ctx, stats)
            probs = F.softmax(logits[..., :16].float(), dim=-1).squeeze(1).cpu().numpy()
            probs_list.append(probs)
    
    all_probs = np.vstack(probs_list)
    print(f"  [TabDPT] Inference completed in {time.time() - t1:.2f}s ({((time.time() - t1) / len(X_test)) * 1000:.2f} ms/query)")
    return all_probs


def rank_products_standalone(probs: np.ndarray, held_products: list[list[str]], top_k: int = 7) -> list[list[str]]:
    """Standard ranking: order classes 1..15 by probability descending, filter held."""
    recs = []
    for i in range(len(probs)):
        held_set = set(held_products[i]) if i < len(held_products) else set()
        p = probs[i]
        # Sort classes 1..15
        ranked_classes = np.argsort(p[1:])[::-1] + 1
        ranked_prods = [INDEX_TO_TARGET[c] for c in ranked_classes]
        filtered = [prod for prod in ranked_prods if prod not in held_set][:top_k]
        recs.append(filtered)
    return recs


def rank_twostage_rerank(
    retrieval_probs: np.ndarray,
    reranker_probs: np.ndarray,
    held_products: list[list[str]],
    top_candidates_m: int = 7,
    top_k: int = 7,
) -> list[list[str]]:
    """
    Two-Stage Pipeline:
    1. TabDPT generates Top-M candidates (excluding already held products).
    2. LightGBM reranks ONLY within those M candidates using LightGBM probabilities.
    3. Return top_k from the reranked list.
    """
    recs = []
    for i in range(len(retrieval_probs)):
        held_set = set(held_products[i]) if i < len(held_products) else set()
        p_retrieval = retrieval_probs[i]
        p_rerank = reranker_probs[i]
        
        # Stage 1: Get Top-M candidate products from TabDPT
        ranked_classes_retrieval = np.argsort(p_retrieval[1:])[::-1] + 1
        candidate_classes = [c for c in ranked_classes_retrieval if INDEX_TO_TARGET[c] not in held_set][:top_candidates_m]
        
        # Stage 2: LightGBM reranks the candidate classes
        candidate_classes_reranked = sorted(candidate_classes, key=lambda c: p_rerank[c], reverse=True)
        
        # Map back to product names
        filtered = [INDEX_TO_TARGET[c] for c in candidate_classes_reranked][:top_k]
        
        # If fewer than top_k candidates were available, pad from remaining LightGBM ranked products
        if len(filtered) < top_k:
            remaining_classes = np.argsort(p_rerank[1:])[::-1] + 1
            for c in remaining_classes:
                prod = INDEX_TO_TARGET[c]
                if prod not in held_set and prod not in filtered:
                    filtered.append(prod)
                if len(filtered) == top_k:
                    break
        recs.append(filtered)
    return recs


def rank_score_fusion(
    probs_tabdpt: np.ndarray,
    probs_lgbm: np.ndarray,
    held_products: list[list[str]],
    alpha: float = 0.5,
    top_k: int = 7,
) -> list[list[str]]:
    """
    Score Fusion:
    Blends TabDPT prior and LightGBM discriminative probabilities:
    P_fused = (1 - alpha) * P_lgbm + alpha * P_tabdpt
    """
    probs_fused = (1.0 - alpha) * probs_lgbm + alpha * probs_tabdpt
    return rank_products_standalone(probs_fused, held_products, top_k=top_k)


def rank_rrf(
    probs_tabdpt: np.ndarray,
    probs_lgbm: np.ndarray,
    held_products: list[list[str]],
    k_constant: int = 60,
    top_k: int = 7,
) -> list[list[str]]:
    """Reciprocal Rank Fusion (RRF) between TabDPT and LightGBM."""
    recs = []
    for i in range(len(probs_tabdpt)):
        held_set = set(held_products[i]) if i < len(held_products) else set()
        p_tab = probs_tabdpt[i]
        p_lgb = probs_lgbm[i]
        
        order_tab = (np.argsort(p_tab[1:])[::-1] + 1).tolist()
        order_lgb = (np.argsort(p_lgb[1:])[::-1] + 1).tolist()
        
        # Calculate RRF score for each class 1..15
        scores = {}
        for c in range(1, 16):
            prod = INDEX_TO_TARGET[c]
            if prod in held_set:
                continue
            rank_tab = order_tab.index(c) + 1
            rank_lgb = order_lgb.index(c) + 1
            scores[prod] = (1.0 / (k_constant + rank_tab)) + (1.0 / (k_constant + rank_lgb))
        
        sorted_prods = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)[:top_k]
        recs.append(sorted_prods)
    return recs


def compute_metrics_dict(recs_7, cohort, y_probs=None, train_class_dist=None):
    """Compute complete metric bundle for a recommendation output."""
    is_cold = cohort.is_cold
    is_warm = ~is_cold
    cold_actuals = [cohort.actual_products[i] for i in range(len(cohort.y)) if is_cold[i]]
    warm_actuals = [cohort.actual_products[i] for i in range(len(cohort.y)) if is_warm[i]]
    
    cold_recs_7 = [recs_7[i] for i in range(len(cohort.y)) if is_cold[i]]
    warm_recs_7 = [recs_7[i] for i in range(len(cohort.y)) if is_warm[i]]
    recs_3 = [r[:3] for r in recs_7]
    recs_4 = [r[:4] for r in recs_7]
    
    map7 = compute_map_at_k(recs_7, cohort.actual_products, k=7)
    warm_map7 = compute_map_at_k(warm_recs_7, warm_actuals, k=7)
    cold_map7 = compute_map_at_k(cold_recs_7, cold_actuals, k=7)
    ndcg7 = compute_ndcg_at_k(recs_7, cohort.actual_products, k=7)
    ndcg4 = compute_ndcg_at_k(recs_4, cohort.actual_products, k=4)
    mrr = compute_mrr(recs_7, cohort.actual_products, k=7)
    
    rec7, hit7 = compute_recall_and_hit_rate_at_k(recs_7, cohort.actual_products, k=7)
    rec3, hit3 = compute_recall_and_hit_rate_at_k(recs_3, cohort.actual_products, k=3)
    gini = compute_catalog_gini(recs_7, SELECTED_15_TARGETS)
    
    # Buyer top-1 accuracy (when actual item matches first recommendation)
    buyer_hits = 0
    buyer_count = 0
    for i in range(len(cohort.y)):
        if len(cohort.actual_products[i]) > 0:
            buyer_count += 1
            if len(recs_7[i]) > 0 and recs_7[i][0] in cohort.actual_products[i]:
                buyer_hits += 1
    buyer_top1 = (buyer_hits / buyer_count * 100.0) if buyer_count > 0 else 0.0
    
    return {
        "MAP@7": map7,
        "Warm MAP@7": warm_map7,
        "Cold MAP@7": cold_map7,
        "Hit Rate@7 (%)": hit7 * 100.0,
        "Hit Rate@3 (%)": hit3 * 100.0,
        "NDCG@7": ndcg7,
        "NDCG@4": ndcg4,
        "MRR": mrr,
        "Buyer Top-1 (%)": buyer_top1,
        "Catalog Gini": gini,
    }


def run_experiment(eval_samples: int = 1000):
    print("=" * 85)
    print(" EXPERIMENT: LIGHTGBM AS A RERANKER FOR TABDPT OUTPUTS")
    print("=" * 85)
    
    print(f"\n[1/4] Preparing dataset (eval_samples = {eval_samples})...")
    X_train, y_train, test_cohort = prepare_benchmark_datasets(
        train_ratio=0.8, eval_sample_size=eval_samples, seed=42
    )
    print(f"  Train samples: {X_train.shape[0]} | Test cohort: {test_cohort.X.shape[0]}")
    print(f"  Test buyers: {sum(1 for a in test_cohort.actual_products if len(a) > 0)} | Cold: {sum(test_cohort.is_cold)}")
    
    # 2. Train LightGBM
    print("\n[2/4] Training LightGBM baseline on 81 tabular features...")
    lgbm = LightGBMRecommender(n_estimators=100, learning_rate=0.08)
    t0 = time.time()
    lgbm.fit(X_train[:, :81], y_train)
    print(f"  LightGBM trained in {time.time() - t0:.2f}s")
    probs_lgbm = lgbm.predict_proba(test_cohort.X[:, :81])
    
    # 3. Compute TabDPT probabilities
    print("\n[3/4] Initializing TabDPT Foundation Model & computing test probabilities...")
    tabdpt_model = load_tabdpt_model_cpu()
    probs_tabdpt = get_tabdpt_probabilities(
        tabdpt_model, X_train, y_train, test_cohort.X, n_context=2000, batch_size=250
    )
    
    # 4. Run Strategy Evaluations
    print("\n[4/4] Evaluating Standalone Baselines and Reranking Configurations...")
    results = {}
    
    # A. Standalone LightGBM
    recs_lgbm = rank_products_standalone(probs_lgbm, test_cohort.held_products, top_k=7)
    results["1. Standalone LightGBM"] = compute_metrics_dict(recs_lgbm, test_cohort)
    
    # B. Standalone TabDPT
    recs_tabdpt = rank_products_standalone(probs_tabdpt, test_cohort.held_products, top_k=7)
    results["2. Standalone TabDPT"] = compute_metrics_dict(recs_tabdpt, test_cohort)
    
    # C. Two-Stage Pipeline: TabDPT Top-M Retrieval -> LightGBM Reranker
    for m in [5, 7, 8, 10, 12]:
        recs_twostage = rank_twostage_rerank(
            probs_tabdpt, probs_lgbm, test_cohort.held_products, top_candidates_m=m, top_k=7
        )
        results[f"3. Two-Stage (TabDPT Top-{m} -> LightGBM Rerank)"] = compute_metrics_dict(recs_twostage, test_cohort)
    
    # D. Score Fusion / Blending
    for alpha in [0.2, 0.35, 0.5, 0.65]:
        recs_fusion = rank_score_fusion(
            probs_tabdpt, probs_lgbm, test_cohort.held_products, alpha=alpha, top_k=7
        )
        pct_tab = int(alpha * 100)
        pct_lgb = int((1 - alpha) * 100)
        results[f"4. Hybrid Linear Fusion ({pct_lgb}% LGBM + {pct_tab}% Tab)"] = compute_metrics_dict(recs_fusion, test_cohort)

    # E. Log-Odds Bayesian Prior Fusion (TabDPT as contextual prior)
    eps = 1e-6
    for weight in [0.15, 0.25, 0.35]:
        log_odds = np.log(probs_lgbm + eps) + weight * np.log(probs_tabdpt + eps)
        exp_lo = np.exp(log_odds - np.max(log_odds, axis=1, keepdims=True))
        probs_log_odds = exp_lo / np.sum(exp_lo, axis=1, keepdims=True)
        recs_lo = rank_products_standalone(probs_log_odds, test_cohort.held_products, top_k=7)
        results[f"5. Log-Odds Prior Fusion (LGBM + {weight}*TabLogOdds)"] = compute_metrics_dict(recs_lo, test_cohort)

    # F. Reciprocal Rank Fusion
    recs_rrf = rank_rrf(probs_tabdpt, probs_lgbm, test_cohort.held_products, k_constant=60, top_k=7)
    results["6. Reciprocal Rank Fusion (RRF)"] = compute_metrics_dict(recs_rrf, test_cohort)
    
    # Display Results Table
    print("\n" + "=" * 125)
    print(" EXPERIMENT RESULTS SUMMARY TABLE")
    print("=" * 125)
    headers = ["Architecture / Configuration", "MAP@7", "Warm MAP", "Cold MAP", "Hit@7 (%)", "Hit@3 (%)", "NDCG@7", "MRR", "Buyer Top-1", "Gini"]
    print(f"{headers[0]:<46} | {headers[1]:<7} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<9} | {headers[5]:<9} | {headers[6]:<7} | {headers[7]:<6} | {headers[8]:<11} | {headers[9]:<6}")
    print("-" * 125)
    
    for name, m in results.items():
        print(
            f"{name:<46} | "
            f"{m['MAP@7']:.4f}  | "
            f"{m['Warm MAP@7']:.4f}   | "
            f"{m['Cold MAP@7']:.4f}   | "
            f"{m['Hit Rate@7 (%)']:>6.2f}%   | "
            f"{m['Hit Rate@3 (%)']:>6.2f}%   | "
            f"{m['NDCG@7']:.4f}  | "
            f"{m['MRR']:.4f} | "
            f"{m['Buyer Top-1 (%)']:>8.2f}%   | "
            f"{m['Catalog Gini']:.4f}"
        )
    print("=" * 125)
    
    return results


if __name__ == "__main__":
    run_experiment(eval_samples=1000)

"""
Comprehensive Evaluation Metrics Suite for TabDPT & Benchmark Models.

Computes:
1. Ranking: MAP@7, MAP@4 (Warm & Cold), NDCG@4, NDCG@5, NDCG@7, MRR.
2. Hit Rates & Coverage: Recall@3, Hit Rate@3, Recall@4, Hit Rate@4, Recall@7, Hit Rate@7.
3. Calibration & Stability: Expected Calibration Error (ECE), Population Stability Index (PSI).
4. Alert & Production Dynamics: Active vs. Silent Alert distribution, Buyer & Non-buyer Top-1 accuracies.
5. Latency Profiling: Mean, p50, p95, p99 (ms) per inference query.
"""

import numpy as np


def compute_ap_at_k(recommended: list[str], actual: list[str], k: int = 7) -> float:
    """
    Computes Average Precision at K for a single customer.
    
    Args:
        recommended: List of recommended product names in descending rank order.
        actual: List of ground-truth product names added by customer.
        k: Ranking cutoff depth (e.g. 4 or 7).
        
    Returns:
        AP@K float value in [0.0, 1.0].
    """
    if not actual:
        return 0.0

    hits = 0
    score = 0.0
    top_recs = recommended[:k]

    for i, item in enumerate(top_recs):
        if item in actual:
            hits += 1
            score += hits / (i + 1.0)

    return score / min(len(actual), k)


def compute_map_at_k(
    all_recommended: list[list[str]], all_actual: list[list[str]], k: int = 7
) -> float:
    """
    Computes Mean Average Precision at K across all customers with actual product additions.
    """
    scores = []
    for recs, actual in zip(all_recommended, all_actual):
        if actual:
            scores.append(compute_ap_at_k(recs, actual, k=k))

    return float(np.mean(scores)) if scores else 0.0


def compute_ndcg_at_k(
    all_recommended: list[list[str]], all_actual: list[list[str]], k: int = 7
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K across all customers.
    """
    scores = []
    for recs, actual in zip(all_recommended, all_actual):
        if not actual:
            continue

        actual_set = set(actual)
        top_recs = recs[:k]

        # Calculate DCG@K
        dcg = 0.0
        for i, item in enumerate(top_recs):
            if item in actual_set:
                dcg += 1.0 / np.log2(i + 2.0)  # rank i+1 -> log2(i+1 + 1)

        # Calculate Ideal DCG@K
        ideal_hits = min(len(actual), k)
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))

        scores.append(dcg / idcg if idcg > 0.0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def compute_mrr(
    all_recommended: list[list[str]], all_actual: list[list[str]], k: int = 7
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) within top K candidates.
    """
    reciprocal_ranks = []
    for recs, actual in zip(all_recommended, all_actual):
        if not actual:
            continue

        actual_set = set(actual)
        rr = 0.0
        for i, item in enumerate(recs[:k]):
            if item in actual_set:
                rr = 1.0 / (i + 1.0)
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def compute_recall_and_hit_rate_at_k(
    all_recommended: list[list[str]], all_actual: list[list[str]], k: int = 7
) -> tuple[float, float]:
    """
    Computes (Recall@K, Hit Rate@K) across all customers with actual additions.
    - Recall@K: fraction of added products captured in top K.
    - Hit Rate@K: fraction of customers who have at least one hit in top K.
    """
    recalls = []
    hits = []

    for recs, actual in zip(all_recommended, all_actual):
        if not actual:
            continue

        actual_set = set(actual)
        top_recs = set(recs[:k])
        matched = top_recs.intersection(actual_set)

        recalls.append(len(matched) / len(actual))
        hits.append(1.0 if len(matched) > 0 else 0.0)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_hit_rate = float(np.mean(hits)) if hits else 0.0

    return mean_recall, mean_hit_rate


def compute_ece(
    y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class probability outputs.
    
    Args:
        y_true: 1D NumPy array of ground truth class indices (0..15).
        y_probs: 2D NumPy array of predicted class probabilities (N, 16).
        n_bins: Number of confidence interval bins.
        
    Returns:
        ECE float score in [0.0, 1.0].
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_psi(
    expected_dist: np.ndarray, actual_dist: np.ndarray, eps: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) between two discrete distributions.
    
    Args:
        expected_dist: 1D array of expected/baseline probabilities across 16 classes.
        actual_dist: 1D array of actual/predicted probabilities across 16 classes.
        eps: Small smoothing constant to avoid log(0) or division by zero.
        
    Returns:
        PSI float value. (PSI < 0.1: stable; 0.1-0.25: slight drift; > 0.25: significant drift).
    """
    p = np.clip(expected_dist, eps, None)
    q = np.clip(actual_dist, eps, None)

    p = p / np.sum(p)
    q = q / np.sum(q)

    psi_val = np.sum((q - p) * np.log(q / p))
    return float(psi_val)


def compute_alert_distribution(
    predictions_or_probs: np.ndarray,
    held_products: list[list[str]] | None = None,
    classes_map: dict[int, str] | None = None,
) -> dict[str, float]:
    """
    Computes distribution between Active Alerts (proactive product recommendations)
    and Silent Alerts (status quo / hold / class 0).
    
    Args:
        predictions_or_probs: Either 1D array of predicted class indices, or 2D probability matrix.
        held_products: Optional list of held products.
        classes_map: Optional mapping from index to class name.
        
    Returns:
        Dict containing:
        - active_alerts_pct: Percentage of instances with active recommendation
        - silent_alerts_pct: Percentage of instances with suppressed/hold recommendation
    """
    if predictions_or_probs.ndim == 2:
        top_classes = np.argmax(predictions_or_probs, axis=1)
    else:
        top_classes = predictions_or_probs

    total = len(top_classes)
    if total == 0:
        return {"active_alerts_pct": 0.0, "silent_alerts_pct": 100.0}

    # Class 0 represents "do_nothing" / hold status quo
    silent_count = int(np.sum(top_classes == 0))
    active_count = total - silent_count

    return {
        "active_alerts_pct": round(float((active_count / total) * 100.0), 2),
        "silent_alerts_pct": round(float((silent_count / total) * 100.0), 2),
    }


def compute_buyer_accuracy_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_recommended: list[list[str]] | None = None,
    all_actual: list[list[str]] | None = None,
) -> dict[str, float]:
    """
    Computes fine-grained accuracy breakdown matching the baseline benchmark:
    - Active Buyers Top-7 Hit Rate (Recall@7)
    - Active Buyers MAP@7
    - Active Buyers Top-1 Accuracy
    - Non-Buyers (Class 0 / Hold) Accuracy
    - Overall Top-1 Accuracy
    """
    is_buyer = y_true != 0
    is_non_buyer = y_true == 0

    buyer_count = np.sum(is_buyer)
    non_buyer_count = np.sum(is_non_buyer)
    total_count = len(y_true)

    buyer_top1_acc = (
        float(np.mean(y_pred[is_buyer] == y_true[is_buyer]) * 100.0)
        if buyer_count > 0
        else 0.0
    )
    non_buyer_top1_acc = (
        float(np.mean(y_pred[is_non_buyer] == y_true[is_non_buyer]) * 100.0)
        if non_buyer_count > 0
        else 0.0
    )
    overall_top1_acc = (
        float(np.mean(y_pred == y_true) * 100.0) if total_count > 0 else 0.0
    )

    buyer_map7 = 0.0
    buyer_hit_rate7 = 0.0
    if all_recommended is not None and all_actual is not None:
        buyer_map7 = compute_map_at_k(all_recommended, all_actual, k=7)
        _, buyer_hit_rate7 = compute_recall_and_hit_rate_at_k(
            all_recommended, all_actual, k=7
        )

    return {
        "active_buyers_top1_acc": round(buyer_top1_acc, 2),
        "non_buyers_accuracy": round(non_buyer_top1_acc, 2),
        "overall_top1_acc": round(overall_top1_acc, 2),
        "active_buyers_map7": round(buyer_map7, 4),
        "active_buyers_hit_rate7": round(buyer_hit_rate7 * 100.0, 2),
    }


class LatencyProfiler:
    """Profiles per-query latency and batch throughput."""

    def __init__(self):
        self.latencies_ms: list[float] = []

    def record(self, duration_sec: float, batch_size: int = 1) -> None:
        per_query_ms = (duration_sec * 1000.0) / max(1, batch_size)
        self.latencies_ms.append(per_query_ms)

    def summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        arr = np.array(self.latencies_ms)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }

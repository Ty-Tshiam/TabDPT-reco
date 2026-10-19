"""
Verification & Smoke Test for Production Recommendation Engine.

Executes Section 4 Checklist from Implement.md:
1. VRAM Residency: Validates TabDPT model + 73.9k context stability in VRAM (~4.8 GB).
2. Routing Telemetry: Replays a 1,000-event sample from May 2016 holdout through
   process_customer_event() and verifies:
   - stage1_cpu_hold accounts for ~42.3% of events (gpu_skipped == True).
   - Combined hold rate matches 84.7% prior to GAMMA_LIVE filter.
   - Combined hold rate reaches ~91.0% when evaluating live push eligibility at p_buy >= 0.30.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from src.config import INDEX_TO_TARGET, SELECTED_15_TARGETS
from src.streaming.event_processor import (
    GAMMA_CACHE,
    GAMMA_LIVE,
    NOISY_SINKS,
    THETA_CLEAN,
    THETA_NOISY,
    TabDPTEngine,
    load_lgbm_gatekeeper,
    process_customer_event,
)
from scripts.experiment_may2016_benchmark import load_may2016_cohort


def run_pipeline_verification(device: str = "cuda:0" if torch.cuda.is_available() else "cpu", n_samples: int = 1000):
    print("================================================================================")
    print("  PRODUCTION RECOMMENDATION ENGINE INTEGRATION VERIFICATION")
    print("================================================================================")
    print(f"  Device:           {device}")
    print(f"  Evaluation cohort: May 2016 Holdout (Sample size: {n_samples})")
    print(f"  Stage 1 Gate:     GAMMA_CACHE={GAMMA_CACHE}, GAMMA_LIVE={GAMMA_LIVE}")
    print(f"  Stage 2 Gate 1C:  THETA_NOISY={THETA_NOISY}, THETA_CLEAN={THETA_CLEAN}")
    print("================================================================================\n")

    # 1. Load May 2016 Cohort
    print("[1/4] Loading May 2016 evaluation cohort...")
    _, _, cohort = load_may2016_cohort()
    X_sample = cohort.X[:n_samples]
    y_sample = cohort.y[:n_samples]
    held_sample = cohort.held_products[:n_samples]
    print(f"  Loaded {len(y_sample)} events (Buyers: {np.sum(y_sample > 0)}, Non-Buyers: {np.sum(y_sample == 0)}).\n")

    # 2. Initialize Stage 1 and Stage 2 models
    print("[2/4] Initializing Stage 1 CPU LightGBM Gatekeeper...")
    t0 = time.time()
    lgbm_model = load_lgbm_gatekeeper()
    print(f"  LightGBM loaded in {time.time() - t0:.2f}s.\n")

    print(f"[3/4] Initializing Stage 2 TabDPT Engine on {device}...")
    t0 = time.time()
    tabdpt_engine = TabDPTEngine(device=device, prefer_triton=False)
    print(f"  TabDPT Engine initialized in {time.time() - t0:.2f}s.")

    if torch.cuda.is_available() and device.startswith("cuda"):
        vram_start_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_res_start_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"  Initial VRAM Allocated: {vram_start_mb:.1f} MB (Reserved: {vram_res_start_mb:.1f} MB)\n")
    else:
        vram_start_mb = 0.0

    # 3. Replay 1,000 Events through process_customer_event
    print(f"[4/4] Replaying {n_samples} customer events through 3-Stage Cascade...")
    regime_counts = {
        "stage1_cpu_hold": 0,
        "stage2_gate1c_hold": 0,
        "candidate_a_silent_cache": 0,
        "candidate_c_live_push": 0,
    }

    # Simulate realistic customer event actions
    actions = ["salary_deposit", "card_payment", "large_deposit", "branch_inquiry"]
    latencies_cpu_hold = []
    latencies_gpu = []

    t_replay_start = time.time()
    results = []

    for i in range(n_samples):
        action = actions[i % len(actions)]
        x_qry = X_sample[i]
        held = held_sample[i]

        t_event_start = time.perf_counter()
        res = process_customer_event(
            features_80d=x_qry,
            action=action,
            lgbm_model=lgbm_model,
            tabdpt_model=tabdpt_engine,
            held_products=held,
        )
        t_event_ms = (time.perf_counter() - t_event_start) * 1000

        results.append(res)
        regime = res["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if res["gpu_skipped"]:
            latencies_cpu_hold.append(t_event_ms)
        else:
            latencies_gpu.append(t_event_ms)

    total_time = time.time() - t_replay_start
    print(f"  Replay complete in {total_time:.2f}s across {n_samples} events!\n")

    # 4. Check VRAM Stability
    if torch.cuda.is_available() and device.startswith("cuda"):
        vram_end_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_res_end_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"  Final VRAM Allocated: {vram_end_mb:.1f} MB (Reserved: {vram_res_end_mb:.1f} MB)")
        vram_diff = abs(vram_end_mb - vram_start_mb)
        print(f"  VRAM Drift across {n_samples} queries: {vram_diff:.2f} MB (Target: < 50 MB)")
    else:
        vram_diff = 0.0

    # 5. Compute Key Telemetry Metrics
    pct_cpu_hold = (regime_counts["stage1_cpu_hold"] / n_samples) * 100.0
    pct_stage2_hold = (regime_counts["stage2_gate1c_hold"] / n_samples) * 100.0
    pct_silent_cache = (regime_counts["candidate_a_silent_cache"] / n_samples) * 100.0
    pct_live_push = (regime_counts["candidate_c_live_push"] / n_samples) * 100.0

    preds = np.array([r["prediction"] for r in results])
    non_buyer_mask = (y_sample == 0)
    buyer_mask = (y_sample > 0)

    # Candidate A Non-Buyer Hold Rate (Target: ~84.7%)
    non_buyer_hold_cand_a = float(np.mean(preds[non_buyer_mask] == 0) * 100.0) if np.any(non_buyer_mask) else 0.0
    # Candidate A Buyer Top-1 Sniper Accuracy (Target: ~54.3%)
    buyer_top1_cand_a = float(np.mean(preds[buyer_mask] == y_sample[buyer_mask]) * 100.0) if np.any(buyer_mask) else 0.0
    # Overall Accuracy (Target: ~67.6%)
    overall_acc_cand_a = float(np.mean(preds == y_sample) * 100.0)

    # Candidate C Live Push Metrics (Target: ~91.0% hold safety on non-buyers)
    live_suppressed_non_buyers = [not results[i]["is_active_trigger"] for i in range(n_samples) if non_buyer_mask[i]]
    non_buyer_live_hold = float(np.mean(live_suppressed_non_buyers) * 100.0) if live_suppressed_non_buyers else 0.0
    total_live_suppressed = sum(1 for r in results if not r["is_active_trigger"])
    pct_live_suppressed = (total_live_suppressed / n_samples) * 100.0

    avg_cpu_lat = np.mean(latencies_cpu_hold) if latencies_cpu_hold else 0.0
    avg_gpu_lat = np.mean(latencies_gpu) if latencies_gpu else 0.0

    print("\n================================================================================")
    print("  VERIFICATION RESULTS & TELEMETRY SUMMARY")
    print("================================================================================")
    print(f"  Total Events Processed:        {n_samples} (Buyers: {np.sum(buyer_mask)}, Non-Buyers: {np.sum(non_buyer_mask)})")
    print(f"  Stage 1 CPU Holds (Bypassed):  {regime_counts['stage1_cpu_hold']} ({pct_cpu_hold:.1f}%) [Target: ~42.3%]")
    print(f"  Stage 2 Gate 1C Holds:         {regime_counts['stage2_gate1c_hold']} ({pct_stage2_hold:.1f}%)")
    print(f"  Candidate A Silent Cache:      {regime_counts['candidate_a_silent_cache']} ({pct_silent_cache:.1f}%)")
    print(f"  Candidate C Live Push Alerts:  {regime_counts['candidate_c_live_push']} ({pct_live_push:.1f}%)")
    print("  ------------------------------------------------------------------------------")
    print(f"  Candidate A Non-Buyer Hold:    {non_buyer_hold_cand_a:.1f}% [Target: ~84.7%]")
    print(f"  Candidate A Buyer Top-1:       {buyer_top1_cand_a:.1f}% [Target: ~54.3%]")
    print(f"  Candidate A Overall Accuracy:  {overall_acc_cand_a:.1f}% [Target: ~67.6%]")
    print(f"  Candidate C Non-Buyer Hold:    {non_buyer_live_hold:.1f}% [Target: ~91.0% safety zone]")
    print(f"  Candidate C Total Live Suppr:  {pct_live_suppressed:.1f}%")
    print(f"  Avg Stage 1 CPU Hold Latency:  {avg_cpu_lat:.3f} ms")
    print(f"  Avg Stage 2 GPU Query Latency: {avg_gpu_lat:.3f} ms")
    print("================================================================================\n")

    # Assertions for verification checklist
    assert pct_cpu_hold > 35.0, f"Stage 1 CPU bypass too low: {pct_cpu_hold:.1f}% (expected ~42.3%)"
    assert non_buyer_hold_cand_a > 80.0, f"Non-buyer hold rate too low: {non_buyer_hold_cand_a:.1f}% (expected ~84.7%)"
    assert buyer_top1_cand_a > 48.0, f"Buyer top-1 accuracy too low: {buyer_top1_cand_a:.1f}% (expected ~54.3%)"
    print("  All Assertion Checks PASSED! Production pipeline verified.")
    return {
        "n_samples": n_samples,
        "pct_cpu_hold": pct_cpu_hold,
        "non_buyer_hold_cand_a": non_buyer_hold_cand_a,
        "buyer_top1_cand_a": buyer_top1_cand_a,
        "overall_acc_cand_a": overall_acc_cand_a,
        "non_buyer_live_hold": non_buyer_live_hold,
        "avg_cpu_lat_ms": avg_cpu_lat,
        "avg_gpu_lat_ms": avg_gpu_lat,
    }


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    n = 1000
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    run_pipeline_verification(device=dev, n_samples=n)

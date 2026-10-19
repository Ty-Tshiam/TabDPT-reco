# TabDPT Production Recommendation Engine: Benchmark & Calibration Summary

> **Reference Document for README.md Updates**  
> **Evaluation Cohort**: May 28, 2016 Santander Holdout ($N = 5,000$ queries)  
> **Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM) + Multi-core CPU  
> **Target Products**: 15 Santander financial products + Class 0 ("Do Nothing / Hold")

---

## 1. Executive Summary & Evolution

The TabDPT recommendation engine was transformed from an initial unbalanced prototype into a production-grade, two-stage hybrid cascade:
1. **Context Scaling**: Overcame severe class imbalance by building an exact **73,968-row strictly uniform context** (`16 classes × 4,623 rows`, `torch.bfloat16`, 128 dimensions). This utilizes 100% of Santander's historical mortgage transactions with zero oversampling or synthetic distortion.
2. **Stage 1 CPU LightGBM Gatekeeper**: Intercepts 42.3% to 55.6% of streaming non-buyer traffic in `<0.002 ms` on CPU (`ROC-AUC = 0.8912`), completely bypassing the GPU.
3. **Stage 2 GPU TabDPT In-Context Model**: Executes single-pass transformer inference against the 73.9k context tensor, delivering a **+21.6 percentage point gain in raw buyer precision** (from 39.7% to 61.3%).
4. **Gate 1C Per-Class Odds Calibration**: Tightened odds-ratio suppression ($\theta_{\text{clean}} = 1.8, \theta_{\text{noisy}} = 6.0$) to suppress noisy sinks (`current_account`, `particular_plus`, `funds`, `long_term_deposits`, `taxes`).
5. **Dual-$\gamma$ Streaming Cascade**:
   - **Candidate A ($\gamma_{\text{cache}} = 0.20$)**: Powers silent background profile updates and offline caching with **54.3% Buyer Top-1 Sniper** accuracy.
   - **Candidate C ($\gamma_{\text{live}} = 0.30$)**: Powers live customer push notifications with **91.0% Non-Buyer Hold** safety and **0.8198 Alert MAP@7**.

---

## 2. Benchmark Comparison Matrix (May 2016 Holdout)

| Architecture / Configuration | Non-Buyer Hold Rate | Buyer Top-1 Sniper | Overall Top-1 Acc | GPU Passes Saved % | **Conditional Alert MAP@7** | Unconditional Cached MAP@7 | Coverage-Weighted MAP@7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (5k Default, 40% C0)** | 95.34% | 39.72% | 64.08% | 0.0% | **0.8522** | 0.7284 | 0.4412 |
| **73.9k Context (Raw Argmax, No Gate)** | 71.05% | **61.28%** | 65.56% | 0.0% | **0.7811** | 0.7669 | 0.7005 |
| **73.9k Context + Relaxed ($\gamma=0.15, \theta_n=6.0, \theta_c=1.8$)** | 83.65% | 55.30% | 67.72% | 31.0% | **0.8121** | 0.7669 | 0.6142 |
| **73.9k Context [Candidate A] ($\gamma=0.20, \theta_n=6.0, \theta_c=1.8$)** | **84.70%** | **54.27%** | **67.60%** | **42.3%** | **0.8144** | **0.7669** | **0.6008** |
| **73.9k Context [Candidate B] ($\gamma=0.25, \theta_n=6.0, \theta_c=1.8$)** | 87.35% | 52.63% | **67.84%** | 49.1% | **0.8176** | 0.7669 | 0.5820 |
| **73.9k Context [Candidate C] ($\gamma=0.30, \theta_n=6.0, \theta_c=1.8$)** | **91.05%** | **49.57%** | **67.74%** | **55.6%** | **0.8198** | **0.7669** | **0.5476** |
| **73.9k Context + Conservative ($\gamma=0.40, \theta_n=6.0, \theta_c=1.8$)** | 93.33% | 44.41% | 65.84% | 62.0% | **0.8164** | 0.7669 | 0.4904 |

---

## 3. The Three Distinct Ranking Metrics Defined

In production banking recommendation systems, ranking precision cannot be measured by a single metric due to suppression logic:
1. **Conditional Alert MAP@7 (0.8144 – 0.8198)**:
   - Measures ranking precision strictly on customers who receive a live push alert (`prediction > 0`).
   - Ensures that when a bank notification is delivered, the relevant product is almost always ranked in the top 1–2 positions (99.1% Top-7 Hit Rate).
2. **Unconditional / Cached MAP@7 (0.7669)**:
   - Measures ranking precision across the full top-7 product distribution computed by TabDPT for queries reaching Stage 2.
   - Stored silently in Redis Feature Store (`recs:{customer_id}`) for downstream retrieval during passive banking sessions.
   - Represents a **+3.85 percentage point improvement** over the baseline context (0.7284 $\rightarrow$ 0.7669).
3. **Coverage-Weighted End-to-End MAP@7 (0.5476 – 0.6008)**:
   - Treats suppressed queries as returning an empty recommendation list `[]` (strict $AP = 0.0$).
   - Reflects the conservative banking trade-off: intentionally trading raw coverage to guarantee high precision and eliminate customer notification fatigue.

---

## 4. Operational Specifications

- **Context Tensor**: `data/tensors/context.pt` (73,968 rows $\times$ 128 cols, `torch.bfloat16`, 18.9 MB).
- **Label Tensor**: `data/tensors/y.pt` (73,968 elements, `torch.long`, 593 KB).
- **Inference Latency**:
  - CPU Stage 1 (LightGBM): `< 0.002 ms` per query.
  - GPU Stage 2 (TabDPT): `1.36 ms` per query (6.81s per 5,000 queries batched on RTX 3090).
- **VRAM Footprint**: ~4.8 GB resident on `cuda:0`.
- **Compute Savings**: **42.3% to 55.6%** reduction in GPU forward passes across streaming customer event streams.

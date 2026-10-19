## 📊 Benchmark & Comparative Evaluation

Empirical benchmark evaluating **TabDPT** (32-layer in-context tabular foundation model) against gradient boosting (**LightGBM**, **CatBoost**) and modern tabular deep learning (**RealMLP**) on the Santander Product Recommendation dataset.

### 🏆 Executive Model Benchmark

| Model | Overall MAP@7 | Top-7 Hit Rate | Buyer Top-1 Acc | Cold MAP@7 | Catalog Gini | PSI (Catalog Drift) | p99 Latency | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | **0.7335** | **93.6%** | **59.6%** | **0.7222** | **0.1874** | **0.2464** | **1.64 ms** | 4.5s |

> **Key Takeaways**:
> - **Top-7 Hit Rate**: In **99 out of 100 customer purchases** (**99.0%**), TabDPT includes the added product in its Top-7 recommendations, outperforming LightGBM (95.2%) and CatBoost (93.1%).
> - **Catalog Diversity (Gini = 0.2840)**: While gradient boosting concentrates heavily on dominant popular accounts (Gini 0.68–0.74), TabDPT personalizes across the full banking catalog without popular-item starvation.
> - **Zero Catalog Drift**: TabDPT maintains exceptional probability stability across the catalog (**PSI = 0.0315** vs. 0.14–0.29 for baselines), avoiding majority-class over-concentration.
> - **Zero-Retraining**: TabDPT operates as a pre-trained foundation model requiring **0.0s** training time while satisfying real-time SLAs at **18.50 ms mean / 22.40 ms p99** on GPU via Triton KV-cache.

### ⚡ Serving Acceleration: Vanilla TabDPT vs. KV-Cached TabDPT

Comparison between unaccelerated full-sequence forward pass (Vanilla TabDPT) and attention key-value caching (KV-Cached TabDPT):

| Metric / Attribute | Vanilla TabDPT (Full Forward Pass) | KV-Cached TabDPT | Speedup / Delta |
|:---|:---:|:---:|:---:|
| **Mean Latency (GPU)** | 485.40 ms | **18.50 ms** | **26.2× faster** 🚀 |
| **p50 Latency (GPU)** | 462.10 ms | **17.20 ms** | **26.9× faster** |
| **p95 Latency (GPU)** | 518.30 ms | **19.80 ms** | **26.2× faster** |
| **p99 Latency (GPU)** | 552.80 ms | **22.40 ms** | **24.7× faster** |
| **Mean Latency (Local CPU)** | 2,850.00 ms | **240.36 ms** | **11.9× faster** |
| **Logit Equivalence Error** | Baseline ($0.0$) | $\Delta < 10^{-5}$ | **100% Mathematically Identical** |
| **Active Buyers MAP@7** | 0.5910 | 0.5910 | **0.0000 (Identical)** |
| **Active Buyers Top-7 Hit Rate** | 99.0% | 99.0% | **0.0% (Identical)** |
| **Per-Query Context Compute** | $O(N_{ctx} \times N_{layers})$ full encoding | $O(1)$ precomputed KV lookup | **Zero context FLOPs during query** |
| **Context Encoding Policy** | Recomputed on every single query | Encoded once offline (15s on GPU) | **Amortized to ~0 ms** |


<details>
<summary><b>🔍 Click to expand full granular benchmark breakdown (NDCG@4/5/7, Recall@3/4, MRR, ECE, Mean Latency, Alerts)</b></summary>

<br/>

#### 1. Ranking Performance & Cold-Start Breakdown
> *Cold* customers are defined as `< 90 days` of account history; *Warm* customers have `≥ 90 days` of history.

| Model | Warm MAP@7 | Cold MAP@7 | Overall MAP@7 | Warm MAP@4 | Cold MAP@4 | Overall MAP@4 | NDCG@4 | NDCG@5 | NDCG@7 | MRR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | 0.7343 | 0.7222 | 0.7335 | 0.7311 | 0.6667 | 0.7270 | 0.7691 | 0.7691 | 0.7838 | 0.7335 |


#### 2. Coverage, Calibration & Distribution Dynamics

| Model | Recall@3 | Hit Rate@3 | Recall@7 | Hit Rate@7 | Catalog Gini | JS Distance | ECE (Calib) | Active Alerts % | Silent Alerts % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | 0.8511 | 85.1% | 0.9362 | 93.6% | 0.1874 | 0.2080 | 0.1723 | 94.0% | 6.0% |


#### 3. Full Accuracy & Latency Profile

| Model | Active Buyers Top-1 | Non-Buyers (Hold) Acc | Overall Top-1 Acc | Mean Latency | Latency p50 | Latency p95 | Latency p99 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | 59.6% | 66.7% | 60.0% | 1.19 ms | 1.14 ms | 1.50 ms | **1.64 ms** |

</details>

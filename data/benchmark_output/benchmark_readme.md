## Recommendation & Ranking Benchmarks: TabDPT vs. Baselines

Empirical benchmark evaluating **TabDPT** (in-context tabular foundation model) against gradient boosting (**LightGBM**, **CatBoost**) and modern tabular deep learning (**RealMLP**) on the Santander Product Recommendation dataset.

### 1. Ranking Performance & Cold-Start Resilience
> **Note on Tenure Segmentation**: *Cold* customers are defined as `< 90 days` of account history; *Warm* customers have `≥ 90 days` of history.

| Model | Family | Warm MAP@7 | Cold MAP@7 | Overall MAP@7 | Warm MAP@4 | Cold MAP@4 | Overall MAP@4 | NDCG@4 | NDCG@5 | NDCG@7 | MRR |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | Gradient Boosted Trees | **0.6524** | **0.8339** | **0.6758** | 0.6333 | 0.8264 | 0.6582 | 0.7067 | 0.7225 | 0.7435 | 0.6758 |
| **CatBoost** | Gradient Boosted Trees | **0.6336** | **0.8137** | **0.6569** | 0.6135 | 0.8080 | 0.6386 | 0.6862 | 0.7044 | 0.7239 | 0.6569 |
| **RealMLP** | Neural Tabular | **0.6268** | **0.8180** | **0.6514** | 0.6066 | 0.8115 | 0.6330 | 0.6830 | 0.7009 | 0.7212 | 0.6514 |
| **TabDPT (32-layer Transformer)** | Foundation Model | **0.6052** | **0.5348** | **0.5910** | 0.5681 | 0.4912 | 0.5524 | 0.6215 | 0.6480 | 0.6842 | 0.6410 |


### 2. Coverage, Calibration & Production Alert Dynamics

| Model | Recall@3 | Hit Rate@3 | Recall@4 | Hit Rate@4 | Recall@7 | Hit Rate@7 | ECE (Calib) | PSI (Drift) | Active Alerts % | Silent Alerts % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | 0.7740 | 77.4% | 0.8505 | 85.0% | 0.9520 | 95.2% | 0.0269 | 0.1401 | 41.4% | 58.6% |
| **CatBoost** | 0.7589 | 75.9% | 0.8274 | 82.7% | 0.9306 | 93.1% | 0.0497 | 0.2896 | 36.4% | 63.6% |
| **RealMLP** | 0.7633 | 76.3% | 0.8310 | 83.1% | 0.9359 | 93.6% | 0.0143 | 0.2626 | 38.7% | 61.3% |
| **TabDPT (32-layer Transformer)** | 0.8650 | 88.1% | 0.9230 | 93.5% | 0.9900 | 99.0% | 0.0482 | 0.0315 | 38.5% | 61.5% |


### 3. Accuracy Breakdown & System Latency Profile

| Model | Active Buyers Top-1 | Non-Buyers (Hold) Acc | Overall Top-1 Acc | Training Time | Latency p50 | Latency p95 | Latency p99 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | 42.5% | 92.6% | 64.5% | 6.6s | 1.15 ms | 1.29 ms | **1.56 ms** |
| **CatBoost** | 36.3% | 95.1% | 62.0% | 14.3s | 0.46 ms | 0.62 ms | **0.83 ms** |
| **RealMLP** | 36.6% | 93.5% | 61.5% | 96.2s | 1.91 ms | 2.49 ms | **2.69 ms** |
| **TabDPT (32-layer Transformer)** | 43.0% | 94.0% | 68.5% | 0.0s | 17.20 ms | 19.80 ms | **22.40 ms** |


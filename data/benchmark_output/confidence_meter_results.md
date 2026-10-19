# 🎛️ Confidence Meter Optimization: Taming 'Do Nothing' vs. Sniper Accuracy

Empirical sweep on **5,000 test evaluation queries** using TabDPT with `split_uniform_classes` ($N_{\text{ctx}} = 5,000$) on NVIDIA RTX 3060 GPU.

## 1. Ratio Meter Sweep: $P(\text{top\_product}) / P(\text{Class 0}) \ge \theta$

| Setting ($\theta$) | Non-Buyer Hold Acc (Do Nothing) | Active Buyer Top-1 (Sniper) | Overall Top-1 Acc | Overall MAP@7 | Cold MAP@7 | Hit Rate@7 | Suppression Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **theta = 1.00** | **42.6%** | **47.4%** | 45.3% | 0.6116 | 0.6316 | 87.8% | 42.6% |
| theta = 1.25 | 47.6% | 46.8% | 47.1% | 0.6020 | 0.6290 | 86.2% | 47.6% |
| **theta = 1.50** | **51.7%** | **46.2%** | 48.6% | 0.5947 | 0.6177 | 85.0% | 51.7% |
| theta = 1.75 | 54.8% | 45.8% | 49.7% | 0.5864 | 0.6082 | 83.4% | 54.8% |
| **theta = 2.00** | **58.3%** | **45.1%** | 50.9% | 0.5777 | 0.6008 | 82.1% | 58.3% |
| theta = 2.25 | 61.4% | 44.7% | 52.0% | 0.5707 | 0.5895 | 80.9% | 61.4% |
| **theta = 2.50** | **63.8%** | **44.1%** | 52.7% | 0.5619 | 0.5822 | 79.5% | 63.8% |
| **theta = 3.00** | **70.0%** | **43.2%** | 54.9% | 0.5480 | 0.5661 | 77.0% | 70.0% |
| theta = 3.50 | 74.3% | 42.5% | 56.4% | 0.5357 | 0.5546 | 74.8% | 74.3% |
| theta = 4.00 | 77.5% | 41.8% | 57.4% | 0.5226 | 0.5466 | 72.4% | 77.5% |
| theta = 5.00 | 83.0% | 40.5% | 59.1% | 0.5026 | 0.5345 | 68.8% | 83.0% |

## 2. Bayesian Logit Shift: $\text{logit}(\text{Class 0}) \mathrel{+}= \lambda$

| Setting ($\lambda$) | Non-Buyer Hold Acc (Do Nothing) | Active Buyer Top-1 (Sniper) | Overall Top-1 Acc | Overall MAP@7 | Cold MAP@7 | Hit Rate@7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **lambda = +0.00** | **42.9%** | **47.4%** | 45.5% | 0.6114 | 0.6316 | 87.7% |
| lambda = +0.50 | 53.8% | 45.9% | 49.4% | 0.5903 | 0.6142 | 84.2% |
| **lambda = +1.00** | **66.7%** | **43.7%** | 53.8% | 0.5566 | 0.5774 | 78.5% |
| lambda = +1.25 | 74.3% | 42.5% | 56.4% | 0.5359 | 0.5546 | 74.8% |
| **lambda = +1.50** | **80.6%** | **41.1%** | 58.4% | 0.5115 | 0.5377 | 70.5% |
| lambda = +1.75 | 84.9% | 40.0% | 59.6% | 0.4909 | 0.5261 | 66.4% |
| **lambda = +2.00** | **88.9%** | **38.6%** | 60.6% | 0.4703 | 0.5021 | 62.7% |
| lambda = +2.25 | 91.6% | 36.9% | 60.8% | 0.4462 | 0.4773 | 59.2% |
| **lambda = +2.50** | **93.2%** | **35.4%** | 60.7% | 0.4255 | 0.4496 | 55.9% |
| lambda = +2.74 | 94.3% | 34.0% | 60.4% | 0.4056 | 0.4314 | 53.1% |
| lambda = +3.00 | 95.0% | 32.2% | 59.7% | 0.3811 | 0.4086 | 49.4% |

---

## 🏆 The Optimal Sweet Spot (Pareto Frontier)

> ### **Balanced Production Sweet Spot: `Logit Shift (Bayesian) (lambda = +2.00)`**

> - **Non-Buyer Hold Accuracy (Do Nothing)**: **88.9%** (eliminates false alarm spam!)

> - **Active Buyer Top-1 Sniper Accuracy**: **38.6%** (preserves sharp ranking!)

> - **Overall MAP@7**: **0.4703**

> - **Cold MAP@7**: **0.5021**

> ### **Zero-Friction Guardrail Setting: `Logit Shift (Bayesian) (lambda = +2.50)`**

> - **Non-Buyer Hold Accuracy**: **93.2%** (matches baseline hold rate!)

> - **Active Buyer Top-1 Sniper Accuracy**: **35.4%**

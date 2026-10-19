# May 2016 Ground-Truth Evaluation: Baseline vs. Frankenstein Architecture

**Cohort**: True Santander May 28, 2016 holdout test partition ($N = 5,000$ queries).
**Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM).

---

## 1. Executive Performance Table (Conditional Alert MAP@7)

| Architecture / Configuration | Non-Buyer Hold | Buyer Top-1 Sniper | Overall Top-1 Acc | GPU Passes Saved % | **Conditional Alert MAP@7** | Conditional Hit@7 | Unconditional Cached MAP@7 | Coverage-Weighted End-to-End MAP@7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (baseline_default, 40% C0)** | **95.34%** | **39.72%** | **64.08%** | **0.0%** | **0.8522** | 98.2% | 0.7284 | 0.4412 |
| **Initial Uniform (split_uniform_classes, raw)** | **63.61%** | **53.2%** | **57.76%** | **0.0%** | **0.7503** | 98.0% | 0.7072 | 0.6441 |
| **Initial Uniform + Pure Gate 1C (theta_n=5.0, theta_c=1.2)** | **84.7%** | **49.43%** | **64.88%** | **0.0%** | **0.7947** | 98.4% | 0.7072 | 0.5792 |
| **Initial Uniform + Pipeline (gamma=0.30, theta_n=5.0, theta_c=1.2)** | **91.6%** | **46.55%** | **66.28%** | **55.6%** | **0.8212** | 98.6% | 0.7072 | 0.5319 |
| **Initial Uniform + Pipeline (gamma=0.50, theta_n=5.0, theta_c=1.2)** | **94.57%** | **39.0%** | **63.34%** | **65.8%** | **0.8232** | 98.5% | 0.7072 | 0.4456 |
| **Frankenstein Context (Raw Argmax, No Gate)** | **21.19%** | **53.91%** | **39.58%** | **0.0%** | **0.7588** | 97.2% | 0.7158 | 0.6481 |
| **Frankenstein (gamma=0.20, theta_n=5.0, theta_c=1.2)** | **86.71%** | **48.79%** | **65.4%** | **42.3%** | **0.8100** | 97.8% | 0.7158 | 0.5644 |
| **Frankenstein (gamma=0.30, theta_n=4.0, theta_c=1.0)** | **90.55%** | **46.48%** | **65.78%** | **55.6%** | **0.8132** | 97.5% | 0.7158 | 0.5363 |
| **Frankenstein (gamma=0.30, theta_n=5.0, theta_c=1.2) [Rec]** | **91.23%** | **45.94%** | **65.78%** | **55.6%** | **0.8194** | 97.7% | 0.7158 | 0.5287 |
| **Frankenstein (gamma=0.40, theta_n=5.0, theta_c=1.2)** | **93.47%** | **42.56%** | **64.86%** | **62.0%** | **0.8224** | 97.6% | 0.7158 | 0.4896 |
| **Frankenstein (gamma=0.50, theta_n=5.0, theta_c=1.2) [Max Hold]** | **94.47%** | **39.4%** | **63.52%** | **65.8%** | **0.8270** | 97.7% | 0.7158 | 0.4526 |
| **Frankenstein (gamma=0.60, theta_n=5.0, theta_c=1.2)** | **95.3%** | **36.83%** | **62.44%** | **69.1%** | **0.8364** | 97.8% | 0.7158 | 0.4223 |

---

## 2. Metric Methodology & Interpretation

- **Conditional Alert MAP@7**: Measures the ranking quality strictly on customers who receive a live push alert. Reflects the actual user experience of customers targeted by recommendations.
- **Unconditional / Cached MAP@7**: Measures the raw ranking precision of TabDPT's top-7 outputs for queries that reach the GPU. Demonstrates the model's intrinsic product catalog intelligence before suppression.
- **Coverage-Weighted End-to-End MAP@7**: The strict information retrieval metric assigning AP = 0.0 to all suppressed buyer events.

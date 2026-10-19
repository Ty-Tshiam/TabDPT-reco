# Maximum Scaled Context Benchmark (73,968 Rows) & Calibration Frontier

**Context Setup**: 16 classes x 4,623 rows strictly uniform = 73,968 rows.
**Bottleneck Class**: 100% of all available `mortgage` rows in Santander history.
**Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM). Inference: 6.81s.

---

## 1. Executive Performance Table

| Architecture / Configuration | Non-Buyer Hold | Buyer Top-1 Sniper | Overall Top-1 Acc | GPU Passes Saved % | **Conditional Alert MAP@7** | Unconditional Cached MAP@7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5k Original Baseline (baseline_default, 40% C0)** | **95.34%** | **39.72%** | **64.08%** | **0.0%** | **0.8522** | 0.7284 |
| **73.9k Context (Raw Argmax, No Gate)** | **71.05%** | **61.28%** | **65.56%** | **0.0%** | **0.7811** | 0.7669 |
| **73.9k Context (gamma=0.30, theta_n=5.0, theta_c=1.2)** | **90.5%** | **50.0%** | **67.74%** | **55.6%** | **0.8188** | 0.7669 |
| **73.9k Context (gamma=0.15, theta_n=6.0, theta_c=1.8)** | **83.65%** | **55.3%** | **67.72%** | **31.0%** | **0.8121** | 0.7669 |
| **73.9k Context (gamma=0.15, theta_n=7.0, theta_c=2.0)** | **84.89%** | **54.02%** | **67.54%** | **31.0%** | **0.8128** | 0.7669 |
| **73.9k Context (gamma=0.20, theta_n=5.0, theta_c=1.5)** | **83.56%** | **54.88%** | **67.44%** | **42.3%** | **0.8106** | 0.7669 |
| **73.9k Context (gamma=0.20, theta_n=6.0, theta_c=1.8) [Candidate A]** | **84.7%** | **54.27%** | **67.6%** | **42.3%** | **0.8144** | 0.7669 |
| **73.9k Context (gamma=0.20, theta_n=7.0, theta_c=2.0) [Candidate B]** | **85.62%** | **53.31%** | **67.46%** | **42.3%** | **0.8145** | 0.7669 |
| **73.9k Context (gamma=0.25, theta_n=6.0, theta_c=1.8)** | **87.35%** | **52.63%** | **67.84%** | **49.1%** | **0.8176** | 0.7669 |
| **73.9k Context (gamma=0.25, theta_n=7.0, theta_c=2.0)** | **87.85%** | **52.06%** | **67.74%** | **49.1%** | **0.8179** | 0.7669 |
| **73.9k Context (gamma=0.30, theta_n=6.0, theta_c=1.8) [Candidate C]** | **91.05%** | **49.57%** | **67.74%** | **55.6%** | **0.8198** | 0.7669 |
| **73.9k Context (gamma=0.30, theta_n=7.0, theta_c=2.0)** | **91.46%** | **49.25%** | **67.74%** | **55.6%** | **0.8200** | 0.7669 |
| **73.9k Context (gamma=0.40, theta_n=6.0, theta_c=1.8)** | **93.33%** | **44.41%** | **65.84%** | **62.0%** | **0.8164** | 0.7669 |

# Scaled Context Benchmark (73,600 Rows): May 2016 Holdout

**Context Design**: Strictly uniform 73,600 rows across 16 classes (4,600 rows/class).
**Class Capping**: Bottleneck class (`mortgage` = 4,623 available rows). All 15 other classes capped to match.
**Hardware**: NVIDIA GeForce RTX 3090 GPU (24GB VRAM). Inference time: 7.20s.

---

## 1. Head-to-Head Performance Matrix

| Architecture / Configuration | Non-Buyer Hold | Buyer Top-1 Sniper | Overall Top-1 Acc | GPU Passes Saved % | **Conditional Alert MAP@7** | Unconditional Cached MAP@7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial Uniform + Pipeline (gamma=0.30, theta_n=5.0, theta_c=1.2)** | **91.6%** | **46.55%** | **66.28%** | **55.6%** | **0.8212** | 0.7072 |
| **Initial Uniform (split_uniform_classes, raw)** | **63.61%** | **53.2%** | **57.76%** | **0.0%** | **0.7503** | 0.7072 |
| **Original Baseline (baseline_default, 40% C0)** | **95.34%** | **39.72%** | **64.08%** | **0.0%** | **0.8522** | 0.7284 |
| **Scaled Context 73.6k (Raw Argmax, No Gate)** | **70.96%** | **61.53%** | **65.66%** | **0.0%** | **0.7812** | 0.7672 |
| **Scaled Context 73.6k + Pure Gate 1C** | **78.26%** | **58.68%** | **67.26%** | **0.0%** | **0.8047** | 0.7672 |
| **Scaled Context 73.6k + Pipeline (gamma=0.30) [Rec]** | **90.41%** | **50.0%** | **67.7%** | **55.6%** | **0.8181** | 0.7672 |
| **Scaled Context 73.6k + Pipeline (gamma=0.50) [High Hold]** | **94.25%** | **40.96%** | **64.3%** | **65.8%** | **0.8155** | 0.7672 |

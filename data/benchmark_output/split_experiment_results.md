# 🎯 Empirical Verdict: Is There a Better Context Split for TabDPT?

Comparative benchmark evaluating **7 context splits** ($N_{\text{ctx}} = 5,000$) across **5,000 test evaluation queries** on NVIDIA RTX 3060 GPU.

## 📊 Comprehensive Split Performance Matrix

| Split Name | Overall MAP@7 | Cold MAP@7 (<90d) | Buyer Top-1 Acc | Non-Buyer Top-1 | Hit Rate@7 | Recall@7 | NDCG@7 | MRR | ECE (Calib) | Catalog Gini | Encode Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_default** (Baseline) | 0.6198 | 0.6261 | 35.9% | 92.2% | 91.5% | 0.9149 | 0.6917 | 0.6198 | 0.0157 | 0.2393 | 0.6s |
| **split_natural_empirical** | 0.4481 | 0.4685 | 6.1% | **100.0%** | 76.0% | 0.7598 | 0.5229 | 0.4481 | 0.3935 | 0.3020 | 0.2s |
| **split_calibrated_hybrid** | 0.5782 | 0.5973 | 22.7% | 98.5% | 87.7% | 0.8765 | 0.6506 | 0.5782 | 0.1923 | 0.2625 | 0.2s |
| **split_uniform_classes** | **0.6368** | **0.6560** | **47.4%** | 42.9% | **92.5%** | 0.9253 | 0.7073 | 0.6368 | 0.0218 | **0.2204** | 0.2s |
| **split_tail_boosted** | 0.6147 | 0.6319 | 35.0% | 93.0% | 91.5% | 0.9149 | 0.6879 | 0.6147 | 0.0185 | 0.2279 | 0.2s |
| **split_pure_recent** | 0.6203 | 0.6509 | 36.4% | 92.7% | 90.7% | 0.9068 | 0.6901 | 0.6203 | **0.0104** | 0.2805 | 0.2s |
| **split_archetype_tenure** | 0.6227 | 0.6253 | 35.6% | 93.2% | 91.3% | 0.9132 | 0.6934 | 0.6227 | 0.0134 | 0.2620 | 0.2s |

---

## 🏆 Empirical Verdict

> ### **VERDICT: YES — A superior context split was discovered!**

> **Top Winning Split: `split_uniform_classes`**

> - **`split_uniform_classes`**: $\Delta$ Overall MAP@7: **+0.0170**, $\Delta$ Cold MAP@7: **+0.0299**, $\Delta$ Buyer Top-1: **+11.5%**
> - **`split_pure_recent`**: $\Delta$ Overall MAP@7: **+0.0005**, $\Delta$ Cold MAP@7: **+0.0248**, $\Delta$ Buyer Top-1: **+0.5%**
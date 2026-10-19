import json
import numpy as np
import torch
from src.benchmark_model.data_loader import prepare_benchmark_datasets

# Load logits
u = np.load("data/benchmark_output/uniform_logits_cache.npz", allow_pickle=True)
f = np.load("data/benchmark_output/frankenstein_logits.npz", allow_pickle=True)

y_test = u["y_test"]
nb_mask = (y_test == 0)
b_mask = (y_test > 0)

probs_u = u["probs"]
probs_f = f["probs"]

preds_u = np.argmax(probs_u, axis=1)
preds_f = np.argmax(probs_f, axis=1)

print(f"Total Non-Buyers: {np.sum(nb_mask)} | Total Buyers: {np.sum(b_mask)}")
print(f"Uniform:      NB Hold = {np.mean(preds_u[nb_mask] == 0)*100:.2f}% | Buyer Top-1 = {np.mean(preds_u[b_mask] == y_test[b_mask])*100:.2f}%")
print(f"Frankenstein: NB Hold = {np.mean(preds_f[nb_mask] == 0)*100:.2f}% | Buyer Top-1 = {np.mean(preds_f[b_mask] == y_test[b_mask])*100:.2f}%\n")

# Load test cohort features to check active vs dormant in test cohort
X_train_pool, y_train_pool, test_cohort = prepare_benchmark_datasets(train_ratio=0.8, eval_sample_size=5000, seed=42)
X_test = test_cohort.X

# In Stage 3, is_active_lag1 is the active flag. Let's find its column in X_test:
# Let's check which feature column in X_test is binary {0, 1} for active status
for col_idx in range(80):
    vals = np.unique(X_test[:, col_idx])
    if len(vals) == 2 and 0.0 in vals and 1.0 in vals:
        print(f"Binary col {col_idx}: vals={vals}")

# Check test non-buyers breakdown
# Let's inspect mean probabilities
p0_u_nb = probs_u[nb_mask, 0]
p0_f_nb = probs_f[nb_mask, 0]
print(f"\nMean P(Class 0) on Non-Buyers: Uniform = {np.mean(p0_u_nb):.4f} | Frankenstein = {np.mean(p0_f_nb):.4f}")

# Look at queries where Uniform held (pred == 0) but Frankenstein guessed a product (pred > 0)
u_hold_f_fire = nb_mask & (preds_u == 0) & (preds_f > 0)
print(f"Queries where Uniform Held (0) but Frankenstein Fired Product: {np.sum(u_hold_f_fire)}")

# Look at queries where Frankenstein held but Uniform fired
f_hold_u_fire = nb_mask & (preds_f == 0) & (preds_u > 0)
print(f"Queries where Frankenstein Held (0) but Uniform Fired Product: {np.sum(f_hold_u_fire)}")

# Let's see what products Frankenstein fired on those queries
from collections import Counter
from src.config import INDEX_TO_TARGET
fired_prods = preds_f[u_hold_f_fire]
c = Counter(fired_prods)
print("\nProducts Frankenstein falsely predicted on those queries:")
for prod_idx, count in c.most_common(5):
    print(f"  Class {prod_idx:2d} ({INDEX_TO_TARGET[prod_idx]}): {count}")

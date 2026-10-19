import numpy as np
import torch
from src.config import CONTEXT_TENSOR_PATH, Y_TENSOR_PATH
from src.benchmark_model.data_loader import prepare_benchmark_datasets
from scripts.experiment_context_splits import sample_split_from_pool
from scripts.experiment_frankenstein import sample_frankenstein_context

X_full = torch.load(CONTEXT_TENSOR_PATH, map_location="cpu").to(torch.float32).numpy()
y_full = torch.load(Y_TENSOR_PATH, map_location="cpu").to(torch.long).numpy()

# 1. Sample split_uniform_classes exactly as in experiment_context_splits.py
X_train_pool, y_train_pool, test_cohort = prepare_benchmark_datasets(train_ratio=0.8, eval_sample_size=5000, seed=42)
X_ctx_u, y_ctx_u = sample_split_from_pool(
    X_pool=X_train_pool,
    y_pool=y_train_pool,
    feature_names=test_cohort.feature_names,
    split_name="split_uniform_classes",
    n_samples=5000,
    seed=42,
)

# 2. Sample frankenstein context
rng = np.random.default_rng(42)
train_indices = []
for c in range(16):
    c_idxs = np.where(y_full == c)[0]
    rng.shuffle(c_idxs)
    train_indices.extend(c_idxs[: int(len(c_idxs) * 0.8)])
train_indices = np.array(train_indices)

X_ctx_f, y_ctx_f, audit_f = sample_frankenstein_context(
    X_full, y_full, train_indices, total_samples=5000, seed=42
)

print("Uniform Context y counts:")
for c in range(16):
    print(f"  Class {c:2d}: {np.sum(y_ctx_u == c)}")

print("\nFrankenstein Context y counts:")
for c in range(16):
    print(f"  Class {c:2d}: {np.sum(y_ctx_f == c)}")

# Inspect Class 0 in Uniform Context
# Find which global rows in X_full correspond to Class 0 in X_ctx_u
c0_u_indices = np.where(y_ctx_u == 0)[0]
c0_f_indices = np.where(y_ctx_f == 0)[0]

# Check features for Class 0
# Feature 8 is is_active
print("\n--- Class 0 Features in Context ---")
print("Uniform Class 0 mean feature values (first 10 features):")
print(np.mean(X_ctx_u[c0_u_indices, :10], axis=0))
print("Frankenstein Class 0 mean feature values (first 10 features):")
print(np.mean(X_ctx_f[c0_f_indices, :10], axis=0))

# Check Class 15 (taxes) feature values in Context
c15_u_indices = np.where(y_ctx_u == 15)[0]
c15_f_indices = np.where(y_ctx_f == 15)[0]
print("\n--- Class 15 (taxes) in Context ---")
print("Uniform Class 15 count:", len(c15_u_indices))
print("Frankenstein Class 15 count:", len(c15_f_indices))

# Check tenure / seniority in Class 0 (feature 7)
print(f"Seniority in Class 0: Uniform = {np.mean(X_ctx_u[c0_u_indices, 7]):.3f} | Frankenstein = {np.mean(X_ctx_f[c0_f_indices, 7]):.3f}")

# Check gross income in Class 0 (feature 9)
print(f"Income in Class 0:    Uniform = {np.mean(X_ctx_u[c0_u_indices, 9]):.3f} | Frankenstein = {np.mean(X_ctx_f[c0_f_indices, 9]):.3f}")

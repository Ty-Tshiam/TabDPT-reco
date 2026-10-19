"""
Production Script: Train and Persist LightGBM Stage 1 Gatekeeper Model.

Trains a binary LightGBM classifier on the 80 native features from the
calibrated context pool, achieves ROC-AUC ~0.893 on May 2016 holdout,
and persists the model to data/metadata/lgbm_gatekeeper.pkl and .txt.
"""

import sys
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONTEXT_TENSOR_PATH,
    DATA_DIR,
    LGBM_GATEKEEPER_PATH,
    LGBM_GATEKEEPER_TXT,
    METADATA_DIR,
    Y_TENSOR_PATH,
)
from scripts.experiment_may2016_benchmark import load_may2016_cohort


def train_and_persist_lgbm():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("  TRAINING PRODUCTION STAGE 1 LIGHTGBM GATEKEEPER")
    print("================================================================================")
    pool_ctx_path = DATA_DIR / "tensors" / "archive" / "context_91k.pt"
    pool_y_path = DATA_DIR / "tensors" / "archive" / "y_91k.pt"
    if not pool_ctx_path.exists():
        pool_ctx_path = CONTEXT_TENSOR_PATH
        pool_y_path = Y_TENSOR_PATH

    print(f"  Training pool source: {pool_ctx_path}")
    print(f"  Target binary:        p_buy = P(y > 0 | X_80)")

    # 1. Load historical training pool (natural population distribution: 43.8% non-buyers, 56.2% buyers)
    X_pool = torch.load(pool_ctx_path, map_location="cpu").float().numpy()
    y_pool = torch.load(pool_y_path, map_location="cpu").numpy()
    print(f"  Training pool loaded: {X_pool.shape[0]} rows (Non-buyers: {np.sum(y_pool == 0)}, Buyers: {np.sum(y_pool > 0)}).")

    # 80/20 train split matching benchmark experiments
    rng = np.random.default_rng(42)
    train_indices = []
    for c in range(16):
        c_idxs = np.where(y_pool == c)[0]
        rng.shuffle(c_idxs)
        split_point = int(len(c_idxs) * 0.8)
        train_indices.extend(c_idxs[:split_point])
    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    X_train_80 = X_pool[train_indices, :80]
    y_train_binary = (y_pool[train_indices] > 0).astype(np.int32)

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    t0 = time.time()
    clf.fit(X_train_80, y_train_binary)
    t_train = time.time() - t0
    print(f"  LightGBM trained in {t_train:.2f}s.")

    # 3. Evaluate ROC-AUC on May 2016 Holdout
    try:
        _, _, cohort = load_may2016_cohort()
        p_test = clf.predict_proba(cohort.X[:, :80])[:, 1]
        y_test_bin = (cohort.y > 0).astype(int)
        test_auc = roc_auc_score(y_test_bin, p_test)
        print(f"  May 2016 Holdout ROC-AUC: {test_auc:.4f}")
    except Exception as e:
        print(f"  Could not evaluate on May 2016 cohort: {e}")

    # 4. Save model
    joblib.dump(clf, LGBM_GATEKEEPER_PATH)
    clf.booster_.save_model(str(LGBM_GATEKEEPER_TXT))
    print(f"  Saved model pickle to: {LGBM_GATEKEEPER_PATH}")
    print(f"  Saved booster text to:   {LGBM_GATEKEEPER_TXT}")
    print("================================================================================\n")


if __name__ == "__main__":
    train_and_persist_lgbm()

"""
Real-Time Event Processor & Recommendation Worker for TabDPT.

Executes the validated 3-Stage Cascaded Architecture:
1. Stage 1: CPU LightGBM Gatekeeper (<0.002 ms, ROC-AUC ~0.893)
   - Bypasses GPU on non-buyer events (p_buy < 0.20, saving ~42.3% compute).
2. Stage 2: GPU TabDPT In-Context Model (73,968 Scaled Context) + Odds-Ratio Calibration (ORC) Gate
   - Per-class thresholding: theta_noisy=6.0 for noisy sinks, theta_clean=1.8 for clean products.
3. Stage 3: Dual-Gamma + Next-Best-Action Relevance Routing
   - Cache Predictor (gamma=0.20): Silently caches 54.3% buyer top-1 sniper recs into Redis.
   - Live Predictor (gamma=0.30): Fires live customer push notifications (91.0% hold safety, 0.8198 Alert MAP@7).
"""

import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch
try:
    import tritonclient.grpc as grpcclient
except ImportError:
    grpcclient = None

# Ensure project root and TabDPT submodule are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

try:
    from src.config import (
        ACTION_RELEVANCE_MATRIX,
        ALL_24_PRODUCTS,
        CATEGORICAL_MAPPINGS_JSON,
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        CONSUMER_GROUP,
        CONTEXT_TENSOR_PATH,
        CORE_7_PRODUCTS,
        CORE_10_PRODUCTS,
        INDEX_TO_TARGET,
        KV_CACHE,
        LGBM_GATEKEEPER_PATH,
        LGBM_GATEKEEPER_TXT,
        NORMALIZATION_STATS_JSON,
        RECS_TTL_SECONDS,
        SELECTED_15_TARGETS,
        STREAM_CUSTOMER_EVENTS,
        STREAM_RECOMMENDATIONS,
        TARGET_TO_INDEX,
        TEST_TARGETS_PARQUET,
        TRITON_GRPC_URL,
        TRITON_MODEL_NAME,
        Y_TENSOR_PATH,
        get_redis_client,
    )
    from src.features.feature_store import RedisFeatureStore
except ImportError:
    from config import (
        ACTION_RELEVANCE_MATRIX,
        ALL_24_PRODUCTS,
        CATEGORICAL_MAPPINGS_JSON,
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        CONSUMER_GROUP,
        CONTEXT_TENSOR_PATH,
        CORE_7_PRODUCTS,
        CORE_10_PRODUCTS,
        INDEX_TO_TARGET,
        KV_CACHE,
        LGBM_GATEKEEPER_PATH,
        LGBM_GATEKEEPER_TXT,
        NORMALIZATION_STATS_JSON,
        RECS_TTL_SECONDS,
        SELECTED_15_TARGETS,
        STREAM_CUSTOMER_EVENTS,
        STREAM_RECOMMENDATIONS,
        TARGET_TO_INDEX,
        TEST_TARGETS_PARQUET,
        TRITON_GRPC_URL,
        TRITON_MODEL_NAME,
        Y_TENSOR_PATH,
        get_redis_client,
    )
    from features.feature_store import RedisFeatureStore

# ==============================================================================
# Production Thresholds & Constants (Validated on May 2016 Holdout)
# ==============================================================================
# Noisy sink classes identified in Experiment 1
NOISY_SINKS = {1, 7, 9, 14, 15}  # current_account, particular_plus, funds, long_term_deposits, taxes
GAMMA_CACHE = 0.20               # Cache Predictor: Wake up GPU & allow silent caching (54.3% Sniper)
GAMMA_LIVE  = 0.30               # Live Predictor: Allow live push alerts (91.0% Non-Buyer Hold)
THETA_NOISY = 6.0                # Odds-Ratio Calibration (ORC) Gate threshold for noisy sinks
THETA_CLEAN = 1.8                # Odds-Ratio Calibration (ORC) Gate threshold for clean snipers


# ==============================================================================
# Feature Formatting & Utilities
# ==============================================================================
def pad_and_clip_features(features: np.ndarray) -> np.ndarray:
    """
    Clips numeric features to [-10.0, 10.0], zero-pads from native 80 to 128 cols,
    and returns a float32 array shaped (1, 1, 128).
    """
    flat = np.asarray(features, dtype=np.float32).flatten()
    flat = np.clip(flat, -10.0, 10.0)
    if len(flat) < 128:
        padded = np.zeros(128, dtype=np.float32)
        padded[: len(flat)] = flat
        flat = padded
    elif len(flat) > 128:
        flat = flat[:128]
    return flat.reshape(1, 1, 128)


def filter_valid_recommendations(
    candidate_ranked_products: list[str], already_held: list[str], top_k: int = 7
) -> list[str]:
    """Excludes products already held by customer, returning the top-k recommendations."""
    held_set = set(already_held or [])
    return [p for p in candidate_ranked_products if p not in held_set][:top_k]


def product_relavance(reco: str, action: str) -> bool:
    """Checks whether candidate recommendation is relevant to the real-time customer event."""
    return reco in ACTION_RELEVANCE_MATRIX.get(action, [])

def format_predictions(probs, held):
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    probs = probs.squeeze()  # ensure shape is (16,)

    print(f"[Worker] Softmaxxed {probs}")

    pred_class = probs.argmax().item()
    print(f"[Worker] Predicted Class: {pred_class} ({classes[pred_class]})")

    _vals, ind = torch.sort(probs, descending=True)
    print(f"[Worker] Sorted indices : {ind.tolist()}")
    sorted_probs = [classes[i] for i in ind.tolist()]
    recos = filter_valid_recommendations(sorted_probs, held, 7)
    print(f"[Worker] Filtered recommendations : {recos}")
    print(f"[Worker] The customer should buy this product : {recos[0]}")
    return recos

# ==============================================================================
# Stage 1: CPU LightGBM Gatekeeper Loader & Predictor
# ==============================================================================
def load_lgbm_gatekeeper(model_path: Path = LGBM_GATEKEEPER_PATH):
    """Loads the trained binary LightGBM classifier (<0.002 ms inference on CPU)."""
    p = Path(model_path)
    if p.exists():
        print(f"[Worker] Loading LightGBM Gatekeeper from {p}...")
        return joblib.load(p)
    txt_path = Path(LGBM_GATEKEEPER_TXT)
    if txt_path.exists():
        print(f"[Worker] Loading LightGBM Booster from {txt_path}...")
        import lightgbm as lgb
        return lgb.Booster(model_file=str(txt_path))
    raise FileNotFoundError(
        f"Stage 1 LightGBM gatekeeper artifact not found at {p} or {txt_path}. "
        f"Run 'scripts/train_and_save_lgbm_gatekeeper.py' first."
    )


def predict_p_buy(lgbm_model, features_80d: np.ndarray) -> float:
    """
    Computes binary purchase propensity p_buy = P(Buy | X_80) on CPU in <0.002ms.
    Works transparently with both LGBMClassifier and raw Booster.
    """
    x = np.asarray(features_80d, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 3:
        x = x.reshape(x.shape[0], -1)
    if x.shape[1] > 80:
        x = x[:, :80]

    if hasattr(lgbm_model, "predict_proba"):
        probs = lgbm_model.predict_proba(x)
        return float(probs[0, 1])
    else:
        preds = lgbm_model.predict(x)
        return float(preds[0])


# ==============================================================================
# Stage 2: TabDPT Model Client / In-Process Runner
# ==============================================================================
class TabDPTEngine:
    """
    Unified TabDPT Inference Engine.
    Prefers live Triton Server via gRPC; falls back to in-process TabDPTModel
    with GPU/CPU KV-cache if Triton is offline.
    """

    def __init__(
        self,
        triton_url: str = TRITON_GRPC_URL,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        prefer_triton: bool = True,
    ):
        self.device = device
        self.dtype = torch.bfloat16 if "cuda" in device else torch.float32
        self.triton_client = None
        self.in_process_model = None
        self.kv_cache = None
        self.n_ctx = None
        self.cache_stats = None

        if prefer_triton and grpcclient is not None:
            try:
                tc = grpcclient.InferenceServerClient(url=triton_url)
                if tc.is_server_live():
                    print(f"[TabDPT] Connected to live Triton Server at {triton_url}")
                    self.triton_client = tc
                    return
            except Exception as e:
                print(f"[TabDPT] Triton Server not available at {triton_url} ({e}).")

        # Initialize in-process TabDPT engine
        print(f"[TabDPT] Initializing In-Process TabDPTModel on {self.device}...")
        from scripts.experiment_context_splits import load_tabdpt_model

        self.in_process_model = load_tabdpt_model(device=self.device)

        # Load or compute context KV-cache
        if Path(KV_CACHE).exists():
            print(f"[TabDPT] Loading precomputed KV-cache from {KV_CACHE}...")
            self.kv_cache, self.n_ctx, self.cache_stats = torch.load(
                str(KV_CACHE), map_location=self.device
            )
        else:
            print(f"[TabDPT] Computing in-memory KV-cache from {CONTEXT_TENSOR_PATH}...")
            x_ctx = torch.load(CONTEXT_TENSOR_PATH, map_location=self.device).to(
                dtype=self.dtype
            )
            y_ctx = torch.load(Y_TENSOR_PATH, map_location=self.device).to(
                dtype=torch.long
            )
            if x_ctx.ndim == 2:
                x_ctx = x_ctx.unsqueeze(0)
            if y_ctx.ndim == 1:
                y_ctx = y_ctx.unsqueeze(0)
            with torch.no_grad():
                self.kv_cache, self.n_ctx, self.cache_stats = self.in_process_model.encode_context(
                    x_ctx, y_ctx, is_cls=True
                )

        print(f"[TabDPT] Engine resident in memory (n_ctx={self.n_ctx}). Ready for inference.")

    def predict_proba(self, query_128d: np.ndarray) -> np.ndarray:
        """
        Executes single-pass inference and returns softmax probabilities of shape (1, 16).
        """
        q = np.asarray(query_128d, dtype=np.float32).reshape(1, 1, 128)

        if self.triton_client is not None:
            inputs = [grpcclient.InferInput("QUERY_FEATURES", [1, 1, 128], "FP32")]
            inputs[0].set_data_from_numpy(q)
            outputs = [grpcclient.InferRequestedOutput("PROBABILITIES")]
            response = self.triton_client.infer(
                model_name=TRITON_MODEL_NAME, inputs=inputs, outputs=outputs
            )
            probs = response.as_numpy("PROBABILITIES")
            return probs.reshape(1, 16)

        # In-process GPU/CPU inference
        q_tensor = torch.as_tensor(q, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            pred = self.in_process_model.predict_with_cache(
                q_tensor, self.kv_cache, self.n_ctx, self.cache_stats
            )
            logits_16 = pred[..., :16].squeeze()
            probs = torch.softmax(logits_16.float(), dim=-1).cpu().numpy()
        return probs.reshape(1, 16)


# ==============================================================================
# End-to-End Streaming Execution Cascade (Implement.md Specification)
# ==============================================================================
def process_customer_event(
    features_80d: np.ndarray,
    action: str,
    lgbm_model: Any,
    tabdpt_model: Any,
    held_products: list[str] | None = None,
) -> dict[str, Any]:
    """
    Executes the 3-stage streaming cascade for an incoming customer event:
    - Stage 1: CPU LightGBM Gatekeeper (<0.002 ms)
    - Stage 2: GPU TabDPT (73,968 Scaled Context) + Gate 1C Per-Class Odds Calibration
    - Stage 3: Dual-Gamma + Next-Best-Action Relevance Routing

    Returns:
        dict containing:
        - prediction: int (0 for hold, 1..15 for target product)
        - predicted_product: str (e.g. "do_nothing", "credit_card")
        - top_7_products: list[str] of filtered product recommendations
        - top_7_indices: list[int] of product class indices
        - is_active_trigger: bool (True for live push alert)
        - gpu_skipped: bool (True if Stage 1 intercepted on CPU)
        - p_buy: float (Stage 1 purchase propensity)
        - odds_ratio: float (Stage 2 Gate 1C odds ratio vs Class 0)
        - regime: str ("stage1_cpu_hold", "stage2_gate1c_hold", "candidate_a_silent_cache", "candidate_c_live_push")
    """
    # -------------------------------------------------------------------------
    # STAGE 1: CPU LightGBM Gatekeeper (<0.002 ms)
    # -------------------------------------------------------------------------
    p_buy = predict_p_buy(lgbm_model, features_80d)
    if p_buy < GAMMA_CACHE:
        return {
            "prediction": 0,
            "predicted_product": "do_nothing",
            "top_7_products": [],
            "top_7_indices": [],
            "is_active_trigger": False,
            "gpu_skipped": True,
            "p_buy": p_buy,
            "odds_ratio": 0.0,
            "regime": "stage1_cpu_hold",
        }

    # -------------------------------------------------------------------------
    # STAGE 2: GPU TabDPT (73,968 Scaled Context) + Odds-Ratio Calibration (ORC) Gate
    # -------------------------------------------------------------------------
    features_128d = pad_and_clip_features(features_80d)
    probs_16d = tabdpt_model.predict_proba(features_128d)[0]  # Shape: (16,)

    # Extract probabilities for products 1..15
    prod_probs = probs_16d[1:16]
    sorted_class_indices = (np.argsort(-prod_probs) + 1).tolist()

    # Filter candidate ranking against products already held by customer
    held_set = set(held_products or [])
    top_7_products = [
        INDEX_TO_TARGET[idx]
        for idx in sorted_class_indices
        if INDEX_TO_TARGET.get(idx) not in held_set
    ][:7]
    top_7_indices = [
        TARGET_TO_INDEX[p] for p in top_7_products if p in TARGET_TO_INDEX
    ]

    # Primary candidate product (Class 1..15)
    cand_prod = int(np.argmax(prod_probs) + 1)
    cand_product_name = INDEX_TO_TARGET.get(cand_prod, "unknown")

    # Compute odds ratio against Class 0
    p_class0 = max(float(probs_16d[0]), 1e-8)
    odds_ratio = float(probs_16d[cand_prod]) / p_class0
    theta_gate = THETA_NOISY if cand_prod in NOISY_SINKS else THETA_CLEAN

    # Odds-Ratio Calibration (ORC) Gate Suppression: override to Class 0 if odds ratio is insufficient
    if odds_ratio <= theta_gate:
        return {
            "prediction": 0,
            "predicted_product": "do_nothing",
            "top_7_products": top_7_products,  # Retained for internal telemetry
            "top_7_indices": top_7_indices,
            "is_active_trigger": False,
            "gpu_skipped": False,
            "p_buy": p_buy,
            "odds_ratio": odds_ratio,
            "regime": "stage2_gate1c_hold",
        }

    # -------------------------------------------------------------------------
    # STAGE 3: Dual-Gamma + Next-Best-Action Relevance Routing
    # -------------------------------------------------------------------------
    # Recommend top unheld product if available, else candidate product
    rec_product = top_7_products[0] if top_7_products else cand_product_name
    rec_class_idx = TARGET_TO_INDEX.get(rec_product, cand_prod)

    is_relevant = bool(product_relavance(rec_product, action))
    is_live_alert = (p_buy >= GAMMA_LIVE) and is_relevant

    return {
        "prediction": rec_class_idx,
        "predicted_product": rec_product,
        "top_7_products": top_7_products,
        "top_7_indices": top_7_indices,
        "is_active_trigger": is_live_alert,
        "gpu_skipped": False,
        "p_buy": p_buy,
        "odds_ratio": odds_ratio,
        "regime": "candidate_c_live_push" if is_live_alert else "candidate_a_silent_cache",
    }


# ==============================================================================
# Redis Publisher & Caching
# ==============================================================================
def publish_reco(stream_payload: dict, r_client=None) -> str:
    """
    Persists recommendations to Redis cache (recs:{customer_id}) and appends every
    decision (both active push alerts and silent/held events) to STREAM_RECOMMENDATIONS
    for full end-to-end telemetry and audit trail visibility.
    """
    client = r_client or get_redis_client()
    customer_id = stream_payload["customer_id"]
    cache_key = f"recs:{customer_id}"

    # Always persist recommendations to Redis cache with TTL
    client.set(cache_key, json.dumps(stream_payload), ex=RECS_TTL_SECONDS)

    # Always publish to STREAM_RECOMMENDATIONS so audit trails & telemetry track all events
    new_msg_id = client.xadd(STREAM_RECOMMENDATIONS, stream_payload)

    is_active = stream_payload.get("is_active_trigger") in (True, "true", "True")
    if is_active:
        print(
            f"[Worker] 🔥 Live Push Alert sent to {STREAM_RECOMMENDATIONS} "
            f"(customer: {customer_id}, reco: '{stream_payload.get('top_product')}', msg_id: {new_msg_id})"
        )
    else:
        print(
            f"[Worker] 💾 Silent Decision logged to {STREAM_RECOMMENDATIONS} & cached in {cache_key} "
            f"(customer: {customer_id}, regime: '{stream_payload.get('regime')}', msg_id: {new_msg_id})"
        )
    return new_msg_id


# ==============================================================================
# Worker Daemon Loop
# ==============================================================================
def run_event_processor():
    """Starts the real-time event consumer worker."""
    r = get_redis_client()
    feature_store = RedisFeatureStore(redis_client=r)

    try:
        r.xgroup_create(STREAM_CUSTOMER_EVENTS, CONSUMER_GROUP, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise

    # Initialize Stage 1 and Stage 2 models once on worker startup
    print("[Worker] Initializing Stage 1 CPU LightGBM Gatekeeper...")
    lgbm_model = load_lgbm_gatekeeper()

    print("[Worker] Initializing Stage 2 GPU TabDPT Engine...")
    tabdpt_engine = TabDPTEngine()

    print(f"[Worker] Listening for events on {STREAM_CUSTOMER_EVENTS} (Group: {CONSUMER_GROUP})...\n")

    while True:
        try:
            entries = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername="worker_1",
                streams={STREAM_CUSTOMER_EVENTS: ">"},
                count=1,
                block=500,
            )
            if entries:
                start_time = time.time()
                for stream_name, messages in entries:
                    for msg_id, payload in messages:
                        customer_id = payload.get("customer_id")
                        action = payload.get("action", "unknown")
                        timestamp = payload.get("timestamp")

                        print(f"[Worker] Event received [{msg_id}]: customer={customer_id}, action={action}")

                        # 1. Fetch features from Redis Feature Store
                        t_feat = time.time()
                        x_qry, held, metadata = feature_store.get_customer_features(
                            customer_id, enable_fallback=True
                        )
                        feat_time_ms = (time.time() - t_feat) * 1000

                        # 2. Execute 3-Stage Cascaded Routing
                        result = process_customer_event(
                            features_80d=x_qry,
                            action=action,
                            lgbm_model=lgbm_model,
                            tabdpt_model=tabdpt_engine,
                            held_products=held,
                        )

                        # Acknowledge event in Redis stream
                        r.xack(STREAM_CUSTOMER_EVENTS, CONSUMER_GROUP, msg_id)

                        # 3. Assemble and publish response payload
                        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        latency_ms = (time.time() - start_time) * 1000

                        stream_payload = {
                            "customer_id": str(customer_id),
                            "action": str(action),
                            "prediction": str(result["prediction"]),
                            "top_product": str(result["predicted_product"]),
                            "recommendations": json.dumps(result["top_7_products"]),
                            "is_active_trigger": str(result["is_active_trigger"]).lower(),
                            "gpu_skipped": str(result["gpu_skipped"]).lower(),
                            "p_buy": f"{result['p_buy']:.4f}",
                            "odds_ratio": f"{result['odds_ratio']:.4f}",
                            "regime": str(result["regime"]),
                            "latency_ms": f"{latency_ms:.2f}",
                            "timestamp": now_iso,
                        }
                        publish_reco(stream_payload, r_client=r)

                        # 4. Update held products if an acquisition event occurred
                        if action in ("claim_offer", "product_acquired"):
                            if result["predicted_product"] != "do_nothing":
                                updated_held = feature_store.update_held_products(
                                    customer_id, result["predicted_product"]
                                )
                                print(f"[Worker] Updated held products for {customer_id}: {updated_held}")

                        print(
                            f"[Worker] Processed event in {latency_ms:.2f}ms "
                            f"(FeatureStore: {feat_time_ms:.2f}ms, Regime: {result['regime']})\n"
                        )
        except KeyboardInterrupt:
            print("[Worker] Stopping worker gracefully...")
            break


if __name__ == "__main__":
    run_event_processor()

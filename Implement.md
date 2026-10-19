### AGENT EXECUTION BRIEF: FINAL PRODUCTION WIRING (`src/streaming/event_processor.py`)

**Objective:** Wire the validated **73,968-row Scaled Uniform TabDPT Context**, the **Stage 1 CPU LightGBM Pre-Filter**, and the **Stage 2/3 Dual-$\gamma$ + Gate 1C Routing Cascade** into `src/streaming/event_processor.py`.

This single pipeline runs **Candidate A (Max Sniper: 54.3% Buyer Top-1, 84.7% Hold, 42.3% GPU Saved)** for silent background caching and **Candidate C (High Hold: 91.0% Non-Buyer Hold, 49.6% Buyer Top-1, 0.8198 Alert MAP@7)** for live customer push alerts simultaneously.

---

### 1. ARTIFACTS & MODEL INITIALIZATION (ON WORKER STARTUP)

Ensure the following artifacts are loaded once at service startup and pinned in memory (zero per-event disk I/O):

1. **TabDPT Context Tensor (`context_scaled_73968.pt`):**
* Load the `73,968 × 128` tensor (`16 classes × 4,623 rows`, `torch.bfloat16`) and its corresponding `y` label tensor directly onto the **NVIDIA RTX 3090 GPU** (`cuda:0`).


* Keep the model and encoded context resident in VRAM (~4.8 GB footprint).


2. **Stage 1 CPU LightGBM Gatekeeper:**
* Load the trained binary LightGBM classifier (`0.8912` ROC-AUC, `0.002 ms` inference latency) on **CPU**.


* Input schema: Native **80 normalized numeric features** (prior to 128-column zero-padding).




3. Gate 1C Per-Class Threshold Constants ($\vec{\theta} \in \mathbb{R}^{15}$):


* Define the noisy sink classes identified in Experiment 1 (`Class 1: current_account`, `Class 7: particular_plus_account`, `Class 9: funds`, `Class 14: long_term_deposits`, `Class 15: taxes`).


* Set $\theta_{\text{noisy}} = 6.0$ for noisy sinks and $\theta_{\text{clean}} = 1.8$ for all other product classes (`Classes 2, 3, 4, 5, 6, 8, 10, 11, 12, 13`).


4. **Dual-Tier LightGBM Thresholds ($\gamma$):**
* `GAMMA_CACHE = 0.20` (Candidate A: GPU wake-up & silent cache threshold).
* `GAMMA_LIVE = 0.30` (Candidate C: Live push notification safety threshold).



---

### 2. END-TO-END STREAMING EXECUTION LOGIC

Update the event processing handler in `src/streaming/event_processor.py` to execute the following 3-stage cascade on every incoming customer event:

#### Stage 1: CPU LightGBM Pre-Filter (`< 0.002 ms`)



1. Extract the 80 native normalized features for the incoming customer event and compute the binary purchase propensity on CPU:



$$p_{\text{buy}} = P_{\text{LGBM}}(\text{Buy} \mid X_{80})$$


2. **GPU Bypass Check (`p_buy < 0.20`):**
* If `p_buy < 0.20`, immediately return `Class 0` ("Do Nothing / Hold") with `is_active_trigger = False` and `gpu_skipped = True`.


* Do **not** invoke the RTX 3090 GPU. (This intercepts ~42.3% of all streaming traffic on CPU).





#### Stage 2: GPU TabDPT Inference & Gate 1C Calibration

1. **TabDPT Forward Pass (`p_buy >= 0.20`):**
* Clip numeric features to `[-10.0, 10.0]`, zero-pad from 80 to 128 columns, cast to `torch.bfloat16`, and execute the single-pass TabDPT forward pass against `context_scaled_73968.pt` on the RTX 3090.


* Extract the 16-class softmax probability vector $P = [P(0), P(1), \dots, P(15)]$.
*Rank the 15 financial products (`Classes 1..15`) in descending order of probability to form the `top_7_recommendations` list (which preserves the `0.7672` Unconditional MAP@7 / `0.8198` Alert MAP@7 ranking).




2. Gate 1C Per-Class Odds-Ratio Check:


* Identify the #1 product candidate: $\text{top\_product} = \arg\max_{i \in \{1..15\}} P(\text{Class } i)$.


* Compute the odds ratio against `Class 0`:

$$\text{odds\_ratio} = \frac{P(\text{top\_product})}{\max(P(\text{Class 0}), 10^{-8})}$$


* Look up the product's Gate 1C threshold:

$$\theta_{\text{gate}} = 6.0 \text{ if } \text{top\_product} \in \{1, 7, 9, 14, 15\} \text{ else } 1.8$$


* **Gate 1C Suppression:** If $\text{odds\_ratio} \le \theta_{\text{gate}}$, override the top-1 prediction to `Class 0` ("Do Nothing / Hold"), set `is_active_trigger = False`, and exit.





#### Stage 3: Dual-$\gamma$ + Action Relevance Routing (Live Push vs. Silent Cache)

If the query clears Stage 1 (`p_buy >= 0.20`) AND clears Stage 2 (`odds_ratio > theta_gate`), we have a validated product recommendation (`top_product`). Determine delivery mode using the existing `p_buy` score and `product_relavance(top_product, action)`:

1. **Live Push Alert Check (Candidate C Regime — 91.0% Hold / 49.6% Sniper):**
* Evaluate: `is_active_trigger = (p_buy >= 0.30) and bool(product_relavance(top_product, action))`.


* If `True`: Fire live customer notification (`prediction = top_product`, `is_active_trigger = True`, `regime = "candidate_c_live_push"`).




2. **Silent Background Cache (Candidate A Regime — 54.3% Max Sniper / 84.7% Hold):**
* If `False` (either `0.20 <= p_buy < 0.30`, OR `p_buy >= 0.30` on a routine non-relevant action like a standard card swipe):


* Silently cache `top_product` and `top_7_recommendations` (`prediction = top_product`, `is_active_trigger = False`, `regime = "candidate_a_silent_cache"`).





---

### 3. REFERENCE IMPLEMENTATION FOR `src/streaming/event_processor.py`

```python
NOISY_SINKS = {1, 7, 9, 14, 15}  # current_account, particular_plus, funds, long_term_deposits, taxes
GAMMA_CACHE = 0.20               # Candidate A: Wake up GPU & allow silent caching (54.3% Sniper)
GAMMA_LIVE  = 0.30               # Candidate C: Allow live push alerts (91.0% Non-Buyer Hold)
THETA_NOISY = 6.0                # Gate 1C odds-ratio threshold for noisy sinks
THETA_CLEAN = 1.8                # Gate 1C odds-ratio threshold for clean snipers

def process_customer_event(features_80d, action, lgbm_model, tabdpt_model):
    # --- STAGE 1: CPU LightGBM Gatekeeper (<0.002 ms) ---
    p_buy = float(lgbm_model.predict_proba(features_80d)[0, 1])
    if p_buy < GAMMA_CACHE:
        return {
            "prediction": 0,
            "top_7_products": [],
            "is_active_trigger": False,
            "gpu_skipped": True,
            "p_buy": p_buy,
            "odds_ratio": 0.0,
            "regime": "stage1_cpu_hold"
        }

    # --- STAGE 2: GPU TabDPT (73,968 Context) + Gate 1C ---
    features_128d = pad_and_clip_features(features_80d)  # Clip [-10, 10], pad to 128, bfloat16
    probs_16d = tabdpt_model.predict_proba(features_128d)[0]  # Shape: (16,)

    top_7_indices = (probs_16d.argsort()[::-1][:7] + 1).tolist()
    top_product = top_7_indices[0]

    p_class0 = max(float(probs_16d[0]), 1e-8)
    odds_ratio = float(probs_16d[top_product]) / p_class0
    theta_gate = THETA_NOISY if top_product in NOISY_SINKS else THETA_CLEAN

    if odds_ratio <= theta_gate:
        return {
            "prediction": 0,
            "top_7_products": top_7_indices,  # Retained for internal telemetry
            "is_active_trigger": False,
            "gpu_skipped": False,
            "p_buy": p_buy,
            "odds_ratio": odds_ratio,
            "regime": "stage2_gate1c_hold"
        }

    # --- STAGE 3: Dual-Gamma + Relevance Routing ---
    is_relevant = bool(product_relavance(top_product, action))
    is_live_alert = (p_buy >= GAMMA_LIVE) and is_relevant

    return {
        "prediction": top_product,
        "top_7_products": top_7_indices,
        "is_active_trigger": is_live_alert,
        "gpu_skipped": False,
        "p_buy": p_buy,
        "odds_ratio": odds_ratio,
        "regime": "candidate_c_live_push" if is_live_alert else "candidate_a_silent_cache"
    }

```

---

### 4. VERIFICATION & SMOKE TEST CHECKLIST

Before closing the task, run an end-to-end integration test verifying:

1. **VRAM Residency:** Confirm `context_scaled_73968.pt` loads once at startup and GPU memory remains stable at ~4.8 GB across sequential events.
2. **Routing Telemetry:** Replay a 1,000-event sample from the May 2016 holdout through `process_customer_event()` and assert that:
* `stage1_cpu_hold` accounts for **~42.3%** of events (`gpu_skipped == True`).
* Combined Hold rate (`prediction == 0`) matches **84.7%** prior to the `GAMMA_LIVE` push filter, and **91.0%** when evaluating live push eligibility at `p_buy >= 0.30`.


3. **Documentation Update:** Record the final production architecture and thresholds in `walkthrough.md` and `problem_summary.txt`.
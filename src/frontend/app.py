"""
Santander Next-Best-Action Banking Recommender
Interactive Real-Time Demonstration Dashboard

Powered by:
- TabDPT: Foundation Tabular Transformer with KV-Cache
- Triton Inference Server (gRPC port 8001)
- Redis Streams: Event-Driven Customer Actions & Output Broadcasts
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path

# Add project root to sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import torch

try:
    import tritonclient.grpc as grpcclient
    TRITON_CLIENT_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_AVAILABLE = False

from src.config import (
    get_redis_client,
    STREAM_CUSTOMER_EVENTS,
    STREAM_RECOMMENDATIONS,
    CONSUMER_GROUP,
    KEY_CUSTOMER_RECS,
    RECS_TTL_SECONDS,
    ACTION_RELEVANCE_MATRIX,
    TRITON_GRPC_URL,
    CLEAN_TEST_PARQUET,
    CLEAN_TRAIN_PARQUET,
    TEST_TARGETS_PARQUET,
    SELECTED_15_TARGETS,
    INDEX_TO_TARGET,
    TARGET_TO_INDEX
)

LOGO_PATH = _repo_root / "docs" / "TYLOGO.png"

# Friendly display metadata for products
PRODUCT_METADATA = {
    "current_account": {"name": "Current Account", "icon": "💳", "category": "Daily Banking"},
    "payroll_account": {"name": "Payroll Account", "icon": "💼", "category": "Daily Banking"},
    "credit_card": {"name": "Credit Card", "icon": "💳", "category": "Financing"},
    "pensions": {"name": "Pension Plan", "icon": "🛡️", "category": "Retirement & Security"},
    "mortgage": {"name": "Mortgage Loan", "icon": "🏠", "category": "Financing"},
    "securities": {"name": "Securities & Stocks", "icon": "📈", "category": "Wealth & Investment"},
    "particular_plus_account": {"name": "Particular Plus Account", "icon": "⭐", "category": "Savings & Premium"},
    "particular_account": {"name": "Particular Account", "icon": "👤", "category": "Daily Banking"},
    "funds": {"name": "Investment Funds", "icon": "📊", "category": "Wealth & Investment"},
    "direct_debit": {"name": "Direct Debit Service", "icon": "🔄", "category": "Daily Banking"},
    "payroll": {"name": "Payroll Direct Deposit", "icon": "💵", "category": "Daily Banking"},
    "more_particular_account": {"name": "Particular More Account", "icon": "🌟", "category": "Savings & Premium"},
    "e_account": {"name": "Digital E-Account", "icon": "🌐", "category": "Digital Banking"},
    "long_term_deposits": {"name": "Long-Term Fixed Deposit", "icon": "⏳", "category": "Savings & Deposits"},
    "taxes": {"name": "Tax Management Account", "icon": "📑", "category": "Specialized Services"},
    "do_nothing": {"name": "Hold / No Action", "icon": "⏸️", "category": "Hold Baseline"}
}

ACTION_DESCRIPTIONS = {
    "salary_deposit": ("💰 Salary / Payroll Credited", "Incoming monthly compensation deposited into account"),
    "large_deposit": ("📈 Large Capital Inflow (> €10,000)", "High-value external wire transfer or deposit"),
    "branch_inquiry": ("🏛️ Branch Financing & Mortgage Inquiry", "Customer consulted relationship manager regarding credit/real-estate"),
    "tax_season_login": ("📅 Tax Season Active Portal Login", "Customer logged in during active tax filing window (April–June)"),
    "card_payment": ("💳 Routine Card Payment (Passive)", "Routine merchant purchase; recalibrates propensity silently"),
    "dormant_reactivation": ("🔄 Dormant Customer Reactivation", "Customer returned after >90 days of inactivity")
}

# Curated customer archetypes
CURATED_CUSTOMERS = {
    "1009063": "🛡️ 1009063 - Active Contributor (Pensions)",
    "1002447": "🔄 1002447 - Urban Resident (Direct Debit)",
    "1166753": "🎓 1166753 - University Student (Payroll / Pensions)",
    "1002271": "🏦 1002271 - Mid-Career (Current)",
    "1000008": "⏸️ 1000008 - Dormant Account (No Action)",
    "658033": "⏸️ 658033 - Established Retail (No Action)",
}

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS (Green Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Santander AI | Next Best Action Recommender",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #064E3B 0%, #059669 45%, #0F172A 100%);
        padding: 22px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .main-header p {
        color: #E2E8F0;
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.95;
    }
    .status-badge-ok {
        background-color: #DCFCE7;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .status-badge-warn {
        background-color: #F1F5F9;
        color: #475569;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #CBD5E1;
    }
    .reco-hero-card-active {
        background: white;
        border-radius: 12px;
        padding: 26px;
        border-left: 6px solid #059669;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.08);
        margin-bottom: 20px;
    }
    .reco-hero-card-suppressed {
        background: white;
        border-radius: 12px;
        padding: 26px;
        border-left: 6px solid #64748B;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .trigger-active-banner {
        background: linear-gradient(90deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 14px 18px;
        color: #065F46;
        font-weight: 600;
        margin-bottom: 18px;
        font-size: 1.0rem;
    }
    .trigger-passive-banner {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px 18px;
        color: #475569;
        font-weight: 500;
        margin-bottom: 18px;
        font-size: 0.95rem;
    }
    .product-pill {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px 4px 2px 0;
        display: inline-block;
        font-weight: 500;
    }
    .awaiting-card {
        text-align: center;
        padding: 56px 24px;
        background: white;
        border-radius: 12px;
        border: 2px dashed #A7F3D0;
        margin: 20px 0;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# System Helpers & Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_redis():
    """Cached Redis client instance."""
    return get_redis_client()

def check_triton_status():
    """Checks if Triton gRPC server is responding."""
    if not TRITON_CLIENT_AVAILABLE:
        return False, "tritonclient not installed"
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        if client.is_server_live() and client.is_model_ready("tabdpt"):
            return True, "Online & Ready (tabdpt v3)"
        return False, "Server reachable, model not ready"
    except Exception as e:
        return False, f"Offline ({e.__class__.__name__})"

@st.cache_data(show_spinner=False)
def load_customer_dossier(customer_id: str):
    """Loads customer snapshot and current holdings from test dataset."""
    try:
        test_df = pl.scan_parquet(str(CLEAN_TEST_PARQUET)).filter(pl.col("customer_id") == customer_id).collect()
        if test_df.is_empty():
            return None
        row = test_df.to_dicts()[0]

        held = []
        for p in SELECTED_15_TARGETS:
            if row.get(p, 0) == 1:
                held.append(p)
        return {
            "row": row,
            "held_products": held
        }
    except Exception as e:
        return None

def execute_pipeline_for_event(customer_id: str, action: str):
    """
    Executes end-to-end inference and business logic:
    1. Extracts customer test & historical snapshots via Polars.
    2. Runs feature engineering, normalization, and 128-dim padding.
    3. Calls Triton gRPC server (port 8001) with KV-cache.
    4. Filters already-held products.
    5. Applies Action-Intent Relevance Logic (relavant -> active vs no action).
    6. Publishes to STREAM_RECOMMENDATIONS and updates KEY_CUSTOMER_RECS.
    """
    try:
        from src.streaming.event_processor import (
            get_customer,
            get_customer_history,
            feature_engineering_pipeline,
            encode_and_normalize,
            get_already_held_mask,
            prepare_query,
            format_predictions,
            product_relavance,
            df,
            history,
            device,
            dtype,
            client as triton_client
        )

        start_time = time.time()
        c_info = get_customer(customer_id, df)
        c_hist = get_customer_history(customer_id, history)
        customer = pl.concat([c_info, c_hist], how="vertical")
        engineered = feature_engineering_pipeline(customer)
        held = get_already_held_mask(engineered)
        processed = encode_and_normalize(engineered)
        query = processed.drop("customer_id", "snapshot_date").to_torch().to(torch.float32)
        x_qry = prepare_query(query.to(device, dtype=dtype)).to(torch.float32).cpu().numpy()

        inputs = [grpcclient.InferInput("QUERY_FEATURES", [1, 1, 128], "FP32")]
        inputs[0].set_data_from_numpy(x_qry)
        outputs = [grpcclient.InferRequestedOutput("PROBABILITIES")]
        response = triton_client.infer(model_name="tabdpt", inputs=inputs, outputs=outputs)
        probs = response.as_numpy("PROBABILITIES")

        recos = format_predictions(probs, held)
        top_product = recos[0] if recos else "do_nothing"
        relevant = product_relavance(top_product, action)
        latency = (time.time() - start_time) * 1000

        probs_1d = probs.squeeze()
        prob_dict = {INDEX_TO_TARGET[i]: float(probs_1d[i]) for i in range(16)}

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        stream_payload = {
            "customer_id": str(customer_id),
            "action": str(action),
            "top_product": str(top_product),
            "recommendations": json.dumps(recos),
            "is_active_trigger": "true" if relevant else "false",
            "latency_ms": f"{latency:.2f}",
            "timestamp": now_iso,
            "probabilities": json.dumps(prob_dict)
        }

        # 1. Push to stream:recommendations
        r = get_redis()
        r.xadd(STREAM_RECOMMENDATIONS, stream_payload)

        # 2. Update recs:{customer_id} cache
        cache_key = f"recs:{customer_id}"
        r.set(cache_key, json.dumps(stream_payload), ex=RECS_TTL_SECONDS)

        return stream_payload, None

    except Exception as e:
        return None, str(e)


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div>
        <h1>🏦 Santander Next-Best-Action Recommender</h1>
        <p>Real-Time Tabular Foundation Model (TabDPT) Serving & Redis Streaming Event Hub</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Status KPI row
r_client = get_redis()
triton_ok, triton_msg = check_triton_status()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Model Architecture", value="TabDPT (128-dim)", delta="32-layer Transformer")
with col2:
    if triton_ok:
        st.metric(label="Triton Server", value="Port 8001", delta="Ready (gRPC)")
    else:
        st.metric(label="Triton Server", value="Offline", delta=triton_msg, delta_color="inverse")
with col3:
    st.metric(label="Streaming Engine", value="Redis Streams", delta="stream:customer_events")
with col4:
    stream_count = r_client.xlen(STREAM_CUSTOMER_EVENTS) if r_client else 0
    st.metric(label="Events Processed", value=f"{stream_count} events", delta="Consumer: reco_workers")

st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar: Customer Profile & Action Simulator
# -----------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    st.header("👤 Customer Dossier")
    
    preset_choice = st.selectbox(
        "Select Archetype Customer:",
        options=list(CURATED_CUSTOMERS.keys()),
        format_func=lambda k: CURATED_CUSTOMERS[k],
        index=0
    )
    
    custom_id_input = st.text_input("Or enter custom Customer ID:", value=preset_choice)
    active_cust_id = custom_id_input.strip() or preset_choice

    # Detect customer switch to clear previous popups
    if "current_customer" not in st.session_state:
        st.session_state["current_customer"] = active_cust_id
    elif st.session_state["current_customer"] != active_cust_id:
        st.session_state["current_customer"] = active_cust_id
        # Clear previous reco payload on profile switch so nothing pops up automatically
        st.session_state.pop("last_emitted_payload", None)

    dossier = load_customer_dossier(active_cust_id)

    if dossier:
        row = dossier["row"]
        st.markdown(f"**Customer ID:** `{active_cust_id}`")
        st.markdown(f"**Age:** {row.get('age', 'N/A')} yrs &nbsp;|&nbsp; **Gender:** {row.get('gender', 'N/A')}")
        st.markdown(f"**Province:** {row.get('province_name', 'N/A')} &nbsp;|&nbsp; **Seniority:** {row.get('seniority_months', 'N/A')} mo")
        st.markdown(f"**Segment:** {row.get('customer_segment', 'N/A')}")
        
        income = row.get('gross_household_income')
        if income is not None:
            st.markdown(f"**Household Gross Income:** €{income:,.2f}")
        else:
            st.markdown("**Household Gross Income:** N/A")
        
        st.subheader("📦 Currently Held Products")
        held = dossier["held_products"]
        if held:
            held_html = "".join([f"<span class='product-pill'>✓ {PRODUCT_METADATA.get(p, {}).get('name', p)}</span>" for p in held])
            st.markdown(held_html, unsafe_allow_html=True)
        else:
            st.caption("No active products currently held.")
    else:
        st.warning(f"Customer {active_cust_id} not found in test partition.")

    st.markdown("---")
    st.header("⚡ Live Action Simulator")
    st.caption("Trigger an action to test real-time Next-Best-Action relevance matching.")

    selected_action = st.selectbox(
        "Simulate Banking Action:",
        options=list(ACTION_DESCRIPTIONS.keys()),
        format_func=lambda a: ACTION_DESCRIPTIONS[a][0]
    )
    st.caption(f"ℹ️ {ACTION_DESCRIPTIONS[selected_action][1]}")

    emit_btn = st.button("🚀 Emit Event to Redis Stream", type="primary", use_container_width=True)

    if emit_btn:
        event_payload = {
            "customer_id": str(active_cust_id),
            "action": str(selected_action),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        msg_id = r_client.xadd(STREAM_CUSTOMER_EVENTS, event_payload)
        
        # Execute pipeline to process event immediately
        with st.spinner("Processing event through Triton TabDPT & Next-Best-Action Filter..."):
            res, err = execute_pipeline_for_event(active_cust_id, selected_action)
            if err:
                st.error(f"Inference error: {err}")
            else:
                st.session_state["last_emitted_customer"] = active_cust_id
                st.session_state["last_emitted_payload"] = res
                st.success(f"Emitted to `{STREAM_CUSTOMER_EVENTS}` (ID: {msg_id}) & Recalibrated!")
            st.rerun()


# -----------------------------------------------------------------------------
# Main Tabs View
# -----------------------------------------------------------------------------
tab_reco, tab_stream, tab_arch = st.tabs([
    "🎯 Live Next-Best-Action Hub",
    "⚡ Redis Stream Monitor",
    "🧠 Architecture & MLOps Blueprint"
])

# =============================================================================
# TAB 1: Live Next-Best-Action Hub
# =============================================================================
with tab_reco:
    # Only show recommendation if an event was emitted for this customer
    has_emitted_for_cust = (
        st.session_state.get("last_emitted_customer") == active_cust_id
        and st.session_state.get("last_emitted_payload") is not None
    )

    if has_emitted_for_cust:
        data = st.session_state["last_emitted_payload"]
        top_prod = data.get("top_product", "do_nothing")
        recos_list = json.loads(data.get("recommendations", "[]"))
        is_active = data.get("is_active_trigger") in ("true", True)
        last_action = data.get("action", "query")
        latency = data.get("latency_ms", "N/A")
        ts = data.get("timestamp", "Just now")
        
        meta = PRODUCT_METADATA.get(top_prod, {"name": top_prod, "icon": "📦", "category": "General"})
        action_title = ACTION_DESCRIPTIONS.get(last_action, (last_action, ""))[0]

        # Extract probabilities
        prob_dict_str = data.get("probabilities")
        if prob_dict_str:
            probs_data = json.loads(prob_dict_str)
        else:
            probs_data = {p: (1.0 / (idx + 1)) for idx, p in enumerate(recos_list[:7])}
            probs_data["do_nothing"] = 0.85

        p_hold = probs_data.get("do_nothing", 0.0)
        p_act = 1.0 - p_hold

        # Decision & Action Trigger Banner
        if is_active:
            st.markdown(f"""
            <div class="trigger-active-banner">
                🔥 <strong>ACTIONABLE TRIGGER MATCH:</strong> Action <code>{action_title}</code> matches recommended product 
                <code>{meta['name']}</code>! Next-Best-Action decision: <strong>PITCH PRODUCT TO CUSTOMER</strong> (Push notification dispatched).
            </div>
            """, unsafe_allow_html=True)
            card_class = "reco-hero-card-active"
            status_text = "🔥 ACTIVE NEXT-BEST-ACTION MATCH"
            status_badge = f"<span class='status-badge-ok'>Decision: Active Pitch</span>"
        else:
            st.markdown(f"""
            <div class="trigger-passive-banner">
                ⏸️ <strong>Decision: No Action</strong> &nbsp;|&nbsp; Action: <code>{action_title}</code>
            </div>
            """, unsafe_allow_html=True)
            card_class = "reco-hero-card-suppressed"
            status_text = "⏸️ DECISION: NO ACTION"
            status_badge = f"<span class='status-badge-warn'>No Action</span>"

        # Hero Card: Top Recommendation Decision
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">{status_text}</span>
                {status_badge}
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px;">
                <div>
                    <h2 style="margin: 0; color: #1E293B; font-size: 2rem;">{meta['icon']} {meta['name']}</h2>
                    <span style="color: #64748B; font-size: 0.95rem;">Category: <strong>{meta['category']}</strong> &nbsp;|&nbsp; Customer ID: <strong>{active_cust_id}</strong></span>
                </div>
                <div style="text-align: right;">
                    <span class="status-badge-ok">Inference: {latency} ms</span><br/>
                    <small style="color: #64748B;">Event: {action_title} ({ts})</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Propensity KPI Gauges
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(label="Decision Status", value="Active Pitch 🔥" if is_active else "No Action ⏸️")
        with kpi2:
            st.metric(label="Top Product Propensity", value=f"{probs_data.get(top_prod, 0.0)*100:.1f}%", delta=f"{meta['name']}")
        with kpi3:
            st.metric(label="Acquisition Propensity (Any)", value=f"{p_act*100:.1f}%")
        with kpi4:
            st.metric(label="Hold / No-Action Propensity", value=f"{p_hold*100:.1f}%")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📋 Top Candidate Recommendations")
            st.caption("Ranked by model propensity; already-held products are strictly filtered out.")
            
            reco_rows = []
            for rank, prod_key in enumerate(recos_list[:7], 1):
                p_meta = PRODUCT_METADATA.get(prod_key, {"name": prod_key, "icon": "📦", "category": "General"})
                is_top = (rank == 1)
                reco_rows.append({
                    "Rank": f"#{rank}",
                    "Product": f"{p_meta['icon']} {p_meta['name']}",
                    "Category": p_meta["category"],
                    "Propensity": f"{probs_data.get(prod_key, 0.0)*100:.2f}%",
                    "Action Status": "🔥 Actionable" if (is_top and is_active) else ("⏸️ Suppressed" if is_top else "Secondary")
                })
            st.table(pd.DataFrame(reco_rows))

        with col_right:
            st.subheader("📊 Class Propensity Distribution")
            
            # User toggle for Hold / Do Nothing
            show_hold = st.checkbox("Include 'Hold / No Action' in Distribution Chart", value=False)
            
            chart_items = []
            for k, v in probs_data.items():
                if not show_hold and k == "do_nothing":
                    continue
                chart_items.append({
                    "Product": PRODUCT_METADATA.get(k, {}).get("name", k),
                    "Propensity (%)": float(v) * 100
                })
            
            chart_df = pd.DataFrame(chart_items).sort_values(by="Propensity (%)", ascending=False).head(10)
            st.bar_chart(chart_df.set_index("Product"), color="#059669")

    else:
        # Clean awaiting state on startup & profile switch
        st.markdown(f"""
        <div class="awaiting-card">
            <span style="font-size: 3rem;">⚡</span>
            <h3 style="color: #065F46; margin: 12px 0 6px 0; font-weight: 700;">Awaiting Banking Event Simulation</h3>
            <p style="color: #64748B; max-width: 520px; margin: 0 auto 16px auto; font-size: 1.0rem;">
                Select a customer profile and a simulated banking action in the sidebar, then click 
                <strong style="color: #059669;">🚀 Emit Event to Redis Stream</strong> to trigger real-time TabDPT Next-Best-Action evaluation.
            </p>
            <span class="status-badge-ok">Target Customer: {active_cust_id}</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: Redis Stream Monitor (Live Audit Feed)
# =============================================================================
with tab_stream:
    st.subheader("📡 Live Redis Streams Audit Trail")
    st.caption(f"Real-time event feed monitoring `{STREAM_CUSTOMER_EVENTS}` and `{STREAM_RECOMMENDATIONS}`.")

    stream_col1, stream_col2 = st.columns([3, 1])
    with stream_col2:
        if st.button("🔄 Refresh Stream Logs", use_container_width=True):
            st.rerun()

    # Read latest messages from stream:recommendations
    try:
        recs_events = r_client.xrevrange(STREAM_RECOMMENDATIONS, count=25)
    except Exception:
        recs_events = []

    if recs_events:
        stream_table = []
        for msg_id, payload in recs_events:
            is_active_str = payload.get("is_active_trigger", "false")
            badge = "🔥 Active Pitch" if is_active_str == "true" else "⏸️ No Action (Suppressed)"
            top_p = payload.get("top_product", "N/A")
            top_name = PRODUCT_METADATA.get(top_p, {}).get("name", top_p)
            act_raw = payload.get("action", "N/A")
            act_title = ACTION_DESCRIPTIONS.get(act_raw, (act_raw, ""))[0]

            stream_table.append({
                "Message ID": msg_id,
                "Customer ID": payload.get("customer_id", "N/A"),
                "Simulated Action": act_title,
                "Model Top Product": top_name,
                "Next-Best-Action Decision": badge,
                "Latency (ms)": payload.get("latency_ms", "N/A"),
                "Timestamp": payload.get("timestamp", "N/A")
            })
        st.dataframe(pd.DataFrame(stream_table), use_container_width=True)
    else:
        st.caption("No stream events recorded yet. Emit an event from the sidebar to populate the feed.")


# =============================================================================
# TAB 3: Architecture & MLOps Blueprint
# =============================================================================
with tab_arch:
    st.subheader("🏛️ Architecture Overview: Real-Time TabDPT RecSys")
    
    st.markdown("""
    ```text
    ┌─────────────────────────┐          XADD            ┌─────────────────────────────┐
    │  Customer Action Stream │ ───────────────────────► │   Redis Stream Hub          │
    │  (UI / Webhook Events)  │                          │   'stream:customer_events'  │
    └─────────────────────────┘                          └──────────────┬──────────────┘
                                                                        │
                                                                   XREADGROUP
                                                                        ▼
    ┌─────────────────────────┐       ModelInfer         ┌─────────────────────────────┐
    │   Triton gRPC Server    │ ◄─────────────────────── │   Streaming Worker          │
    │   (TabDPT + KV-Cache)   │ ───────────────────────► │   (event_processor.py)      │
    └─────────────────────────┘      [Batch, 16]         └──────────────┬──────────────┘
                                                                        │
                                                          Filter Held & Action Match
                                                                        ▼
                                                         ┌─────────────────────────────┐
                                                         │   Next-Best-Action Filter   │
                                                         │   - Relevant: Active Pitch  │
                                                         │   - Irrelevant: No Action   │
                                                         └──────────────┬──────────────┘
                                                                        │
                                                                   XADD & Cache
                                                                        ▼
                                                         ┌─────────────────────────────┐
                                                         │   Redis Stream & Cache      │
                                                         │   - stream:recommendations  │
                                                         │   - recs:{customer_id}      │
                                                         └─────────────────────────────┘
    ```
    """)

    st.markdown("""
    ### Key Engineering Highlights
    1. **Tabular Deep Pretrained Transformer (TabDPT)**:
       - Uses a 32-layer transformer column-attention foundation model trained on multi-table tasks.
       - Tensors are padded to **128 dimensions** (native column dimension).
    2. **Precomputed KV-Cache Speedup**:
       - Context samples are pre-encoded in an attention KV-cache (`context_kv_cache.pt`).
       - Inference latency drops from ~500ms down to **< 20ms**, enabling real-time stream processing.
    3. **Next-Best-Action Relevance Masking (Business Logic)**:
       - The TabDPT model predicts raw propensity over 15 financial products.
       - The business logic evaluates whether the customer's current action warrants engaging them:
         - **Match found $\rightarrow$ Active Pitch** (e.g. `salary_deposit` matching `payroll` or `pensions`).
         - **No match $\rightarrow$ No Action** (e.g. routine `card_payment` suppresses active popups).
    """)

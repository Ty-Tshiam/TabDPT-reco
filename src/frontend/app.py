"""
Santander Next-Best-Action Banking Recommender
Interactive Real-Time Bank Portal & AI Serving Simulator

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

# Real-world banking activities for simulator
BANKING_ACTIVITIES = {
    "salary_deposit": {
        "title": "Deposit Monthly Paycheck",
        "icon": "💼",
        "amount": "+€2,850.00",
        "delta": 2850.00,
        "tx_name": "Corporate Payroll Direct Deposit",
        "tx_sub": "Santander Employer SEPA Credit",
        "description": "Credit monthly corporate payroll directly to account",
        "type": "credit"
    },
    "large_deposit": {
        "title": "High-Value Capital Wire",
        "icon": "🏧",
        "amount": "+€25,000.00",
        "delta": 25000.00,
        "tx_name": "SEPA Capital Wire Inflow",
        "tx_sub": "Interbank Liquidity Transfer",
        "description": "Inflow wire transfer of liquidity capital",
        "type": "credit"
    },
    "branch_inquiry": {
        "title": "Explore Home Mortgage Rates",
        "icon": "🏠",
        "amount": "Inquiry",
        "delta": 0.0,
        "tx_name": "Mortgage Rate Simulation Inquiry",
        "tx_sub": "Branch Digital Advisory Session",
        "description": "Simulate mortgage rate calculation & advisor request",
        "type": "inquiry"
    },
    "tax_season_login": {
        "title": "Open Tax Filing Portal",
        "icon": "📅",
        "amount": "Portal",
        "delta": 0.0,
        "tx_name": "AEAT Fiscal Certificate Access",
        "tx_sub": "Tax Filing Portal Session",
        "description": "Access fiscal documents for annual tax declaration",
        "type": "inquiry"
    },
    "card_payment": {
        "title": "Pay Monthly Utility Bill",
        "icon": "💳",
        "amount": "-€68.40",
        "delta": -68.40,
        "tx_name": "Iberdrola Clientes S.A.",
        "tx_sub": "Utility Bill Direct Debit",
        "description": "Routine merchant payment (Iberdrola Electricity)",
        "type": "debit"
    },
    "dormant_reactivation": {
        "title": "Security Re-authentication",
        "icon": "🔄",
        "amount": "Login",
        "delta": 0.0,
        "tx_name": "Portal Security Re-Authentication",
        "tx_sub": "Biometric 2FA Verification",
        "description": "Reactivate account portal after inactivity",
        "type": "inquiry"
    }
}

# Curated customer archetypes verified from test targets
CURATED_CUSTOMERS = {
    "1009063": "🛡️ 1009063 - Active Contributor (Pensions)",
    "1002447": "🔄 1002447 - Urban Resident (Direct Debit)",
    "1166753": "🎓 1166753 - University Student (Payroll / Pensions)",
    "1001710": "💳 1001710 - Young Professional",
    "1002271": "🏦 1002271 - Mid-Career",
    "1011464": "📑 1011464 - Seasoned Taxpayer",
    "1011230": "💼 1011230 - Early Career (No Action)",
    "1000008": "⏸️ 1000008 - Dormant Account (No Action)",
    "658033": "⏸️ 658033 - Established Retail (No Action)",
}

# Detailed bank accounts & personalized transaction ledgers per customer persona
CUSTOMER_ACCOUNTS_DATA = {
    "1009063": {
        "account_name": "Santander Premier Wealth Account",
        "iban": "ES76 0049 1500 2312 8491",
        "status": "Private Banking",
        "balance": 14850.20,
        "transactions": [
            {"title": "Vanguard S&P 500 Index SIP", "subtitle": "Monthly Investment Direct Debit", "amount": "-€500.00", "delta": -500.0, "time": "2 days ago"},
            {"title": "Sanitas Private Healthcare", "subtitle": "Health Insurance Policy", "amount": "-€115.40", "delta": -115.4, "time": "5 days ago"},
            {"title": "Mercadona Gourmet Madrid", "subtitle": "POS Contactless Swipe", "amount": "-€84.60", "delta": -84.6, "time": "1 week ago"},
            {"title": "Repsol AutoFuel Estación", "subtitle": "Debit Card Fuel Payment", "amount": "-€65.00", "delta": -65.0, "time": "1 week ago"}
        ]
    },
    "1002447": {
        "account_name": "Santander Smart Daily Checking",
        "iban": "ES42 0049 2831 9901 4427",
        "status": "Active Resident",
        "balance": 3420.50,
        "transactions": [
            {"title": "Metro de Madrid Bono Transporte", "subtitle": "Public Transit Card Recharge", "amount": "-€54.60", "delta": -54.6, "time": "Yesterday"},
            {"title": "Carrefour Market Chamberí", "subtitle": "Supermarket Grocery Debit", "amount": "-€38.25", "delta": -38.25, "time": "3 days ago"},
            {"title": "Endesa Energía Renovable", "subtitle": "Electricity Direct Debit", "amount": "-€72.10", "delta": -72.1, "time": "6 days ago"},
            {"title": "Bizum Split - Dinner with Marcos", "subtitle": "P2P Instant Mobile Payment", "amount": "+€24.00", "delta": 24.0, "time": "1 week ago"}
        ]
    },
    "1166753": {
        "account_name": "Santander Smart Joven Account",
        "iban": "ES19 0049 0012 3345 6789",
        "status": "University Student",
        "balance": 842.10,
        "transactions": [
            {"title": "Complutense University Library", "subtitle": "Academic Course Material", "amount": "-€22.50", "delta": -22.5, "time": "Yesterday"},
            {"title": "Cafetería Universitaria Moncloa", "subtitle": "Contactless Student Debit", "amount": "-€3.40", "delta": -3.4, "time": "2 days ago"},
            {"title": "Family Allowance (SEPA Inflow)", "subtitle": "Monthly Student Support Wire", "amount": "+€250.00", "delta": 250.0, "time": "5 days ago"},
            {"title": "Spotify Student Premium", "subtitle": "Digital Streaming Subscription", "amount": "-€5.99", "delta": -5.99, "time": "1 week ago"}
        ]
    },
    "1001710": {
        "account_name": "Santander One Professional Account",
        "iban": "ES88 0049 7621 5532 9910",
        "status": "Young Professional",
        "balance": 6120.00,
        "transactions": [
            {"title": "Zara Gran Vía Flagship", "subtitle": "Apparel & Retail Card Payment", "amount": "-€89.95", "delta": -89.95, "time": "Yesterday"},
            {"title": "Uber Spain Urban Mobility", "subtitle": "Ride-Hailing App Charge", "amount": "-€18.40", "delta": -18.4, "time": "3 days ago"},
            {"title": "Amazon Prime Subscription", "subtitle": "E-Commerce Digital Membership", "amount": "-€4.99", "delta": -4.99, "time": "6 days ago"},
            {"title": "Gourmet Dining Gastrobar", "subtitle": "Restaurant POS Card Payment", "amount": "-€62.00", "delta": -62.0, "time": "1 week ago"}
        ]
    },
    "1002271": {
        "account_name": "Santander Classic Family Checking",
        "iban": "ES54 0049 8830 1209 7721",
        "status": "Mid-Career",
        "balance": 8750.30,
        "transactions": [
            {"title": "Colegio San Patricio School Tuition", "subtitle": "Monthly Direct Debit Education", "amount": "-€340.00", "delta": -340.0, "time": "3 days ago"},
            {"title": "El Corte Inglés Castellana", "subtitle": "Department Store Card Swipe", "amount": "-€142.80", "delta": -142.8, "time": "5 days ago"},
            {"title": "Vodafone Fibra Óptica 1Gbps", "subtitle": "Home Broadband & Telecom", "amount": "-€52.00", "delta": -52.0, "time": "1 week ago"},
            {"title": "Gas Natural Fenosa", "subtitle": "Home Gas Direct Debit", "amount": "-€48.15", "delta": -48.15, "time": "10 days ago"}
        ]
    },
    "1011464": {
        "account_name": "Santander Senior Privilege Account",
        "iban": "ES63 0049 4410 9823 1156",
        "status": "Senior Member",
        "balance": 24850.00,
        "transactions": [
            {"title": "Agencia Tributaria (AEAT Quarterly)", "subtitle": "Fiscal Tax Settlement Payment", "amount": "-€1,450.00", "delta": -1450.0, "time": "4 days ago"},
            {"title": "Farmacia Mayor Madrid", "subtitle": "Healthcare & Prescriptions", "amount": "-€31.20", "delta": -31.2, "time": "5 days ago"},
            {"title": "Santander Dividend Yield Payout", "subtitle": "Securities Portfolio Dividend", "amount": "+€620.00", "delta": 620.0, "time": "1 week ago"},
            {"title": "Alcampo Hipermercado", "subtitle": "Household Groceries Payment", "amount": "-€112.50", "delta": -112.5, "time": "10 days ago"}
        ]
    },
    "1011230": {
        "account_name": "Santander Starter Account",
        "iban": "ES21 0049 6102 3381 4059",
        "status": "Standard Account",
        "balance": 2180.75,
        "transactions": [
            {"title": "Decathlon Sports Equipment", "subtitle": "Sporting Goods Card Payment", "amount": "-€45.00", "delta": -45.0, "time": "2 days ago"},
            {"title": "Starbucks Paseo del Prado", "subtitle": "Cafeteria Contactless Swipe", "amount": "-€4.80", "delta": -4.8, "time": "3 days ago"},
            {"title": "Gympass Fitness Subscription", "subtitle": "Monthly Fitness Direct Debit", "amount": "-€29.99", "delta": -29.99, "time": "1 week ago"},
            {"title": "Mercadona Express", "subtitle": "Weekly Groceries Debit", "amount": "-€26.40", "delta": -26.4, "time": "1 week ago"}
        ]
    },
    "1000008": {
        "account_name": "Santander Standard Inactive Ledger",
        "iban": "ES90 0049 0001 0000 8123",
        "status": "Dormant (Inactive)",
        "balance": 124.50,
        "transactions": [
            {"title": "Account Administration Fee", "subtitle": "Santander Quarterly Maintenance Fee", "amount": "-€12.00", "delta": -12.0, "time": "3 months ago"},
            {"title": "Santander ATM Cash Withdrawal", "subtitle": "Automated Teller Cash Out", "amount": "-€50.00", "delta": -50.0, "time": "7 months ago"},
            {"title": "Annual Account Yield Interest", "subtitle": "Deposit Yield Payout", "amount": "+€0.12", "delta": 0.12, "time": "9 months ago"}
        ]
    },
    "658033": {
        "account_name": "Santander Commercial Retail Checking",
        "iban": "ES33 0049 5521 8763 0214",
        "status": "Commercial Retail",
        "balance": 19450.00,
        "transactions": [
            {"title": "Distribuciones Hostelería Madrid", "subtitle": "Commercial Supplier Wire", "amount": "-€620.00", "delta": -620.0, "time": "3 days ago"},
            {"title": "POS Merchant Card Settlement", "subtitle": "Daily Credit Terminal Batch Out", "amount": "+€1,840.00", "delta": 1840.0, "time": "4 days ago"},
            {"title": "Canal de Isabel II Water Utility", "subtitle": "Commercial Water Direct Debit", "amount": "-€88.40", "delta": -88.4, "time": "1 week ago"},
            {"title": "Mutua Madrileña Commercial Insurance", "subtitle": "Annual Liability Premium", "amount": "-€145.00", "delta": -145.0, "time": "2 weeks ago"}
        ]
    }
}

def get_or_init_customer_ledger(customer_id: str):
    """Retrieves or initializes mutable customer account ledger in session state."""
    if "customer_ledgers" not in st.session_state:
        st.session_state["customer_ledgers"] = {}
    if customer_id not in st.session_state["customer_ledgers"]:
        import copy
        if customer_id in CUSTOMER_ACCOUNTS_DATA:
            st.session_state["customer_ledgers"][customer_id] = copy.deepcopy(CUSTOMER_ACCOUNTS_DATA[customer_id])
        else:
            cid_hash = abs(hash(str(customer_id))) % 9000 + 1000
            st.session_state["customer_ledgers"][customer_id] = {
                "account_name": "Santander Personal Checking",
                "iban": f"ES85 0049 1000 {cid_hash} 9921",
                "status": "Standard Account",
                "balance": 4850.20,
                "transactions": [
                    {"title": "Supermercados Mercadona", "subtitle": "Point of Sale Swipe", "amount": "-€42.50", "delta": -42.50, "time": "2 days ago"},
                    {"title": "Cafeteria Gran Vía", "subtitle": "Contactless Payment", "amount": "-€3.80", "delta": -3.80, "time": "3 days ago"},
                    {"title": "Gasolinera Repsol", "subtitle": "Debit Fuel Card", "amount": "-€65.00", "delta": -65.00, "time": "5 days ago"}
                ]
            }
    return st.session_state["customer_ledgers"][customer_id]

def reset_customer_ledger(customer_id: str):
    """Resets customer balance and transactions back to archetype defaults."""
    if "customer_ledgers" in st.session_state and customer_id in st.session_state["customer_ledgers"]:
        import copy
        if customer_id in CUSTOMER_ACCOUNTS_DATA:
            st.session_state["customer_ledgers"][customer_id] = copy.deepcopy(CUSTOMER_ACCOUNTS_DATA[customer_id])
        else:
            del st.session_state["customer_ledgers"][customer_id]

# -----------------------------------------------------------------------------
# Page Configuration & Emerald Green Theme
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
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.2);
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
    .bank-card {
        background: white;
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .bank-balance-card {
        background: linear-gradient(135deg, #064E3B 0%, #047857 100%);
        color: white;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 14px rgba(4, 120, 87, 0.25);
        margin-bottom: 20px;
    }
    .bank-balance-card h4 {
        color: #A7F3D0 !important;
        font-size: 0.95rem;
        margin: 0 0 4px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .bank-balance-card h2 {
        color: white !important;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
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
        padding: 24px;
        border-left: 6px solid #059669;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.08);
        margin-bottom: 20px;
    }
    .reco-hero-card-suppressed {
        background: white;
        border-radius: 12px;
        padding: 24px;
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
    }
    .trigger-passive-banner {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 12px 16px;
        color: #475569;
        font-weight: 500;
        margin-bottom: 18px;
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
    .tx-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #F1F5F9;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3) !important;
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

@st.cache_data(show_spinner=False)
def get_cached_customer_features(customer_id: str):
    """
    Extracts customer historical series, runs feature engineering,
    and returns the 128-dim normalized query tensor and held products mask.
    Cached across user interactions so 185MB Parquet scan is executed only once.
    """
    from src.streaming.event_processor import (
        get_customer,
        get_customer_history,
        feature_engineering_pipeline,
        encode_and_normalize,
        get_already_held_mask,
        prepare_query,
        df,
        history,
        device,
        dtype
    )
    c_info = get_customer(customer_id, df)
    c_hist = get_customer_history(customer_id, history)
    customer = pl.concat([c_info, c_hist], how="vertical")
    engineered = feature_engineering_pipeline(customer)
    held = get_already_held_mask(engineered)
    processed = encode_and_normalize(engineered)
    query = processed.drop("customer_id", "snapshot_date").to_torch().to(torch.float32)
    x_qry = prepare_query(query.to(device, dtype=dtype)).to(torch.float32).cpu().numpy()
    return x_qry, held

@st.cache_data(show_spinner=False)
def run_customer_model_inference(customer_id: str):
    """
    Runs Triton gRPC inference for the customer's feature tensor.
    Cached in memory so repeated interactions (paycheck, deposit, utility bill)
    can evaluate next-best-action relevance and serve offers with sub-millisecond latency.
    """
    from src.streaming.event_processor import format_predictions, client as triton_client

    t_start = time.time()
    x_qry, held = get_cached_customer_features(customer_id)
    feat_time_ms = (time.time() - t_start) * 1000

    t_infer_start = time.time()
    inputs = [grpcclient.InferInput("QUERY_FEATURES", [1, 1, 128], "FP32")]
    inputs[0].set_data_from_numpy(x_qry)
    outputs = [grpcclient.InferRequestedOutput("PROBABILITIES")]
    response = triton_client.infer(model_name="tabdpt", inputs=inputs, outputs=outputs)
    probs = response.as_numpy("PROBABILITIES")
    triton_time_ms = (time.time() - t_infer_start) * 1000

    recos = format_predictions(probs, held)
    top_product = recos[0] if recos else "do_nothing"
    probs_1d = probs.squeeze()
    prob_dict = {INDEX_TO_TARGET[i]: float(probs_1d[i]) for i in range(16)}

    return {
        "recos": recos,
        "top_product": top_product,
        "prob_dict": prob_dict,
        "triton_ms": round(triton_time_ms, 2),
        "feat_ms": round(feat_time_ms, 2)
    }

def execute_pipeline_for_event(customer_id: str, action: str):
    """
    Executes end-to-end inference and business logic:
    1. Retrieves precomputed query tensor & Triton predictions (cached).
    2. Evaluates Action-Intent Relevance Matrix (relevant -> active vs no action).
    3. Publishes event to STREAM_RECOMMENDATIONS and updates KEY_CUSTOMER_RECS.
    """
    try:
        from src.streaming.event_processor import product_relavance

        start_time = time.time()
        infer_data = run_customer_model_inference(customer_id)
        
        recos = infer_data["recos"]
        top_product = infer_data["top_product"]
        prob_dict = infer_data["prob_dict"]
        triton_ms = infer_data["triton_ms"]

        relevant = product_relavance(top_product, action)
        gateway_latency = (time.time() - start_time) * 1000

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        stream_payload = {
            "customer_id": str(customer_id),
            "action": str(action),
            "top_product": str(top_product),
            "recommendations": json.dumps(recos),
            "is_active_trigger": "true" if relevant else "false",
            "latency_ms": f"{gateway_latency:.2f}",
            "triton_ms": f"{triton_ms:.2f}",
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
# Dynamic Ad Modal Dialog (@st.dialog)
# -----------------------------------------------------------------------------
@st.dialog("🎉 Personalized Offer Just for You!")
def render_offer_modal(product_key: str, action_key: str):
    """Pops up a realistic banking offer modal when an action matches product intent."""
    meta = PRODUCT_METADATA.get(product_key, {"name": product_key, "icon": "🎁", "category": "Special Offer"})
    act_info = BANKING_ACTIVITIES.get(action_key, {"title": action_key})
    
    st.markdown(f"## {meta.get('icon', '🎁')} {meta.get('name', product_key)}")
    st.caption(f"Category: **{meta.get('category')}** &nbsp;|&nbsp; Verified Santander Product")
    
    st.markdown(f"""
    Because you just completed **{act_info['title']}**, our Next-Best-Action engine matched your profile 
    for an exclusive financial upgrade with preferred member rates.
    """)
    
    st.info("""
    ✓ **Pre-Approved with 1-Click Setup**: Zero paperwork required.  
    ✓ **Exclusive Promotional Terms**: 0% introductory maintenance fees.  
    ✓ **Seamless Integration**: Directly connects with your existing Santander accounts.
    """)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🚀 Claim & Open Account", type="primary", use_container_width=True):
            st.session_state["claimed_offer"] = meta.get("name", product_key)
            st.session_state.pop("popup_offer_to_render", None)
            st.toast("🎉 Product application submitted! Your relationship manager will confirm.", icon="✅")
            st.rerun()
    with c2:
        if st.button("✕ Dismiss", use_container_width=True):
            st.session_state.pop("popup_offer_to_render", None)
            st.rerun()


# -----------------------------------------------------------------------------
# Sidebar: Logo & Customer Dossier
# -----------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    st.header("👤 Customer Dossier")
    
    preset_choice = st.selectbox(
        "Select Customer Profile:",
        options=list(CURATED_CUSTOMERS.keys()),
        format_func=lambda k: CURATED_CUSTOMERS[k],
        index=0
    )
    
    custom_id_input = st.text_input("Or enter custom Customer ID:", value=preset_choice)
    active_cust_id = custom_id_input.strip() or preset_choice

    # Detect customer switch to clear previous session states
    if "current_customer" not in st.session_state:
        st.session_state["current_customer"] = active_cust_id
    elif st.session_state["current_customer"] != active_cust_id:
        st.session_state["current_customer"] = active_cust_id
        st.session_state.pop("last_emitted_payload", None)
        st.session_state.pop("last_tx_receipt", None)
        st.session_state.pop("popup_offer_to_render", None)
        st.session_state.pop("claimed_offer", None)

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


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div>
        <h1>🏦 Santander Next-Best-Action Banking Recommender</h1>
        <p>Real-Time Tabular Foundation Model (TabDPT) Serving & Redis Streaming Event Hub</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Status KPI row
r_client = get_redis()
triton_ok, triton_msg = check_triton_status()

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.metric(label="Model Architecture", value="TabDPT (128-dim)", delta="32-layer Transformer")
with kpi_c2:
    if triton_ok:
        st.metric(label="Triton Server", value="Port 8001", delta="Ready (gRPC)")
    else:
        st.metric(label="Triton Server", value="Offline", delta=triton_msg, delta_color="inverse")
with kpi_c3:
    st.metric(label="Streaming Engine", value="Redis Streams", delta="stream:customer_events")
with kpi_c4:
    stream_count = r_client.xlen(STREAM_CUSTOMER_EVENTS) if r_client else 0
    st.metric(label="Events Ingested", value=f"{stream_count} events", delta="Consumer: reco_workers")

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_portal, tab_ai, tab_stream, tab_arch = st.tabs([
    "🏦 Customer Online Bank Portal",
    "🧠 AI & Next-Best-Action Hub",
    "⚡ Redis Stream Monitor",
    "🏛️ Architecture Blueprint"
])


# =============================================================================
# TAB 1: Customer Online Bank Portal (The Simulator)
# =============================================================================
with tab_portal:
    portal_left, portal_right = st.columns([1.6, 1.0])

    cust_ledger = get_or_init_customer_ledger(active_cust_id)
    current_balance = cust_ledger["balance"]
    account_name = cust_ledger["account_name"]
    iban = cust_ledger["iban"]
    account_status = cust_ledger.get("status", "Active Member")

    with portal_left:
        # Bank Account Summary Card with Reset action
        bal_c1, bal_c2 = st.columns([3.2, 1.0])
        with bal_c1:
            bal_html = (
                f'<div class="bank-balance-card">'
                f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
                f'<div><h4>{account_name}</h4><h2>€{current_balance:,.2f}</h2></div>'
                f'<div style="text-align: right;"><span class="status-badge-ok">{account_status}</span></div>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; margin-top: 14px; font-size: 0.9rem; color: #D1FAE5; border-top: 1px solid rgba(255,255,255,0.25); padding-top: 10px;">'
                f'<span>Account Holder: <strong>Customer #{active_cust_id}</strong></span>'
                f'<span>IBAN: <strong>{iban}</strong></span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(bal_html, unsafe_allow_html=True)
        with bal_c2:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            if st.button("↺ Reset Ledger", key="reset_ledger_btn", use_container_width=True, help="Reset balance and transactions to default initial state"):
                reset_customer_ledger(active_cust_id)
                st.session_state.pop("last_tx_receipt", None)
                st.toast("Account ledger reset to initial state.", icon="🔄")
                st.rerun()

        st.subheader("⚡ Customer Activity Center")
        st.caption("Click any banking action to simulate real-time behavior. Balance and ledger update instantly!")

        # Dynamic Notification or Transaction Receipt
        if "last_tx_receipt" in st.session_state and st.session_state.get("last_emitted_customer") == active_cust_id:
            tx = st.session_state["last_tx_receipt"]
            if tx.get("is_active"):
                st.markdown(
                    f'<div class="trigger-active-banner">'
                    f'💥 <strong>MATCH DETECTED!</strong> Action <strong>{tx["title"]}</strong> processed ({tx["amount"]}). '
                    f'Balance updated to <strong>€{tx["new_balance"]:,.2f}</strong>. Pop-up ad dispatched!'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="trigger-passive-banner">'
                    f'✅ <strong>Transaction Confirmed:</strong> {tx["title"]} processed ({tx["amount"]}). '
                    f'Balance updated to <strong>€{tx["new_balance"]:,.2f}</strong>. '
                    f'<em>Decision: No promotional popup triggered.</em>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # 6 Interactive Banking Activity Buttons (2x3 Grid)
        act_c1, act_c2 = st.columns(2)
        
        act_keys = list(BANKING_ACTIVITIES.keys())
        triggered_action_key = None

        with act_c1:
            for k in act_keys[:3]:
                act = BANKING_ACTIVITIES[k]
                if st.button(f"{act['icon']} {act['title']}\n({act['amount']})", key=f"btn_{k}", use_container_width=True):
                    triggered_action_key = k

        with act_c2:
            for k in act_keys[3:]:
                act = BANKING_ACTIVITIES[k]
                if st.button(f"{act['icon']} {act['title']}\n({act['amount']})", key=f"btn_{k}", use_container_width=True):
                    triggered_action_key = k

        # Execute event if clicked
        if triggered_action_key:
            act_info = BANKING_ACTIVITIES[triggered_action_key]
            
            # 1. Update customer balance and transactions in session state
            delta = act_info.get("delta", 0.0)
            cust_ledger["balance"] += delta
            
            new_tx = {
                "title": act_info.get("tx_name", act_info["title"]),
                "subtitle": act_info.get("tx_sub", "Online Banking Activity"),
                "amount": act_info["amount"],
                "delta": delta,
                "time": time.strftime("%H:%M:%S Today")
            }
            cust_ledger["transactions"].insert(0, new_tx)

            # 2. Push to Redis Stream
            event_payload = {
                "customer_id": str(active_cust_id),
                "action": str(triggered_action_key),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            r_client.xadd(STREAM_CUSTOMER_EVENTS, event_payload)

            # 3. Run TabDPT Inference Pipeline
            with st.spinner(f"Processing {act_info['title']} through TabDPT & Triton..."):
                res, err = execute_pipeline_for_event(active_cust_id, triggered_action_key)
                if not err and res:
                    st.session_state["last_emitted_customer"] = active_cust_id
                    st.session_state["last_emitted_payload"] = res
                    is_active = (res.get("is_active_trigger") in ("true", True))
                    
                    st.session_state["last_tx_receipt"] = {
                        "title": act_info["title"],
                        "amount": act_info["amount"],
                        "delta": delta,
                        "new_balance": cust_ledger["balance"],
                        "is_active": is_active,
                        "time": time.strftime("%H:%M:%S")
                    }

                    # If active match, trigger pop-up offer!
                    if is_active:
                        st.session_state["popup_offer_to_render"] = (res.get("top_product"), triggered_action_key)
                    else:
                        st.session_state.pop("popup_offer_to_render", None)

            st.rerun()

    with portal_right:
        st.subheader("📜 Live Transaction Ledger")
        st.caption("Updated dynamically as money is deposited or debited.")

        tx_list = cust_ledger.get("transactions", [])
        tx_html_items = []
        for tx in tx_list[:10]:
            delta = tx.get("delta", 0.0)
            if delta > 0:
                color = "#059669"  # emerald green
            elif delta < 0:
                color = "#DC2626"  # red
            else:
                color = "#64748B"  # slate gray
            
            tx_html_items.append(
                f'<div class="tx-item">'
                f'<div><strong>{tx["title"]}</strong><br/>'
                f'<small style="color: #64748B;">{tx["subtitle"]} • {tx["time"]}</small></div>'
                f'<span style="color: {color}; font-weight: 700; font-size: 0.95rem;">{tx["amount"]}</span>'
                f'</div>'
            )

        ledger_content = "".join(tx_html_items)
        st.markdown(
            f'<div class="bank-card" style="max-height: 440px; overflow-y: auto;">{ledger_content}</div>',
            unsafe_allow_html=True
        )

    # Check if pop-up dialog should be launched (consume flag so it pops up strictly once)
    if "popup_offer_to_render" in st.session_state and st.session_state["popup_offer_to_render"] is not None:
        prod_k, act_k = st.session_state.pop("popup_offer_to_render")
        render_offer_modal(prod_k, act_k)


# =============================================================================
# TAB 2: AI & Next-Best-Action Hub (The Developer Console)
# =============================================================================
with tab_ai:
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
        act_info = BANKING_ACTIVITIES.get(last_action, {"title": last_action})

        prob_dict_str = data.get("probabilities")
        if prob_dict_str:
            probs_data = json.loads(prob_dict_str)
        else:
            probs_data = {p: (1.0 / (idx + 1)) for idx, p in enumerate(recos_list[:7])}
            probs_data["do_nothing"] = 0.85

        p_hold = probs_data.get("do_nothing", 0.0)
        p_act = 1.0 - p_hold

        # Concise decision badge
        if is_active:
            card_class = "reco-hero-card-active"
            status_text = "🔥 ACTIVE NEXT-BEST-ACTION MATCH"
            status_badge = "<span class='status-badge-ok'>Active Offer Triggered</span>"
        else:
            card_class = "reco-hero-card-suppressed"
            status_text = "⏸️ DECISION: NO ACTION"
            status_badge = "<span class='status-badge-warn'>No Action</span>"

        # Hero Card
        hero_html = (
            f'<div class="{card_class}">'
            f'<div style="display: flex; align-items: center; justify-content: space-between;">'
            f'<span style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">{status_text}</span>'
            f'{status_badge}'
            f'</div>'
            f'<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px;">'
            f'<div>'
            f'<h2 style="margin: 0; color: #1E293B; font-size: 2rem;">{meta["icon"]} {meta["name"]}</h2>'
            f'<span style="color: #64748B; font-size: 0.95rem;">Category: <strong>{meta["category"]}</strong> &nbsp;|&nbsp; Customer ID: <strong>{active_cust_id}</strong></span>'
            f'</div>'
            f'<div style="text-align: right;">'
            f'<span class="status-badge-ok">Serving: {latency} ms</span><br/>'
            f'<small style="color: #64748B;">Model (Triton): {data.get("triton_ms", "N/A")} ms</small>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

        # Propensity KPI Gauges
        ai_kpi1, ai_kpi2, ai_kpi3, ai_kpi4 = st.columns(4)
        with ai_kpi1:
            st.metric(label="Decision Status", value="Active Pitch 🔥" if is_active else "No Action ⏸️")
        with ai_kpi2:
            st.metric(label="Top Product Propensity", value=f"{probs_data.get(top_prod, 0.0)*100:.1f}%", delta=f"{meta['name']}")
        with ai_kpi3:
            st.metric(label="Acquisition Propensity (Any)", value=f"{p_act*100:.1f}%")
        with ai_kpi4:
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
                    "Status": "🔥 Actionable" if (is_top and is_active) else ("⏸️ Suppressed" if is_top else "Secondary")
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
        await_html = (
            f'<div style="text-align: center; padding: 48px 20px; background: white; border-radius: 12px; border: 2px dashed #A7F3D0;">'
            f'<span style="font-size: 2.5rem;">⚡</span>'
            f'<h3 style="color: #065F46; margin: 12px 0 6px 0; font-weight: 700;">Awaiting Customer Activity</h3>'
            f'<p style="color: #64748B; max-width: 500px; margin: 0 auto; font-size: 0.95rem;">'
            f'Go to the <strong>🏦 Customer Online Bank Portal</strong> tab and simulate an activity to inspect real-time TabDPT predictions.'
            f'</p>'
            f'</div>'
        )
        st.markdown(await_html, unsafe_allow_html=True)


# =============================================================================
# TAB 3: Redis Stream Monitor (Live Audit Feed)
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
            badge = "🔥 Active Offer" if is_active_str == "true" else "⏸️ No Action"
            top_p = payload.get("top_product", "N/A")
            top_name = PRODUCT_METADATA.get(top_p, {}).get("name", top_p)
            act_raw = payload.get("action", "N/A")
            act_title = BANKING_ACTIVITIES.get(act_raw, {}).get("title", act_raw)

            stream_table.append({
                "Message ID": msg_id,
                "Customer ID": payload.get("customer_id", "N/A"),
                "Banking Activity": act_title,
                "Top Recommendation": top_name,
                "NBA Decision": badge,
                "Serving Latency (ms)": payload.get("latency_ms", "N/A"),
                "Model Latency (ms)": payload.get("triton_ms", "N/A"),
                "Timestamp": payload.get("timestamp", "N/A")
            })
        st.dataframe(pd.DataFrame(stream_table), use_container_width=True)
    else:
        st.caption("No stream events recorded yet. Trigger an activity in the Bank Portal to populate the feed.")


# =============================================================================
# TAB 4: Architecture Blueprint
# =============================================================================
with tab_arch:
    st.subheader("🏛️ Architecture Overview: Real-Time TabDPT RecSys")
    
    st.markdown("""
    ```text
    ┌──────────────────────────────┐        XADD         ┌─────────────────────────────┐
    │  Customer Bank Portal (UI)   │ ──────────────────► │   Redis Stream Hub          │
    │  - Paycheck / Deposit / Bill │                     │   'stream:customer_events'  │
    └──────────────────────────────┘                     └──────────────┬──────────────┘
                                                                        │
                                                                   XREADGROUP
                                                                        ▼
    ┌──────────────────────────────┐     ModelInfer      ┌─────────────────────────────┐
    │   Triton gRPC Server         │ ◄────────────────── │   Streaming Worker          │
    │   (TabDPT + KV-Cache)        │ ──────────────────► │   (event_processor.py)      │
    └──────────────────────────────┘    [Batch, 16]      └──────────────┬──────────────┘
                                                                        │
                                                       Filter Held & Evaluate Relevance
                                                                        ▼
                                                         ┌─────────────────────────────┐
                                                         │   Next-Best-Action Gate     │
                                                         │   - Match: Pop-up Ad Modal  │
                                                         │   - Routine: Silent Receipt │
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
    1. **Interactive Bank Simulator with `@st.dialog`**:
       - Simulates a real digital banking portal (checking balance, recent transactions, quick activity center).
       - When a relevant action occurs, an in-app promotional offer dynamically pops up on screen!
       - Routine transactions (e.g. paying utility bill) execute quietly with **No Action** (no annoying popups).
    2. **Tabular Foundation Transformer (TabDPT)**:
       - 32-layer transformer column-attention foundation model operating on 128-dim padded tensors.
       - Precomputed KV-cache drops attention inference down to **< 20ms**.
    3. **Redis Streams Decoupling**:
       - User actions are written directly to `stream:customer_events`.
       - Decoupled worker handles inference and writes to `stream:recommendations` with sub-millisecond pub/sub latency.
    """)

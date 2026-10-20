"""
Santander Next-Best-Action Banking Recommender
Interactive Real-Time Bank Portal & AI Serving Simulator

Powered by:
- TabDPT: Foundation Tabular Transformer with KV-Cache
- Triton Inference Server (gRPC port 8001)
- Redis Streams: Event-Driven Customer Actions & Output Broadcasts
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import torch

try:
    import tritonclient.grpc as grpcclient

    TRITON_CLIENT_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_AVAILABLE = False

from src.config import (
    CLEAN_TEST_PARQUET,
    INDEX_TO_TARGET,
    RECS_TTL_SECONDS,
    SELECTED_15_TARGETS,
    STREAM_CUSTOMER_EVENTS,
    STREAM_RECOMMENDATIONS,
    TRITON_GRPC_URL,
    get_redis_client,
)

LOGO_PATH = _repo_root / "docs" / "TYLOGO.png"

# Friendly display metadata for products
PRODUCT_METADATA = {
    "current_account": {
        "name": "Current Account",
        "icon": "💳",
        "category": "Daily Banking",
    },
    "payroll_account": {
        "name": "Payroll Account",
        "icon": "💼",
        "category": "Daily Banking",
    },
    "credit_card": {"name": "Credit Card", "icon": "💳", "category": "Financing"},
    "pensions": {
        "name": "Pension Plan",
        "icon": "🛡️",
        "category": "Retirement & Security",
    },
    "mortgage": {"name": "Mortgage Loan", "icon": "🏠", "category": "Financing"},
    "securities": {
        "name": "Securities & Stocks",
        "icon": "📈",
        "category": "Wealth & Investment",
    },
    "particular_plus_account": {
        "name": "Particular Plus Account",
        "icon": "⭐",
        "category": "Savings & Premium",
    },
    "particular_account": {
        "name": "Particular Account",
        "icon": "👤",
        "category": "Daily Banking",
    },
    "funds": {
        "name": "Investment Funds",
        "icon": "📊",
        "category": "Wealth & Investment",
    },
    "direct_debit": {
        "name": "Direct Debit Service",
        "icon": "🔄",
        "category": "Daily Banking",
    },
    "payroll": {
        "name": "Payroll Direct Deposit",
        "icon": "💵",
        "category": "Daily Banking",
    },
    "more_particular_account": {
        "name": "Particular More Account",
        "icon": "🌟",
        "category": "Savings & Premium",
    },
    "e_account": {
        "name": "Digital E-Account",
        "icon": "🌐",
        "category": "Digital Banking",
    },
    "long_term_deposits": {
        "name": "Long-Term Fixed Deposit",
        "icon": "⏳",
        "category": "Savings & Deposits",
    },
    "taxes": {
        "name": "Tax Management Account",
        "icon": "📑",
        "category": "Specialized Services",
    },
    "do_nothing": {
        "name": "Hold / No Action",
        "icon": "⏸️",
        "category": "Hold Baseline",
    },
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
        "type": "credit",
    },
    "large_deposit": {
        "title": "High-Value Capital Wire",
        "icon": "🏧",
        "amount": "+€25,000.00",
        "delta": 25000.00,
        "tx_name": "SEPA Capital Wire Inflow",
        "tx_sub": "Interbank Liquidity Transfer",
        "description": "Inflow wire transfer of liquidity capital",
        "type": "credit",
    },
    "branch_inquiry": {
        "title": "Explore Home Mortgage Rates",
        "icon": "🏠",
        "amount": "Inquiry",
        "delta": 0.0,
        "tx_name": "Mortgage Rate Simulation Inquiry",
        "tx_sub": "Branch Digital Advisory Session",
        "description": "Simulate mortgage rate calculation & advisor request",
        "type": "inquiry",
    },
    "tax_season_login": {
        "title": "Open Tax Filing Portal",
        "icon": "📅",
        "amount": "Portal",
        "delta": 0.0,
        "tx_name": "AEAT Fiscal Certificate Access",
        "tx_sub": "Tax Filing Portal Session",
        "description": "Access fiscal documents for annual tax declaration",
        "type": "inquiry",
    },
    "card_payment": {
        "title": "Pay Monthly Utility Bill",
        "icon": "💳",
        "amount": "-€68.40",
        "delta": -68.40,
        "tx_name": "Iberdrola Clientes S.A.",
        "tx_sub": "Utility Bill Direct Debit",
        "description": "Routine merchant payment (Iberdrola Electricity)",
        "type": "debit",
    },
    "dormant_reactivation": {
        "title": "Security Re-authentication",
        "icon": "🔄",
        "amount": "Login",
        "delta": 0.0,
        "tx_name": "Portal Security Re-Authentication",
        "tx_sub": "Biometric 2FA Verification",
        "description": "Reactivate account portal after inactivity",
        "type": "inquiry",
    },
}

# Curated customer archetypes verified from test targets
CURATED_CUSTOMERS = {
    "106596": "💳 106596 - Top Executive (Credit Card)",
    "1273930": "⏳ 1273930 - Senior Capital Investor (Long-Term Deposit)",
    "663201": "⏳ 663201 - Mature Saver (Long-Term Deposit)",
    "392818": "👤 392818 - Established Client (Particular Account)",
    "374737": "⭐ 374737 - Wealth Client (Particular Plus Account)",
    "1542491": "🌟 1542491 - Affluent Resident (More Particular Account)",
    "1221443": "🌐 1221443 - Tech Resident (Digital E-Account)",
    "1034229": "💼 1034229 - Recent Graduate (Payroll Account)",
    "1009063": "🛡️ 1009063 - Active Contributor (Pensions)",
    "1002447": "🔄 1002447 - Urban Resident (Direct Debit)",
    "1002271": "🏦 1002271 - Mid-Career Family (Current Account)",
    "1166753": "🎓 1166753 - University Student (Payroll / Pensions)",
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
            {
                "title": "Vanguard S&P 500 Index SIP",
                "subtitle": "Monthly Investment Direct Debit",
                "amount": "-€500.00",
                "delta": -500.0,
                "time": "2 days ago",
            },
            {
                "title": "Sanitas Private Healthcare",
                "subtitle": "Health Insurance Policy",
                "amount": "-€115.40",
                "delta": -115.4,
                "time": "5 days ago",
            },
            {
                "title": "Mercadona Gourmet Madrid",
                "subtitle": "POS Contactless Swipe",
                "amount": "-€84.60",
                "delta": -84.6,
                "time": "1 week ago",
            },
            {
                "title": "Repsol AutoFuel Estación",
                "subtitle": "Debit Card Fuel Payment",
                "amount": "-€65.00",
                "delta": -65.0,
                "time": "1 week ago",
            },
        ],
    },
    "1002447": {
        "account_name": "Santander Smart Daily Checking",
        "iban": "ES42 0049 2831 9901 4427",
        "status": "Active Resident",
        "balance": 3420.50,
        "transactions": [
            {
                "title": "Metro de Madrid Bono Transporte",
                "subtitle": "Public Transit Card Recharge",
                "amount": "-€54.60",
                "delta": -54.6,
                "time": "Yesterday",
            },
            {
                "title": "Carrefour Market Chamberí",
                "subtitle": "Supermarket Grocery Debit",
                "amount": "-€38.25",
                "delta": -38.25,
                "time": "3 days ago",
            },
            {
                "title": "Endesa Energía Renovable",
                "subtitle": "Electricity Direct Debit",
                "amount": "-€72.10",
                "delta": -72.1,
                "time": "6 days ago",
            },
            {
                "title": "Bizum Split - Dinner with Marcos",
                "subtitle": "P2P Instant Mobile Payment",
                "amount": "+€24.00",
                "delta": 24.0,
                "time": "1 week ago",
            },
        ],
    },
    "1166753": {
        "account_name": "Santander Smart Joven Account",
        "iban": "ES19 0049 0012 3345 6789",
        "status": "University Student",
        "balance": 842.10,
        "transactions": [
            {
                "title": "Complutense University Library",
                "subtitle": "Academic Course Material",
                "amount": "-€22.50",
                "delta": -22.5,
                "time": "Yesterday",
            },
            {
                "title": "Cafetería Universitaria Moncloa",
                "subtitle": "Contactless Student Debit",
                "amount": "-€3.40",
                "delta": -3.4,
                "time": "2 days ago",
            },
            {
                "title": "Family Allowance (SEPA Inflow)",
                "subtitle": "Monthly Student Support Wire",
                "amount": "+€250.00",
                "delta": 250.0,
                "time": "5 days ago",
            },
            {
                "title": "Spotify Student Premium",
                "subtitle": "Digital Streaming Subscription",
                "amount": "-€5.99",
                "delta": -5.99,
                "time": "1 week ago",
            },
        ],
    },
    "106596": {
        "account_name": "Santander Private Banking Wealth Account",
        "iban": "ES12 0049 8812 4001 0659",
        "status": "Private Banking (Executive)",
        "balance": 42150.80,
        "transactions": [
            {
                "title": "Club de Campo Villa de Madrid",
                "subtitle": "Annual Country Club Membership",
                "amount": "-€380.00",
                "delta": -380.0,
                "time": "Yesterday",
            },
            {
                "title": "Iberia Business Class Madrid-Frankfurt",
                "subtitle": "Airlines Corporate Booking",
                "amount": "-€640.50",
                "delta": -640.50,
                "time": "3 days ago",
            },
            {
                "title": "Restaurante Zalacaín Madrid",
                "subtitle": "Executive POS Dining Swipe",
                "amount": "-€215.00",
                "delta": -215.0,
                "time": "5 days ago",
            },
            {
                "title": "Loewe Serrano Flagship",
                "subtitle": "Luxury Retail Card Swipe",
                "amount": "-€490.00",
                "delta": -490.0,
                "time": "1 week ago",
            },
        ],
    },
    "1273930": {
        "account_name": "Santander Senior Premier Portfolio",
        "iban": "ES81 0049 4432 1098 7393",
        "status": "Senior Premier Wealth",
        "balance": 85420.00,
        "transactions": [
            {
                "title": "Notaría González & Asociados",
                "subtitle": "Estate Advisory & Legal Fees",
                "amount": "-€450.00",
                "delta": -450.0,
                "time": "2 days ago",
            },
            {
                "title": "Hospital San Rafael Madrid",
                "subtitle": "Private Medical Consultation",
                "amount": "-€180.00",
                "delta": -180.0,
                "time": "4 days ago",
            },
            {
                "title": "Santander Treasury Yield Inflow",
                "subtitle": "Fixed Income Semi-Annual Yield",
                "amount": "+€1,850.00",
                "delta": 1850.0,
                "time": "1 week ago",
            },
            {
                "title": "Mutua Madrileña Seguros",
                "subtitle": "Home & Property Insurance",
                "amount": "-€320.00",
                "delta": -320.0,
                "time": "10 days ago",
            },
        ],
    },
    "663201": {
        "account_name": "Santander Fixed Capital Deposit",
        "iban": "ES48 0049 6632 0100 2319",
        "status": "Long-Term Investor",
        "balance": 48500.00,
        "transactions": [
            {
                "title": "Fixed Term Deposit Interest Payout",
                "subtitle": "Quarterly Liquidity Yield",
                "amount": "+€540.00",
                "delta": 540.0,
                "time": "2 days ago",
            },
            {
                "title": "Repsol AutoGas Estación",
                "subtitle": "Contactless Fuel Debit",
                "amount": "-€58.40",
                "delta": -58.4,
                "time": "4 days ago",
            },
            {
                "title": "Farmacia Central Murcia",
                "subtitle": "Health & Prescription POS",
                "amount": "-€28.60",
                "delta": -28.6,
                "time": "1 week ago",
            },
            {
                "title": "Iberdrola Clientes S.A.",
                "subtitle": "Home Electricity Direct Debit",
                "amount": "-€64.20",
                "delta": -64.2,
                "time": "10 days ago",
            },
        ],
    },
    "1221443": {
        "account_name": "Santander Online Smart Checking",
        "iban": "ES65 0049 7120 4455 1443",
        "status": "Active Digital Resident",
        "balance": 5340.20,
        "transactions": [
            {
                "title": "Apple Services Dublin",
                "subtitle": "iCloud & Developer Membership",
                "amount": "-€12.99",
                "delta": -12.99,
                "time": "Yesterday",
            },
            {
                "title": "Cabify Valencia Mobility",
                "subtitle": "Urban Rideshare Charge",
                "amount": "-€14.50",
                "delta": -14.5,
                "time": "2 days ago",
            },
            {
                "title": "Mercadona Ruzafa Valencia",
                "subtitle": "Supermarket Contactless Swipe",
                "amount": "-€52.30",
                "delta": -52.3,
                "time": "4 days ago",
            },
            {
                "title": "Amazon Web Services EMEA",
                "subtitle": "Cloud Hosting Direct Debit",
                "amount": "-€34.20",
                "delta": -34.2,
                "time": "1 week ago",
            },
        ],
    },
    "1034229": {
        "account_name": "Santander Smart University Graduate",
        "iban": "ES39 0049 0154 8820 4229",
        "status": "University Graduate",
        "balance": 3890.40,
        "transactions": [
            {
                "title": "Inditex Corporate Vigo",
                "subtitle": "Graduate Salary SEPA Direct Credit",
                "amount": "+€1,650.00",
                "delta": 1650.0,
                "time": "Yesterday",
            },
            {
                "title": "Zara Home Pontevedra",
                "subtitle": "Home Decor POS Contactless",
                "amount": "-€48.90",
                "delta": -48.9,
                "time": "3 days ago",
            },
            {
                "title": "Café Central Pontevedra",
                "subtitle": "Contactless POS Swipe",
                "amount": "-€4.20",
                "delta": -4.2,
                "time": "4 days ago",
            },
            {
                "title": "Renfe Alvia Vigo - Madrid",
                "subtitle": "High-Speed Rail Travel",
                "amount": "-€42.00",
                "delta": -42.0,
                "time": "1 week ago",
            },
        ],
    },
    "1002271": {
        "account_name": "Santander Classic Family Checking",
        "iban": "ES54 0049 8830 1209 7721",
        "status": "Mid-Career",
        "balance": 8750.30,
        "transactions": [
            {
                "title": "Colegio San Patricio School Tuition",
                "subtitle": "Monthly Direct Debit Education",
                "amount": "-€340.00",
                "delta": -340.0,
                "time": "3 days ago",
            },
            {
                "title": "El Corte Inglés Castellana",
                "subtitle": "Department Store Card Swipe",
                "amount": "-€142.80",
                "delta": -142.8,
                "time": "5 days ago",
            },
            {
                "title": "Vodafone Fibra Óptica 1Gbps",
                "subtitle": "Home Broadband & Telecom",
                "amount": "-€52.00",
                "delta": -52.0,
                "time": "1 week ago",
            },
            {
                "title": "Gas Natural Fenosa",
                "subtitle": "Home Gas Direct Debit",
                "amount": "-€48.15",
                "delta": -48.15,
                "time": "10 days ago",
            },
        ],
    },
    "392818": {
        "account_name": "Santander Personal Particular Checking",
        "iban": "ES51 0049 3928 1800 5512",
        "status": "Established Personal",
        "balance": 14250.00,
        "transactions": [
            {
                "title": "El Corte Inglés Castellana",
                "subtitle": "Department Store POS Debit",
                "amount": "-€134.60",
                "delta": -134.6,
                "time": "Yesterday",
            },
            {
                "title": "Repsol AutoFuel Estación",
                "subtitle": "Fuel Card Payment",
                "amount": "-€68.00",
                "delta": -68.0,
                "time": "3 days ago",
            },
            {
                "title": "Mercadona Chamberí",
                "subtitle": "Weekly Household Groceries",
                "amount": "-€82.40",
                "delta": -82.4,
                "time": "5 days ago",
            },
            {
                "title": "Bizum Instant Transfer - Carlos",
                "subtitle": "Peer-to-Peer Mobile Credit",
                "amount": "+€45.00",
                "delta": 45.0,
                "time": "1 week ago",
            },
        ],
    },
    "374737": {
        "account_name": "Santander Particular Plus Premium",
        "iban": "ES29 0049 3747 3701 8834",
        "status": "Premium Banking",
        "balance": 31800.50,
        "transactions": [
            {
                "title": "Quarterly Wealth Advisory Retainer",
                "subtitle": "Private Portfolio Management",
                "amount": "-€250.00",
                "delta": -250.0,
                "time": "2 days ago",
            },
            {
                "title": "Real Club de Golf Sevilla",
                "subtitle": "Sports Facility Fee",
                "amount": "-€165.00",
                "delta": -165.0,
                "time": "4 days ago",
            },
            {
                "title": "Hospital Quirónsalud Sevilla",
                "subtitle": "Private Health Payment",
                "amount": "-€95.00",
                "delta": -95.0,
                "time": "1 week ago",
            },
            {
                "title": "Santander Preferred Dividend Yield",
                "subtitle": "Quarterly Equity Distribution",
                "amount": "+€480.00",
                "delta": 480.0,
                "time": "10 days ago",
            },
        ],
    },
    "1542491": {
        "account_name": "Santander Particular More Account",
        "iban": "ES72 0049 1542 4910 9941",
        "status": "Affluent Resident",
        "balance": 22640.00,
        "transactions": [
            {
                "title": "Telefónica Movistar Fibra 1Gb",
                "subtitle": "Home Broadband Direct Debit",
                "amount": "-€74.90",
                "delta": -74.9,
                "time": "Yesterday",
            },
            {
                "title": "Mercado San Miguel Gourmet",
                "subtitle": "Contactless POS Dining Swipe",
                "amount": "-€88.00",
                "delta": -88.0,
                "time": "3 days ago",
            },
            {
                "title": "Parador de Turismo Toledo",
                "subtitle": "Weekend Hospitality Booking",
                "amount": "-€195.00",
                "delta": -195.0,
                "time": "6 days ago",
            },
            {
                "title": "Abono Nómina Profesional",
                "subtitle": "Corporate Salary Credit",
                "amount": "+€3,450.00",
                "delta": 3450.0,
                "time": "1 week ago",
            },
        ],
    },
    "1011230": {
        "account_name": "Santander Starter Account",
        "iban": "ES21 0049 6102 3381 4059",
        "status": "Standard Account",
        "balance": 2180.75,
        "transactions": [
            {
                "title": "Decathlon Sports Equipment",
                "subtitle": "Sporting Goods Card Payment",
                "amount": "-€45.00",
                "delta": -45.0,
                "time": "2 days ago",
            },
            {
                "title": "Starbucks Paseo del Prado",
                "subtitle": "Cafeteria Contactless Swipe",
                "amount": "-€4.80",
                "delta": -4.8,
                "time": "3 days ago",
            },
            {
                "title": "Gympass Fitness Subscription",
                "subtitle": "Monthly Fitness Direct Debit",
                "amount": "-€29.99",
                "delta": -29.99,
                "time": "1 week ago",
            },
            {
                "title": "Mercadona Express",
                "subtitle": "Weekly Groceries Debit",
                "amount": "-€26.40",
                "delta": -26.4,
                "time": "1 week ago",
            },
        ],
    },
    "1000008": {
        "account_name": "Santander Standard Inactive Ledger",
        "iban": "ES90 0049 0001 0000 8123",
        "status": "Dormant (Inactive)",
        "balance": 124.50,
        "transactions": [
            {
                "title": "Account Administration Fee",
                "subtitle": "Santander Quarterly Maintenance Fee",
                "amount": "-€12.00",
                "delta": -12.0,
                "time": "3 months ago",
            },
            {
                "title": "Santander ATM Cash Withdrawal",
                "subtitle": "Automated Teller Cash Out",
                "amount": "-€50.00",
                "delta": -50.0,
                "time": "7 months ago",
            },
            {
                "title": "Annual Account Yield Interest",
                "subtitle": "Deposit Yield Payout",
                "amount": "+€0.12",
                "delta": 0.12,
                "time": "9 months ago",
            },
        ],
    },
    "658033": {
        "account_name": "Santander Commercial Retail Checking",
        "iban": "ES33 0049 5521 8763 0214",
        "status": "Commercial Retail",
        "balance": 19450.00,
        "transactions": [
            {
                "title": "Distribuciones Hostelería Madrid",
                "subtitle": "Commercial Supplier Wire",
                "amount": "-€620.00",
                "delta": -620.0,
                "time": "3 days ago",
            },
            {
                "title": "POS Merchant Card Settlement",
                "subtitle": "Daily Credit Terminal Batch Out",
                "amount": "+€1,840.00",
                "delta": 1840.0,
                "time": "4 days ago",
            },
            {
                "title": "Canal de Isabel II Water Utility",
                "subtitle": "Commercial Water Direct Debit",
                "amount": "-€88.40",
                "delta": -88.4,
                "time": "1 week ago",
            },
            {
                "title": "Mutua Madrileña Commercial Insurance",
                "subtitle": "Annual Liability Premium",
                "amount": "-€145.00",
                "delta": -145.0,
                "time": "2 weeks ago",
            },
        ],
    },
}


def get_or_init_customer_ledger(customer_id: str):
    """Retrieves or initializes mutable customer account ledger in session state."""
    if "customer_ledgers" not in st.session_state:
        st.session_state["customer_ledgers"] = {}
    if customer_id not in st.session_state["customer_ledgers"]:
        import copy

        if customer_id in CUSTOMER_ACCOUNTS_DATA:
            st.session_state["customer_ledgers"][customer_id] = copy.deepcopy(
                CUSTOMER_ACCOUNTS_DATA[customer_id]
            )
        else:
            cid_hash = abs(hash(str(customer_id))) % 9000 + 1000
            st.session_state["customer_ledgers"][customer_id] = {
                "account_name": "Santander Personal Checking",
                "iban": f"ES85 0049 1000 {cid_hash} 9921",
                "status": "Standard Account",
                "balance": 4850.20,
                "transactions": [
                    {
                        "title": "Supermercados Mercadona",
                        "subtitle": "Point of Sale Swipe",
                        "amount": "-€42.50",
                        "delta": -42.50,
                        "time": "2 days ago",
                    },
                    {
                        "title": "Cafeteria Gran Vía",
                        "subtitle": "Contactless Payment",
                        "amount": "-€3.80",
                        "delta": -3.80,
                        "time": "3 days ago",
                    },
                    {
                        "title": "Gasolinera Repsol",
                        "subtitle": "Debit Fuel Card",
                        "amount": "-€65.00",
                        "delta": -65.00,
                        "time": "5 days ago",
                    },
                ],
            }
    return st.session_state["customer_ledgers"][customer_id]


def reset_customer_ledger(customer_id: str):
    """Resets customer balance and transactions back to archetype defaults."""
    if (
        "customer_ledgers" in st.session_state
        and customer_id in st.session_state["customer_ledgers"]
    ):
        import copy

        if customer_id in CUSTOMER_ACCOUNTS_DATA:
            st.session_state["customer_ledgers"][customer_id] = copy.deepcopy(
                CUSTOMER_ACCOUNTS_DATA[customer_id]
            )
        else:
            del st.session_state["customer_ledgers"][customer_id]

    if (
        "claimed_products" in st.session_state
        and customer_id in st.session_state["claimed_products"]
    ):
        del st.session_state["claimed_products"][customer_id]


# -----------------------------------------------------------------------------
# Page Configuration & Emerald Green Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Santander AI | Next Best Action Recommender",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
    .regime-badge-cpu {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #BFDBFE;
        display: inline-block;
    }
    .regime-badge-gate {
        background-color: #F1F5F9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #CBD5E1;
        display: inline-block;
    }
    .regime-badge-cache {
        background-color: #F5F3FF;
        color: #6D28D9;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #DDD6FE;
        display: inline-block;
    }
    .regime-badge-live {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #86EFAC;
        display: inline-block;
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
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# System Helpers & Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_redis():
    """Cached Redis client instance."""
    return get_redis_client()


@st.cache_resource(show_spinner=False)
def get_serving_pipeline():
    """
    Initializes and caches the Stage 1 CPU LightGBM gatekeeper and
    Stage 2 GPU TabDPT Engine with precomputed KV-cache.
    """
    from src.streaming.event_processor import TabDPTEngine, load_lgbm_gatekeeper

    lgbm = load_lgbm_gatekeeper()
    tabdpt = TabDPTEngine(prefer_triton=True)
    return lgbm, tabdpt


def check_triton_status():
    """Checks if Triton gRPC server is responding or if in-process GPU/CPU mode is active."""
    if not TRITON_CLIENT_AVAILABLE:
        return False, "In-Process GPU Mode (tritonclient not installed)"

    try:
        client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        if client.is_server_live() and client.is_model_ready("tabdpt"):
            return True, "Online & Ready (tabdpt v3)"
        return False, "In-Process GPU Mode (Triton server offline)"
    except Exception as e:  # noqa: BLE001
        return False, f"In-Process GPU Mode ({e.__class__.__name__})"


@st.cache_data(show_spinner=False)
def load_customer_dossier(customer_id: str):
    """Loads customer snapshot and current holdings from Redis Feature Store."""
    from src.features.feature_store import RedisFeatureStore

    fs = RedisFeatureStore(redis_client=get_redis())
    try:
        _, held, metadata = fs.get_customer_features(customer_id, enable_fallback=True)
        if metadata.get("unknown"):
            return None
        return {"row": metadata, "held_products": held}
    except Exception:
        test_df = (
            pl.scan_parquet(str(CLEAN_TEST_PARQUET))
            .filter(pl.col("customer_id") == customer_id)
            .collect()
        )
        if test_df.is_empty():
            return None
        row = test_df.to_dicts()[0]
        held = [p for p in SELECTED_15_TARGETS if row.get(p, 0) == 1]
        return {"row": row, "held_products": held}


@st.cache_data(show_spinner=False)
def get_cached_customer_features(customer_id: str):
    """
    Retrieves the 128-dim normalized query tensor and held products mask
    directly from the Redis Feature Store (<1ms point lookup).
    """
    from src.features.feature_store import RedisFeatureStore

    fs = RedisFeatureStore(redis_client=get_redis())
    x_qry, held, _ = fs.get_customer_features(customer_id, enable_fallback=True)
    return x_qry, held


@st.cache_data(show_spinner=False)
def run_customer_model_inference(customer_id: str, claimed_tuple: tuple = (), action: str = "query"):
    """
    Executes the validated 3-Stage Cascaded Routing Architecture:
    - Stage 1: CPU LightGBM Gatekeeper (<0.002 ms)
    - Stage 2: GPU TabDPT (73,968 Scaled Context) + Odds-Ratio Calibration (ORC) Gate
    - Stage 3: Dual-Predictor + Next-Best-Action Relevance Routing
    """
    from src.streaming.event_processor import (
        GAMMA_CACHE,
        GAMMA_LIVE,
        NOISY_SINKS,
        THETA_CLEAN,
        THETA_NOISY,
        process_customer_event,
    )

    t_start = time.time()
    lgbm_model, tabdpt_engine = get_serving_pipeline()
    x_qry, held = get_cached_customer_features(customer_id)
    feat_time_ms = (time.time() - t_start) * 1000

    # Combine static historical holdings with newly claimed demo accounts
    all_held = list(dict.fromkeys(held + list(claimed_tuple)))

    t_cascade_start = time.time()
    res = process_customer_event(
        features_80d=x_qry,
        action=action,
        lgbm_model=lgbm_model,
        tabdpt_model=tabdpt_engine,
        held_products=all_held,
    )
    cascade_time_ms = (time.time() - t_cascade_start) * 1000

    # Extract or compute probability vector for visualization
    prob_dict = {}
    try:
        p16 = tabdpt_engine.predict_proba(x_qry)[0]
        prob_dict = {INDEX_TO_TARGET[i]: float(p16[i]) for i in range(16)}
    except Exception:
        prob_dict = {"do_nothing": 1.0 - float(res.get("p_buy", 0.0))}
        if res.get("predicted_product") != "do_nothing":
            prob_dict[res.get("predicted_product")] = float(res.get("p_buy", 0.1))

    top_product = res["predicted_product"]
    recos = res["top_7_products"]
    if not recos and top_product != "do_nothing":
        recos = [top_product]

    return {
        "recos": recos,
        "top_product": top_product,
        "prediction": res["prediction"],
        "prob_dict": prob_dict,
        "p_buy": float(res["p_buy"]),
        "odds_ratio": float(res["odds_ratio"]),
        "regime": res["regime"],
        "gpu_skipped": res["gpu_skipped"],
        "is_active_trigger": res["is_active_trigger"],
        "cascade_ms": round(cascade_time_ms, 2),
        "feat_ms": round(feat_time_ms, 2),
    }


def execute_pipeline_for_event(customer_id: str, action: str):
    """
    Executes end-to-end inference and business logic:
    1. Executes 3-Stage Cascade (CPU LightGBM -> GPU TabDPT + ORC Gate -> Dual-Predictor Routing).
    2. Publishes event to STREAM_RECOMMENDATIONS and updates KEY_CUSTOMER_RECS in Redis.
    """
    from src.streaming.event_processor import publish_reco

    start_time = time.time()
    claimed = tuple(
        sorted(st.session_state.get("claimed_products", {}).get(str(customer_id), []))
    )
    infer_data = run_customer_model_inference(customer_id, claimed, action=action)

    gateway_latency = (time.time() - start_time) * 1000
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    stream_payload = {
        "customer_id": str(customer_id),
        "action": str(action),
        "prediction": str(infer_data["prediction"]),
        "top_product": str(infer_data["top_product"]),
        "recommendations": json.dumps(infer_data["recos"]),
        "is_active_trigger": "true" if infer_data["is_active_trigger"] else "false",
        "gpu_skipped": "true" if infer_data["gpu_skipped"] else "false",
        "p_buy": f"{infer_data['p_buy']:.4f}",
        "odds_ratio": f"{infer_data['odds_ratio']:.4f}",
        "regime": str(infer_data["regime"]),
        "latency_ms": f"{gateway_latency:.2f}",
        "cascade_ms": f"{infer_data['cascade_ms']:.2f}",
        "feat_ms": f"{infer_data['feat_ms']:.2f}",
        "timestamp": now_iso,
        "probabilities": json.dumps(infer_data["prob_dict"]),
    }

    # Publish to Redis streams & cache
    publish_reco(stream_payload, r_client=get_redis())

    return stream_payload, None


# -----------------------------------------------------------------------------
# Dynamic Ad Modal Dialog (@st.dialog)
# -----------------------------------------------------------------------------
@st.dialog("🎉 Personalized Offer Just for You!")
def render_offer_modal(product_key: str, action_key: str):
    """Pops up a realistic banking offer modal when an action matches product intent."""
    meta = PRODUCT_METADATA.get(
        product_key, {"name": product_key, "icon": "🎁", "category": "Special Offer"}
    )
    act_info = BANKING_ACTIVITIES.get(action_key, {"title": action_key})

    st.markdown(f"## {meta.get('icon', '🎁')} {meta.get('name', product_key)}")
    st.caption(
        f"Category: **{meta.get('category')}** &nbsp;|&nbsp; Verified Santander Product"
    )

    st.markdown(f"""
    Because you just completed **{act_info["title"]}**, our Next-Best-Action engine matched your profile 
    for an exclusive financial upgrade with preferred member rates.
    """)

    st.info("""
    ✓ **Pre-Approved with 1-Click Setup**: Zero paperwork required.  
    ✓ **Exclusive Promotional Terms**: 0% introductory maintenance fees.  
    ✓ **Seamless Integration**: Directly connects with your existing Santander accounts.
    """)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(
            "🚀 Claim & Open Account", type="primary", use_container_width=True
        ):
            active_cid = st.session_state.get("current_customer", "")
            if "claimed_products" not in st.session_state:
                st.session_state["claimed_products"] = {}
            if active_cid not in st.session_state["claimed_products"]:
                st.session_state["claimed_products"][active_cid] = set()
            st.session_state["claimed_products"][active_cid].add(product_key)
            try:
                from src.features.feature_store import RedisFeatureStore

                fs = RedisFeatureStore(redis_client=get_redis())
                fs.update_held_products(active_cid, product_key)
            except Exception:
                pass

            # Record welcome activation in customer live ledger
            if active_cid:
                ledger = get_or_init_customer_ledger(active_cid)
                ledger["transactions"].insert(
                    0,
                    {
                        "title": f"Account Activated: {meta.get('name', product_key)}",
                        "subtitle": "1-Click Digital Setup • Zero Maintenance Fee",
                        "amount": "Activated",
                        "delta": 0.0,
                        "time": time.strftime("%H:%M:%S Today"),
                    },
                )

            st.session_state["claimed_offer"] = meta.get("name", product_key)
            st.session_state.pop("popup_offer_to_render", None)
            # Retain st.session_state["last_emitted_payload"] so AI NBA Hub keeps full telemetry & prediction context
            st.toast(
                f"🎉 {meta.get('name', product_key)} opened! Added to your Santander portfolio.",
                icon="✅",
            )
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
        index=0,
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
        st.markdown(
            f"**Age:** {row.get('age', 'N/A')} yrs &nbsp;|&nbsp; **Gender:** {row.get('gender', 'N/A')}"
        )
        st.markdown(
            f"**Province:** {row.get('province_name', 'N/A')} &nbsp;|&nbsp; **Seniority:** {row.get('seniority_months', 'N/A')} mo"
        )
        st.markdown(f"**Segment:** {row.get('customer_segment', 'N/A')}")

        income = row.get("gross_household_income")
        if income is not None:
            st.markdown(f"**Household Gross Income:** €{income:,.2f}")
        else:
            st.markdown("**Household Gross Income:** N/A")

        st.subheader("📦 Currently Held Products")
        held = dossier["held_products"]
        claimed = list(
            st.session_state.get("claimed_products", {}).get(active_cust_id, [])
        )
        all_held = list(dict.fromkeys(held + claimed))
        if all_held:
            held_html = "".join(
                [
                    f"<span class='product-pill' style='background-color: #DCFCE7; border: 1.5px solid #059669; color: #065F46; font-weight: 700;'>✨ {PRODUCT_METADATA.get(p, {}).get('name', p)} (New)</span>"
                    if p in claimed
                    else f"<span class='product-pill'>✓ {PRODUCT_METADATA.get(p, {}).get('name', p)}</span>"
                    for p in all_held
                ]
            )
            st.markdown(held_html, unsafe_allow_html=True)
        else:
            st.caption("No active products currently held.")
    else:
        st.warning(f"Customer {active_cust_id} not found in test partition.")


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <div>
        <h1>🏦 Santander Next-Best-Action Banking Recommender</h1>
        <p>Real-Time Tabular Foundation Model (TabDPT) Serving & Redis Streaming Event Hub</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Status KPI row
r_client = get_redis()
triton_ok, triton_msg = check_triton_status()

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.metric(
        label="TabDPT In-Context",
        value="73,968 Rows",
        delta="16 classes × 4,623 uniform",
    )
with kpi_c2:
    st.metric(
        label="Stage 1 Gatekeeper",
        value="LightGBM Gate",
        delta="0.8912 AUC (<0.002 ms)",
    )
with kpi_c3:
    if triton_ok:
        st.metric(label="Model Serving", value="Triton Server", delta="Port 8001 (KV-Cache)")
    else:
        st.metric(
            label="Model Serving",
            value="Triton Server",
            delta="In-Process Fallback",
        )
with kpi_c4:
    st.metric(
        label="Event Hub",
        value="Redis Engine Streams",
        delta="Feature Store + Streams",
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_portal, tab_ai, tab_stream, tab_arch = st.tabs(
    [
        "🏦 Customer Online Bank Portal",
        "🧠 AI & Next-Best-Action Hub",
        "⚡ Redis Stream & Telemetry Monitor",
        "🏛️ Architecture Blueprint",
    ]
)


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
                f"<div><h4>{account_name}</h4><h2>€{current_balance:,.2f}</h2></div>"
                f'<div style="text-align: right;"><span class="status-badge-ok">{account_status}</span></div>'
                f"</div>"
                f'<div style="display: flex; justify-content: space-between; margin-top: 14px; font-size: 0.9rem; color: #D1FAE5; border-top: 1px solid rgba(255,255,255,0.25); padding-top: 10px;">'
                f"<span>Account Holder: <strong>Customer #{active_cust_id}</strong></span>"
                f"<span>IBAN: <strong>{iban}</strong></span>"
                f"</div>"
                f"</div>"
            )
            st.markdown(bal_html, unsafe_allow_html=True)
        with bal_c2:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            if st.button(
                "↺ Reset Ledger",
                key="reset_ledger_btn",
                use_container_width=True,
                help="Reset balance and transactions to default initial state",
            ):
                reset_customer_ledger(active_cust_id)
                st.session_state.pop("last_tx_receipt", None)
                st.toast("Account ledger reset to initial state.", icon="🔄")
                st.rerun()

        st.subheader("⚡ Customer Activity Center")
        st.caption(
            "Click any banking action to simulate real-time behavior. Balance and ledger update instantly!"
        )

        # Dynamic Notification or Transaction Receipt with Cascaded Telemetry
        if (
            "last_tx_receipt" in st.session_state
            and st.session_state.get("last_emitted_customer") == active_cust_id
        ):
            tx = st.session_state["last_tx_receipt"]
            reg = tx.get("regime", "")
            p_b = tx.get("p_buy", "N/A")
            o_r = tx.get("odds_ratio", "N/A")
            top_p_name = PRODUCT_METADATA.get(tx.get("top_product", ""), {}).get("name", tx.get("top_product", ""))

            if tx.get("is_active"):
                st.markdown(
                    f'<div class="trigger-active-banner">'
                    f"💥 <strong>LIVE OFFER TRIGGERED!</strong> Action <strong>{tx['title']}</strong> processed ({tx['amount']}). "
                    f"Balance: <strong>€{tx['new_balance']:,.2f}</strong> &nbsp;|&nbsp; "
                    f"<span class='regime-badge-live'>🔥 Live Predictor Push (p_buy={p_b} ≥ 0.30)</span><br/>"
                    f"<small>Recommended: <strong>{top_p_name}</strong> • Promotional modal dispatched to customer screen.</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif reg == "stage1_cpu_hold":
                st.markdown(
                    f'<div class="trigger-passive-banner">'
                    f"⚡ <strong>Transaction Confirmed:</strong> {tx['title']} processed ({tx['amount']}). "
                    f"Balance: <strong>€{tx['new_balance']:,.2f}</strong> &nbsp;|&nbsp; "
                    f"<span class='regime-badge-cpu'>⚡ Stage 1 CPU Hold</span><br/>"
                    f"<small style='color: #475569;'>Non-buyer propensity <strong>p_buy = {p_b} < 0.20</strong>. "
                    f"GPU inference completely bypassed on CPU (<0.002 ms). Zero notification friction.</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif reg == "stage2_gate1c_hold":
                st.markdown(
                    f'<div class="trigger-passive-banner">'
                    f"🛡️ <strong>Transaction Confirmed:</strong> {tx['title']} processed ({tx['amount']}). "
                    f"Balance: <strong>€{tx['new_balance']:,.2f}</strong> &nbsp;|&nbsp; "
                    f"<span class='regime-badge-gate'>🛡️ ORC Gate Hold</span><br/>"
                    f"<small style='color: #475569;'>Odds ratio <strong>{o_r}</strong> below noise barrier for candidate <strong>{top_p_name}</strong>. "
                    f"Suppressed to preserve high precision.</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif reg == "candidate_a_silent_cache":
                st.markdown(
                    f'<div class="trigger-passive-banner">'
                    f"💾 <strong>Transaction Confirmed:</strong> {tx['title']} processed ({tx['amount']}). "
                    f"Balance: <strong>€{tx['new_balance']:,.2f}</strong> &nbsp;|&nbsp; "
                    f"<span class='regime-badge-cache'>💾 Cache Predictor (Silent Cache)</span><br/>"
                    f"<small style='color: #475569;'>Recommendation <strong>{top_p_name}</strong> (p_buy={p_b}, odds={o_r}) silently stored in Redis <code>recs:{active_cust_id}</code>. "
                    f"Action '{tx['title']}' is not relevant, so live notification was suppressed.</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="trigger-passive-banner">'
                    f"✅ <strong>Transaction Confirmed:</strong> {tx['title']} processed ({tx['amount']}). "
                    f"Balance: <strong>€{tx['new_balance']:,.2f}</strong>. "
                    f"<em>Decision: Suppressed (No action).</em>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # 6 Interactive Banking Activity Buttons (2x3 Grid)
        act_c1, act_c2 = st.columns(2)

        act_keys = list(BANKING_ACTIVITIES.keys())
        triggered_action_key = None

        with act_c1:
            for k in act_keys[:3]:
                act = BANKING_ACTIVITIES[k]
                if st.button(
                    f"{act['icon']} {act['title']}\n({act['amount']})",
                    key=f"btn_{k}",
                    use_container_width=True,
                ):
                    triggered_action_key = k

        with act_c2:
            for k in act_keys[3:]:
                act = BANKING_ACTIVITIES[k]
                if st.button(
                    f"{act['icon']} {act['title']}\n({act['amount']})",
                    key=f"btn_{k}",
                    use_container_width=True,
                ):
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
                "time": time.strftime("%H:%M:%S Today"),
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
            with st.spinner(
                f"Processing {act_info['title']} through TabDPT & Triton..."
            ):
                res, err = execute_pipeline_for_event(
                    active_cust_id, triggered_action_key
                )
                if not err and res:
                    st.session_state["last_emitted_customer"] = active_cust_id
                    st.session_state["last_emitted_payload"] = res
                    is_active = res.get("is_active_trigger") in ("true", True)

                    st.session_state["last_tx_receipt"] = {
                        "title": act_info["title"],
                        "amount": act_info["amount"],
                        "delta": delta,
                        "new_balance": cust_ledger["balance"],
                        "is_active": is_active,
                        "regime": res.get("regime", ""),
                        "p_buy": res.get("p_buy", "N/A"),
                        "odds_ratio": res.get("odds_ratio", "N/A"),
                        "gpu_skipped": res.get("gpu_skipped", "false"),
                        "top_product": res.get("top_product", "do_nothing"),
                        "time": time.strftime("%H:%M:%S"),
                    }

                    # If active match, trigger pop-up offer!
                    if is_active:
                        st.session_state["popup_offer_to_render"] = (
                            res.get("top_product"),
                            triggered_action_key,
                        )
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
                f"<div><strong>{tx['title']}</strong><br/>"
                f'<small style="color: #64748B;">{tx["subtitle"]} • {tx["time"]}</small></div>'
                f'<span style="color: {color}; font-weight: 700; font-size: 0.95rem;">{tx["amount"]}</span>'
                f"</div>"
            )

        ledger_content = "".join(tx_html_items)
        st.markdown(
            f'<div class="bank-card" style="max-height: 440px; overflow-y: auto;">{ledger_content}</div>',
            unsafe_allow_html=True,
        )

    # Check if pop-up dialog should be launched (consume flag so it pops up strictly once)
    if (
        "popup_offer_to_render" in st.session_state
        and st.session_state["popup_offer_to_render"] is not None
    ):
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
        regime = data.get("regime", "candidate_c_live_push" if is_active else "stage2_gate1c_hold")
        gpu_skipped = data.get("gpu_skipped") in ("true", True, "True")

        try:
            p_buy_val = float(data.get("p_buy", 0.0))
        except (ValueError, TypeError):
            p_buy_val = 0.0

        try:
            odds_val = float(data.get("odds_ratio", 1.0))
        except (ValueError, TypeError):
            odds_val = 1.0

        meta = PRODUCT_METADATA.get(
            top_prod, {"name": top_prod, "icon": "📦", "category": "General"}
        )
        act_info = BANKING_ACTIVITIES.get(last_action, {"title": last_action})

        prob_dict_str = data.get("probabilities")
        if prob_dict_str:
            probs_data = json.loads(prob_dict_str)
        else:
            probs_data = {p: (1.0 / (idx + 1)) for idx, p in enumerate(recos_list[:7])}
            probs_data["do_nothing"] = 0.85

        p_hold = probs_data.get("do_nothing", 0.0)
        p_act = 1.0 - p_hold

        # Regime badge & status formatting
        if regime == "stage1_cpu_hold":
            card_class = "reco-hero-card-suppressed"
            status_text = "🛑 STAGE 1 CPU GATEKEEPER HOLD (p_buy < 0.20)"
            status_badge = "<span class='regime-badge-cpu'>Stage 1 CPU Hold</span>"
            decision_kpi = "CPU Hold 🛑"
            decision_sub = "p_buy < 0.20 (Bypassed)"
            gpu_kpi = "⚡ Bypassed (<1.1 ms)"
            gpu_sub = "42.3% GPU Compute Saved"
        elif regime == "stage2_gate1c_hold":
            card_class = "reco-hero-card-suppressed"
            status_text = "⏸️ ORC GATE HOLD (Odds Ratio < θ)"
            status_badge = "<span class='regime-badge-gate'>ORC Gate Hold</span>"
            decision_kpi = "ORC Gate Hold ⏸️"
            decision_sub = "Odds Ratio < θ"
            gpu_kpi = "🖥️ Invoked RTX 3090"
            gpu_sub = "73,968 Rows Context"
        elif regime == "candidate_a_silent_cache":
            card_class = "reco-hero-card-suppressed"
            status_text = "💾 CACHE PREDICTOR: SILENT REDIS UPDATE (p_buy ≥ 0.20)"
            status_badge = "<span class='regime-badge-cache'>Cache Predictor: Silent Update</span>"
            decision_kpi = "Cache Predictor 💾"
            decision_sub = "recs:{cust_id} Stored"
            gpu_kpi = "🖥️ Invoked RTX 3090"
            gpu_sub = "54.3% Sniper Precision"
        elif regime == "candidate_c_live_push":
            card_class = "reco-hero-card-active"
            status_text = "🔥 LIVE PREDICTOR: ACTIVE PUSH ALERT (p_buy ≥ 0.30 + Matrix Match)"
            status_badge = "<span class='regime-badge-live'>Live Predictor: Push Alert</span>"
            decision_kpi = "Live Alert Push 🔥"
            decision_sub = "MAP@7: 0.8198"
            gpu_kpi = "🖥️ Invoked RTX 3090"
            gpu_sub = "High Relevance Match"
        else:
            card_class = "reco-hero-card-active" if is_active else "reco-hero-card-suppressed"
            status_text = "🔥 ACTIVE NEXT-BEST-ACTION MATCH" if is_active else "⏸️ DECISION: NO ACTION"
            status_badge = "<span class='status-badge-ok'>Active Trigger</span>" if is_active else "<span class='status-badge-warn'>No Action</span>"
            decision_kpi = "Active Pitch 🔥" if is_active else "No Action ⏸️"
            decision_sub = regime
            gpu_kpi = "⚡ Bypassed" if gpu_skipped else "🖥️ Invoked RTX 3090"
            gpu_sub = "In-Context Model"

        # Hero Card
        is_already_claimed = top_prod in st.session_state.get(
            "claimed_products", {}
        ).get(active_cust_id, set())
        claimed_badge = (
            ' &nbsp;|&nbsp; <span style="background: #DCFCE7; color: #065F46; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem;">✅ Claimed in Portfolio</span>'
            if is_already_claimed
            else ""
        )

        hero_html = (
            f'<div class="{card_class}">'
            f'<div style="display: flex; align-items: center; justify-content: space-between;">'
            f'<span style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">{status_text}</span>'
            f"{status_badge}"
            f"</div>"
            f'<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px;">'
            f"<div>"
            f'<h2 style="margin: 0; color: #1E293B; font-size: 2rem;">{meta["icon"]} {meta["name"]}</h2>'
            f'<span style="color: #64748B; font-size: 0.95rem;">Category: <strong>{meta["category"]}</strong> &nbsp;|&nbsp; Customer ID: <strong>{active_cust_id}</strong> &nbsp;|&nbsp; Trigger Activity: <strong>{act_info.get("title", last_action)}</strong>{claimed_badge}</span>'
            f"</div>"
            f'<div style="text-align: right;">'
            f'<span class="status-badge-ok">Serving: {latency} ms</span><br/>'
            f'<small style="color: #64748B;">GPU Engine: {"⚡ Bypassed (<1.1 ms)" if gpu_skipped else "🖥️ Invoked (KV-Cache)"}</small>'
            f"</div>"
            f"</div>"
            f"</div>"
        )
        st.markdown(hero_html, unsafe_allow_html=True)

        # Propensity KPI Gauges
        ai_kpi1, ai_kpi2, ai_kpi3, ai_kpi4 = st.columns(4)
        with ai_kpi1:
            st.metric(
                label="Cascaded Decision",
                value=decision_kpi,
                delta=decision_sub,
            )
        with ai_kpi2:
            st.metric(
                label="Stage 1 Purchase Propensity",
                value=f"{p_buy_val * 100:.1f}%",
                delta="≥ 0.20 (Stage 2 Pass)" if p_buy_val >= 0.20 else "< 0.20 (Bypassed)",
                delta_color="normal" if p_buy_val >= 0.20 else "inverse",
            )
        with ai_kpi3:
            st.metric(
                label="Stage 2 Calibrated Odds Ratio",
                value=f"{odds_val:.2f}x",
                delta="Signal Strong (≥ 1.8x)" if odds_val >= 1.8 else "Suppressed (< 1.8x)",
                delta_color="normal" if odds_val >= 1.8 else "inverse",
            )
        with ai_kpi4:
            st.metric(
                label="GPU Pipeline Status",
                value=gpu_kpi,
                delta=gpu_sub,
            )

        # Real-Time Decision & Routing Status Box (Single Unified Box)
        if regime == "candidate_c_live_push":
            border_box = "#059669"
            bg_box = "#F0FDF4"
            badge_box = "<span class='regime-badge-live'>LIVE PREDICTOR ALERT</span>"
            desc_text = (
                f"<strong>Channel:</strong> Live Redis Stream (<code>stream:recommendations</code>) &amp; Customer In-App Dialog<br/>"
                f"<strong>Routing Logic:</strong> Propensity (p_buy = {p_buy_val:.3f} ≥ 0.30) cleared the Live Predictor threshold with action relevance match. "
                f"High-impact recommendation delivered in real time (91.0% Hold Safety, 0.8198 Alert MAP@7)."
            )
        elif regime == "candidate_a_silent_cache":
            border_box = "#0284C7"
            bg_box = "#F0F9FF"
            badge_box = "<span class='regime-badge-cache'>CACHE PREDICTOR (SILENT UPDATE)</span>"
            desc_text = (
                f"<strong>Channel:</strong> Redis Feature Store (<code>recs:{active_cust_id}</code>)<br/>"
                f"<strong>Routing Logic:</strong> Propensity (p_buy = {p_buy_val:.3f} ≥ 0.20) cleared the Cache Predictor threshold. "
                f"Silently updated customer recommendation profile for headless retrieval without pop-up friction (54.3% Sniper Precision)."
            )
        elif regime == "stage2_gate1c_hold":
            border_box = "#F59E0B"
            bg_box = "#FFFBEB"
            badge_box = "<span class='regime-badge-gate'>ORC GATE SUPPRESSION</span>"
            desc_text = (
                f"<strong>Channel:</strong> Suppressed (Zero Customer Disturbance)<br/>"
                f"<strong>Routing Logic:</strong> TabDPT odds ratio ({odds_val:.2f}) was below the Odds-Ratio Calibration noise barrier (θ = 6.0/1.8). "
                f"Suppressed candidate to Hold (Class 0) to eliminate noisy sink false positives."
            )
        else:
            border_box = "#E2E8F0"
            bg_box = "#F8FAFC"
            badge_box = "<span class='regime-badge-cpu'>STAGE 1 CPU HOLD</span>"
            desc_text = (
                f"<strong>Channel:</strong> GPU Bypassed on CPU (&lt; 0.002 ms)<br/>"
                f"<strong>Routing Logic:</strong> Non-buyer propensity (p_buy = {p_buy_val:.3f} &lt; 0.20) intercepted on CPU. "
                f"Completely bypassed GPU TabDPT forward pass, conserving compute while maintaining zero notification fatigue."
            )

        routing_html = f"""
        <div style="background-color: {bg_box}; border: 1.5px solid {border_box}; border-radius: 8px; padding: 12px 18px; margin: 12px 0 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #1E293B; font-size: 0.95rem;">🎯 Real-Time Decision &amp; Routing Status</span>
                {badge_box}
            </div>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #334155; line-height: 1.45;">
                {desc_text}
            </p>
        </div>
        """
        st.markdown(routing_html, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📋 Top Recommendations")
            st.caption(
                "Ranked by model propensity against 73,968 in-context rows. Held products are strictly filtered."
            )

            reco_rows = []
            for rank, prod_key in enumerate(recos_list[:7], 1):
                p_meta = PRODUCT_METADATA.get(
                    prod_key, {"name": prod_key, "icon": "📦", "category": "General"}
                )
                is_top = rank == 1
                if regime == "candidate_c_live_push" and is_top:
                    status_col = "🔥 Live Alert Pushed"
                elif regime == "candidate_a_silent_cache" and is_top:
                    status_col = "💾 Cached Top-1"
                elif regime == "stage1_cpu_hold":
                    status_col = "🛑 CPU Gatekeeper Hold"
                elif regime == "stage2_gate1c_hold":
                    status_col = "⏸️ ORC Gate Suppressed"
                else:
                    status_col = f"Rank #{rank} Candidate"

                reco_rows.append(
                    {
                        "Rank": f"#{rank}",
                        "Product": f"{p_meta['icon']} {p_meta['name']}",
                        "Category": p_meta["category"],
                        "Propensity": f"{probs_data.get(prod_key, 0.0) * 100:.2f}%",
                        "Decision": status_col,
                    }
                )
            st.table(pd.DataFrame(reco_rows))

        with col_right:
            st.subheader("📊 Class Propensity Distribution")

            # User toggle for Hold / Do Nothing
            show_hold = st.checkbox(
                "Include 'Hold / No Action' in Distribution Chart", value=False
            )

            chart_items = []
            for k, v in probs_data.items():
                if not show_hold and k == "do_nothing":
                    continue
                chart_items.append(
                    {
                        "Product": PRODUCT_METADATA.get(k, {}).get("name", k),
                        "Propensity (%)": float(v) * 100,
                    }
                )

            chart_df = (
                pd.DataFrame(chart_items)
                .sort_values(by="Propensity (%)", ascending=False)
                .head(10)
            )
            st.bar_chart(chart_df.set_index("Product"), color="#059669")

    else:
        await_html = (
            '<div style="text-align: center; padding: 48px 20px; background: white; border-radius: 12px; border: 2px dashed #A7F3D0;">'
            '<span style="font-size: 2.5rem;">⚡</span>'
            '<h3 style="color: #065F46; margin: 12px 0 6px 0; font-weight: 700;">Awaiting Customer Activity</h3>'
            '<p style="color: #64748B; max-width: 500px; margin: 0 auto; font-size: 0.95rem;">'
            "Go to the <strong>🏦 Customer Online Bank Portal</strong> tab and simulate an activity to inspect real-time TabDPT predictions."
            "</p>"
            "</div>"
        )
        st.markdown(await_html, unsafe_allow_html=True)


# =============================================================================
# TAB 3: Redis Stream & Telemetry Monitor (Live Observability)
# =============================================================================
with tab_stream:
    st.subheader("📡 Live Redis Streams Telemetry & Model Observability")
    st.caption(
        f"Real-time event feed, serving latency percentiles & model stability monitoring for `{STREAM_CUSTOMER_EVENTS}` and `{STREAM_RECOMMENDATIONS}`."
    )

    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2.5, 1.0, 1.0])
    with ctrl_c1:
        stream_window = st.select_slider(
            "Telemetry Sample Window (Recent Events):",
            options=[10, 25, 50, 100],
            value=25,
            key="stream_window_slider",
        )
    with ctrl_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Telemetry", use_container_width=True):
            st.rerun()
    with ctrl_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear Stream",
            use_container_width=True,
            help="Clear in-memory stream events for this session",
        ):
            if r_client:
                r_client.delete(STREAM_RECOMMENDATIONS)
                r_client.delete(STREAM_CUSTOMER_EVENTS)
            st.toast("Stream buffer cleared.", icon="🗑️")
            st.rerun()

    # Read latest messages from stream:recommendations
    recs_events = (
        r_client.xrevrange(STREAM_RECOMMENDATIONS, count=stream_window)
        if r_client
        else []
    )
    if not recs_events:
        recs_events = []

    total_stream_len = r_client.xlen(STREAM_RECOMMENDATIONS) if r_client else 0

    if recs_events:
        # Extract telemetry vectors
        latencies = []
        triton_latencies = []
        active_count = 0
        action_counts = {}
        stream_table = []

        for idx, (msg_id, payload) in enumerate(recs_events, 1):
            is_active = payload.get("is_active_trigger") in ("true", True, "True")
            if is_active:
                active_count += 1

            top_p = payload.get("top_product", "N/A")
            top_name = PRODUCT_METADATA.get(top_p, {}).get("name", top_p)
            act_raw = payload.get("action", "N/A")
            act_title = BANKING_ACTIVITIES.get(act_raw, {}).get("title", act_raw)
            action_counts[act_title] = action_counts.get(act_title, 0) + 1

            regime_raw = payload.get("regime", "candidate_c_live_push" if is_active else "stage2_gate1c_hold")
            gpu_skipped = payload.get("gpu_skipped") in ("true", True, "True")

            if regime_raw == "stage1_cpu_hold":
                regime_label = "🛑 CPU Gatekeeper Hold"
            elif regime_raw == "stage2_gate1c_hold":
                regime_label = "⏸️ ORC Gate Hold"
            elif regime_raw == "candidate_a_silent_cache":
                regime_label = "💾 Cache Predictor (Silent)"
            elif regime_raw == "candidate_c_live_push":
                regime_label = "🔥 Live Predictor Alert"
            else:
                regime_label = regime_raw

            p_buy_val = None
            try:
                p_buy_val = float(payload.get("p_buy", 0.0))
            except (ValueError, TypeError):
                pass

            odds_val = None
            try:
                odds_val = float(payload.get("odds_ratio", 0.0))
            except (ValueError, TypeError):
                pass

            lat_val = None
            try:
                lat_val = float(payload.get("latency_ms", 0.0))
                latencies.append(lat_val)
            except (ValueError, TypeError):
                pass

            triton_val = None
            try:
                triton_val = float(payload.get("triton_ms", 0.0))
                triton_latencies.append(triton_val)
            except (ValueError, TypeError):
                pass

            stream_table.append(
                {
                    "Message ID": msg_id,
                    "Customer ID": payload.get("customer_id", "N/A"),
                    "Banking Activity": act_title,
                    "Top Recommendation": top_name,
                    "Execution Regime": regime_label,
                    "p(Buy)": f"{p_buy_val:.3f}" if p_buy_val is not None else "N/A",
                    "Odds Ratio": f"{odds_val:.2f}x" if odds_val is not None else "N/A",
                    "GPU Status": "⚡ Bypassed (<1.1ms)" if gpu_skipped else "🖥️ Invoked",
                    "Serving Latency (ms)": f"{lat_val:.2f}"
                    if lat_val is not None
                    else "N/A",
                    "Timestamp": payload.get("timestamp", "N/A"),
                }
            )

        n_events = len(recs_events)
        p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        active_pct = (active_count / n_events) * 100.0 if n_events > 0 else 0.0
        silent_pct = 100.0 - active_pct

        # Top 4 KPI Metrics Strip
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.metric(
                label="Stream Ingestion",
                value=f"{total_stream_len} events",
                delta=f"Window: {n_events} loaded",
            )
        with kpi_c2:
            st.metric(
                label="Serving Latency (p50 / p95)",
                value=f"{p50_lat:.1f} ms",
                delta=f"p95: {p95_lat:.1f} ms",
            )
        with kpi_c3:
            st.metric(
                label="NBA Offer Match Rate",
                value=f"{active_pct:.1f}% Active",
                delta=f"{silent_pct:.1f}% Suppressed / Silent",
            )
        with kpi_c4:
            st.metric(
                label="Holdout Alert MAP@7",
                value="0.8198 🟢",
                delta="+28.4% vs Baseline (0.6385)",
            )

        # Telemetry & PSI Stability Banner
        st.markdown(
            """
            <div style="background-color: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 8px; padding: 12px 18px; margin: 10px 0 16px 0; color: #065F46; font-size: 0.93rem; line-height: 1.5;">
                <strong>🛡️ Real-Time Stream Telemetry &amp; Catalog Stability (PSI):</strong>
                GBDTs (LightGBM/CatBoost) over-concentrate on popular products (Gini 0.68–0.74, PSI 0.14–0.29), while
                <strong>TabDPT stays stable (PSI = 0.0315, Gini = 0.2840)</strong> across all 15 banking products with no popular-item starvation.
                With durable Redis Streams (<code>stream:customer_events</code>) and sub-0.1 ms feature lookups, the pipeline serves personalized Next-Best-Action recommendations at &lt;20 ms p95.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2-Column Telemetry Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("⚡ Real-Time Serving Latencies (ms)")
            # Build chronological series
            rev_items = list(reversed(recs_events))
            timeline_items = []
            for i, (m_id, p_load) in enumerate(rev_items, 1):
                try:
                    g_val = float(p_load.get("latency_ms", 0.0))
                    t_val = float(p_load.get("triton_ms", 0.0))
                    timeline_items.append(
                        {
                            "Event": f"#{i}",
                            "Gateway Latency (ms)": g_val,
                            "Triton Model Latency (ms)": t_val,
                        }
                    )
                except (ValueError, TypeError):
                    continue
            if timeline_items:
                chart_df = pd.DataFrame(timeline_items).set_index("Event")
                st.line_chart(chart_df, color=["#059669", "#3B82F6"])

        with chart_col2:
            st.subheader("📊 Ingested Activity Mix")
            if action_counts:
                mix_df = (
                    pd.DataFrame(
                        [
                            {"Banking Activity": k, "Count": v}
                            for k, v in action_counts.items()
                        ]
                    )
                    .sort_values(by="Count", ascending=False)
                    .set_index("Banking Activity")
                )
                st.bar_chart(mix_df, color="#0F766E")

        st.subheader("📜 Live Event Audit Trail")
        st.dataframe(pd.DataFrame(stream_table), use_container_width=True)

    else:
        # Graceful Standby & Baseline Card when stream is empty
        standby_html = (
            '<div style="text-align: center; padding: 40px 24px; background: white; border-radius: 12px; border: 2px dashed #A7F3D0; margin-top: 14px;">'
            '<span style="font-size: 2.5rem;">📡</span>'
            '<h3 style="color: #065F46; margin: 12px 0 6px 0; font-weight: 700;">Live Telemetry & Stream Ingestion Standby</h3>'
            '<p style="color: #64748B; max-width: 620px; margin: 0 auto 20px auto; font-size: 0.95rem;">'
            'Redis Streams engine is online and listening on <code>stream:customer_events</code> and <code>stream:recommendations</code>.<br/>'
            'Go to the <strong>🏦 Customer Online Bank Portal</strong> tab and trigger any banking action (e.g. Deposit Monthly Paycheck, High-Value Capital Wire, Utility Payment) to stream real-time telemetry.'
            '</p>'
            '<div style="display: flex; justify-content: center; gap: 32px; margin-top: 16px; font-size: 0.92rem; color: #334155; flex-wrap: wrap;">'
            '<div><strong>In-Context Size:</strong> <span style="color: #059669; font-weight: 700;">73,968 Uniform Rows</span></div>'
            '<div><strong>GPU Compute Saved:</strong> <span style="color: #059669; font-weight: 700;">42.3%–55.6% Bypassed</span></div>'
            '<div><strong>Conditional Alert MAP@7:</strong> <span style="color: #059669; font-weight: 700;">0.8198 🟢 (Top-1 Sniper)</span></div>'
            '<div><strong>Non-Buyer Hold Rate:</strong> <span style="color: #059669; font-weight: 700;">91.05% Safe</span></div>'
            '<div><strong>Benchmark Latency:</strong> <span style="color: #059669; font-weight: 700;">&lt;1.1 ms CPU / 17.2 ms GPU</span></div>'
            '</div>'
            '</div>'
        )
        st.markdown(standby_html, unsafe_allow_html=True)


# =============================================================================
# TAB 4: Architecture Blueprint
# =============================================================================
with tab_arch:
    st.subheader("🏛️ Architecture Blueprint: Real-Time Event Recommender")
    st.caption("High-level overview of the event-driven serving pipeline from bank action trigger to Next-Best-Action recommendation.")

    st.markdown("""
    ```text
    ┌──────────────────────────────┐        XADD         ┌─────────────────────────────┐
    │  Customer Bank Portal (UI)   │ ──────────────────► │   Redis Stream Hub          │
    │  - Paycheck / Deposit / Bill │                     │   'stream:customer_events'  │
    └──────────────────────────────┘                     └──────────────┬──────────────┘
                                                                        │
                                                                   XREADGROUP
                                                                        ▼
                                                         ┌─────────────────────────────┐
                                                         │   Streaming Worker          │
                                                         │   (event_processor.py)      │
                                                         │   - Point Lookup (<0.1 ms)  │
                                                         └──────────────┬──────────────┘
                                                                        │
                                                            p_buy < 0.20 (Hold / Bypass)
                                                                        │
    ┌──────────────────────────────┐     ModelInfer      ┌──────────────┴──────────────┐
    │   Triton gRPC Server         │ ◄────────────────── │   Stage 1 CPU Gatekeeper    │
    │   (TabDPT + KV-Cache)        │ ──────────────────► │   - LightGBM (ROC-AUC 0.8912│
    │   - Port 8001 | Latency <20ms│    [Batch, 16]      │   - Intercepts 42-56% on CPU│
    └──────────────────────────────┘                     └──────────────┬──────────────┘
                                                                        │
                                                       Filter Held & Odds Calibration
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

    st.markdown(r"""
    ### ⚙️ Production Engineering Highlights
    1. **100% Native Polars Feature Pipeline**:
       - Cleans multi-year transactional banking data (~2.4 GB) and engineers 80 features (lag-1/2 holdings, velocity deltas, composite bundles, seasonality) with zero JVM dependency.
    2. **Online Redis Feature Store & Streams**:
       - Compact 512-byte FP32 vectors with sub-0.1ms point lookups (0.080 ms) and durable pub/sub event logs via Redis Streams (`stream:customer_events`).
    3. **Stage 1 CPU LightGBM Gatekeeper (`ROC-AUC = 0.8912`)**:
       - Intercepts non-buyer traffic in `< 0.002 ms` on CPU, completely bypassing GPU TabDPT inference on 42.3% to 55.6% of streaming queries.
    4. **Stage 2 GPU TabDPT Serving + Precomputed KV-Cache**:
       - 32-layer column-attention tabular transformer served via Triton gRPC against the 73,968 scaled uniform context. KV-caching provides a **26.2× speedup** (<20 ms).
    5. **Odds-Ratio Calibration (ORC) Gate**:
       - Per-class odds ratio calibration ($\theta_{\text{clean}} = 1.8, \theta_{\text{noisy}} = 6.0$) suppresses noisy product sinks (`current_account`, `particular_plus`, `funds`, `taxes`), preserving top-tier precision.
    6. **Dual-Predictor Next-Best-Action Routing**:
       - Automatically bifurcates into the **Cache Predictor** (silently updating `recs:{customer_id}` with 54.3% sniper precision) and the **Live Predictor** (delivering real-time in-app alerts with 91.0% hold safety and 0.8198 Alert MAP@7).
    """)

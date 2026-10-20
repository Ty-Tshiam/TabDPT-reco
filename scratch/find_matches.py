import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from src.features.feature_store import RedisFeatureStore
from src.streaming.event_processor import load_lgbm_gatekeeper, TabDPTEngine, process_customer_event
from src.config import INDEX_TO_TARGET

fs = RedisFeatureStore()
lgbm_model = load_lgbm_gatekeeper()
tabdpt_engine = TabDPTEngine()

# Find matches for particular_plus_account (Class 7)
plus_df = (
    pl.scan_parquet('data/processed/test_targets/*.parquet')
    .filter(pl.col('target_class') == 7)
    .select(['customer_id', 'added_products'])
    .limit(40)
    .collect()
)

print(f"Checking {len(plus_df)} customers with target_class == 7 (particular_plus_account):")
for row in plus_df.iter_rows(named=True):
    cid = str(row['customer_id'])
    try:
        x_qry, held, meta = fs.get_customer_features(cid, enable_fallback=True)
        res = process_customer_event(
            features_80d=x_qry,
            action='query',
            lgbm_model=lgbm_model,
            tabdpt_model=tabdpt_engine,
            held_products=held,
        )
        pred = res['predicted_product']
        top7 = res['top_7_products']
        if pred == 'particular_plus_account':
            print(f"  >>> EXACT MATCH: Customer {cid} -> pred={pred}, p_buy={res['p_buy']:.3f}, odds={res['odds_ratio']:.2f}, held={held}")
            break
        else:
            rank = top7.index('particular_plus_account') + 1 if 'particular_plus_account' in top7 else -1
            print(f"  Customer {cid}: pred={pred}, rank_of_target={rank}, p_buy={res['p_buy']:.3f}")
    except Exception as e:
        print(f"  Customer {cid} error: {e}")
        continue

# Find a genuine early-career / retail non-buyer (target_class == 0, p_buy < 0.20)
non_buyer_df = (
    pl.scan_parquet('data/processed/test_targets/*.parquet')
    .filter(pl.col('target_class') == 0)
    .select(['customer_id', 'added_products'])
    .limit(40)
    .collect()
)

print(f"\nChecking customers with target_class == 0 (do_nothing):")
for row in non_buyer_df.iter_rows(named=True):
    cid = str(row['customer_id'])
    try:
        x_qry, held, meta = fs.get_customer_features(cid, enable_fallback=True)
        res = process_customer_event(
            features_80d=x_qry,
            action='query',
            lgbm_model=lgbm_model,
            tabdpt_model=tabdpt_engine,
            held_products=held,
        )
        if res['prediction'] == 0:
            age = meta.get('age', 'N/A')
            segment = meta.get('customer_segment', 'N/A')
            print(f"  >>> TRUE NON-BUYER MATCH: Customer {cid} (Age: {age}, Segment: {segment}) -> pred={res['predicted_product']}, p_buy={res['p_buy']:.3f}, held={held}")
            break
    except Exception as e:
        continue

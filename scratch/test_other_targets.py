import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from src.features.feature_store import RedisFeatureStore
from src.streaming.event_processor import load_lgbm_gatekeeper, TabDPTEngine, process_customer_event

fs = RedisFeatureStore()
lgbm_model = load_lgbm_gatekeeper()
tabdpt_engine = TabDPTEngine()

targets_to_test = [
    (7, 'particular_plus_account'),
    (6, 'securities'),
    (9, 'funds'),
    (15, 'taxes'),
    (5, 'mortgage')
]

for target_class, prod_name in targets_to_test:
    df = (
        pl.scan_parquet('data/processed/test_targets/*.parquet')
        .filter(pl.col('target_class') == target_class)
        .select(['customer_id', 'added_products'])
        .limit(100)
        .collect()
    )
    found = False
    for row in df.iter_rows(named=True):
        cid = str(row['customer_id'])
        x_qry, held, meta = fs.get_customer_features(cid, enable_fallback=True)
        res = process_customer_event(x_qry, 'query', lgbm_model, tabdpt_engine, held)
        if res['predicted_product'] == prod_name:
            print(f"MATCH for {prod_name} (Class {target_class}): Customer {cid}, p_buy={res['p_buy']:.3f}, odds={res['odds_ratio']:.2f}, held={held}")
            found = True
            break
    if not found:
        print(f"No exact Top-1 match found in sample for {prod_name}")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from src.features.feature_store import RedisFeatureStore
from src.streaming.event_processor import load_lgbm_gatekeeper, TabDPTEngine, process_customer_event
from src.config import INDEX_TO_TARGET, TARGET_TO_INDEX

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

curated_cids = list(CURATED_CUSTOMERS.keys())

targets_df = (
    pl.scan_parquet('data/processed/test_targets/*.parquet')
    .filter(pl.col('customer_id').cast(pl.Utf8).is_in(curated_cids))
    .collect()
)
targets_dict = {
    str(r['customer_id']): {
        'target_class': r['target_class'],
        'added_products': r['added_products']
    }
    for r in targets_df.iter_rows(named=True)
}

fs = RedisFeatureStore()
lgbm_model = load_lgbm_gatekeeper()
tabdpt_engine = TabDPTEngine()

results = []
for cid in curated_cids:
    x_qry, held, meta = fs.get_customer_features(cid, enable_fallback=True)
    real_info = targets_dict.get(cid, {'target_class': -1, 'added_products': []})
    real_class = real_info['target_class']
    real_prod = INDEX_TO_TARGET.get(real_class, 'unknown')
    added_prods = real_info['added_products']

    res = process_customer_event(
        features_80d=x_qry,
        action='query',
        lgbm_model=lgbm_model,
        tabdpt_model=tabdpt_engine,
        held_products=held,
    )

    pred_prod = res['predicted_product']
    pred_class = res['prediction']
    top7 = res['top_7_products']
    p_buy = res['p_buy']
    odds = res['odds_ratio']
    regime = res['regime']

    # Evaluation
    is_top1_match = False
    if real_class == 0 and pred_class == 0:
        is_top1_match = True
    elif pred_prod in added_prods or pred_prod == real_prod:
        is_top1_match = True

    is_top7_match = False
    target_rank = -1
    if is_top1_match:
        is_top7_match = True
        target_rank = 1 if real_class != 0 else 0
    elif any(p in top7 for p in added_prods) or real_prod in top7:
        is_top7_match = True
        for p in added_prods + [real_prod]:
            if p in top7:
                target_rank = top7.index(p) + 1
                break

    results.append({
        'cid': cid,
        'label': CURATED_CUSTOMERS[cid],
        'held': held,
        'real_class': real_class,
        'real_prod': real_prod,
        'added_prods': added_prods,
        'pred_class': pred_class,
        'pred_prod': pred_prod,
        'top7': top7,
        'p_buy': p_buy,
        'odds': odds,
        'regime': regime,
        'is_top1_match': is_top1_match,
        'is_top7_match': is_top7_match,
        'target_rank': target_rank,
    })

print(f"{'CID':<10} | {'Real Value (Added)':<32} | {'Model Top-1 Prediction':<22} | {'Top-1 Match?':<12} | {'Rank in Top-7'}")
print("-" * 95)
for r in results:
    real_str = f"{r['real_prod']} ({r['added_prods']})" if r['added_prods'] else "do_nothing []"
    top1_str = "YES" if r['is_top1_match'] else "NO"
    rank_str = f"Rank #{r['target_rank']}" if r['target_rank'] > 0 else ("Hold/No Action" if r['target_rank'] == 0 else "Not in Top 7")
    print(f"{r['cid']:<10} | {real_str:<32} | {r['pred_prod']:<22} | {top1_str:<12} | {rank_str}")

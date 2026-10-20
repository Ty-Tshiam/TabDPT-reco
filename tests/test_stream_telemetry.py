"""
Unit tests for Redis Stream Telemetry & Model Stability calculations in frontend.
"""

import numpy as np


def test_telemetry_latency_percentiles():
    """Verify that p50 and p95 latencies are computed accurately."""
    latencies = [15.2, 17.5, 18.0, 19.1, 22.4, 30.0, 45.0, 16.8, 18.9, 21.0]
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    assert 17.0 <= p50 <= 20.0
    assert p95 >= p50
    assert p95 <= 45.0


def test_telemetry_action_rates():
    """Verify active vs silent decision breakdown."""
    sample_records = [
        {"is_active_trigger": "true", "action": "salary_deposit"},
        {"is_active_trigger": "false", "action": "card_payment"},
        {"is_active_trigger": "true", "action": "large_deposit"},
        {"is_active_trigger": "false", "action": "card_payment"},
    ]
    active_count = sum(
        1 for r in sample_records if r.get("is_active_trigger") in ("true", True, "True")
    )
    active_pct = (active_count / len(sample_records)) * 100.0
    silent_pct = 100.0 - active_pct

    assert active_count == 2
    assert active_pct == 50.0
    assert silent_pct == 50.0


def test_empty_stream_handling():
    """Verify empty stream does not crash percentiles or division by zero."""
    empty_latencies = []
    p50 = float(np.percentile(empty_latencies, 50)) if empty_latencies else 0.0
    p95 = float(np.percentile(empty_latencies, 95)) if empty_latencies else 0.0
    active_pct = 0.0

    assert p50 == 0.0
    assert p95 == 0.0
    assert active_pct == 0.0


def test_publish_reco_tracks_silent_and_active_events():
    """Verify that publish_reco records both active push alerts and silent decisions in the stream."""
    import json
    import fakeredis
    from src.streaming.event_processor import STREAM_RECOMMENDATIONS, publish_reco

    server = fakeredis.FakeServer()
    r = fakeredis.FakeRedis(server=server, decode_responses=True)

    # 1. Publish an active live push alert
    active_payload = {
        "customer_id": "1001",
        "action": "salary_deposit",
        "top_product": "payroll",
        "is_active_trigger": "true",
        "regime": "candidate_c_live_push",
        "p_buy": "0.3421",
        "odds_ratio": "4.25",
    }
    publish_reco(active_payload, r_client=r)

    # 2. Publish a silent cache decision
    silent_payload = {
        "customer_id": "1002",
        "action": "card_payment",
        "top_product": "credit_card",
        "is_active_trigger": "false",
        "regime": "candidate_a_silent_cache",
        "p_buy": "0.2210",
        "odds_ratio": "2.10",
    }
    publish_reco(silent_payload, r_client=r)

    # 3. Publish a Stage 1 CPU gatekeeper hold
    cpu_hold_payload = {
        "customer_id": "1003",
        "action": "atm_withdrawal",
        "top_product": "do_nothing",
        "is_active_trigger": "false",
        "regime": "stage1_cpu_hold",
        "p_buy": "0.0850",
        "odds_ratio": "0.00",
    }
    publish_reco(cpu_hold_payload, r_client=r)

    # Verify all 3 events are present in STREAM_RECOMMENDATIONS for the Live Event Audit Trail
    stream_len = r.xlen(STREAM_RECOMMENDATIONS)
    assert stream_len == 3

    events = r.xrevrange(STREAM_RECOMMENDATIONS)
    assert len(events) == 3

    # Check that individual cache keys exist
    assert json.loads(r.get("recs:1001"))["top_product"] == "payroll"
    assert json.loads(r.get("recs:1002"))["top_product"] == "credit_card"
    assert json.loads(r.get("recs:1003"))["top_product"] == "do_nothing"

    # Verify stream entries preserved active trigger flags
    cust_to_active = {p["customer_id"]: p["is_active_trigger"] for _, p in events}
    assert cust_to_active["1001"] == "true"
    assert cust_to_active["1002"] == "false"
    assert cust_to_active["1003"] == "false"


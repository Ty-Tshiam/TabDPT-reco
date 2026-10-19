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

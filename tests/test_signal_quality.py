"""Tests for advanced signal-quality diagnostics."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.signal_quality import SignalQualityEngine, SignalInputs


def test_signal_quality_identifies_positive_actionable_edge():
    engine = SignalQualityEngine()
    result = engine.evaluate(
        SignalInputs(
            market_price=8.0,
            forecast_fair_value=9.2,
            bid=7.9,
            ask=8.1,
            mc_ci=(7.8, 8.2),
            confidence=0.8,
        )
    )
    assert result["expected_return_pct"] > 0
    assert result["quality_tier"] in {"A", "B", "C"}
    assert result["prob_edge_real"] > 0


def test_signal_quality_penalizes_high_cost_low_edge_setups():
    engine = SignalQualityEngine()
    result = engine.evaluate(
        SignalInputs(
            market_price=8.0,
            forecast_fair_value=8.05,
            bid=7.2,
            ask=8.8,
            mc_ci=(6.0, 10.0),
            confidence=0.4,
        )
    )
    assert result["expected_return_pct"] <= 0
    assert result["quality_tier"] == "D"
    assert result["actionable"] is False

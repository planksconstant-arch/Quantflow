"""Tests for research-grade walk-forward validator."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from analysis.research_validation import ResearchValidator, ValidationConfig


def _sample_df(n: int = 80, positive_edge: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    market = 5.0 + rng.normal(0, 0.1, n)
    edge = 0.08 if positive_edge else 0.0
    fair = market * (1.0 + edge)
    bid = market - 0.05
    ask = market + 0.05
    future = market * (1.03 if positive_edge else 0.99)
    return pd.DataFrame(
        {
            "date": dates,
            "market_price": market,
            "fair_value": fair,
            "bid": bid,
            "ask": ask,
            "future_option_price": future,
        }
    )


def test_walk_forward_backtest_generates_significant_report_for_positive_edge():
    validator = ResearchValidator(ValidationConfig(signal_threshold_pct=2.0, bootstrap_samples=300))
    report = validator.walk_forward_backtest(_sample_df(positive_edge=True))

    assert report["n_trades"] > 0
    assert report["avg_return"] > 0
    assert report["economic_significance"] in {"moderate", "high"}


def test_walk_forward_backtest_returns_empty_when_no_signals():
    validator = ResearchValidator(ValidationConfig(signal_threshold_pct=15.0, bootstrap_samples=200))
    report = validator.walk_forward_backtest(_sample_df(positive_edge=False))

    assert report["n_trades"] == 0
    assert report["reason"] == "no_signals"

"""Tests for pricing signal guardrails in QuantFlow main workflow."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import QuantFlow


def test_market_price_validation_requires_positive_and_valid_flag():
    assert QuantFlow._is_market_price_valid(1.25, True) is True
    assert QuantFlow._is_market_price_valid(0.0, True) is False
    assert QuantFlow._is_market_price_valid(1.25, False) is False


def test_uncertainty_adjusted_score_penalizes_noisy_signal():
    low_noise_score = QuantFlow._uncertainty_adjusted_score(
        divergence_pct=20.0,
        divergence_dollars=3.0,
        bid=9.9,
        ask=10.1,
        mc_ci=(9.8, 10.2),
    )
    high_noise_score = QuantFlow._uncertainty_adjusted_score(
        divergence_pct=20.0,
        divergence_dollars=0.5,
        bid=8.0,
        ask=12.0,
        mc_ci=(6.0, 14.0),
    )

    assert low_noise_score > high_noise_score
    assert 0 <= high_noise_score <= 100

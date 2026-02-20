"""Unit tests for core pricing and Greeks interfaces."""

import math
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.greeks.calculator import calculate_greeks
from models.pricing.black_scholes import BlackScholesModel, black_scholes


class TestBlackScholesPricing:
    def test_call_option_reference_value(self):
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        assert price == pytest.approx(10.4506, rel=2e-3)

    def test_put_option_reference_value(self):
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        assert price == pytest.approx(5.5735, rel=2e-3)

    def test_put_call_parity(self):
        model = BlackScholesModel(102.5, 100, 0.75, 0.03, 0.22, q=0.01)
        parity_gap = model.call_price() - model.put_price() - (
            model.S * math.exp(-model.q * model.T) - model.K * math.exp(-model.r * model.T)
        )
        assert abs(parity_gap) < 1e-10

    def test_no_arbitrage_bounds(self):
        model = BlackScholesModel(100, 100, 1.0, 0.02, 0.3)
        lower, upper = model.no_arbitrage_bounds("call")
        call_price = model.call_price()
        assert lower <= call_price <= upper

    def test_invalid_market_price_rejected_in_iv_solver(self):
        with pytest.raises(ValueError, match="no-arbitrage"):
            BlackScholesModel.implied_volatility(
                market_price=150,
                S=100,
                K=100,
                T=1.0,
                r=0.02,
                option_type="call",
            )


    def test_terminal_payoff_when_time_is_zero(self):
        assert black_scholes(110, 100, 0.0, 0.05, 0.2, "call") == 10.0
        assert black_scholes(110, 100, 0.0, 0.05, 0.2, "put") == 0.0


class TestGreeks:
    def test_first_order_greeks_are_well_behaved(self):
        greeks = calculate_greeks(100, 100, 1.0, 0.05, 0.2, "call")
        assert 0 < greeks["delta"] < 1
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0

    def test_higher_order_greeks_exist_and_are_finite(self):
        greeks = calculate_greeks(95, 100, 0.8, 0.02, 0.3, "put")
        for key in ("vanna", "vomma", "charm"):
            assert key in greeks
            assert math.isfinite(greeks[key])

    def test_implied_volatility_round_trip(self):
        target_sigma = 0.27
        model = BlackScholesModel(110, 100, 0.6, 0.04, target_sigma, q=0.01)
        market_price = model.call_price()

        recovered_sigma = BlackScholesModel.implied_volatility(
            market_price=market_price,
            S=110,
            K=100,
            T=0.6,
            r=0.04,
            option_type="call",
            q=0.01,
        )

        assert recovered_sigma == pytest.approx(target_sigma, rel=1e-3)


    def test_price_and_greeks_snapshot_contains_expected_keys(self):
        model = BlackScholesModel(100, 100, 1.0, 0.05, 0.2)
        snapshot = model.price_and_greeks("call")
        for key in ("price", "delta", "gamma", "vega", "vomma"):
            assert key in snapshot

    def test_risk_neutral_density_is_positive_and_mass_is_discounted_one(self):
        model = BlackScholesModel(100, 100, 1.0, 0.03, 0.2)
        strikes = [k for k in range(30, 301)]
        densities = [model.risk_neutral_density(float(k)) for k in strikes]

        assert min(densities) > 0

        approx_mass = sum(densities)  # ΔK = 1
        assert approx_mass == pytest.approx(math.exp(-model.r * model.T), rel=1e-2)


class TestInputValidation:
    def test_negative_strike(self):
        with pytest.raises(ValueError):
            black_scholes(100, -50, 1.0, 0.05, 0.2, "call")

    def test_negative_time(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, -1.0, 0.05, 0.2, "call")

    def test_invalid_option_type(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, 1.0, 0.05, 0.2, "invalid")

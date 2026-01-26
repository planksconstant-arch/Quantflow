"""
Unit Tests for QuantFlow Pricing Models
Tests Black-Scholes, Greeks, and basic validation
"""
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pricing.black_scholes import BlackScholesModel
from models.greeks.greeks_calculator import GreeksCalculator


class TestBlackScholes:
    """Test suite for Black-Scholes pricing"""
    
    def test_call_option_basic(self):
        """Test basic call option pricing"""
        model = BlackScholesModel(100, 100, 1.0, 0.05, 0.2)
        price = model.call_price()
        assert 8 < price < 12, f"Expected ~10, got {price}"
    
    def test_put_option_basic(self):
        """Test basic put option pricing"""
        model = BlackScholesModel(100, 100, 1.0, 0.05, 0.2)
        price = model.put_price()
        assert 4 < price < 8, f"Expected ~6, got {price}"
    
    def test_deep_itm_call(self):
        """Deep in-the-money call should be near intrinsic value"""
        model = BlackScholesModel(150, 100, 0.1, 0.05, 0.2)
        price = model.call_price()
        assert price > 49, "Deep ITM call should be > $49"
    
    def test_deep_otm_call(self):
        """Deep out-of-the-money call should be near zero"""
        model = BlackScholesModel(50, 100, 0.1, 0.05, 0.2)
        price = model.call_price()
        assert price < 1, "Deep OTM call should be < $1"
    
    def test_zero_volatility(self):
        """Zero volatility should give intrinsic value"""
        model = BlackScholesModel(110, 100, 0.0001, 0.05, 0.0001) # Use small values instead of 0 to avoid division by zero if not handled
        price = model.call_price()
        assert abs(price - 10) < 0.1, "Zero vol should give intrinsic"


class TestGreeks:
    """Test suite for Greeks calculations"""
    
    def test_delta_range(self):
        """Delta should be between 0 and 1 for calls"""
        calc = GreeksCalculator(100, 100, 1.0, 0.05, 0.2, 'call')
        greeks = calc.get_analytical_greeks()
        assert 0 <= greeks['delta'] <= 1, f"Delta {greeks['delta']} out of range"
    
    def test_gamma_positive(self):
        """Gamma should always be positive"""
        calc = GreeksCalculator(100, 100, 1.0, 0.05, 0.2, 'call')
        greeks = calc.get_analytical_greeks()
        assert greeks['gamma'] > 0, "Gamma should be positive"
    
    def test_vega_positive(self):
        """Vega should always be positive"""
        calc = GreeksCalculator(100, 100, 1.0, 0.05, 0.2, 'call')
        greeks = calc.get_analytical_greeks()
        assert greeks['vega'] > 0, "Vega should be positive"
    
    def test_atm_delta(self):
        """At-the-money call delta should be near 0.5"""
        # Use r=0 so forward price equals spot price
        calc = GreeksCalculator(100, 100, 1.0, 0.0, 0.2, 'call')
        greeks = calc.get_analytical_greeks()
        assert 0.4 < greeks['delta'] < 0.6, f"ATM delta {greeks['delta']} not near 0.5"


class TestInputValidation:
    """Test edge cases and validation"""
    
    def test_negative_strike(self):
        """Negative strike should raise error"""
        with pytest.raises(ValueError):
            BlackScholesModel(100, -50, 1.0, 0.05, 0.2)
    
    def test_invalid_option_type(self):
        """Invalid option type should raise error"""
        model = BlackScholesModel(100, 100, 1.0, 0.05, 0.2)
        with pytest.raises(ValueError):
            model.price('invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

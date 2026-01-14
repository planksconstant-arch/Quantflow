"""
Unit Tests for QuantFlow Pricing Models
Tests Black-Scholes, Greeks, and basic validation
"""
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pricing.black_scholes import black_scholes
from models.greeks.calculator import calculate_greeks


class TestBlackScholes:
    """Test suite for Black-Scholes pricing"""
    
    def test_call_option_basic(self):
        """Test basic call option pricing"""
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
        assert 8 < price < 12, f"Expected ~10, got {price}"
    
    def test_put_option_basic(self):
        """Test basic put option pricing"""
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'put')
        assert 4 < price < 8, f"Expected ~6, got {price}"
    
    def test_deep_itm_call(self):
        """Deep in-the-money call should be near intrinsic value"""
        price = black_scholes(150, 100, 0.1, 0.05, 0.2, 'call')
        assert price > 49, "Deep ITM call should be > $49"
    
    def test_deep_otm_call(self):
        """Deep out-of-the-money call should be near zero"""
        price = black_scholes(50, 100, 0.1, 0.05, 0.2, 'call')
        assert price < 1, "Deep OTM call should be < $1"
    
    def test_zero_volatility(self):
        """Zero volatility should give intrinsic value"""
        price = black_scholes(110, 100, 0.0, 0.05, 0.0, 'call')
        assert abs(price - 10) < 0.1, "Zero vol should give intrinsic"


class TestGreeks:
    """Test suite for Greeks calculations"""
    
    def test_delta_range(self):
        """Delta should be between 0 and 1 for calls"""
        greeks = calculate_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
        assert 0 <= greeks['delta'] <= 1, f"Delta {greeks['delta']} out of range"
    
    def test_gamma_positive(self):
        """Gamma should always be positive"""
        greeks = calculate_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
        assert greeks['gamma'] > 0, "Gamma should be positive"
    
    def test_vega_positive(self):
        """Vega should always be positive"""
        greeks = calculate_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
        assert greeks['vega'] > 0, "Vega should be positive"
    
    def test_atm_delta(self):
        """At-the-money call delta should be near 0.5"""
        greeks = calculate_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
        assert 0.4 < greeks['delta'] < 0.6, f"ATM delta {greeks['delta']} not near 0.5"


class TestInputValidation:
    """Test edge cases and validation"""
    
    def test_negative_strike(self):
        """Negative strike should raise error"""
        with pytest.raises(ValueError):
            black_scholes(100, -50, 1.0, 0.05, 0.2, 'call')
    
    def test_negative_time(self):
        """Negative time should raise error"""
        with pytest.raises(ValueError):
            black_scholes(100, 100, -1.0, 0.05, 0.2, 'call')
    
    def test_invalid_option_type(self):
        """Invalid option type should raise error"""
        with pytest.raises(ValueError):
            black_scholes(100, 100, 1.0, 0.05, 0.2, 'invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Greeks package"""
from .greeks_calculator import GreeksCalculator


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str, q: float = 0.0):
    """Functional helper returning analytical Greeks for convenience."""
    calculator = GreeksCalculator(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
    return calculator.get_analytical_greeks()


__all__ = ['GreeksCalculator', 'calculate_greeks']

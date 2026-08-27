"""Backward-compatible Greeks API."""
from models.greeks.greeks_calculator import GreeksCalculator


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call', q: float = 0.0):
    calc = GreeksCalculator(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
    return calc.get_analytical_greeks()


__all__ = ["calculate_greeks", "GreeksCalculator"]

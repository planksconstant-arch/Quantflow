"""
Helper utility functions for QuantFlow
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional
import joblib
import os


def time_to_maturity(expiry: Union[str, datetime], current_date: Optional[datetime] = None) -> float:
    """
    Calculate time to maturity in years (annualized)
    
    Parameters:
    -----------
    expiry : str or datetime
        Expiration date
    current_date : datetime, optional
        Current date (defaults to now)
    
    Returns:
    --------
    float : Time to maturity in years
    """
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry, "%Y-%m-%d")
    
    if current_date is None:
        current_date = datetime.now()
    
    days_to_expiry = (expiry - current_date).days
    return max(days_to_expiry / 365.0, 0.0001)  # Avoid division by zero


def annualized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate annualized rolling volatility
    
    Parameters:
    -----------
    returns : pd.Series
        Log returns
    window : int
        Rolling window size
    
    Returns:
    --------
    pd.Series : Annualized volatility
    """
    return returns.rolling(window).std() * np.sqrt(252)


def log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series"""
    return np.log(prices / prices.shift(1))


def cache_to_disk(func):
    """
    Decorator to cache function results to disk
    
    Usage:
        @cache_to_disk
        def expensive_function(arg1, arg2):
            ...
    """
    def wrapper(*args, **kwargs):
        # Create cache key from function name and arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        cache_file = os.path.join("data", "cache", f"{cache_key}.pkl")
        
        # Check if cached result exists
        if os.path.exists(cache_file):
            # Check if cache is still valid (less than 24 hours old)
            cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
            if cache_age_hours < 24:
                print(f"Loading cached result for {func.__name__}")
                return joblib.load(cache_file)
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        joblib.dump(result, cache_file)
        return result
    
    return wrapper


def normal_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution"""
    return (1.0 + np.erf(x / np.sqrt(2.0))) / 2.0


def normal_pdf(x: float) -> float:
    """Probability density function for standard normal distribution"""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def format_currency(value: float) -> str:
    """Format number as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage"""
    return f"{value * 100:.2f}%"


def validate_positive(value: float, name: str = "Value") -> float:
    """Validate that a value is positive"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def calculate_returns_from_prices(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Calculate simple returns from prices"""
    if isinstance(prices, pd.Series):
        prices = prices.values
    return np.diff(prices) / prices[:-1]


def black_scholes_d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
    """Calculate d1 parameter for Black-Scholes formula"""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def black_scholes_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
    """Calculate d2 parameter for Black-Scholes formula"""
    return black_scholes_d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

"""
Black-Scholes Option Pricing Model
Implements classical BS pricing with analytical Greeks
"""

import numpy as np
from typing import Dict, Tuple
from scipy.stats import norm

from utils.helpers import validate_positive


class BlackScholesModel:
    """
    Black-Scholes option pricing and Greeks calculation
    
    Supports European-style options with dividends
    """
    
    MIN_TIME = 1e-10

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        Initialize BS model
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate (annualized)
        sigma : float
            Volatility (annualized)
        q : float, optional
            Dividend yield (annualized), default 0
        """
        self.S = validate_positive(S, "Spot price")
        self.K = validate_positive(K, "Strike price")
        self.T = max(float(T), self.MIN_TIME)  # Avoid division by zero
        self.r = r
        self.sigma = validate_positive(sigma, "Volatility")
        self.q = q
        
        self.sqrt_T = np.sqrt(self.T)
        self.discount_r = np.exp(-self.r * self.T)
        self.discount_q = np.exp(-self.q * self.T)

        # Calculate d1 and d2
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()
    
    def _calculate_d1(self) -> float:
        """Calculate d1 parameter"""
        numerator = np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        denominator = self.sigma * self.sqrt_T
        return numerator / denominator
    
    def _calculate_d2(self) -> float:
        """Calculate d2 parameter"""
        return self.d1 - self.sigma * self.sqrt_T

    def _ensure_option_type(self, option_type: str) -> str:
        option_type_normalized = option_type.lower()
        if option_type_normalized not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        return option_type_normalized

    def no_arbitrage_bounds(self, option_type: str) -> Tuple[float, float]:
        """Return theoretical lower/upper bounds for European options."""
        option_type = self._ensure_option_type(option_type)
        spot_pv = self.S * self.discount_q
        strike_pv = self.K * self.discount_r

        if option_type == "call":
            lower = max(spot_pv - strike_pv, 0.0)
            upper = spot_pv
        else:
            lower = max(strike_pv - spot_pv, 0.0)
            upper = strike_pv

        return lower, upper
    
    def call_price(self) -> float:
        """
        Calculate European call option price
        
        Returns:
        --------
        float : Call option value
        """
        S_pv = self.S * self.discount_q
        K_pv = self.K * self.discount_r
        
        price = S_pv * norm.cdf(self.d1) - K_pv * norm.cdf(self.d2)
        return max(price, 0.0)
    
    def put_price(self) -> float:
        """
        Calculate European put option price
        
        Returns:
        --------
        float : Put option value
        """
        S_pv = self.S * self.discount_q
        K_pv = self.K * self.discount_r
        
        price = K_pv * norm.cdf(-self.d2) - S_pv * norm.cdf(-self.d1)
        return max(price, 0.0)
    
    def price(self, option_type: str) -> float:
        """Generic price call for either option type"""
        option_type = self._ensure_option_type(option_type)
        if option_type == 'call':
            return self.call_price()
        return self.put_price()
    
    # ========== GREEKS ==========
    
    def delta(self, option_type: str) -> float:
        """
        Calculate Delta: ∂V/∂S
        
        Interpretation: Change in option price per $1 change in stock price
        Also represents hedge ratio (shares to hedge one option)
        
        Returns:
        --------
        float : Delta value
        """
        option_type = self._ensure_option_type(option_type)
        if option_type == 'call':
            return self.discount_q * norm.cdf(self.d1)
        else:  # put
            return -self.discount_q * norm.cdf(-self.d1)
    
    def gamma(self) -> float:
        """
        Calculate Gamma: ∂²V/∂S²
        
        Interpretation: Rate of change of Delta
        High gamma = Delta changes rapidly = need frequent rehedging
        
        Returns:
        --------
        float : Gamma value (same for calls and puts)
        """
        numerator = norm.pdf(self.d1) * self.discount_q
        denominator = self.S * self.sigma * self.sqrt_T
        return numerator / denominator
    
    def theta(self, option_type: str) -> float:
        """
        Calculate Theta: ∂V/∂t
        
        Interpretation: Time decay (negative for long positions)
        Theta is typically quoted per day, so divide by 365
        
        Returns:
        --------
        float : Theta value (annualized)
        """
        option_type = self._ensure_option_type(option_type)
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma * self.discount_q) / (2 * self.sqrt_T)
        
        if option_type == 'call':
            term2 = self.r * self.K * self.discount_r * norm.cdf(self.d2)
            term3 = -self.q * self.S * self.discount_q * norm.cdf(self.d1)
            theta = term1 - term2 + term3
        else:  # put
            term2 = self.r * self.K * self.discount_r * norm.cdf(-self.d2)
            term3 = self.q * self.S * self.discount_q * norm.cdf(-self.d1)
            theta = term1 + term2 - term3
        
        return theta
    
    def theta_per_day(self, option_type: str) -> float:
        """Theta expressed as daily decay"""
        return self.theta(option_type) / 365.0
    
    def vega(self) -> float:
        """
        Calculate Vega: ∂V/∂σ
        
        Interpretation: Sensitivity to volatility
        Vega is typically quoted per 1% change in vol
        
        Returns:
        --------
        float : Vega value (per 100% vol change)
        """
        vega = self.S * norm.pdf(self.d1) * self.sqrt_T * self.discount_q
        return vega
    
    def vega_percent(self) -> float:
        """Vega per 1% change in volatility"""
        return self.vega() / 100.0
    
    def rho(self, option_type: str) -> float:
        """
        Calculate Rho: ∂V/∂r
        
        Interpretation: Sensitivity to interest rate changes
        Rho is typically quoted per 1% change in rate
        
        Returns:
        --------
        float : Rho value (per 100% rate change)
        """
        option_type = self._ensure_option_type(option_type)
        if option_type == 'call':
            rho = self.K * self.T * self.discount_r * norm.cdf(self.d2)
        else:  # put
            rho = -self.K * self.T * self.discount_r * norm.cdf(-self.d2)
        
        return rho
    
    def rho_percent(self, option_type: str) -> float:
        """Rho per 1% change in interest rate"""
        return self.rho(option_type) / 100.0
    
    def vanna(self) -> float:
        """Cross-sensitivity d²V/(dS dσ)."""
        return -self.discount_q * norm.pdf(self.d1) * self.d2 / self.sigma

    def vomma(self) -> float:
        """Second-order volatility sensitivity d²V/dσ²."""
        return self.vega() * self.d1 * self.d2 / self.sigma

    def charm(self, option_type: str) -> float:
        """Delta decay with respect to time dΔ/dt (calendar time)."""
        option_type = self._ensure_option_type(option_type)
        common = (
            self.discount_q * norm.pdf(self.d1)
            * (2 * (self.r - self.q) * self.T - self.d2 * self.sigma * self.sqrt_T)
            / (2 * self.T * self.sigma * self.sqrt_T)
        )
        if option_type == "call":
            return self.q * self.discount_q * norm.cdf(self.d1) - common
        return -self.q * self.discount_q * norm.cdf(-self.d1) - common


    def risk_neutral_density(self, strike: float) -> float:
        """
        Breeden-Litzenberger risk-neutral density at maturity for a given strike.

        For Black-Scholes this is simply the lognormal density under the
        risk-neutral measure, discounted by exp(-rT).
        """
        k = validate_positive(strike, "strike")
        d2_k = (np.log(self.S / k) + (self.r - self.q - 0.5 * self.sigma**2) * self.T) / (self.sigma * self.sqrt_T)
        return self.discount_r * norm.pdf(d2_k) / (k * self.sigma * self.sqrt_T)

    def price_and_greeks(self, option_type: str) -> Dict[str, float]:
        """Return a compact pricing + Greeks snapshot used in risk reports."""
        option_type = self._ensure_option_type(option_type)
        snapshot = {'price': self.price(option_type)}
        snapshot.update(self.all_greeks(option_type))
        return snapshot

    def all_greeks(self, option_type: str) -> Dict[str, float]:
        """
        Calculate all Greeks at once
        
        Returns:
        --------
        dict : All Greek values
        """
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'theta_per_day': self.theta_per_day(option_type),
            'vega': self.vega(),
            'vega_percent': self.vega_percent(),
            'rho': self.rho(option_type),
            'rho_percent': self.rho_percent(option_type),
            'vanna': self.vanna(),
            'vomma': self.vomma(),
            'charm': self.charm(option_type),
        }
    
    # ========== IMPLIED VOLATILITY ==========
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str, q: float = 0.0,
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Parameters:
        -----------
        market_price : float
            Observed market price of option
        S, K, T, r, q : float
            BS model parameters
        option_type : str
            'call' or 'put'
        max_iterations : int
            Maximum iterations for convergence
        tolerance : float
            Convergence tolerance
        
        Returns:
        --------
        float : Implied volatility
        """
        probe_model = BlackScholesModel(S, K, T, r, sigma=max(0.2, 1e-3), q=q)
        lower_bound, upper_bound = probe_model.no_arbitrage_bounds(option_type)
        if not (lower_bound <= market_price <= upper_bound):
            raise ValueError(
                f"Market price {market_price:.6f} violates no-arbitrage bounds "
                f"[{lower_bound:.6f}, {upper_bound:.6f}]"
            )

        if T <= 0:
            raise ValueError("Time to maturity must be positive for implied volatility")

        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
        sigma = max(sigma, 0.01)  # Ensure positive
        
        for i in range(max_iterations):
            bs = BlackScholesModel(S, K, T, r, sigma, q)
            
            # Calculate price and vega
            price = bs.price(option_type)
            vega = bs.vega()
            
            # Check convergence
            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma
            
            # Newton-Raphson update
            if vega > 1e-10:  # Avoid division by zero
                sigma = sigma + diff / vega
                sigma = max(sigma, 0.001)  # Keep positive
            else:
                break
        
        # If didn't converge, try bisection as fallback
        return BlackScholesModel._implied_vol_bisection(
            market_price, S, K, T, r, option_type, q
        )
    
    @staticmethod
    def _implied_vol_bisection(market_price: float, S: float, K: float, T: float, 
                               r: float, option_type: str, q: float = 0.0) -> float:
        """Bisection method fallback for IV calculation"""
        sigma_low = 0.001
        sigma_high = 5.0
        
        for _ in range(100):
            sigma_mid = (sigma_low + sigma_high) / 2
            bs = BlackScholesModel(S, K, T, r, sigma_mid, q)
            price_mid = bs.price(option_type)
            
            if abs(price_mid - market_price) < 1e-6:
                return sigma_mid
            
            if price_mid < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
        
        return sigma_mid
    
    def __repr__(self) -> str:
        return (f"BlackScholesModel(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, σ={self.sigma:.4f}, q={self.q:.4f})")


if __name__ == "__main__":
    # Test Black-Scholes model
    bs = BlackScholesModel(S=100, K=105, T=0.25, r=0.05, sigma=0.30)
    
    print("Black-Scholes Model Test:")
    print(f"Call Price: ${bs.call_price():.2f}")
    print(f"Put Price: ${bs.put_price():.2f}")
    print("\nCall Greeks:")
    for greek, value in bs.all_greeks('call').items():
        print(f"  {greek}: {value:.4f}")


def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str, q: float = 0.0) -> float:
    """Functional API wrapper kept for backwards compatibility."""
    if T < 0:
        raise ValueError(f"Time to maturity must be non-negative, got {T}")
    normalized_type = option_type.lower()
    if normalized_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    if T == 0:
        if normalized_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    model = BlackScholesModel(S=S, K=K, T=T, r=r, sigma=max(sigma, 1e-12), q=q)
    return model.price(normalized_type)

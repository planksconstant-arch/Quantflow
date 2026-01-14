"""
Greeks Calculator
Aggregates Greeks from different models
"""

import numpy as np
import pandas as pd
from typing import Dict

from models.pricing import BlackScholesModel, BinomialTreeModel, MonteCarloSimulation
from utils.helpers import time_to_maturity


class GreeksCalculator:
    """
    Calculate option Greeks using multiple methods
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str, q: float = 0.0):
        """
        Initialize Greeks calculator
        
        Parameters:
        -----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        q : float
            Dividend yield
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.q = q
        
        # Initialize BS model for analytical Greeks
        self.bs_model = BlackScholesModel(S, K, T, r, sigma, q)
    
    def get_analytical_greeks(self) -> Dict[str, float]:
        """
        Get analytical Greeks from Black-Scholes
        
        Returns:
        --------
        dict : All Greeks
        """
        return self.bs_model.all_greeks(self.option_type)
    
    def get_numerical_greeks_mc(self, epsilon: float = 0.01, n_sims: int = 50000) -> Dict[str, float]:
        """
        Get numerical Greeks from Monte Carlo
        
        Parameters:
        -----------
        epsilon : float
            Bump size for finite difference
        n_sims : int
            Number of MC simulations
        
        Returns:
        --------
        dict : Numerical Greeks
        """
        mc = MonteCarloSimulation(self.S, self.K, self.T, self.r, self.sigma, n_sims, self.q)
        return mc.greeks_numerical(self.option_type, epsilon)
    
    def get_all_greeks_summary(self) -> pd.DataFrame:
        """
        Get comprehensive Greeks summary from all methods
        
        Returns:
        --------
        pd.DataFrame : Greeks comparison
        """
        analytical = self.get_analytical_greeks()
        
        greeks_data = {
            'Greek': ['Delta', 'Gamma', 'Theta (annual)', 'Theta (per day)', 'Vega', 'Vega (%)', 'Rho', 'Rho (%)'],
            'Value': [
                analytical['delta'],
                analytical['gamma'],
                analytical['theta'],
                analytical['theta_per_day'],
                analytical['vega'],
                analytical['vega_percent'],
                analytical['rho'],
                analytical['rho_percent'],
            ],
            'Interpretation': [
                f"Position moves ${analytical['delta']:.3f} per $1 stock move",
                f"Delta changes by {analytical['gamma']:.4f} per $1 stock move",
                f"Option loses ${abs(analytical['theta']):.2f} per year due to time decay",
                f"Daily time decay: ${abs(analytical['theta_per_day']):.4f}",
                f"Option gains ${analytical['vega']:.2f} per 100% vol increase",
                f"Option gains ${analytical['vega_percent']:.4f} per 1% vol increase",
                f"Option gains ${analytical['rho']:.2f} per 100% rate increase",
                f"Option gains ${analytical['rho_percent']:.4f} per 1% rate increase",
            ]
        }
        
        return pd.DataFrame(greeks_data)
    
    def greeks_vs_spot(self, spot_range: tuple = (0.7, 1.3), n_points: int = 100) -> pd.DataFrame:
        """
        Calculate Greeks across a range of spot prices
        
        Parameters:
        -----------
        spot_range : tuple
            (min_ratio, max_ratio) relative to current spot
        n_points : int
            Number of points to calculate
        
        Returns:
        --------
        pd.DataFrame : Greeks at different spot prices
        """
        spot_prices = np.linspace(
            self.S * spot_range[0],
            self.S * spot_range[1],
            n_points
        )
        
        results = []
        
        for S_test in spot_prices:
            bs = BlackScholesModel(S_test, self.K, self.T, self.r, self.sigma, self.q)
            greeks = bs.all_greeks(self.option_type)
            price = bs.price(self.option_type)
            
            results.append({
                'spot_price': S_test,
                'moneyness': S_test / self.K,
                'option_price': price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta_per_day': greeks['theta_per_day'],
                'vega_percent': greeks['vega_percent'],
            })
        
        return pd.DataFrame(results)
    
    def greeks_vs_time(self, n_points: int = 50) -> pd.DataFrame:
        """
        Calculate Greeks as time passes (time decay)
        
        Parameters:
        -----------
        n_points : int
            Number of time points
        
        Returns:
        --------
        pd.DataFrame : Greeks over time
        """
        time_points = np.linspace(self.T, 0.001, n_points)  # From now to near expiry
        days_to_expiry = time_points * 365
        
        results = []
        
        for T_test, days in zip(time_points, days_to_expiry):
            bs = BlackScholesModel(self.S, self.K, T_test, self.r, self.sigma, self.q)
            greeks = bs.all_greeks(self.option_type)
            price = bs.price(self.option_type)
            
            results.append({
                'time_to_maturity': T_test,
                'days_to_expiry': days,
                'option_price': price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta_per_day': greeks['theta_per_day'],
                'vega_percent': greeks['vega_percent'],
            })
        
        return pd.DataFrame(results)
    
    def option_surface(self, spot_range: tuple = (0.7, 1.3), 
                      spot_points: int = 50, time_points: int = 30) -> Dict:
        """
        Generate 3D option surface: Value(S, T)
        
        Parameters:
        -----------
        spot_range : tuple
            Spot price range (relative to current)
        spot_points : int
            Number of spot price points
        time_points : int
            Number of time points
        
        Returns:
        --------
        dict : Surface data (S, T, V)
        """
        spot_prices = np.linspace(
            self.S * spot_range[0],
            self.S * spot_range[1],
            spot_points
        )
        
        time_values = np.linspace(self.T, 0.001, time_points)
        
        # Create meshgrid
        S_grid, T_grid = np.meshgrid(spot_prices, time_values)
        V_grid = np.zeros_like(S_grid)
        
        for i in range(time_points):
            for j in range(spot_points):
                bs = BlackScholesModel(
                    S_grid[i, j], self.K, T_grid[i, j], self.r, self.sigma, self.q
                )
                V_grid[i, j] = bs.price(self.option_type)
        
        return {
            'spot_prices': S_grid,
            'time_to_maturity': T_grid,
            'option_values': V_grid,
            'days_to_expiry': T_grid * 365
        }
    
    def delta_neutral_hedge(self) -> Dict[str, float]:
        """
        Calculate Delta-neutral hedge position
        
        Returns:
        --------
        dict : Hedge details
        """
        greeks = self.get_analytical_greeks()
        delta = greeks['delta']
        
        # For 1 long option, need to short delta shares to be delta-neutral
        hedge_shares = -delta
        hedge_notional = hedge_shares * self.S
        
        # Cost of hedging (transaction costs assuming 0.1% each way)
        transaction_cost_pct = 0.001
        hedge_cost = abs(hedge_notional) * transaction_cost_pct
        
        # Rehedging frequency (based on Gamma)
        gamma = greeks['gamma']
        # Higher gamma = more frequent rehedging needed
        # Rough heuristic: rehedge when spot moves > 1% if gamma is low, < 1% if gamma is high
        if gamma > 0.05:
            rehedge_threshold_pct = 0.5
        elif gamma > 0.02:
            rehedge_threshold_pct = 1.0
        else:
            rehedge_threshold_pct = 2.0
        
        return {
            'delta': delta,
            'hedge_position': hedge_shares,
            'hedge_notional': hedge_notional,
            'transaction_cost': hedge_cost,
            'gamma': gamma,
            'rehedge_threshold_pct': rehedge_threshold_pct,
            'rehedge_threshold_price': self.S * (rehedge_threshold_pct / 100),
            'recommendation': f"To hedge 1 {'long' if delta > 0 else 'short'} {self.option_type}, "
                            f"{'short' if delta > 0 else 'long'} {abs(hedge_shares):.3f} shares at ${self.S:.2f}. "
                            f"Rehedge when stock moves ${self.S * rehedge_threshold_pct / 100:.2f} "
                            f"({rehedge_threshold_pct}% threshold due to gamma={gamma:.4f})."
        }
    
    def __repr__(self) -> str:
        return f"GreeksCalculator({self.option_type.upper()}, S=${self.S:.2f}, K=${self.K:.2f}, T={self.T:.4f})"


if __name__ == "__main__":
    # Test Greeks calculator
    calc = GreeksCalculator(S=100,  K=105, T=0.25, r=0.05, sigma=0.30, option_type='call')
    
    print("Greeks Summary:")
    print(calc.get_all_greeks_summary().to_string(index=False))
    
    print("\nDelta-Neutral Hedge:")
    hedge = calc.delta_neutral_hedge()
    print(hedge['recommendation'])

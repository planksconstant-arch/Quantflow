"""
Scenario Analysis Engine
Monte Carlo stress testing with shock scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats

from models.pricing import MonteCarloSimulation
from utils.helpers import format_currency, format_percentage


class ScenarioAnalyzer:
    """
    Advanced scenario analysis and stress testing
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str, q: float = 0.0, position_size: int = 1):
        """
        Initialize scenario analyzer
        
        Parameters:
        -----------
        S, K, T, r, sigma : float
            Standard option parameters
        option_type : str
            'call' or 'put'
        q : float
            Dividend yield
        position_size : int
            Number of contracts (1 contract = 100 shares)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.q = q
        self.position_size = position_size
        
    def shock_scenario(self, stock_shock_pct: float, vol_shock_pct: float,
                       time_passed_days: int = 0) -> Dict:
        """
        Analyze single shock scenario
        
        Parameters:
        -----------
        stock_shock_pct : float
            Stock price change (e.g., 0.10 for +10%)
        vol_shock_pct : float
            Volatility change (e.g., 0.20 for +20%)
        time_passed_days : int
            Days that have passed
        
        Returns:
        --------
        dict : Scenario results
        """
        # New parameters
        S_new = self.S * (1 + stock_shock_pct)
        sigma_new = self.sigma * (1 + vol_shock_pct)
        T_new = max(self.T - time_passed_days / 365, 0.001)
        
        # Price with new parameters
        from models.pricing import BlackScholesModel
        bs = BlackScholesModel(S_new, self.K, T_new, self.r, sigma_new, self.q)
        new_price = bs.price(self.option_type)
        
        # Original price
        bs_orig = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, self.q)
        orig_price = bs_orig.price(self.option_type)
        
        # P&L calculation (per contract, 100 shares)
        pnl_per_contract = (new_price - orig_price) * 100
        total_pnl = pnl_per_contract * self.position_size
        
        return {
            'stock_shock_pct': stock_shock_pct,
            'vol_shock_pct': vol_shock_pct,
            'S_new': S_new,
            'sigma_new': sigma_new,
            'original_price': orig_price,
            'new_price': new_price,
            'pnl_per_contract': pnl_per_contract,
            'total_pnl': total_pnl,
            'return_pct': (total_pnl / (orig_price * 100 * self.position_size)) * 100
        }
    
    def run_standard_scenarios(self, time_horizon_days: int = 30) -> pd.DataFrame:
        """
        Run standard shock scenarios
        
        Parameters:
        -----------
        time_horizon_days : int
            Time horizon for scenarios
        
        Returns:
        --------
        pd.DataFrame : Scenario results
        """
        print(f"\n{'='*70}")
        print(f"📊 SCENARIO ANALYSIS ({time_horizon_days}-day horizon)")
        print(f"{'='*70}\n")
        
        scenarios = [
            {'name': 'Bull', 'stock': 0.10, 'vol': -0.20, 'description': 'Stock +10%, Vol -20%'},
            {'name': 'Mild Bull', 'stock': 0.05, 'vol': -0.10, 'description': 'Stock +5%, Vol -10%'},
            {'name': 'Base', 'stock': 0.00, 'vol': 0.00, 'description': 'No change'},
            {'name': 'Mild Bear', 'stock': -0.05, 'vol': 0.10, 'description': 'Stock -5%, Vol +10%'},
            {'name': 'Bear', 'stock': -0.10, 'vol': 0.30, 'description': 'Stock -10%, Vol +30%'},
            {'name': 'Crisis', 'stock': -0.20, 'vol': 0.50, 'description': 'Stock -20%, Vol +50%'},
        ]
        
        results = []
        
        for scenario in scenarios:
            result = self.shock_scenario(
                scenario['stock'],
                scenario['vol'],
                time_horizon_days
            )
            result['scenario_name'] = scenario['name']
            result['description'] = scenario['description']
            results.append(result)
            
            pnl_str = format_currency(result['total_pnl'])
            return_str = f"{result['return_pct']:+.1f}%"
            
            print(f"{scenario['name']:12s}: {pnl_str:>12s} ({return_str:>8s})  |  {scenario['description']}")
        
        return pd.DataFrame(results)
    
    def monte_carlo_distribution(self, n_simulations: int = 10000, 
                                 time_horizon_days: int = 30) -> Dict:
        """
        Generate P&L distribution via Monte Carlo
        
        Parameters:
        -----------
        n_simulations : int
            Number of Monte Carlo paths
        time_horizon_days : int
            Time horizon
        
        Returns:
        --------
        dict : Distribution results
        """
        print(f"\n📈 Monte Carlo P&L Distribution ({n_simulations:,} simulations)...")
        
        # Simulate stock paths
        dt = time_horizon_days / 365
        T_new = max(self.T - dt, 0.001)
        
        # Generate stock prices at horizon
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        shock = self.sigma * np.sqrt(dt) * np.random.standard_normal(n_simulations)
        S_future = self.S * np.exp(drift + shock)
        
        # Price options at each future stock price
        from models.pricing import BlackScholesModel
        bs_orig = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, self.q)
        orig_price = bs_orig.price(self.option_type)
        
        future_prices = np.zeros(n_simulations)
        for i, S_fut in enumerate(S_future):
            bs_fut = BlackScholesModel(S_fut, self.K, T_new, self.r, self.sigma, self.q)
            future_prices[i] = bs_fut.price(self.option_type)
        
        # P&L distribution
        pnl_per_contract = (future_prices - orig_price) * 100
        total_pnl = pnl_per_contract * self.position_size
        
        # Calculate metrics
        mean_pnl = np.mean(total_pnl)
        median_pnl = np.median(total_pnl)
        std_pnl = np.std(total_pnl)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(total_pnl, 5)  # 5th percentile loss
        var_99 = np.percentile(total_pnl, 1)  # 1st percentile loss
        
        # Conditional VaR (CVaR / Expected Shortfall)
        cvar_95 = np.mean(total_pnl[total_pnl <= var_95])
        cvar_99 = np.mean(total_pnl[total_pnl <= var_99])
        
        # Probability of profit
        prob_profit = np.sum(total_pnl > 0) / n_simulations
        
        # Percentiles
        percentiles = {
            'p1': np.percentile(total_pnl, 1),
            'p5': np.percentile(total_pnl, 5),
            'p25': np.percentile(total_pnl, 25),
            'p50': np.percentile(total_pnl, 50),
            'p75': np.percentile(total_pnl, 75),
            'p95': np.percentile(total_pnl, 95),
            'p99': np.percentile(total_pnl, 99),
        }
        
        print(f"\n✓ Distribution Statistics:")
        print(f"   Mean P&L: {format_currency(mean_pnl)}")
        print(f"   Median P&L: {format_currency(median_pnl)}")
        print(f"   Std Dev: {format_currency(std_pnl)}")
        print(f"\n📉 Risk Metrics:")
        print(f"   VaR (95%): {format_currency(var_95)}")
        print(f"   VaR (99%): {format_currency(var_99)}")
        print(f"   CVaR (95%): {format_currency(cvar_95)}")
        print(f"   CVaR (99%): {format_currency(cvar_99)}")
        print(f"\n📊 Probability of Profit: {format_percentage(prob_profit)}")
        
        return {
            'mean_pnl': mean_pnl,
            'median_pnl': median_pnl,
            'std_pnl': std_pnl,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'prob_profit': prob_profit,
            'percentiles': percentiles,
            'all_pnl': total_pnl  # Store for visualization
        }
    
    def expected_return_analysis(self, scenarios_df: pd.DataFrame, 
                                  probabilities: Dict = None) -> Dict:
        """
        Calculate expected return across scenarios
        
        Parameters:
        -----------
        scenarios_df : pd.DataFrame
            Scenario results
        probabilities : dict, optional
            Scenario probabilities (default: equal weight)
        
        Returns:
        --------
        dict : Expected value analysis
        """
        if probabilities is None:
            # Equal probabilities
            probabilities = {name: 1/len(scenarios_df) for name in scenarios_df['scenario_name']}
        
        # Expected P&L
        expected_pnl = sum(
            row['total_pnl'] * probabilities.get(row['scenario_name'], 0)
            for _, row in scenarios_df.iterrows()
        )
        
        # Expected return
        orig_price = scenarios_df.iloc[0]['original_price']
        expected_return = (expected_pnl / (orig_price * 100 * self.position_size)) * 100
        
        return {
            'expected_pnl': expected_pnl,
            'expected_return_pct': expected_return,
            'probabilities': probabilities
        }


if __name__ == "__main__":
    # Test scenario analyzer
    analyzer = ScenarioAnalyzer(
        S=145, K=140, T=93/365, r=0.045, sigma=0.35,
        option_type='call', position_size=1
    )
    
    # Standard scenarios
    scenarios = analyzer.run_standard_scenarios(time_horizon_days=30)
    
    # Monte Carlo
    mc_dist = analyzer.monte_carlo_distribution(n_simulations=10000, time_horizon_days=30)

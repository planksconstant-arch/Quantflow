"""
Monte Carlo Simulation for Option Pricing
Implements GBM with variance reduction techniques
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats

from utils.helpers import validate_positive


class MonteCarloSimulation:
    """
    Monte Carlo option pricing using Geometric Brownian Motion
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float,
                 n_simulations: int = 100000, q: float = 0.0, random_seed: int = 42):
        """
        Initialize Monte Carlo simulation
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        n_simulations : int
            Number of Monte Carlo paths
        q : float
            Dividend yield
        random_seed : int
            Random seed for reproducibility
        """
        self.S = validate_positive(S, "Spot price")
        self.K = validate_positive(K, "Strike price")
        self.T = max(T, 0.0001)
        self.r = r
        self.sigma = validate_positive(sigma, "Volatility")
        self.n_simulations = n_simulations
        self.q = q
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
    
    def simulate_paths(self, n_steps: int = 1, antithetic: bool = True) -> np.ndarray:
        """
        Simulate stock price paths using GBM
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps (1 for European options)
        antithetic : bool
            Use antithetic variates for variance reduction
        
        Returns:
        --------
        np.ndarray : Simulated terminal stock prices
        """
        dt = self.T / n_steps
        n_paths = self.n_simulations // 2 if antithetic else self.n_simulations
        
        # Generate random standard normals
        Z = np.random.standard_normal((n_paths, n_steps))
        
        # GBM formula: S_T = S_0 * exp((r - q - 0.5*σ²)*T + σ*sqrt(T)*Z)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        # Calculate cumulative returns
        log_returns = drift + diffusion * Z
        cumulative_returns = np.sum(log_returns, axis=1)
        
        # Terminal stock prices
        S_T = self.S * np.exp(cumulative_returns)
        
        if antithetic:
            # Antithetic paths (use -Z)
            log_returns_anti = drift - diffusion * Z
            cumulative_returns_anti = np.sum(log_returns_anti, axis=1)
            S_T_anti = self.S * np.exp(cumulative_returns_anti)
            
            # Combine original and antithetic paths
            S_T = np.concatenate([S_T, S_T_anti])
        
        return S_T
    
    def price(self, option_type: str, antithetic: bool = True) -> Dict[str, float]:
        """
        Price option using Monte Carlo
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        antithetic : bool
            Use antithetic variance reduction
        
        Returns:
        --------
        dict : Price, standard error, and confidence intervals
        """
        # Simulate terminal stock prices
        S_T = self.simulate_paths(n_steps=1, antithetic=antithetic)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:  # put
            payoffs = np.maximum(self.K - S_T, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        # 95% and 99% confidence intervals
        ci_95 = 1.96 * std_error
        ci_99 = 2.576 * std_error
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_95_lower': price - ci_95,
            'ci_95_upper': price + ci_95,
            'ci_99_lower': price - ci_99,
            'ci_99_upper': price + ci_99,
            'payoffs': payoffs  # Store for distribution analysis
        }
    
    def payoff_distribution(self, option_type: str) -> Dict:
        """
        Analyze payoff distribution
        
        Returns:
        --------
        dict : Distribution statistics
        """
        result = self.price(option_type)
        payoffs = result['payoffs']
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Filter non-zero payoffs (in-the-money cases)
        itm_payoffs = discounted_payoffs[discounted_payoffs > 0]
        
        return {
            'mean': np.mean(discounted_payoffs),
            'median': np.median(discounted_payoffs),
            'std': np.std(discounted_payoffs),
            'percentile_5': np.percentile(discounted_payoffs, 5),
            'percentile_25': np.percentile(discounted_payoffs, 25),
            'percentile_75': np.percentile(discounted_payoffs, 75),
            'percentile_95': np.percentile(discounted_payoffs, 95),
            'prob_itm': len(itm_payoffs) / len(discounted_payoffs),
            'expected_payoff_itm': np.mean(itm_payoffs) if len(itm_payoffs) > 0 else 0,
            'all_payoffs': discounted_payoffs
        }
    
    def greeks_numerical(self, option_type: str, epsilon: float = 0.01) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference (numerical approximation)
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        epsilon : float
            Bump size for finite difference
        
        Returns:
        --------
        dict : Numerical Greeks
        """
        # Base price
        base_price = self.price(option_type)['price']
        
        # Delta: ∂V/∂S
        mc_up = MonteCarloSimulation(
            self.S + epsilon, self.K, self.T, self.r, self.sigma, 
            self.n_simulations, self.q, self.random_seed
        )
        price_up = mc_up.price(option_type)['price']
        delta = (price_up - base_price) / epsilon
        
        # Gamma: ∂²V/∂S²
        mc_down = MonteCarloSimulation(
            self.S - epsilon, self.K, self.T, self.r, self.sigma,
            self.n_simulations, self.q, self.random_seed
        )
        price_down = mc_down.price(option_type)['price']
        gamma = (price_up - 2 * base_price + price_down) / (epsilon**2)
        
        # Vega: ∂V/∂σ
        mc_vega = MonteCarloSimulation(
            self.S, self.K, self.T, self.r, self.sigma + epsilon,
            self.n_simulations, self.q, self.random_seed
        )
        price_vega = mc_vega.price(option_type)['price']
        vega = (price_vega - base_price) / epsilon
        
        # Theta: ∂V/∂T (one day change)
        day_epsilon = 1 / 365
        if self.T > day_epsilon:
            mc_theta = MonteCarloSimulation(
                self.S, self.K, self.T - day_epsilon, self.r, self.sigma,
                self.n_simulations, self.q, self.random_seed
            )
            price_theta = mc_theta.price(option_type)['price']
            theta = (price_theta - base_price) / (-day_epsilon)
        else:
            theta = 0.0
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
    
    def convergence_test(self, option_type: str, 
                        simulation_sizes: list = None) -> list:
        """
        Test convergence as simulation count increases
        
        Returns:
        --------
        list : Convergence results for different simulation sizes
        """
        if simulation_sizes is None:
            simulation_sizes = [1000, 5000, 10000, 50000, 100000]
        
        results = []
        
        for n_sims in simulation_sizes:
            mc = MonteCarloSimulation(
                self.S, self.K, self.T, self.r, self.sigma, 
                n_sims, self.q, self.random_seed
            )
            result = mc.price(option_type)
            
            results.append({
                'n_simulations': n_sims,
                'price': result['price'],
                'std_error': result['std_error'],
                'ci_95_width': result['ci_95_upper'] - result['ci_95_lower']
            })
        
        return results
    
    def __repr__(self) -> str:
        return (f"MonteCarloSimulation(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, σ={self.sigma:.4f}, n={self.n_simulations})")


if __name__ == "__main__":
    # Test Monte Carlo simulation
    mc = MonteCarloSimulation(S=100, K=105, T=0.25, r=0.05, sigma=0.30, n_simulations=100000)
    
    print("Monte Carlo Simulation Test:")
    
    call_result = mc.price('call')
    print(f"\nCall Option:")
    print(f"  Price: ${call_result['price']:.2f} ± ${call_result['std_error']:.4f}")
    print(f"  95% CI: [${call_result['ci_95_lower']:.2f}, ${call_result['ci_95_upper']:.2f}]")
    print(f"  99% CI: [${call_result['ci_99_lower']:.2f}, ${call_result['ci_99_upper']:.2f}]")
    
    put_result = mc.price('put')
    print(f"\nPut Option:")
    print(f"  Price: ${put_result['price']:.2f} ± ${put_result['std_error']:.4f}")
    print(f"  95% CI: [${put_result['ci_95_lower']:.2f}, ${put_result['ci_95_upper']:.2f}]")
    
    print("\nConvergence Test:")
    convergence = mc.convergence_test('call')
    for r in convergence:
        print(f"  n={r['n_simulations']:6d}: Price=${r['price']:.4f}, "
              f"SE={r['std_error']:.4f}, CI_width={r['ci_95_width']:.4f}")

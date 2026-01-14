"""
Binomial Tree Option Pricing Model
Implements Cox-Ross-Rubinstein (CRR) binomial model
"""

import numpy as np
from typing import Dict

from utils.helpers import validate_positive


class BinomialTreeModel:
    """
    Binomial tree option pricing for American and European options
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 n_steps: int = 50, q: float = 0.0):
        """
        Initialize binomial tree model
        
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
        n_steps : int
            Number of time steps
        q : float, optional
            Dividend yield, default 0
        """
        self.S = validate_positive(S, "Spot price")
        self.K = validate_positive(K, "Strike price")
        self.T = max(T, 0.0001)
        self.r = r
        self.sigma = max(validate_positive(sigma, "Volatility"), 0.01)  # Floor at 1% to prevent p > 1 explosion
        self.n_steps = n_steps
        self.q = q
        
        # Calculate tree parameters
        self.dt = T / n_steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt))  # Up factor uses floored sigma
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral prob
        self.discount = np.exp(-r * self.dt)
        
    def build_stock_tree(self) -> np.ndarray:
        """
        Build stock price tree
        
        Returns:
        --------
        np.ndarray : Stock prices at each node
        """
        tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        
        return tree
    
    def price(self, option_type: str, exercise_type: str = "european") -> float:
        """
        Calculate option price using binomial tree
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        exercise_type : str
            'european' or 'american'
        
        Returns:
        --------
        float : Option price
        """
        # Build stock price tree
        stock_tree = self.build_stock_tree()
        
        # Initialize option value tree
        option_tree = np.zeros_like(stock_tree)
        
        # Terminal payoffs
        if option_type.lower() == 'call':
            option_tree[:, self.n_steps] = np.maximum(
                stock_tree[:, self.n_steps] - self.K, 0
            )
        else:  # put
            option_tree[:, self.n_steps] = np.maximum(
                self.K - stock_tree[:, self.n_steps], 0
            )
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # European value (discounted expected value)
                european_value = self.discount * (
                    self.p * option_tree[j, i + 1] +
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                
                if exercise_type.lower() == 'american':
                    # Intrinsic value (immediate exercise)
                    if option_type.lower() == 'call':
                        intrinsic = max(stock_tree[j, i] - self.K, 0)
                    else:  # put
                        intrinsic = max(self.K - stock_tree[j, i], 0)
                    
                    # American: max of hold vs. exercise
                    option_tree[j, i] = max(european_value, intrinsic)
                else:
                    option_tree[j, i] = european_value
        
        return option_tree[0, 0]
    
    def early_exercise_boundary(self, option_type: str = 'put', 
                                 threshold: float = 1e-6) -> Dict:
        """
        Find early exercise boundary for American options
        
        Most relevant for American puts
        
        Returns:
        --------
        dict : Exercise boundary data
        """
        stock_tree = self.build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        
        # Terminal payoffs
        if option_type.lower() == 'call':
            option_tree[:, self.n_steps] = np.maximum(
                stock_tree[:, self.n_steps] - self.K, 0
            )
        else:
            option_tree[:, self.n_steps] = np.maximum(
                self.K - stock_tree[:, self.n_steps], 0
            )
        
        exercise_boundary = []
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                european_value = self.discount * (
                    self.p * option_tree[j, i + 1] +
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                
                if option_type.lower() == 'call':
                    intrinsic = max(stock_tree[j, i] - self.K, 0)
                else:
                    intrinsic = max(self.K - stock_tree[j, i], 0)
                
                option_tree[j, i] = max(european_value, intrinsic)
                
                # Check if early exercise is optimal
                if intrinsic > european_value + threshold:
                    exercise_boundary.append({
                        'time_step': i,
                        'stock_price': stock_tree[j, i],
                        'option_value': option_tree[j, i],
                        'intrinsic_value': intrinsic,
                        'european_value': european_value
                    })
        
        return {
            'boundary_points': exercise_boundary,
            'n_exercise_points': len(exercise_boundary)
        }
    
    def convergence_test(self, option_type: str, step_sizes: list = None) -> Dict:
        """
        Test convergence as number of steps increases
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        step_sizes : list, optional
            List of step sizes to test
        
        Returns:
        --------
        dict : Convergence results
        """
        if step_sizes is None:
            step_sizes = [10, 20, 30, 50, 100, 200]
        
        results = []
        
        for n in step_sizes:
            model = BinomialTreeModel(
                self.S, self.K, self.T, self.r, self.sigma, n, self.q
            )
            price_european = model.price(option_type, 'european')
            price_american = model.price(option_type, 'american')
            
            results.append({
                'n_steps': n,
                'european_price': price_european,
                'american_price': price_american
            })
        
        return results
    
    def __repr__(self) -> str:
        return (f"BinomialTreeModel(S={self.S:.2f}, K={self.K:.2f}, T={self.T:.4f}, "
                f"r={self.r:.4f}, σ={self.sigma:.4f}, steps={self.n_steps})")


if __name__ == "__main__":
    # Test binomial tree
    binomial = BinomialTreeModel(S=100, K=105, T=0.25, r=0.05, sigma=0.30, n_steps=50)
    
    print("Binomial Tree Model Test:")
    print(f"European Call: ${binomial.price('call', 'european'):.2f}")
    print(f"American Call: ${binomial.price('call', 'american'):.2f}")
    print(f"European Put: ${binomial.price('put', 'european'):.2f}")
    print(f"American Put: ${binomial.price('put', 'american'):.2f}")
    
    print("\nConvergence Test:")
    convergence = binomial.convergence_test('call')
    for result in convergence:
        print(f"  Steps={result['n_steps']:3d}: European=${result['european_price']:.4f}, "
              f"American=${result['american_price']:.4f}")

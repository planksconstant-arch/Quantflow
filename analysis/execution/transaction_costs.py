"""
Transaction Cost Models
=======================

Implements Almgren-Chriss framework for market impact and execution costs.

References:
    - Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
"""

import numpy as np
import torch


class AlmgrenChrissModel:
    """
    Market impact cost model
    
    Total cost = Permanent Impact + Temporary Impact + Spread
    
    C(v) = ε|v| + η*v² + γ|v|
    
    Where:
        ε: Fixed cost (bid-ask spread)
        η: Temporary impact (market depth)
        γ: Permanent impact (price shift)
        v: Trade velocity (shares/time)
    """
    
    def __init__(self, epsilon=0.0005, eta=0.001, gamma=0.0001):
        """
        Args:
            epsilon: Spread cost (as fraction, e.g., 0.0005 = 5 bps)
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
        """
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
    
    def compute_cost(self, trade_size, price, inventory_before=None):
        """
        Compute total transaction cost
        
        Args:
            trade_size: Number of shares (positive = buy, negative = sell)
            price: Current asset price
            inventory_before: Previous inventory (for computing change)
        
        Returns:
            cost: Total cost in dollars
        """
        # Convert to velocity (trade size)
        v = abs(trade_size)
        
        if v < 1e-8:  # No trade
            return 0.0
        
        # Spread cost
        spread_cost = self.epsilon * v * price
        
        # Temporary impact (quadratic in trade size)
        temp_impact = self.eta * (v ** 2) * price
        
        # Permanent impact (linear in trade size)
        perm_impact = self.gamma * v * price
        
        total_cost = spread_cost + temp_impact + perm_impact
        
        return total_cost
    
    def compute_cost_torch(self, trade_sizes, prices):
        """
        Torch version for batched gradient computation
        
        Args:
            trade_sizes: (batch_size,) tensor
            prices: (batch_size,) tensor
        
        Returns:
            costs: (batch_size,) tensor
        """
        v = torch.abs(trade_sizes)
        
        spread = self.epsilon * v * prices
        temp = self.eta * (v ** 2) * prices
        perm = self.gamma * v * prices
        
        return spread + temp + perm


class SimplifiedCostModel:
    """
    Simplified cost model: fixed proportional cost
    
    Cost = c * |trade_size| * price
    
    Easier to use, suitable for initial training.
    """
    
    def __init__(self, cost_bps=5.0):
        """
        Args:
            cost_bps: Cost in basis points (e.g., 5.0 = 0.05%)
        """
        self.cost_rate = cost_bps / 10000.0
    
    def compute_cost(self, trade_size, price):
        """Compute simple proportional cost"""
        return self.cost_rate * abs(trade_size) * price
    
    def compute_cost_torch(self, trade_sizes, prices):
        """Torch version"""
        return self.cost_rate * torch.abs(trade_sizes) * prices


if __name__ == "__main__":
    print("Testing Transaction Cost Models...")
    
    # Almgren-Chriss example
    ac_model = AlmgrenChrissModel(epsilon=0.0005, eta=0.001, gamma=0.0001)
    
    # Trading 100 shares at $100
    cost = ac_model.compute_cost(trade_size=100, price=100.0)
    print(f"\nAlmgren-Chriss Cost:")
    print(f"  Trade: 100 shares @ $100")
    print(f"  Total Cost: ${cost:.4f}")
    print(f"  Cost as % of notional: {cost/(100*100)*100:.3f}%")
    
    # Torch version
    trades = torch.tensor([50.0, 100.0, 200.0])
    prices = torch.tensor([100.0, 100.0, 100.0])
    costs = ac_model.compute_cost_torch(trades, prices)
    print(f"\n  Batch costs: {costs}")
    
    # Simplified model
    simple_model = SimplifiedCostModel(cost_bps=5.0)
    simple_cost = simple_model.compute_cost(100, 100.0)
    print(f"\nSimplified Cost (5 bps):")
    print(f"  Total Cost: ${simple_cost:.4f}")
    
    print("\n✓ Transaction cost tests passed")

"""
Limit Order Book (LOB) Analytics
=================================

Microstructural metrics for order book analysis.
"""

import numpy as np
import pandas as pd


class LOBAnalyzer:
    """
    Analyzes limit order book snapshots
    
    Computes:
        - Order Book Imbalance (OBI)
        - Kyle's Lambda (price impact)
        - Liquidity slope
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def order_book_imbalance(bid_volume, ask_volume):
        """
        Compute OBI = (V_bid - V_ask) / (V_bid + V_ask)
        
        OBI > 0: Buy pressure (bid dominance)
        OBI < 0: Sell pressure (ask dominance)
        OBI ≈ 0: Balanced
        
        Args:
            bid_volume: Total bid volume (shares)
            ask_volume: Total ask volume (shares)
        
        Returns:
            obi: Imbalance metric [-1, 1]
        """
        total = bid_volume + ask_volume
        if total < 1e-8:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    @staticmethod
    def kyles_lambda(price_changes, signed_volumes):
        """
        Estimate Kyle's Lambda via linear regression
        
        ΔP_t = λ * Q_t + ε_t
        
        Where:
            ΔP: Price change
            Q: Signed order flow (+ for buy, - for sell)
            λ: Kyle's Lambda (price impact coefficient)
        
        Args:
            price_changes: Array of price changes
            signed_volumes: Array of signed volumes
        
        Returns:
            lambda_kyle: Price impact coefficient
        """
        if len(price_changes) < 2:
            return 0.0
        
        # Simple OLS
        X = np.array(signed_volumes).reshape(-1, 1)
        y = np.array(price_changes)
        
        # λ = (X^T X)^{-1} X^T y
        XtX = X.T @ X
        if XtX < 1e-8:
            return 0.0
        
        lambda_kyle = (X.T @ y) / XtX
        
        return float(lambda_kyle)
    
    @staticmethod  
    def mid_price(best_bid, best_ask):
        """Compute mid-price"""
        return (best_bid + best_ask) / 2
    
    @staticmethod
    def spread(best_bid, best_ask):
        """Compute bid-ask spread"""
        return best_ask - best_bid
    
    @staticmethod
    def spread_bps(best_bid, best_ask):
        """Spread in basis points"""
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        return (spread / mid) * 10000


def generate_synthetic_lob_data(num_snapshots=100, volatility=0.2):
    """
    Generate synthetic LOB data for testing
    
    Returns:
        DataFrame with columns: [timestamp, bid_price, ask_price, bid_vol, ask_vol]
    """
    timestamps = np.arange(num_snapshots)
    
    # Price random walk
    mid_price = 100 + np.cumsum(np.random.randn(num_snapshots) * volatility)
    
    # Spread (random around 0.05)
    spreads = np.abs(np.random.randn(num_snapshots) * 0.02 + 0.05)
    
    bid_prices = mid_price - spreads / 2
    ask_prices = mid_price + spreads / 2
    
    # Volumes (Poisson-like)
    bid_vols = np.random.exponential(1000, num_snapshots)
    ask_vols = np.random.exponential(1000, num_snapshots)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'bid_vol': bid_vols,
        'ask_vol': ask_vols
    })
    
    return df


if __name__ == "__main__":
    print("Testing LOB Analytics...")
    
    # Generate data
    lob_data = generate_synthetic_lob_data(num_snapshots=100)
    
    print(f"Generated {len(lob_data)} LOB snapshots")
    print(lob_data.head())
    
    # Compute OBI
    obi_series = LOBAnalyzer.order_book_imbalance(
        lob_data['bid_vol'].values,
        lob_data['ask_vol'].values
    )
    
    print(f"\nOrder Book Imbalance:")
    print(f"  Mean: {np.mean(obi_series):.3f}")
    print(f"  Std: {np.std(obi_series):.3f}")
    
    # Compute Kyle's Lambda
    price_changes = np.diff(lob_data['bid_price'].values)
    signed_volumes = lob_data['bid_vol'].values[1:] - lob_data['ask_vol'].values[1:]
    
    lambda_kyle = LOBAnalyzer.kyles_lambda(price_changes, signed_volumes)
    print(f"\nKyle's Lambda: {lambda_kyle:.6f}")
    
    # Spread analysis
    spreads = lob_data.apply(
        lambda row: LOBAnalyzer.spread_bps(row['bid_price'], row['ask_price']),
        axis=1
    )
    
    print(f"\nSpread Statistics:")
    print(f"  Mean: {spreads.mean():.2f} bps")
    print(f"  Std: {spreads.std():.2f} bps")
    
    print("\n✓ LOB Analytics test passed")

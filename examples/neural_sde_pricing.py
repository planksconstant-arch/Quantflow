"""
Neural SDE Integration Script
==============================

Demonstrates how to use Neural SDEs for option pricing in QuantFlow.
"""

import torch
import numpy as np
from models.neural_sde import NeuralSDE, NeuralSDETrainer
from models.neural_sde.utils import generate_gbm_paths
from data.fetch_market_data import fetch_option_chain
import yfinance as yf


def train_neural_sde_from_history(ticker, lookback_days=252, epochs=500):
    """
    Train a Neural SDE on historical price data
    
    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        lookback_days: Number of historical days
        epochs: Training epochs
    
    Returns:
        Trained NeuralSDE model
    """
    print(f"Fetching {lookback_days} days of {ticker} data...")
    
    # Fetch historical data
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"{lookback_days}d")
    
    if len(hist) < 50:
        raise ValueError(f"Insufficient data for {ticker}")
    
    prices = hist['Close'].values
    
    # Convert to paths format (adding small synthetic variations)
    num_synthetic_paths = 100
    path_length = min(100, len(prices))
    
    paths = []
    for i in range(num_synthetic_paths):
        # Sample random starting points
        start_idx = np.random.randint(0, len(prices) - path_length)
        path = prices[start_idx:start_idx + path_length]
        
        # Normalize to start at 100
        path = path / path[0] * 100
        paths.append(path)
    
    historical_paths = torch.tensor(paths, dtype=torch.float32).unsqueeze(-1)
    
    print(f"Created {num_synthetic_paths} training paths of length {path_length}")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Create and train Neural SDE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=64, num_layers=3)
    trainer = NeuralSDETrainer(sde, historical_paths, device=device)
    
    print(f"\nTraining Neural SDE for {epochs} epochs...")
    trainer.train(epochs=epochs, batch_size=32, n_critic=5, verbose=True)
    
    print("\n✓ Training complete!")
    
    return trainer.sde, prices[-1]  # Return model and current price


def price_option_with_neural_sde(sde, S0, K, T, r, option_type='call', num_sims=10000):
    """
    Price option using Neural SDE Monte Carlo
    
    Args:
        sde: Trained NeuralSDE
        S0: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        num_sims: Number of Monte Carlo paths
    
    Returns:
        Option price
    """
    device = next(sde.parameters()).device
    
    # Generate paths
    num_steps = max(int(T * 252), 10)  # Daily steps
    paths = sde.sample_paths(S0, num_sims, num_steps, T, device)
    
    # Terminal prices
    ST = paths[:, -1, 0]
    
    # Payoffs
    if option_type.lower() == 'call':
        payoffs = torch.maximum(ST - K, torch.tensor(0.0, device=device))
    else:
        payoffs = torch.maximum(K - ST, torch.tensor(0.0, device=device))
    
    # Discounted expected payoff
    option_price = torch.exp(-r * T) * payoffs.mean()
    
    return option_price.item()


if __name__ == "__main__":
    # Example: Train on NVDA, price an option
    print("=" * 60)
    print("Neural SDE Option Pricing Demo")
    print("=" * 60)
    
    # Train model
    sde, current_price = train_neural_sde_from_history('NVDA', lookback_days=252, epochs=100)
    
    # Price an ATM call
    K = current_price
    T = 0.25  # 3 months
    r = 0.05
    
    print(f"\nPricing NVDA Call Option:")
    print(f"  Strike: ${K:.2f}")
    print(f"  Expiry: {T} years")
    print(f"  Rate: {r*100}%")
    
    price = price_option_with_neural_sde(sde, current_price, K, T, r, 'call', num_sims=5000)
    
    print(f"\n  Neural SDE Price: ${price:.2f}")
    
    # Compare with Black-Scholes
    from models.pricing.black_scholes import BlackScholesModel
    
    # Estimate vol from recent history
    import yfinance as yf
    hist = yf.Ticker('NVDA').history(period='30d')
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    sigma = returns.std() * np.sqrt(252)
    
    bs = BlackScholesModel(current_price, K, T, r, sigma)
    bs_price = bs.call_price()
    
    print(f"  Black-Scholes Price: ${bs_price:.2f}")
    print(f"  Difference: ${abs(price - bs_price):.2f}")
    
    print("\n✓ Demo complete!")

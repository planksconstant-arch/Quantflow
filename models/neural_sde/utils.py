"""
Utility Functions for Neural SDEs
==================================

Brownian motion generation and SDE helper functions.
"""

import torch
import numpy as np


def generate_brownian_paths(num_paths, num_steps, T=1.0, dim=1, device='cpu'):
    """
    Generate Brownian motion paths
    
    Args:
        num_paths: Number of independent paths
        num_steps: Number of time steps
        T: Terminal time
        dim: Dimension of Brownian motion
        device: 'cpu' or 'cuda'
    
    Returns:
        Brownian paths (num_paths, num_steps, dim)
        Time grid (num_steps,)
    """
    dt = T / (num_steps - 1)
    
    # Generate increments
    dW = torch.randn(num_paths, num_steps - 1, dim, device=device) * np.sqrt(dt)
    
    # Cumulative sum to get paths (starting from 0)
    W = torch.cat([
        torch.zeros(num_paths, 1, dim, device=device),
        torch.cumsum(dW, dim=1)
    ], dim=1)
    
    t = torch.linspace(0, T, num_steps, device=device)
    
    return W, t


def generate_gbm_paths(S0, mu, sigma, T, num_paths, num_steps, device='cpu'):
    """
    Generate Geometric Brownian Motion paths (for comparison/baseline)
    
    dS_t = μ S_t dt + σ S_t dW_t
    
    Args:
        S0: Initial price
        mu: Drift
        sigma: Volatility
        T: Terminal time
        num_paths: Number of paths
        num_steps: Number of steps
        device: 'cpu' or 'cuda'
    
    Returns:
        Price paths (num_paths, num_steps, 1)
    """
    dt = T / (num_steps - 1)
    
    # Generate random increments
    dW = torch.randn(num_paths, num_steps - 1, 1, device=device) * np.sqrt(dt)
    
    # Initialize paths
    S = torch.zeros(num_paths, num_steps, 1, device=device)
    S[:, 0, 0] = S0
    
    # Euler-Maruyama scheme
    for i in range(num_steps - 1):
        S[:, i+1, :] = S[:, i, :] + mu * S[:, i, :] * dt + sigma * S[:, i,:] * dW[:, i, :]
    
    return S


def compute_path_signatures(paths):
    """
    Compute statistical signatures of paths for discriminator
    
    Features:
        - Terminal value
        - Max/Min
        - Volatility (realized)
        - Skewness
        - Kurtosis
    
    Args:
        paths: (batch_size, num_steps, state_size)
    
    Returns:
        Features (batch_size, num_features)
    """
    batch_size = paths.shape[0]
    
    # Log returns
    log_returns = torch.diff(torch.log(paths + 1e-8), dim=1)
    
    features = []
    
    # Terminal value (normalized)
    features.append(paths[:, -1, :])
    
    # Max and Min
    features.append(paths.max(dim=1)[0])
    features.append(paths.min(dim=1)[0])
    
    # Realized volatility
    vol = log_returns.std(dim=1)
    features.append(vol)
    
    # Skewness and Kurtosis
    mean_returns = log_returns.mean(dim=1, keepdim=True)
    std_returns = log_returns.std(dim=1, keepdim=True) + 1e-8
    
    normalized = (log_returns - mean_returns) / std_returns
    skew = (normalized ** 3).mean(dim=1)
    kurt = (normalized ** 4).mean(dim=1)
    
    features.append(skew)
    features.append(kurt)
    
    # Concatenate all features
    return torch.cat(features, dim=-1)


if __name__ == "__main__":
    print("Testing Neural SDE utilities...")
    
    # Test Brownian motion generation
    W, t = generate_brownian_paths(num_paths=1000, num_steps=252, T=1.0)
    print(f"Brownian paths shape: {W.shape}")
    print(f"Mean: {W.mean():.4f} (should be ~0)")
    print(f"Std: {W.std():.4f} (should be ~sqrt(T))")
    
    # Test GBM
    S = generate_gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1.0, 
                           num_paths=1000, num_steps=252)
    print(f"\nGBM paths shape: {S.shape}")
    print(f"Initial: {S[:, 0, 0].mean():.2f}")
    print(f"Terminal: {S[:, -1, 0].mean():.2f}")
    
    # Test signatures
    sigs = compute_path_signatures(S)
    print(f"\nPath signatures shape: {sigs.shape}")
    print("✓ All tests passed")

"""
High-Performance Acceleration Engine
====================================
Vectorized operations and Numba JIT routines for ultra-fast LOB computations,
pairwise Mormyrid sensory distance fields, and Hawkes intensity recursions.
"""

import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, fastmath=True)
def fast_pairwise_euclidean(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix for swarm agents.
    positions: (N, D) array
    returns: (N, N) distance matrix
    """
    n = positions.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(positions.shape[1]):
                diff = positions[i, k] - positions[j, k]
                d += diff * diff
            dist = np.sqrt(d)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


@jit(nopython=True, fastmath=True)
def fast_hawkes_intensity_recursive(
    event_times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    eval_times: np.ndarray
) -> np.ndarray:
    """
    Compute Hawkes intensity path efficiently using recursive exponential decay.
    """
    n_eval = eval_times.shape[0]
    intensities = np.zeros(n_eval, dtype=np.float64)
    
    for idx in range(n_eval):
        t = eval_times[idx]
        kernel_sum = 0.0
        for e_idx in range(event_times.shape[0]):
            t_i = event_times[e_idx]
            if t_i < t:
                kernel_sum += alpha * beta * np.exp(-beta * (t - t_i))
        intensities[idx] = mu + kernel_sum
        
    return intensities


@jit(nopython=True, fastmath=True)
def fast_stoikov_reservation_price(
    mid_price: float,
    inventory: float,
    gamma: float,
    sigma: float,
    time_remaining: float,
    swarm_skew: float,
) -> float:
    """
    Fast Numba calculation of reservation price with swarm skew.
    """
    inventory_term = inventory * gamma * (sigma * sigma) * (time_remaining / 252.0) * mid_price
    return (mid_price - inventory_term) + swarm_skew

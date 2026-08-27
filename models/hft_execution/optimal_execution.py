"""
Optimal Execution & Algorithmic Router
======================================
Almgren-Chriss (2000) optimal liquidation trajectories with linear/non-linear
market impact, and TWAP / VWAP algorithmic execution with Swarm timing modulation.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


@dataclass
class ExecutionTrajectory:
    """Optimal trajectory for algorithmic trade execution"""
    times: np.ndarray
    holdings: np.ndarray
    trade_sizes: np.ndarray
    expected_shortfall: float
    variance_shortfall: float
    utility: float
    total_cost: float


class AlmgrenChrissExecution:
    """
    Almgren-Chriss Optimal Liquidation Model
    
    Minimizes E[x] + lambda * Var(x) where:
    - Temporary impact: eta * (n_k / tau)
    - Permanent impact: gamma * n_k
    - Volatility risk: sigma^2 * tau * x_k^2
    """
    def __init__(
        self,
        total_shares: float = 10000.0,
        time_horizon: float = 1.0,      # In hours or normalized trading day
        num_intervals: int = 20,
        risk_aversion: float = 1e-5,    # Lambda
        temp_impact: float = 2.5e-6,    # Eta
        perm_impact: float = 2.5e-7,    # Gamma
        volatility: float = 0.30,       # Annualized sigma
        initial_price: float = 140.0,
    ):
        self.X = total_shares
        self.T = time_horizon
        self.N = num_intervals
        self.tau = time_horizon / num_intervals
        self.lam = risk_aversion
        self.eta = temp_impact
        self.gamma = perm_impact
        self.sigma = volatility * initial_price / np.sqrt(252 * 6.5)  # Per hour vol
        self.S0 = initial_price

    def calculate_trajectory(self, swarm_urgency_multiplier: float = 1.0) -> ExecutionTrajectory:
        """
        Compute optimal trading trajectory.
        swarm_urgency_multiplier: >1.0 accelerates liquidation (e.g. predatory breakout),
                                  <1.0 slows down (passive schooling).
        """
        adjusted_lambda = max(1e-9, self.lam * swarm_urgency_multiplier)
        
        # Kappa parameter: kappa^2 = (lambda * sigma^2) / eta_hat
        eta_hat = self.eta * (1.0 - 0.5 * self.gamma * self.tau / self.eta)
        eta_hat = max(1e-9, eta_hat)
        
        kappa_sq = (adjusted_lambda * (self.sigma ** 2)) / eta_hat
        kappa = np.sqrt(kappa_sq)
        
        # Calculate holdings at each step t_k: x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T)
        times = np.linspace(0, self.T, self.N + 1)
        
        if kappa * self.T < 1e-4:
            # Linear TWAP as kappa -> 0 (risk-neutral)
            holdings = self.X * (1.0 - times / self.T)
        else:
            denom = np.sinh(kappa * self.T)
            holdings = self.X * np.sinh(kappa * (self.T - times)) / denom
            
        trade_sizes = -np.diff(holdings)
        
        # Expected transaction costs
        # Permanent impact cost: 0.5 * gamma * X^2
        # Temporary impact cost: eta_hat * sum(n_j^2 / tau)
        perm_cost = 0.5 * self.gamma * (self.X ** 2)
        temp_cost = eta_hat * np.sum((trade_sizes ** 2) / self.tau)
        expected_shortfall = perm_cost + temp_cost
        
        # Variance of shortfall
        variance_shortfall = (self.sigma ** 2) * self.tau * np.sum(holdings[:-1] ** 2)
        utility = expected_shortfall + adjusted_lambda * variance_shortfall
        
        return ExecutionTrajectory(
            times=times,
            holdings=holdings,
            trade_sizes=trade_sizes,
            expected_shortfall=float(expected_shortfall),
            variance_shortfall=float(variance_shortfall),
            utility=float(utility),
            total_cost=float(expected_shortfall),
        )


class AlgorithmicRouter:
    """
    Algorithmic Order Execution Router (TWAP / VWAP with Swarm Pulse Slicing)
    """
    @staticmethod
    def generate_twap_schedule(total_volume: float, duration_minutes: int, interval_minutes: int = 1) -> pd.DataFrame:
        n_slices = max(1, duration_minutes // interval_minutes)
        slice_vol = total_volume / n_slices
        records = []
        for i in range(n_slices):
            records.append({
                "slice_index": i + 1,
                "minute": (i + 1) * interval_minutes,
                "target_volume": slice_vol,
                "cumulative_volume": (i + 1) * slice_vol,
            })
        return pd.DataFrame(records)

    @staticmethod
    def generate_swarm_vwap_schedule(
        total_volume: float,
        duration_minutes: int,
        swarm_forecast_curve: np.ndarray,
    ) -> pd.DataFrame:
        """
        VWAP execution with dynamic volume tilting based on swarm confidence.
        """
        n_slices = len(swarm_forecast_curve)
        # Weight by U-shape intraday volume + swarm confidence boost
        u_curve = 0.6 + 0.4 * ((np.linspace(-1, 1, n_slices)) ** 2)
        combined_weights = u_curve * (1.0 + 0.3 * swarm_forecast_curve)
        combined_weights /= np.sum(combined_weights)
        
        volumes = total_volume * combined_weights
        records = []
        cum = 0.0
        for i, v in enumerate(volumes):
            cum += v
            records.append({
                "slice_index": i + 1,
                "minute": i + 1,
                "target_volume": round(v, 2),
                "cumulative_volume": round(cum, 2),
                "swarm_weight": round(combined_weights[i], 4),
            })
        return pd.DataFrame(records)

"""
High-Frequency Trading Real-Time Risk Engine
============================================
Institutional risk metrics and safety controls for quantitative HFT systems:
- Parametric & Historical Value at Risk (VaR 95%, 99%)
- Conditional Value at Risk (CVaR / Expected Shortfall)
- Real-Time Sharpe, Sortino, and Calmar Ratios
- Maximum Drawdown Tracker
- Real-Time Toxicity & Inventory Circuit Breakers
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


@dataclass
class RiskMetricsSnapshot:
    """Consolidated real-time risk profile"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    inventory_utilization: float
    circuit_breaker_tripped: bool
    risk_level: str  # "LOW", "MODERATE", "ELEVATED", "CRITICAL"


class HFTRiskEngine:
    """
    Real-Time High-Frequency Risk & Safety Engine.
    """
    def __init__(
        self,
        max_inventory_limit: float = 100.0,
        max_drawdown_limit: float = 5000.0,
        max_vpin_threshold: float = 0.75,
        var_lookback_window: int = 100,
    ):
        self.max_inventory = max_inventory_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_vpin = max_vpin_threshold
        self.lookback = var_lookback_window
        
        self.pnl_history: List[float] = []
        self.peak_pnl: float = 0.0

    def update_pnl(self, current_total_pnl: float) -> None:
        self.pnl_history.append(current_total_pnl)
        if current_total_pnl > self.peak_pnl:
            self.peak_pnl = current_total_pnl

    def compute_risk_profile(
        self,
        current_inventory: float,
        current_vpin: float = 0.2,
        current_pnl: float = 0.0,
    ) -> RiskMetricsSnapshot:
        """Calculate comprehensive risk metrics"""
        self.update_pnl(current_pnl)
        
        # Calculate returns
        if len(self.pnl_history) < 10:
            returns = np.zeros(10)
        else:
            window_pnl = np.array(self.pnl_history[-self.lookback:])
            returns = np.diff(window_pnl)

        # 1. VaR & CVaR (Historical)
        if len(returns) > 5 and np.std(returns) > 1e-8:
            var_95 = float(-np.percentile(returns, 5))
            var_99 = float(-np.percentile(returns, 1))
            
            tail_95 = returns[returns <= -var_95]
            cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else var_95
            
            tail_99 = returns[returns <= -var_99]
            cvar_99 = float(-np.mean(tail_99)) if len(tail_99) > 0 else var_99
        else:
            var_95 = 50.0
            var_99 = 100.0
            cvar_95 = 75.0
            cvar_99 = 150.0

        # 2. Performance Ratios (Sharpe, Sortino, Calmar)
        mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
        std_ret = float(np.std(returns)) if len(returns) > 0 else 1.0
        
        downside_returns = returns[returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else std_ret
        
        ann_factor = np.sqrt(252 * 6.5 * 3600)
        sharpe = float((mean_ret / max(1e-6, std_ret)) * ann_factor)
        sortino = float((mean_ret / max(1e-6, downside_std)) * ann_factor)

        # 3. Drawdown Analysis
        current_dd = float(self.peak_pnl - current_pnl)
        pnl_arr = np.array(self.pnl_history)
        cummax = np.maximum.accumulate(pnl_arr)
        drawdowns = cummax - pnl_arr
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        calmar = float((current_pnl / max(1.0, max_dd))) if max_dd > 0 else 0.0

        # 4. Inventory utilization & Circuit breakers
        inv_util = float(abs(current_inventory) / max(1.0, self.max_inventory))
        
        is_tripped = (
            current_dd >= self.max_drawdown_limit or
            inv_util >= 1.0 or
            current_vpin >= self.max_vpin
        )

        # 5. Risk classification
        if is_tripped or current_dd > 0.8 * self.max_drawdown_limit:
            risk_level = "CRITICAL"
        elif inv_util > 0.7 or current_vpin > 0.6:
            risk_level = "ELEVATED"
        elif inv_util > 0.4 or current_vpin > 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return RiskMetricsSnapshot(
            var_95=round(max(0.0, var_95), 2),
            var_99=round(max(0.0, var_99), 2),
            cvar_95=round(max(0.0, cvar_95), 2),
            cvar_99=round(max(0.0, cvar_99), 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown=round(max_dd, 2),
            current_drawdown=round(current_dd, 2),
            inventory_utilization=round(min(1.0, inv_util), 3),
            circuit_breaker_tripped=is_tripped,
            risk_level=risk_level,
        )

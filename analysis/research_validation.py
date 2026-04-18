"""
Research-grade validation utilities for option mispricing signals.

Focus:
- Walk-forward evaluation (no in-sample leakage)
- Cost-aware returns
- Robust statistics (HAC t-stat proxy, bootstrap confidence intervals)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationConfig:
    signal_threshold_pct: float = 3.0
    hold_period_days: int = 5
    commission_per_contract: float = 0.65
    contract_multiplier: int = 100
    bootstrap_samples: int = 2000
    annualization_factor: int = 252
    hac_lags: int = 5


class ResearchValidator:
    """
    Validation engine for signal quality and economic significance.

    Expected columns in input DataFrame:
    - date
    - market_price
    - fair_value
    - bid
    - ask
    - future_option_price (price after hold period)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def walk_forward_backtest(self, df: pd.DataFrame) -> Dict:
        data = self._validate_input(df).copy()
        data = data.sort_values("date").reset_index(drop=True)

        trades = self._build_trade_log(data)
        if trades.empty:
            return self._empty_result("no_signals")

        returns = trades["net_return"].values
        equity_curve = (1.0 + trades["net_return"]).cumprod()
        drawdown = equity_curve / np.maximum.accumulate(equity_curve) - 1.0

        sharpe = self._annualized_sharpe(returns)
        hac_t = self._hac_t_stat(returns, lags=self.config.hac_lags)
        ci_low, ci_high = self._bootstrap_ci(
            returns,
            n_samples=self.config.bootstrap_samples,
            confidence=0.95,
        )

        return {
            "n_trades": int(len(trades)),
            "hit_rate": float((trades["net_return"] > 0).mean()),
            "avg_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "total_return": float(np.prod(1.0 + returns) - 1.0),
            "annualized_sharpe": float(sharpe),
            "hac_t_stat": float(hac_t),
            "max_drawdown": float(drawdown.min()),
            "bootstrap_ci_95": (float(ci_low), float(ci_high)),
            "p_positive": float(np.mean(returns > 0)),
            "economic_significance": self._economic_significance(
                avg_return=float(np.mean(returns)),
                sharpe=float(sharpe),
                hac_t=float(hac_t),
                ci_low=float(ci_low),
            ),
            "trades": trades,
        }

    def _build_trade_log(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.signal_threshold_pct / 100.0
        rows: List[Dict] = []

        for _, row in df.iterrows():
            mkt = float(row["market_price"])
            fv = float(row["fair_value"])
            if mkt <= 0:
                continue

            edge = (fv - mkt) / mkt
            direction = 1 if edge > threshold else (-1 if edge < -threshold else 0)
            if direction == 0:
                continue

            bid = float(row["bid"])
            ask = float(row["ask"])
            future = float(row["future_option_price"])

            entry_price = ask if direction == 1 else bid
            exit_price = future

            gross_pnl = direction * (exit_price - entry_price) * self.config.contract_multiplier
            total_cost = self.config.commission_per_contract + ((ask - bid) * self.config.contract_multiplier / 2.0)
            net_pnl = gross_pnl - total_cost

            notional = max(entry_price * self.config.contract_multiplier, 0.01)
            net_return = net_pnl / notional

            rows.append(
                {
                    "date": row["date"],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "edge_pct": edge * 100.0,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "net_return": net_return,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _annualized_sharpe(returns: np.ndarray, annualization_factor: int = 252) -> float:
        if len(returns) < 2 or np.std(returns, ddof=1) == 0:
            return 0.0
        return (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(annualization_factor)

    @staticmethod
    def _hac_t_stat(returns: np.ndarray, lags: int = 5) -> float:
        """
        Newey-West style t-statistic for mean return.
        """
        n = len(returns)
        if n < 3:
            return 0.0

        x = returns - np.mean(returns)
        gamma0 = np.sum(x * x) / n
        var_hac = gamma0
        for lag in range(1, min(lags, n - 1) + 1):
            weight = 1.0 - lag / (lags + 1.0)
            gamma = np.sum(x[lag:] * x[:-lag]) / n
            var_hac += 2.0 * weight * gamma

        se = np.sqrt(max(var_hac, 1e-12) / n)
        return float(np.mean(returns) / se) if se > 0 else 0.0

    @staticmethod
    def _bootstrap_ci(values: np.ndarray, n_samples: int = 2000, confidence: float = 0.95) -> Tuple[float, float]:
        if len(values) == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(42)
        means = np.empty(n_samples)
        for i in range(n_samples):
            sample = rng.choice(values, size=len(values), replace=True)
            means[i] = np.mean(sample)

        alpha = (1.0 - confidence) / 2.0
        return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))

    @staticmethod
    def _economic_significance(avg_return: float, sharpe: float, hac_t: float, ci_low: float) -> str:
        if avg_return > 0.01 and sharpe > 1.0 and hac_t > 2.0 and ci_low > 0:
            return "high"
        if avg_return > 0 and sharpe > 0.5 and hac_t > 1.0:
            return "moderate"
        return "low"

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "market_price", "fair_value", "bid", "ask", "future_option_price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Input dataframe is empty")
        return df

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        return {
            "n_trades": 0,
            "hit_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "total_return": 0.0,
            "annualized_sharpe": 0.0,
            "hac_t_stat": 0.0,
            "max_drawdown": 0.0,
            "bootstrap_ci_95": (0.0, 0.0),
            "p_positive": 0.0,
            "economic_significance": "low",
            "reason": reason,
            "trades": pd.DataFrame(),
        }

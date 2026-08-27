"""
High-Frequency Trading (HFT) Simulator & Backtest Harness
=========================================================
Event-driven high-frequency simulation harness that evaluates:
1. Classical Avellaneda-Stoikov Market Making
2. Swarm-Skewed Avellaneda-Stoikov Market Making (with Mormyrid Consensus)
3. Predator Momentum Flow Chasing Strategy

Tracks real-time order fills, queue priorities, inventory drift, adverse selection,
slippage, cash PnL, unrealized PnL, and Sharpe metrics.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from models.microstructure.order_book import (
    LimitOrderBook,
    L2Snapshot,
    OrderSide,
    generate_synthetic_lob_stream,
)
from models.microstructure.signals import (
    MultiLevelOFI,
    VPIN,
    StoikovMicroPrice,
)
from models.microstructure.hawkes import HawkesProcess
from models.swarm.consensus_engine import MormyridSwarmConsensusEngine, SwarmConsensusSignal
from models.hft_execution.avellaneda_stoikov import SwarmAvellanedaStoikov


@dataclass
class SimulationResult:
    """Consolidated backtest and live simulation output"""
    strategy_name: str
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    buy_trades: int
    sell_trades: int
    final_inventory: float
    max_inventory: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float
    total_volume: float
    pnl_series: pd.Series
    inventory_series: pd.Series
    price_series: pd.Series
    swarm_signals: List[Dict]
    fill_log: pd.DataFrame


class HFTSimulator:
    """
    Event-driven simulation harness for High-Frequency Trading strategies.
    """
    def __init__(
        self,
        initial_cash: float = 100000.0,
        latency_ticks: int = 1,
        fill_slippage_bps: float = 0.5,
    ):
        self.initial_cash = initial_cash
        self.latency_ticks = latency_ticks
        self.slippage_bps = fill_slippage_bps

    def run_simulation(
        self,
        snapshots: List[L2Snapshot],
        strategy_type: str = "swarm_as",   # "swarm_as", "classical_as", "momentum_predator"
        swarm_engine: Optional[MormyridSwarmConsensusEngine] = None,
        as_model: Optional[SwarmAvellanedaStoikov] = None,
    ) -> SimulationResult:
        """
        Execute full event-driven simulation through L2 snapshots.
        """
        if swarm_engine is None:
            swarm_engine = MormyridSwarmConsensusEngine()
        if as_model is None:
            as_model = SwarmAvellanedaStoikov()

        ofi_calc = MultiLevelOFI()
        vpin_calc = VPIN()
        hawkes = HawkesProcess(alpha=0.6, beta=1.5, mu=0.8)

        cash = self.initial_cash
        inventory = 0.0
        pnl_history = []
        inventory_history = []
        price_history = []
        swarm_signal_history = []
        fill_records = []

        event_times = []
        n_steps = len(snapshots)

        for i, snap in enumerate(snapshots):
            t = snap.timestamp
            mid = snap.mid_price
            price_history.append(mid)

            # Update microstructure signals
            ofi = ofi_calc.update(snap)
            
            # Update VPIN
            if len(price_history) > 1:
                vpin_calc.update_trade(mid, snap.total_bid_depth * 0.1, price_history[-2])
            vpin = vpin_calc.get_vpin()

            # Update Hawkes point process intensity
            event_times.append(t)
            hawkes_intensity = hawkes.intensity(t, np.array(event_times))
            hawkes_branching = hawkes.branching_ratio()

            # Micro-price deviation (in bps)
            micro_dev_bps = ((snap.micro_price - mid) / mid) * 10000.0
            relative_spread_bps = (snap.spread / mid) * 10000.0

            # Step Mormyrid Swarm Consensus Engine
            swarm_out: SwarmConsensusSignal = swarm_engine.step_market_state(
                ofi=ofi,
                vpin=vpin,
                hawkes_intensity=hawkes_intensity,
                micro_price_dev=micro_dev_bps,
                relative_spread=relative_spread_bps,
                hawkes_branching_ratio=hawkes_branching,
            )

            swarm_signal_history.append({
                "timestamp": t,
                "mid_price": mid,
                "micro_price": snap.micro_price,
                "drift_bps": swarm_out.predicted_drift_bps,
                "jump_prob": swarm_out.jump_probability,
                "crowding_index": swarm_out.market_crowding_index,
                "adverse_risk": swarm_out.adverse_selection_risk,
                "dominant_regime": swarm_out.dominant_regime,
                "confidence": swarm_out.swarm_confidence,
            })

            # Time remaining in session
            time_rem = max(1e-4, 1.0 - (i / n_steps))

            # Strategy quote logic
            if strategy_type == "swarm_as":
                # Swarm-Skewed Avellaneda-Stoikov
                quotes = as_model.calculate_quotes(
                    mid_price=mid,
                    inventory=inventory,
                    time_remaining=time_rem,
                    swarm_drift_bps=swarm_out.predicted_drift_bps,
                    jar_crowding_index=swarm_out.market_crowding_index,
                    adverse_selection_risk=swarm_out.adverse_selection_risk,
                    hawkes_intensity=hawkes_intensity,
                )
                
                # Check for fills on resting quotes
                # Buy fill trigger: If next mid touches bid or Poisson arrival hits bid
                prob_hit_bid = quotes.bid_fill_probability * (1.0 + max(0.0, -ofi))
                prob_hit_ask = quotes.ask_fill_probability * (1.0 + max(0.0, ofi))
                
                # Sample fills
                if np.random.uniform() < prob_hit_bid and inventory < as_model.max_inventory:
                    # Buy fill at quotes.bid_price
                    trade_vol = 10.0
                    fill_price = quotes.bid_price
                    cash -= trade_vol * fill_price
                    inventory += trade_vol
                    fill_records.append({
                        "timestamp": t,
                        "side": "BUY",
                        "price": fill_price,
                        "volume": trade_vol,
                        "inventory_after": inventory,
                        "type": "MAKER_LIMIT",
                    })

                if np.random.uniform() < prob_hit_ask and inventory > -as_model.max_inventory:
                    # Sell fill at quotes.ask_price
                    trade_vol = 10.0
                    fill_price = quotes.ask_price
                    cash += trade_vol * fill_price
                    inventory -= trade_vol
                    fill_records.append({
                        "timestamp": t,
                        "side": "SELL",
                        "price": fill_price,
                        "volume": trade_vol,
                        "inventory_after": inventory,
                        "type": "MAKER_LIMIT",
                    })

            elif strategy_type == "classical_as":
                # Classical Avellaneda-Stoikov (No Swarm Skew, No JAR protection)
                quotes = as_model.calculate_quotes(
                    mid_price=mid,
                    inventory=inventory,
                    time_remaining=time_rem,
                    swarm_drift_bps=0.0,
                    jar_crowding_index=0.0,
                    adverse_selection_risk=0.0,
                    hawkes_intensity=1.0,
                )
                prob_hit_bid = quotes.bid_fill_probability
                prob_hit_ask = quotes.ask_fill_probability

                if np.random.uniform() < prob_hit_bid and inventory < as_model.max_inventory:
                    trade_vol = 10.0
                    cash -= trade_vol * quotes.bid_price
                    inventory += trade_vol
                    fill_records.append({
                        "timestamp": t,
                        "side": "BUY",
                        "price": quotes.bid_price,
                        "volume": trade_vol,
                        "inventory_after": inventory,
                        "type": "CLASSICAL_MAKER",
                    })

                if np.random.uniform() < prob_hit_ask and inventory > -as_model.max_inventory:
                    trade_vol = 10.0
                    cash += trade_vol * quotes.ask_price
                    inventory -= trade_vol
                    fill_records.append({
                        "timestamp": t,
                        "side": "SELL",
                        "price": quotes.ask_price,
                        "volume": trade_vol,
                        "inventory_after": inventory,
                        "type": "CLASSICAL_MAKER",
                    })

            elif strategy_type == "momentum_predator":
                # Takes liquidity aggressively on Hawkes jump bursts + Swarm drift
                if swarm_out.jump_probability > 0.70 and swarm_out.predicted_drift_bps > 3.0 and inventory < 50:
                    trade_vol = 15.0
                    fill_p = mid + snap.spread / 2.0  # Pay the ask
                    cash -= trade_vol * fill_p
                    inventory += trade_vol
                    fill_records.append({
                        "timestamp": t,
                        "side": "BUY",
                        "price": fill_p,
                        "volume": trade_vol,
                        "inventory_after": inventory,
                        "type": "TAKER_MOMENTUM",
                    })
                elif swarm_out.jump_probability > 0.70 and swarm_out.predicted_drift_bps < -3.0 and inventory > -50:
                    trade_vol = 15.0
                    fill_p = mid - snap.spread / 2.0  # Hit the bid
                    cash += trade_vol * fill_p
                    inventory -= trade_vol
                    fill_records.append({
                        "timestamp": t,
                        "side": "SELL",
                        "price": fill_p,
                        "volume": trade_vol,
                        "inventory_after": inventory,
                        "type": "TAKER_MOMENTUM",
                    })

            # Calculate total mark-to-market PnL
            unrealized = inventory * mid
            total_pnl = (cash + unrealized) - self.initial_cash
            pnl_history.append(total_pnl)
            inventory_history.append(inventory)

        # Performance analytics
        pnl_series = pd.Series(pnl_history)
        inv_series = pd.Series(inventory_history)
        price_series = pd.Series(price_history)
        fill_df = pd.DataFrame(fill_records) if fill_records else pd.DataFrame(columns=["timestamp", "side", "price", "volume", "inventory_after", "type"])

        # Returns & Sharpe
        pnl_diffs = pnl_series.diff().fillna(0)
        mean_ret = pnl_diffs.mean()
        std_ret = pnl_diffs.std()
        sharpe = float((mean_ret / (std_ret + 1e-8)) * np.sqrt(252 * 6.5 * 3600)) if std_ret > 0 else 0.0

        # Max Drawdown
        cum_max = pnl_series.cummax()
        drawdown = cum_max - pnl_series
        max_dd = float(drawdown.max())

        total_trades = len(fill_df)
        buy_trades = len(fill_df[fill_df["side"] == "BUY"]) if total_trades > 0 else 0
        sell_trades = len(fill_df[fill_df["side"] == "SELL"]) if total_trades > 0 else 0
        total_vol = float(fill_df["volume"].sum()) if total_trades > 0 else 0.0

        win_trades = (pnl_diffs > 0).sum()
        win_rate = float(win_trades / max(1, (pnl_diffs != 0).sum()))

        return SimulationResult(
            strategy_name=strategy_type,
            total_pnl=float(pnl_series.iloc[-1]),
            realized_pnl=float(cash - self.initial_cash),
            unrealized_pnl=float(inventory * price_series.iloc[-1]),
            total_trades=total_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            final_inventory=float(inventory),
            max_inventory=float(inv_series.abs().max()),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_trade_pnl=float(pnl_series.iloc[-1] / max(1, total_trades)),
            total_volume=total_vol,
            pnl_series=pnl_series,
            inventory_series=inv_series,
            price_series=price_series,
            swarm_signals=swarm_signal_history,
            fill_log=fill_df,
        )

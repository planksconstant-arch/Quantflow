"""
Unit Tests for High-Frequency Trading Execution & Strategy Module
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hft_execution import (
    SwarmAvellanedaStoikov,
    AlmgrenChrissExecution,
    HFTSimulator,
    SimulationResult,
)
from models.risk.hft_risk import HFTRiskEngine
from models.microstructure import generate_synthetic_lob_stream


class TestAvellanedaStoikov:
    """Test Swarm-Skewed Avellaneda-Stoikov market maker"""
    
    def test_quote_generation(self):
        mm = SwarmAvellanedaStoikov(gamma=0.1, kappa=1.5, sigma=0.3)
        quotes = mm.calculate_quotes(
            mid_price=140.0,
            inventory=0,
            time_remaining=0.5,
            swarm_drift_bps=5.0, # Bullish drift
            jar_crowding_index=0.1,
            adverse_selection_risk=0.1,
        )
        assert quotes.bid_price < quotes.mid_price < quotes.ask_price
        assert quotes.optimal_spread > 0
        assert 0.0 <= quotes.bid_fill_probability <= 1.0

    def test_inventory_skew_effect(self):
        mm = SwarmAvellanedaStoikov(gamma=0.2, kappa=1.5, sigma=0.3)
        # When long (+20), reservation price drops, so quotes skew down (lower bid & ask)
        quotes_long = mm.calculate_quotes(mid_price=100.0, inventory=20, time_remaining=0.5)
        # When short (-20), reservation price rises, quotes skew up (higher bid & ask)
        quotes_short = mm.calculate_quotes(mid_price=100.0, inventory=-20, time_remaining=0.5)
        
        assert quotes_long.reservation_price < quotes_short.reservation_price


class TestAlmgrenChriss:
    """Test Almgren-Chriss liquidation trajectory"""
    
    def test_trajectory_bounds(self):
        ac = AlmgrenChrissExecution(total_shares=10000.0, time_horizon=1.0, num_intervals=10)
        traj = ac.calculate_trajectory()
        assert len(traj.holdings) == 11
        assert abs(traj.holdings[0] - 10000.0) < 1e-4
        assert abs(traj.holdings[-1] - 0.0) < 1e-4
        assert traj.total_cost > 0


class TestHFTSimulator:
    """Test full event-driven simulation"""
    
    def test_simulator_run(self):
        snapshots, _ = generate_synthetic_lob_stream(n_ticks=30, initial_price=100.0, seed=99)
        sim = HFTSimulator(initial_cash=50000.0)
        res = sim.run_simulation(snapshots, strategy_type="swarm_as")
        
        assert isinstance(res, SimulationResult)
        assert len(res.pnl_series) == 30
        assert len(res.inventory_series) == 30
        assert -1.0 <= res.win_rate <= 1.0


class TestHFTRiskEngine:
    """Test real-time risk profile calculations"""
    
    def test_risk_metrics(self):
        risk_engine = HFTRiskEngine(max_inventory_limit=100, max_drawdown_limit=2000)
        for pnl in [100, 150, 130, 200, 250, 180, 300, 320, 290, 350]:
            snapshot = risk_engine.compute_risk_profile(current_inventory=10, current_vpin=0.25, current_pnl=pnl)
            
        assert snapshot.var_95 >= 0
        assert snapshot.max_drawdown >= 0
        assert snapshot.risk_level in ["LOW", "MODERATE", "ELEVATED", "CRITICAL"]

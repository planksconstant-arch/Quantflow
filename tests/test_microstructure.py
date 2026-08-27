"""
Unit Tests for Market Microstructure & Signal Suite
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.microstructure import (
    LimitOrderBook,
    OrderSide,
    generate_synthetic_lob_stream,
    MultiLevelOFI,
    VPIN,
    StoikovMicroPrice,
    HawkesProcess,
)


class TestLimitOrderBook:
    """Test Limit Order Book matching and depth calculations"""
    
    def test_order_insertion_and_spread(self):
        book = LimitOrderBook(tick_size=0.01)
        book.add_limit_order("o1", OrderSide.BUY, 100.00, 100)
        book.add_limit_order("o2", OrderSide.SELL, 100.05, 100)
        
        assert book.best_bid == 100.00
        assert book.best_ask == 100.05
        assert book.mid_price == 100.025
        assert abs(book.spread - 0.05) < 1e-4

    def test_market_order_execution(self):
        book = LimitOrderBook(tick_size=0.01)
        book.add_limit_order("a1", OrderSide.SELL, 100.10, 50)
        book.add_limit_order("a2", OrderSide.SELL, 100.20, 50)
        
        fill = book.execute_market_order(OrderSide.BUY, 75)
        assert fill["filled_volume"] == 75
        assert fill["vwap_price"] == (50 * 100.10 + 25 * 100.20) / 75
        assert book.best_ask == 100.20

    def test_synthetic_lob_stream(self):
        snapshots, df = generate_synthetic_lob_stream(n_ticks=50, initial_price=150.0, seed=123)
        assert len(snapshots) == 50
        assert len(df) == 50
        assert "mid_price" in df.columns
        assert "micro_price" in df.columns
        assert "book_imbalance" in df.columns


class TestMicrostructureSignals:
    """Test OFI, VPIN, Stoikov, and Hawkes models"""
    
    def test_multi_level_ofi(self):
        snapshots, _ = generate_synthetic_lob_stream(n_ticks=10, initial_price=100.0, seed=42)
        ofi_engine = MultiLevelOFI(depth_levels=3)
        for snap in snapshots:
            val = ofi_engine.update(snap)
            assert -1.0 <= val <= 1.0

    def test_vpin_calculation(self):
        vpin = VPIN(bucket_size=100.0, n_buckets=5)
        # Feed trades
        vpin.update_trade(100.0, 50.0, 99.9)
        vpin.update_trade(100.2, 80.0, 100.0)
        vpin.update_trade(100.1, 70.0, 100.2)
        score = vpin.get_vpin()
        assert 0.0 <= score <= 1.0

    def test_stoikov_micro_price(self):
        snapshots, _ = generate_synthetic_lob_stream(n_ticks=5, initial_price=120.0, seed=7)
        micro_p = StoikovMicroPrice.calculate(snapshots[0])
        assert abs(micro_p - snapshots[0].mid_price) <= snapshots[0].spread

    def test_hawkes_process_simulation(self):
        hawkes = HawkesProcess(alpha=0.5, beta=2.0, mu=1.0)
        events = hawkes.simulate(T=5.0)
        assert len(events) > 0
        intensity = hawkes.intensity(5.0, events)
        assert intensity >= hawkes.mu

"""
High-Frequency Market Microstructure Signals
============================================
Quantitative microstructure signals calculated on Level 2 / Level 3 LOB data:
1. Multi-Level Order Flow Imbalance (ML-OFI)
2. Volume-Synchronized Probability of Toxicity (VPIN)
3. Stoikov Micro-Price & Markov Queue Imbalance
4. Kyle's Lambda Price Impact Estimator
5. Roll Serial Autocovariance Spread
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from models.microstructure.order_book import L2Snapshot


class MultiLevelOFI:
    """
    Multi-Level Order Flow Imbalance (ML-OFI)
    
    Measures net order flow pressure across multiple price levels:
    OFI_t^{(k)} = I_{P_{b,t}^{(k)} >= P_{b,t-1}^{(k)}} * V_{b,t}^{(k)}
                 - I_{P_{b,t}^{(k)} <= P_{b,t-1}^{(k)}} * V_{b,t-1}^{(k)}
                 - I_{P_{a,t}^{(k)} <= P_{a,t-1}^{(k)}} * V_{a,t}^{(k)}
                 + I_{P_{a,t}^{(k)} >= P_{a,t-1}^{(k)}} * V_{a,t-1}^{(k)}
    """
    def __init__(self, depth_levels: int = 5, level_weights: Optional[np.ndarray] = None):
        self.depth_levels = depth_levels
        if level_weights is not None:
            self.level_weights = np.array(level_weights, dtype=float)
            self.level_weights /= np.sum(self.level_weights)
        else:
            # Exponentially decaying weights by default [1.0, 0.5, 0.25, ...]
            raw_w = np.exp(-0.5 * np.arange(depth_levels))
            self.level_weights = raw_w / np.sum(raw_w)
            
        self.last_snapshot: Optional[L2Snapshot] = None

    def update(self, snapshot: L2Snapshot) -> float:
        """
        Compute ML-OFI between previous and current L2 snapshot.
        Returns composite normalized OFI in [-1.0, 1.0].
        """
        if self.last_snapshot is None:
            self.last_snapshot = snapshot
            return 0.0

        prev = self.last_snapshot
        self.last_snapshot = snapshot
        
        k_max = min(
            self.depth_levels,
            len(snapshot.bid_prices),
            len(prev.bid_prices),
            len(snapshot.ask_prices),
            len(prev.ask_prices)
        )
        if k_max == 0:
            return 0.0

        ofi_levels = np.zeros(k_max)
        for k in range(k_max):
            # Bid side flow
            p_b_curr = snapshot.bid_prices[k]
            p_b_prev = prev.bid_prices[k]
            v_b_curr = snapshot.bid_volumes[k]
            v_b_prev = prev.bid_volumes[k]
            
            if p_b_curr > p_b_prev:
                delta_b = v_b_curr
            elif p_b_curr == p_b_prev:
                delta_b = v_b_curr - v_b_prev
            else:
                delta_b = -v_b_prev

            # Ask side flow
            p_a_curr = snapshot.ask_prices[k]
            p_a_prev = prev.ask_prices[k]
            v_a_curr = snapshot.ask_volumes[k]
            v_a_prev = prev.ask_volumes[k]

            if p_a_curr < p_a_prev:
                delta_a = v_a_curr
            elif p_a_curr == p_a_prev:
                delta_a = v_a_curr - v_a_prev
            else:
                delta_a = -v_a_prev

            ofi_levels[k] = delta_b - delta_a

        # Weighted aggregate OFI
        weights = self.level_weights[:k_max] / np.sum(self.level_weights[:k_max])
        raw_composite_ofi = np.sum(ofi_levels * weights)
        
        # Normalize by total depth
        tot_depth = (snapshot.total_bid_depth + snapshot.total_ask_depth) / 2.0
        normalized_ofi = float(np.tanh(raw_composite_ofi / max(1.0, tot_depth * 0.2)))
        
        return normalized_ofi


class VPIN:
    """
    Volume-Synchronized Probability of Toxicity (VPIN)
    
    Partitions trade stream into equal-volume buckets and measures buy/sell order imbalance.
    VPIN = sum(|V_tau^B - V_tau^S|) / (N * V_bucket)
    """
    def __init__(self, bucket_size: float = 500.0, n_buckets: int = 20):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        
        # State
        self.current_bucket_vol: float = 0.0
        self.current_bucket_buy_vol: float = 0.0
        self.current_bucket_sell_vol: float = 0.0
        self.completed_bucket_imbalances: List[float] = []

    def update_trade(self, price: float, volume: float, prev_price: float) -> float:
        """
        Process incoming trade and compute current rolling VPIN.
        Uses Lee-Ready trade classification (tick test).
        """
        # Determine buyer/seller initiated volume
        if price > prev_price:
            buy_frac = 0.8
        elif price < prev_price:
            buy_frac = 0.2
        else:
            buy_frac = 0.5

        vol_buy = volume * buy_frac
        vol_sell = volume * (1.0 - buy_frac)

        rem_vol = volume
        while rem_vol > 0:
            space = self.bucket_size - self.current_bucket_vol
            fill = min(rem_vol, space)
            
            frac = fill / volume
            self.current_bucket_buy_vol += vol_buy * frac
            self.current_bucket_sell_vol += vol_sell * frac
            self.current_bucket_vol += fill
            rem_vol -= fill

            if self.current_bucket_vol >= self.bucket_size:
                # Bucket completed
                imbalance = abs(self.current_bucket_buy_vol - self.current_bucket_sell_vol)
                self.completed_bucket_imbalances.append(imbalance)
                if len(self.completed_bucket_imbalances) > self.n_buckets:
                    self.completed_bucket_imbalances.pop(0)

                self.current_bucket_vol = 0.0
                self.current_bucket_buy_vol = 0.0
                self.current_bucket_sell_vol = 0.0

        return self.get_vpin()

    def get_vpin(self) -> float:
        """Calculate current VPIN score [0.0, 1.0]"""
        if not self.completed_bucket_imbalances:
            return 0.15  # Baseline default
        tot_imbalance = sum(self.completed_bucket_imbalances)
        total_vol = len(self.completed_bucket_imbalances) * self.bucket_size
        vpin_val = tot_imbalance / total_vol if total_vol > 0 else 0.15
        return float(np.clip(vpin_val, 0.0, 1.0))


class StoikovMicroPrice:
    """
    Stoikov Multi-Level Micro-Price Estimator
    
    P^{micro} = Mid + Imbalance * (Spread / 2)
    Extended with non-linear multi-level queue weighting.
    """
    @staticmethod
    def calculate(snapshot: L2Snapshot, decay_factor: float = 0.6) -> float:
        if len(snapshot.bid_prices) == 0 or len(snapshot.ask_prices) == 0:
            return snapshot.mid_price

        k_max = min(5, len(snapshot.bid_prices), len(snapshot.ask_prices))
        weights = np.array([decay_factor ** i for i in range(k_max)])
        weights /= np.sum(weights)

        weighted_bid_vol = np.sum(snapshot.bid_volumes[:k_max] * weights)
        weighted_ask_vol = np.sum(snapshot.ask_volumes[:k_max] * weights)
        
        tot = weighted_bid_vol + weighted_ask_vol
        if tot <= 1e-6:
            return snapshot.mid_price

        imbalance = (weighted_bid_vol - weighted_ask_vol) / tot
        micro_price = snapshot.mid_price + imbalance * (snapshot.spread / 2.0)
        return round(micro_price, 4)


class KylesLambda:
    """
    Kyle's Lambda (Price Impact Coefficient)
    
    Measures price change per unit of signed order flow:
    Delta P_t = lambda * SignedVolume_t + epsilon_t
    """
    def __init__(self, window_size: int = 50):
        self.window = window_size
        self.delta_p: List[float] = []
        self.signed_v: List[float] = []

    def update(self, price_change: float, signed_volume: float) -> float:
        self.delta_p.append(price_change)
        self.signed_v.append(signed_volume)
        if len(self.delta_p) > self.window:
            self.delta_p.pop(0)
            self.signed_v.pop(0)

        if len(self.delta_p) < 10:
            return 0.001  # Default baseline

        dp = np.array(self.delta_p)
        sv = np.array(self.signed_v)
        
        # OLS slope: Cov(dp, sv) / Var(sv)
        var_sv = np.var(sv)
        if var_sv < 1e-8:
            return 0.001
        cov = np.cov(dp, sv)[0, 1]
        lam = cov / var_sv
        return float(max(1e-5, lam))


class RollSpread:
    """
    Roll's Effective Spread Estimator
    
    Estimates effective bid-ask spread from serial autocovariance of price changes:
    Spread_{Roll} = 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1}))  if Cov < 0 else 0
    """
    def __init__(self, window_size: int = 50):
        self.window = window_size
        self.prices: List[float] = []

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.pop(0)

        if len(self.prices) < 15:
            return 0.02

        dp = np.diff(self.prices)
        if len(dp) < 2:
            return 0.02
            
        cov = np.cov(dp[:-1], dp[1:])[0, 1]
        if cov < 0:
            roll_spread = 2.0 * np.sqrt(-cov)
        else:
            roll_spread = 0.01  # Small positive bound
            
        return float(roll_spread)

"""
Swarm-Skewed Avellaneda-Stoikov Market Making Model
===================================================
High-Frequency Market Making with Biomimetic Mormyrid Swarm Intelligence Skew.

Extends the classical Avellaneda-Stoikov (2008) optimal inventory model:
1. Classical reservation price:
   r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
2. Swarm-Skewed reservation price:
   r_swarm(s, q, t) = r(s, q, t) + kappa_swarm * drift_consensus
3. JAR-modulated dynamic spread widening under high crowding/toxicity:
   delta_spread_jar = jar_crowding_index * max_widening
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Optional


@dataclass
class MarketMakerQuotes:
    """Optimal bid and ask quotes produced by Avellaneda-Stoikov engine"""
    mid_price: float
    reservation_price: float
    optimal_spread: float
    bid_price: float
    ask_price: float
    bid_depth_offset: float    # delta_b (cents or dollars)
    ask_depth_offset: float    # delta_a (cents or dollars)
    inventory: float
    target_inventory: float
    swarm_skew_bps: float
    jar_spread_expansion: float
    bid_fill_probability: float
    ask_fill_probability: float


class SwarmAvellanedaStoikov:
    """
    Institutional Avellaneda-Stoikov Market Maker with Swarm Skew and JAR protection.
    """
    def __init__(
        self,
        gamma: float = 0.1,             # Risk aversion parameter
        kappa: float = 1.5,             # Order arrival liquidity parameter
        sigma: float = 0.30,            # Asset volatility (annualized)
        time_horizon: float = 1.0,      # Trading session horizon (days)
        swarm_skew_multiplier: float = 1.2,
        jar_widening_factor: float = 0.05,
        max_inventory: int = 100,
        tick_size: float = 0.01,
    ):
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.time_horizon = time_horizon
        self.swarm_multiplier = swarm_skew_multiplier
        self.jar_widening = jar_widening_factor
        self.max_inventory = max_inventory
        self.tick_size = tick_size

    def calculate_quotes(
        self,
        mid_price: float,
        inventory: float,
        time_remaining: float,
        swarm_drift_bps: float = 0.0,
        jar_crowding_index: float = 0.0,
        adverse_selection_risk: float = 0.0,
        hawkes_intensity: float = 1.0,
    ) -> MarketMakerQuotes:
        """
        Compute optimal bid and ask quotes for current market & swarm state.
        
        Args:
            mid_price: Current LOB mid-price
            inventory: Current inventory position (-max_inventory to +max_inventory)
            time_remaining: Normalized time remaining in session (0.0 to 1.0)
            swarm_drift_bps: Predicted drift from Mormyrid Swarm Consensus in bps
            jar_crowding_index: Jamming Avoidance crowding index (0.0 to 1.0)
            adverse_selection_risk: Microstructure toxicity/adverse risk (0.0 to 1.0)
            hawkes_intensity: Order arrival rate multiplier
        """
        tau = max(1e-4, time_remaining)
        
        # 1. Classical reservation price: r(s, q, t) = s - q * gamma * sigma^2 * tau
        # Convert annualized sigma to session vol
        session_var = (self.sigma ** 2) * (tau / 252.0)
        inventory_penalty = inventory * self.gamma * session_var * mid_price
        r_classical = mid_price - inventory_penalty
        
        # 2. Swarm Drift Skew Enhancement:
        # Convert basis points drift into absolute price shift
        swarm_drift_dollar = (swarm_drift_bps / 10000.0) * mid_price * self.swarm_multiplier
        # Dampen drift skew if adverse selection risk is extreme
        effective_swarm_skew = swarm_drift_dollar * (1.0 - 0.7 * adverse_selection_risk)
        
        r_swarm = r_classical + effective_swarm_skew
        
        # 3. Optimal Half-Spreads with Hawkes & JAR Widening
        # Base AS spread: s_opt = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)
        base_spread = (self.gamma * session_var * mid_price) + (2.0 / self.gamma) * np.log(1.0 + self.gamma / max(0.01, self.kappa))
        
        # JAR crowding & toxicity spread widening
        jar_widen = (jar_crowding_index * 1.5 + adverse_selection_risk * 2.0) * self.jar_widening * mid_price
        optimal_spread = max(2 * self.tick_size, base_spread + jar_widen)
        
        half_spread = optimal_spread / 2.0
        
        # 4. Asymmetric Quote Calculation
        # delta_a = (r_swarm - mid) + half_spread
        # delta_b = (mid - r_swarm) + half_spread
        delta_a = (r_swarm - mid_price) + half_spread
        delta_b = (mid_price - r_swarm) + half_spread
        
        # Ensure half-spreads are at least one tick
        delta_a = max(self.tick_size, delta_a)
        delta_b = max(self.tick_size, delta_b)
        
        # Enforce inventory threshold boundaries
        if inventory >= self.max_inventory:
            # Extreme long: discourage buys (widen delta_b), aggressively lower ask to dump
            delta_b += 5 * self.tick_size
            delta_a = self.tick_size
        elif inventory <= -self.max_inventory:
            # Extreme short: discourage sells (widen delta_a), aggressively lift bid to cover
            delta_a += 5 * self.tick_size
            delta_b = self.tick_size
            
        raw_ask = mid_price + delta_a
        raw_bid = mid_price - delta_b
        
        # Round to tick size
        ask_quote = round(round(raw_ask / self.tick_size) * self.tick_size, 4)
        bid_quote = round(round(raw_bid / self.tick_size) * self.tick_size, 4)
        
        # Guarantee no negative spread
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - self.tick_size

        # 5. Fill Probabilities: P(fill) = 1 - exp(-A * exp(-k * delta))
        effective_kappa = self.kappa * (1.0 + 0.5 * hawkes_intensity)
        p_fill_ask = float(np.clip(np.exp(-effective_kappa * (delta_a / mid_price) * 100), 0.01, 0.95))
        p_fill_bid = float(np.clip(np.exp(-effective_kappa * (delta_b / mid_price) * 100), 0.01, 0.95))

        return MarketMakerQuotes(
            mid_price=mid_price,
            reservation_price=round(r_swarm, 4),
            optimal_spread=round(ask_quote - bid_quote, 4),
            bid_price=bid_quote,
            ask_price=ask_quote,
            bid_depth_offset=round(delta_b, 4),
            ask_depth_offset=round(delta_a, 4),
            inventory=inventory,
            target_inventory=0.0,
            swarm_skew_bps=swarm_drift_bps,
            jar_spread_expansion=round(jar_widen, 4),
            bid_fill_probability=p_fill_bid,
            ask_fill_probability=p_fill_ask,
        )

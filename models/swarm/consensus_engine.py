"""
Biomimetic Mormyrid Swarm Consensus Engine
===========================================
Decentralized multi-agent predictive engine orchestrating heterogeneous
Mormyrid fish agents.

Implements:
1. Multi-scale active electrolocation probing across Limit Order Book dynamics.
2. Inter-agent Jamming Avoidance Response (JAR).
3. Byzantine-resilient dynamic consensus voting with Kalman-filter confidence weighting.
4. Output signal synthesis: Micro-price drift, Hawkes jump probability, optimal MM skew.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from models.swarm.mormyrid_agent import (
    FishAgent,
    ScoutFishAgent,
    PredatorFishAgent,
    SchoolingFishAgent,
    SentinelFishAgent,
    AgentRole,
    AgentState,
)
from models.swarm.jamming_avoidance import JammingAvoidanceEngine


@dataclass
class SwarmConsensusSignal:
    """Consensus output synthesized by the Mormyrid Swarm Engine"""
    predicted_drift_bps: float        # Predicted mid-price drift in basis points (-50.0 to +50.0)
    jump_probability: float           # Probability of imminent order cascade / volatility jump (0.0 to 1.0)
    optimal_quote_skew: float         # Skew factor for Avellaneda-Stoikov reservation price
    market_crowding_index: float      # Jamming & queue congestion score (0.0 = low, 1.0 = extreme)
    adverse_selection_risk: float     # Toxicity & adverse fill risk (0.0 to 1.0)
    swarm_confidence: float           # Collective consensus confidence (0.0 to 1.0)
    dominant_regime: str              # Active regime: 'HAWKES_MOMENTUM', 'MEAN_REVERSION', 'TOXIC_DRAIN', 'STABLE_LOB'
    active_agent_count: int
    jammed_agent_ratio: float
    role_contributions: Dict[str, float]
    agent_states: List[AgentState]


class MormyridSwarmConsensusEngine:
    """
    Orchestrates the biomimetic swarm of weakly electric fish agents
    to generate real-time predictive microstructure signals.
    """
    def __init__(
        self,
        n_scouts: int = 6,
        n_predators: int = 8,
        n_schoolers: int = 10,
        n_sentinels: int = 4,
        feature_dim: int = 5,
        base_eod_frequency: float = 90.0,
        consensus_prune_alpha: float = 0.15, # Byzantine outlier trimming ratio
    ):
        self.feature_dim = feature_dim
        self.prune_alpha = consensus_prune_alpha
        self.jar_engine = JammingAvoidanceEngine()
        
        # Initialize heterogeneous agent population
        self.agents: List[FishAgent] = []
        agent_id = 0
        
        for _ in range(n_scouts):
            self.agents.append(ScoutFishAgent(agent_id=agent_id, feature_dim=feature_dim, base_frequency=base_eod_frequency + 10.0))
            agent_id += 1
            
        for _ in range(n_predators):
            self.agents.append(PredatorFishAgent(agent_id=agent_id, feature_dim=feature_dim, base_frequency=base_eod_frequency + 25.0))
            agent_id += 1
            
        for _ in range(n_schoolers):
            self.agents.append(SchoolingFishAgent(agent_id=agent_id, feature_dim=feature_dim, base_frequency=base_eod_frequency - 10.0))
            agent_id += 1
            
        for _ in range(n_sentinels):
            self.agents.append(SentinelFishAgent(agent_id=agent_id, feature_dim=feature_dim, base_frequency=base_eod_frequency - 25.0))
            agent_id += 1

        self.global_best_position = np.zeros(feature_dim)
        self.global_best_fitness = -np.inf
        
        # Kalman filter state for smoothing consensus drift
        self._kf_drift_estimate = 0.0
        self._kf_variance = 1.0

    def step_market_state(
        self,
        ofi: float,                     # Order Flow Imbalance (-1.0 to 1.0)
        vpin: float,                    # VPIN toxicity (0.0 to 1.0)
        hawkes_intensity: float,        # Current Hawkes arrival intensity
        micro_price_dev: float,         # (MicroPrice - MidPrice) / MidPrice in bps
        relative_spread: float,         # Spread / MidPrice in bps
        hawkes_branching_ratio: float = 0.65,
        iterations: int = 3,
    ) -> SwarmConsensusSignal:
        """
        Execute sensory perception and swarm consensus step for incoming LOB state.
        """
        # Construct true normalized market feature vector
        # [OFI, VPIN, Hawkes_Norm, Micro_Dev_Norm, Spread_Norm]
        norm_hawkes = float(np.clip(hawkes_intensity / 5.0, 0.0, 3.0))
        norm_micro_dev = float(np.clip(micro_price_dev / 5.0, -3.0, 3.0))
        norm_spread = float(np.clip(relative_spread / 2.0, 0.0, 3.0))
        
        market_features = np.array([
            np.clip(ofi, -1.0, 1.0),
            np.clip(vpin, 0.0, 1.0),
            norm_hawkes,
            norm_micro_dev,
            norm_spread,
        ])

        # Step 1: Active Electrolocation EOD Pulse Discharge
        for agent in self.agents:
            fitness = agent.emit_eod_pulse(market_features)
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = agent.position.copy()

        # Step 2: Jamming Avoidance Response (JAR)
        repulsion_vectors, crowding_index, jammed_ratio = self.jar_engine.process_swarm_jar(self.agents)

        # Step 3: Swarm Kinematics Update
        positions = np.array([a.position for a in self.agents])
        swarm_center = np.mean(positions, axis=0)

        for agent in self.agents:
            repulsion = repulsion_vectors.get(agent.agent_id, np.zeros(self.feature_dim))
            agent.update_kinematics(
                swarm_center=swarm_center,
                best_global_position=self.global_best_position,
                repulsion_vector=repulsion,
            )

        # Step 4: Byzantine-Resilient Consensus Aggregation
        consensus_signal = self._aggregate_consensus(
            market_features=market_features,
            crowding_index=crowding_index,
            jammed_ratio=jammed_ratio,
            hawkes_branching_ratio=hawkes_branching_ratio,
            vpin=vpin,
            ofi=ofi,
        )

        return consensus_signal

    def _aggregate_consensus(
        self,
        market_features: np.ndarray,
        crowding_index: float,
        jammed_ratio: float,
        hawkes_branching_ratio: float,
        vpin: float,
        ofi: float,
    ) -> SwarmConsensusSignal:
        """
        Aggregate agent votes using confidence-weighted trimmed means (Byzantine resilience)
        and Kalman filtering.
        """
        agent_states = [a.get_state() for a in self.agents]
        
        # Extract predictions and confidences
        drifts = np.array([s.local_drift_forecast for s in agent_states])
        confidences = np.array([s.confidence for s in agent_states])
        is_jammed = np.array([s.is_jammed for s in agent_states])
        
        # De-weight jammed agents (sensory degradation)
        effective_weights = confidences * np.where(is_jammed, 0.3, 1.0)
        
        if np.sum(effective_weights) > 0:
            effective_weights /= np.sum(effective_weights)
        else:
            effective_weights = np.ones(len(self.agents)) / len(self.agents)

        # Byzantine Trimming: Sort by drift forecast and trim top/bottom alpha percentile
        k_trim = int(len(self.agents) * self.prune_alpha)
        sorted_indices = np.argsort(drifts)
        
        if k_trim > 0 and len(sorted_indices) > 2 * k_trim:
            valid_indices = sorted_indices[k_trim:-k_trim]
        else:
            valid_indices = sorted_indices

        trimmed_drifts = drifts[valid_indices]
        trimmed_weights = effective_weights[valid_indices]
        if np.sum(trimmed_weights) > 0:
            trimmed_weights /= np.sum(trimmed_weights)
        else:
            trimmed_weights = np.ones_like(trimmed_drifts) / len(trimmed_drifts)

        # Raw consensus drift forecast (basis points)
        raw_consensus_drift = float(np.sum(trimmed_drifts * trimmed_weights) * 10.0)
        
        # Kalman filter update for drift smoothing
        # Process variance Q = 0.05, Measurement variance R = (1.0 - mean_conf) + 0.1
        mean_confidence = float(np.mean(confidences))
        q_var = 0.05
        r_var = max(0.05, 1.0 - mean_confidence)
        
        # Prior update
        p_prior = self._kf_variance + q_var
        # Kalman gain
        k_gain = p_prior / (p_prior + r_var)
        # Posterior update
        self._kf_drift_estimate += k_gain * (raw_consensus_drift - self._kf_drift_estimate)
        self._kf_variance = (1.0 - k_gain) * p_prior
        
        filtered_drift = float(self._kf_drift_estimate)

        # Calculate Hawkes Jump Probability
        # Higher if predators are excited (Hawkes intensity high) and branching ratio >= 0.85
        predator_drifts = [s.local_drift_forecast for s in agent_states if s.role == AgentRole.PREDATOR]
        predator_intensity = float(np.mean(np.abs(predator_drifts))) if predator_drifts else 0.5
        jump_prob = float(np.clip(
            0.4 * hawkes_branching_ratio + 0.4 * (market_features[2] / 2.0) + 0.2 * predator_intensity,
            0.0,
            0.99
        ))

        # Adverse Selection Risk
        adverse_risk = self.jar_engine.compute_adverse_selection_risk(
            vpin=vpin,
            hawkes_branching=hawkes_branching_ratio,
            order_imbalance=ofi,
        )

        # Optimal Quote Skew for Avellaneda-Stoikov
        # Positive skew shifts reservation price up (willing to buy higher, sell higher)
        # Negative skew shifts reservation price down
        # Dampened by adverse selection risk
        skew_factor = np.tanh(filtered_drift / 5.0) * (1.0 - 0.5 * adverse_risk)

        # Role contributions breakdown
        role_contributions = {}
        for role in AgentRole:
            role_drifts = [s.local_drift_forecast for s in agent_states if s.role == role]
            role_contributions[role.value] = float(np.mean(role_drifts)) if role_drifts else 0.0

        # Market Microstructural Regime Detection
        if jump_prob > 0.75 and abs(ofi) > 0.4:
            regime = "HAWKES_MOMENTUM"
        elif adverse_risk > 0.65 or vpin > 0.6:
            regime = "TOXIC_DRAIN"
        elif abs(filtered_drift) < 1.0 and market_features[4] > 0.5:
            regime = "MEAN_REVERSION"
        else:
            regime = "STABLE_LOB"

        return SwarmConsensusSignal(
            predicted_drift_bps=filtered_drift,
            jump_probability=jump_prob,
            optimal_quote_skew=float(skew_factor),
            market_crowding_index=crowding_index,
            adverse_selection_risk=adverse_risk,
            swarm_confidence=mean_confidence,
            dominant_regime=regime,
            active_agent_count=len(self.agents),
            jammed_agent_ratio=jammed_ratio,
            role_contributions=role_contributions,
            agent_states=agent_states,
        )

"""
Biomimetic Mormyrid Swarm Agents
=================================
Sensory fish agents inspired by the active electrolocation capabilities and
collective behavior of Mormyrid weakly electric fish (Mormyridae).

Each agent navigates the multi-dimensional Limit Order Book (LOB) feature space,
emitting Electric Organ Discharge (EOD) probe pulses to sense microstructural
curvature, order flow velocity, queue depletion, and toxicity.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple


class AgentRole(Enum):
    """Specialized biological role in the swarm"""
    SCOUT = "scout"          # Explores depth anomalies, spread expansion, hidden liquidity
    PREDATOR = "predator"    # Detects Hawkes jump bursts, momentum cascades, aggressive flow
    SCHOOLER = "schooler"    # Exploits queue imbalance, mean-reversion, market-making spread
    SENTINEL = "sentinel"    # Monitors VPIN toxicity, jamming risk, adverse selection


@dataclass
class AgentState:
    """Snapshot of a fish agent's sensory and dynamic state"""
    agent_id: int
    role: AgentRole
    position: np.ndarray          # Coordinates in feature space [OFI, VPIN, Hawkes_intensity, micro_drift, spread]
    velocity: np.ndarray          # Velocity in feature space
    eod_frequency: float          # Electric Organ Discharge frequency (Hz)
    pulse_amplitude: float        # Sensory probe power (0.0 to 1.0)
    perceived_fitness: float      # Local alpha / signal fitness
    local_drift_forecast: float   # Predicted price drift (basis points)
    confidence: float             # Agent self-confidence (0.0 to 1.0)
    is_jammed: bool = False       # Whether signal is jammed by nearby peers


class FishAgent:
    """
    Base Mormyrid Fish Agent with Active Electrolocation Sensing.
    """
    def __init__(
        self,
        agent_id: int,
        role: AgentRole,
        feature_dim: int = 5,
        base_frequency: float = 80.0,
        sensory_radius: float = 1.5,
        learning_rate: float = 0.05,
    ):
        self.agent_id = agent_id
        self.role = role
        self.feature_dim = feature_dim
        self.base_frequency = base_frequency
        self.frequency = base_frequency + np.random.normal(0, 2.0)
        self.sensory_radius = sensory_radius
        self.learning_rate = learning_rate
        
        # Position and velocity vectors in normalized feature space
        # Features: [0: OFI, 1: VPIN, 2: Hawkes_Intensity, 3: Stoikov_Micro_Dev, 4: Relative_Spread]
        self.position = np.random.uniform(-1.0, 1.0, size=feature_dim)
        self.velocity = np.random.uniform(-0.1, 0.1, size=feature_dim)
        
        # Internal memory and weights
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.confidence = 0.5
        self.pulse_amplitude = 1.0
        self.local_drift = 0.0
        self.is_jammed = False
        
        # Role-specific sensory weight bias
        self._init_role_weights()

    def _init_role_weights(self):
        """Configure feature sensitivity matrix based on agent role"""
        if self.role == AgentRole.SCOUT:
            # Scouts emphasize depth discovery, spread anomalies, and micro-deviations
            self.role_weights = np.array([0.2, 0.1, 0.1, 0.4, 0.2])
            self.sensory_radius *= 1.4  # Wider field
        elif self.role == AgentRole.PREDATOR:
            # Predators prioritize Hawkes jump intensities and directional OFI bursts
            self.role_weights = np.array([0.45, 0.05, 0.45, 0.05, 0.0])
            self.sensory_radius *= 1.1
        elif self.role == AgentRole.SCHOOLER:
            # Schoolers focus on queue imbalance, spread capture, and micro-price mean-reversion
            self.role_weights = np.array([0.35, 0.15, 0.1, 0.25, 0.15])
            self.sensory_radius *= 0.9  # Tighter schooling
        elif self.role == AgentRole.SENTINEL:
            # Sentinels focus on VPIN toxicity and risk avoidance
            self.role_weights = np.array([0.1, 0.55, 0.25, 0.05, 0.05])
            self.sensory_radius *= 1.2

    def emit_eod_pulse(self, market_feature_vector: np.ndarray) -> float:
        """
        Simulate Active Electrolocation pulse discharge into market feature field.
        Returns the perceived sensory return (fitness).
        """
        # Distance between agent position and true market feature state
        weighted_diff = (self.position - market_feature_vector) * self.role_weights
        dist_sq = np.sum(weighted_diff ** 2)
        
        # Electric field attenuation: 1 / (1 + r^2)
        field_strength = self.pulse_amplitude / (1.0 + dist_sq)
        
        # Compute role-specific directional signal
        if self.role == AgentRole.PREDATOR:
            # Directional momentum push
            drift_signal = market_feature_vector[0] * 1.5 + market_feature_vector[2] * 0.8
        elif self.role == AgentRole.SCHOOLER:
            # Mean-reverting micro-price convergence
            drift_signal = market_feature_vector[3] * 0.9 - market_feature_vector[0] * 0.2
        elif self.role == AgentRole.SCOUT:
            # Volatility expansion breakout
            drift_signal = np.sign(market_feature_vector[0]) * market_feature_vector[4] * 1.2
        else: # SENTINEL
            # Toxicity-penalized conservative estimate
            vpin_penalty = 1.0 - np.clip(market_feature_vector[1], 0.0, 1.0)
            drift_signal = market_feature_vector[0] * vpin_penalty * 0.5
            
        self.local_drift = float(drift_signal * field_strength)
        
        # Calculate fitness
        fitness = float(field_strength * (1.0 - 0.5 * market_feature_vector[1]))
        
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            
        self.confidence = float(np.clip(field_strength * 1.2, 0.1, 0.99))
        return fitness

    def update_kinematics(
        self,
        swarm_center: np.ndarray,
        best_global_position: np.ndarray,
        repulsion_vector: np.ndarray,
        step_size: float = 0.08,
    ):
        """
        Update agent position and velocity using swarm attraction, role exploration,
        and anti-crowding repulsion.
        """
        # Cognitive component (personal best)
        r1 = np.random.uniform(0.0, 1.0, size=self.feature_dim)
        v_cognitive = 0.8 * r1 * (self.best_position - self.position)
        
        # Social component (global best and swarm center)
        r2 = np.random.uniform(0.0, 1.0, size=self.feature_dim)
        v_social = 1.2 * r2 * (best_global_position - self.position)
        
        # Swarming cohesion
        v_cohesion = 0.3 * (swarm_center - self.position)
        
        # Anti-jamming / repulsion component (Jamming Avoidance Response)
        v_repulsion = 1.5 * repulsion_vector
        
        # Update velocity with inertia
        inertia = 0.6 if not self.is_jammed else 0.2
        self.velocity = (
            inertia * self.velocity
            + v_cognitive
            + v_social
            + v_cohesion
            + v_repulsion
        )
        
        # Velocity clamping
        max_v = 0.25
        v_norm = np.linalg.norm(self.velocity)
        if v_norm > max_v:
            self.velocity = (self.velocity / v_norm) * max_v
            
        # Update position
        self.position += self.velocity * step_size
        
        # Boundary bounding in [-2.5, 2.5]
        self.position = np.clip(self.position, -2.5, 2.5)

    def get_state(self) -> AgentState:
        """Return full state representation"""
        return AgentState(
            agent_id=self.agent_id,
            role=self.role,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            eod_frequency=self.frequency,
            pulse_amplitude=self.pulse_amplitude,
            perceived_fitness=self.best_fitness,
            local_drift_forecast=self.local_drift,
            confidence=self.confidence,
            is_jammed=self.is_jammed,
        )


class ScoutFishAgent(FishAgent):
    """Specialized Scout agent for depth and spread anomalies"""
    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.SCOUT, **kwargs)


class PredatorFishAgent(FishAgent):
    """Specialized Predator agent for Hawkes cascades and momentum bursts"""
    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.PREDATOR, **kwargs)


class SchoolingFishAgent(FishAgent):
    """Specialized Schooling agent for mean-reversion and market-making liquidity"""
    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.SCHOOLER, **kwargs)


class SentinelFishAgent(FishAgent):
    """Specialized Sentinel agent for toxicity detection and adverse selection"""
    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.SENTINEL, **kwargs)

"""
Jamming Avoidance Response (JAR) Engine
=======================================
Implements the biological Jamming Avoidance Response found in Mormyrid and
Gymnotiform weakly electric fish.

In quantitative market microstructure, signal jamming corresponds to:
1. Crowded trade congestion (order queue front-running, latency competition).
2. Sensory interference when too many agents cluster in identical feature regions.
3. Adverse selection risk when toxic flow overwhelms a localized price level.

JAR dynamically shifts agent discharge frequencies, introduces repulsive dispersion,
and computes a market crowding index to adjust execution skew and spread width.
"""

import numpy as np
from typing import List, Tuple, Dict
from models.swarm.mormyrid_agent import FishAgent


class JammingAvoidanceEngine:
    """
    Bio-inspired Jamming Avoidance Response (JAR) for Swarm Intelligence.
    """
    def __init__(
        self,
        jamming_frequency_threshold: float = 4.0,   # Hz difference threshold
        spatial_crowding_radius: float = 0.45,       # Distance in feature space
        frequency_shift_step: float = 1.2,           # Frequency adjustment rate (Hz)
        repulsion_strength: float = 0.7,             # Spatial dispersion gain
    ):
        self.jamming_threshold = jamming_frequency_threshold
        self.crowding_radius = spatial_crowding_radius
        self.shift_step = frequency_shift_step
        self.repulsion_strength = repulsion_strength
        
        # Historical metrics
        self.last_crowding_index: float = 0.0
        self.jammed_agent_count: int = 0

    def process_swarm_jar(
        self,
        agents: List[FishAgent]
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray], float]:
        """
        Process JAR across all pairs of agents in the swarm.
        
        Returns:
            repulsion_vectors: Dict[agent_id, np.ndarray] of spatial anti-jamming vectors
            crowding_index: Overall normalized market crowding index (0.0 = clear, 1.0 = heavy congestion)
            jammed_ratio: Proportion of jammed agents
        """
        n = len(agents)
        if n <= 1:
            return {}, 0.0, 0.0

        positions = np.array([agent.position for agent in agents])
        frequencies = np.array([agent.frequency for agent in agents])
        
        repulsion_vectors = {agent.agent_id: np.zeros_like(agent.position) for agent in agents}
        jammed_flags = [False] * n
        total_interactions = 0
        jammed_interactions = 0

        # Pairwise distance matrix in feature space
        diff_matrix = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, D)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)  # (N, N)

        # Pairwise frequency differences
        freq_diff_matrix = frequencies[:, np.newaxis] - frequencies[np.newaxis, :]  # (N, N)

        for i in range(n):
            agent_i = agents[i]
            for j in range(i + 1, n):
                agent_j = agents[j]
                spatial_dist = dist_matrix[i, j]
                delta_f = abs(freq_diff_matrix[i, j])
                
                total_interactions += 1
                
                # Check for sensory interference / jamming
                is_spatially_close = spatial_dist < self.crowding_radius
                is_frequency_jammed = delta_f < self.jamming_threshold
                
                if is_spatially_close or is_frequency_jammed:
                    jammed_interactions += 1
                    jammed_flags[i] = True
                    jammed_flags[j] = True
                    
                    # 1. Biological Frequency Shift (JAR):
                    # Higher frequency shifts upward, lower frequency shifts downward
                    if freq_diff_matrix[i, j] > 0:
                        agent_i.frequency += self.shift_step * np.exp(-spatial_dist)
                        agent_j.frequency -= self.shift_step * np.exp(-spatial_dist)
                    else:
                        agent_i.frequency -= self.shift_step * np.exp(-spatial_dist)
                        agent_j.frequency += self.shift_step * np.exp(-spatial_dist)
                        
                    # Frequency bounds [40.0 Hz, 200.0 Hz]
                    agent_i.frequency = float(np.clip(agent_i.frequency, 40.0, 200.0))
                    agent_j.frequency = float(np.clip(agent_j.frequency, 40.0, 200.0))

                    # 2. Spatial Dispersion Repulsion Vector:
                    # Normalized direction pointing away from neighboring peer
                    if spatial_dist > 1e-6:
                        direction = (positions[i] - positions[j]) / spatial_dist
                        # Repulsive force decays inversely with distance squared
                        magnitude = self.repulsion_strength / (spatial_dist ** 2 + 0.1)
                        repulsion_vectors[agent_i.agent_id] += direction * magnitude
                        repulsion_vectors[agent_j.agent_id] -= direction * magnitude
                    else:
                        # Perturb randomly if exact overlap
                        jitter = np.random.normal(0, 0.05, size=positions[i].shape)
                        repulsion_vectors[agent_i.agent_id] += jitter
                        repulsion_vectors[agent_j.agent_id] -= jitter

        # Update agent jammed status
        for i, agent in enumerate(agents):
            agent.is_jammed = jammed_flags[i]

        self.jammed_agent_count = sum(jammed_flags)
        crowding_index = float(jammed_interactions / max(1, total_interactions))
        self.last_crowding_index = crowding_index
        jammed_ratio = float(self.jammed_agent_count / n)

        return repulsion_vectors, crowding_index, jammed_ratio

    def compute_adverse_selection_risk(
        self,
        vpin: float,
        hawkes_branching: float,
        order_imbalance: float
    ) -> float:
        """
        Compute aggregate adverse selection risk factor combining JAR crowding,
        VPIN toxicity, and Hawkes point-process avalanche risk.
        """
        # Crowding risk
        crowd_risk = self.last_crowding_index
        # Hawkes supercriticality risk (branching ratio approaching or exceeding 1.0)
        hawkes_risk = float(np.clip(hawkes_branching / 1.2, 0.0, 1.0))
        # VPIN toxicity risk
        vpin_risk = float(np.clip(vpin, 0.0, 1.0))
        # Directional imbalance extremity
        imbalance_risk = float(abs(order_imbalance))

        # Composite adverse selection score [0.0, 1.0]
        composite_risk = (
            0.35 * vpin_risk
            + 0.25 * hawkes_risk
            + 0.25 * crowd_risk
            + 0.15 * imbalance_risk
        )
        return float(np.clip(composite_risk, 0.0, 1.0))

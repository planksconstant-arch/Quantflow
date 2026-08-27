"""
QuantFlow Biomimetic Swarm Module
==================================
Swarm intelligence and consensus algorithms inspired by Mormyrid weakly electric
fish active electrolocation and Artificial Fish Swarm Algorithms (AFSA).
"""

from models.swarm.mormyrid_agent import (
    AgentRole,
    AgentState,
    FishAgent,
    ScoutFishAgent,
    PredatorFishAgent,
    SchoolingFishAgent,
    SentinelFishAgent,
)
from models.swarm.jamming_avoidance import JammingAvoidanceEngine
from models.swarm.consensus_engine import (
    MormyridSwarmConsensusEngine,
    SwarmConsensusSignal,
)
from models.swarm.afsa import ArtificialFishSwarmOptimizer

__all__ = [
    "AgentRole",
    "AgentState",
    "FishAgent",
    "ScoutFishAgent",
    "PredatorFishAgent",
    "SchoolingFishAgent",
    "SentinelFishAgent",
    "JammingAvoidanceEngine",
    "MormyridSwarmConsensusEngine",
    "SwarmConsensusSignal",
    "ArtificialFishSwarmOptimizer",
]
